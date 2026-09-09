import argparse
import hashlib
import json
import os
import sys
import tempfile
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_map import (
    ACTIVATION_GEOMETRY_ID_SPACE,
    ACTIVATION_GEOMETRY_SCHEMA_VERSION,
    build_fsm_map,
    expand_log_scales,
    gaussians_to_covariances,
    load_activation_geometry,
    load_fsm_map,
    load_splatam_params,
    normalize_quaternions,
    save_fsm_map,
    selected_ids_sha256,
)
from utils.fsm_observability import (
    compare_replacement_information,
    build_frustum_center_factors,
    load_reference_views,
    select_observable_gaussians,
)
from utils.fsm_pruning import score_opacity, score_scale_opacity, select_topk
from utils.fsm_robust_registration import estimate_normals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory-observable Gaussian selection artifact.")
    parser.add_argument("--params", required=True, help="SplaTAM params.npz with estimated camera poses and keyframes.")
    parser.add_argument("--output", required=True, help="Selection artifact .npz path (not written during --dry-run).")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--count", type=int, default=None, help="Number of Gaussians to retain.")
    budget.add_argument("--ratio", type=float, default=0.2, help="Fraction of Gaussians to retain.")
    parser.add_argument("--map-output", default=None, help="Optional legacy-compatible FSM map output path.")
    parser.add_argument(
        "--activation-geometry-output",
        default=None,
        help="Optional runtime activation geometry sidecar .npz path; requires --map-output.",
    )
    parser.add_argument(
        "--reserve-count",
        type=int,
        default=4096,
        help="Maximum sidecar-only reserve Gaussians outside the selected map; 0 disables reserve.",
    )
    parser.add_argument("--all-estimated-poses", action="store_true", help="Use all estimated poses instead of params keyframes.")
    parser.add_argument("--max-reference-poses", type=int, default=64, help="Uniformly subsample estimated reference keyframes.")
    parser.add_argument("--pair-cap", type=int, default=200_000)
    parser.add_argument("--per-pose-cap", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--candidate-cap", type=int, default=4096)
    parser.add_argument("--greedy-core-cap", type=int, default=128)
    parser.add_argument(
        "--replacement-count", type=int, default=None,
        help="Exact number removed from the cheap top budget and replaced by outside challengers. "
        "Defaults to greedy-core-cap, except cheap mode (core 0) defaults to 0.",
    )
    parser.add_argument("--min-greedy-fraction", type=float, default=0.0)
    parser.add_argument("--fill-strategy", choices=("opacity", "scale_opacity"), default="scale_opacity")
    parser.add_argument("--normal-k", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--min-eigen-every", type=int, default=16)
    parser.add_argument("--min-eigen-bonus", type=float, default=0.1)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating directories or files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a verified matching selection/map pair.")
    parser.add_argument("--save-factors", action="store_true", help="Include raw visibility factors and full scores.")
    return parser.parse_args(argv)


def resolve_budget(num_gaussians: int, count: int | None, ratio: float | None) -> int:
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        return min(int(count), num_gaussians)
    if ratio is None or ratio <= 0:
        raise ValueError("ratio must be positive")
    return min(max(int(np.ceil(float(ratio) * num_gaussians)), 1), num_gaussians)


def estimate_plan(
    num_gaussians: int,
    num_poses: int,
    budget: int,
    pair_cap: int | None,
    candidate_cap: int | None,
    greedy_core_cap: int | None,
    replacement_count: int,
) -> dict[str, Any]:
    available_factor_pool = int(replacement_count) + max(int(num_gaussians) - int(budget), 0)
    configured_cap = available_factor_pool if candidate_cap is None else int(candidate_cap)
    candidates = min(available_factor_pool, configured_cap) if replacement_count else 0
    possible_pairs = int(candidates) * int(num_poses)
    bounded_pairs = possible_pairs if pair_cap is None else min(possible_pairs, int(pair_cap))
    core_budget = int(replacement_count)
    factor_bytes = bounded_pairs * (8 + 6 * 8 + 8) + (num_poses + 1) * 8
    candidate_scans = int(candidates) * int(num_poses)
    factor_recomputations = int(candidates) + int(core_budget) * max(int(candidates) - 1, 0)
    return {
        "num_gaussians": int(num_gaussians),
        "num_reference_poses": int(num_poses),
        "budget": int(budget),
        "possible_visibility_pairs": int(possible_pairs),
        "bounded_visibility_pairs": int(bounded_pairs),
        "candidate_count": int(candidates),
        "candidate_cap": candidate_cap,
        "observable_core_count": int(core_budget),
        "cheap_fill_count": int(budget - core_budget),
        "replacement_count": int(replacement_count),
        "observable_core_fraction": float(core_budget / budget),
        "greedy_core_budget": int(core_budget),
        "greedy_fraction": float(core_budget / budget),
        "algorithm_is_exact_full_greedy": False,
        "objective": "sum_per_pose_logdet",
        "estimated_factor_memory_bytes": int(factor_bytes),
        "estimated_factor_memory_mib": float(factor_bytes / (1024**2)),
        "estimated_candidate_scans": candidate_scans,
        "factor_recomputations_upper_bound": factor_recomputations,
        "complexity": {
            "visibility": f"O({num_poses} * {candidates}) projection tests over bounded candidates",
            "selection_upper_bound": f"lazy greedy, at most {factor_recomputations} candidate gain recomputations",
        },
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _params_record(params_path: str, controls: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(controls, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "params_path": os.path.abspath(params_path),
        "params_sha256": _sha256(params_path),
        "controls": controls,
        "controls_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _verified_existing(
    output_path: str,
    map_output: str | None,
    activation_geometry_output: str | None,
    done_path: str,
    record: dict[str, Any],
) -> bool:
    required_paths = [done_path, output_path]
    if map_output:
        required_paths.append(map_output)
    if activation_geometry_output:
        required_paths.append(activation_geometry_output)
    if not all(os.path.isfile(path) for path in required_paths):
        return False
    try:
        with open(done_path) as stream:
            done = json.load(stream)
        if done.get("provenance") != record:
            return False
        with np.load(output_path, allow_pickle=False) as artifact:
            indices = np.asarray(artifact["selected_indices"], dtype=np.int64)
            if indices.size != int(done["selected_count"]):
                return False
        if map_output:
            fsm_map = load_fsm_map(map_output)
            if not np.array_equal(np.asarray(fsm_map["indices"], dtype=np.int64), indices):
                return False
            if activation_geometry_output:
                load_activation_geometry(
                    activation_geometry_output,
                    fsm_map,
                    source_params_sha256=record["params_sha256"],
                    selection_controls_sha256=record["controls_sha256"],
                )
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _enforce_safe_plan(plan: dict[str, Any]) -> None:
    limits = {
        "estimated_candidate_scans": int(os.environ.get("MAX_CANDIDATE_SCANS", "500000")),
        "factor_recomputations_upper_bound": int(os.environ.get("MAX_FACTOR_RECOMPUTATIONS", "1000000")),
        "estimated_factor_memory_bytes": int(os.environ.get("MAX_FACTOR_MEMORY_BYTES", str(512 * 1024**2))),
    }
    plan["safe_thresholds"] = limits
    exceeded = {key: (int(plan[key]), limit) for key, limit in limits.items() if int(plan[key]) > limit}
    plan["safe_thresholds_exceeded"] = exceeded
    if exceeded and os.environ.get("ALLOW_EXPENSIVE") != "1":
        details = ", ".join(f"{key}={value}>{limit}" for key, (value, limit) in exceeded.items())
        raise ValueError(f"observable plan exceeds safe thresholds: {details}; set ALLOW_EXPENSIVE=1 to override")


def validate_controls(
    pair_cap: int | None,
    per_pose_cap: int | None,
    chunk_size: int,
    candidate_cap: int | None,
    greedy_core_cap: int | None,
    min_greedy_fraction: float,
    normal_k: int,
    regularization: float,
    min_eigen_every: int,
    min_eigen_bonus: float,
    min_depth: float,
    max_depth: float | None,
    replacement_count: int | None,
    reserve_count: int,
) -> None:
    if pair_cap is not None and pair_cap <= 0:
        raise ValueError("pair-cap must be positive")
    if per_pose_cap is not None and per_pose_cap <= 0:
        raise ValueError("per-pose-cap must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if candidate_cap is not None and candidate_cap <= 0:
        raise ValueError("candidate-cap must be positive")
    if greedy_core_cap is not None and greedy_core_cap < 0:
        raise ValueError("greedy-core-cap must be non-negative")
    if replacement_count is not None and replacement_count < 0:
        raise ValueError("replacement-count must be non-negative")
    if reserve_count < 0:
        raise ValueError("reserve-count must be non-negative")
    if not 0.0 <= min_greedy_fraction <= 1.0:
        raise ValueError("min-greedy-fraction must be in [0, 1]")
    if normal_k < 3:
        raise ValueError("normal-k must be at least 3")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if min_eigen_every < 0 or min_eigen_bonus < 0.0:
        raise ValueError("min-eigen controls must be non-negative")
    if min_depth < 0.0 or (max_depth is not None and max_depth <= min_depth):
        raise ValueError("depth limits must satisfy 0 <= min-depth < max-depth")


def _cheap_scores(params: dict[str, Any], strategy: str) -> np.ndarray:
    if strategy == "opacity":
        return score_opacity(params)
    if strategy == "scale_opacity":
        return score_scale_opacity(params)
    raise ValueError(f"Unsupported fill strategy: {strategy}")


def _stable_fill(scores: np.ndarray, excluded: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[np.asarray(excluded, dtype=np.int64)] = False
    eligible = np.flatnonzero(mask)
    order = np.argsort(-np.asarray(scores, dtype=np.float64)[eligible], kind="mergesort")
    return eligible[order[:count]].astype(np.int64)


CANDIDATE_SOURCE_REMOVED = 0
CANDIDATE_SOURCE_QUALITY = 1
CANDIDATE_SOURCE_SPATIAL = 2
CANDIDATE_SOURCE_CHEAP_FILL = 3
CANDIDATE_SOURCE_NAMES = ("removed", "quality_frontier", "spatial_voxel", "cheap_fill")
RESERVE_POLICY = "positive_trajectory_trace_then_outside_quality_spatial_voxel_quota"


def _deterministic_spatial_representatives(
    means: np.ndarray,
    eligible_indices: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> np.ndarray:
    """Choose stable, spatially spread representatives from global eligible IDs."""
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if count <= 0 or eligible.size == 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), eligible.size)
    points = np.asarray(means, dtype=np.float64)[eligible]
    span = np.ptp(points, axis=0)
    positive = span[span > 0]
    if positive.size == 0:
        order = np.lexsort((eligible, -np.asarray(scores)[eligible]))
        return eligible[order[:count]]
    volume = float(np.prod(np.maximum(span, np.min(positive))))
    voxel_size = max((volume / max(count, 1)) ** (1.0 / 3.0), 1e-12)
    grid = np.floor((points - np.min(points, axis=0)) / voxel_size).astype(np.int64)
    quality_order = np.lexsort((eligible, -np.asarray(scores, dtype=np.float64)[eligible]))
    representatives: dict[tuple[int, int, int], int] = {}
    for local_id in quality_order:
        voxel = tuple(int(value) for value in grid[local_id])
        representatives.setdefault(voxel, int(eligible[local_id]))
    ordered = np.asarray([representatives[key] for key in sorted(representatives)], dtype=np.int64)
    if ordered.size > count:
        positions = np.linspace(0, ordered.size - 1, count, dtype=np.int64)
        ordered = ordered[positions]
    return ordered


def build_forced_replacement_candidates(
    means: np.ndarray,
    cheap_scores: np.ndarray,
    baseline_indices: np.ndarray,
    replacement_count: int,
    candidate_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build removed/challenger factor IDs and stable source codes."""
    baseline = np.asarray(baseline_indices, dtype=np.int64)
    replacement_count = int(replacement_count)
    if replacement_count <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    if candidate_cap < 2 * replacement_count:
        raise ValueError("candidate-cap must be at least 2 * replacement-count")
    removed = baseline[-replacement_count:]
    outside_mask = np.ones(means.shape[0], dtype=bool)
    outside_mask[baseline] = False
    outside = np.flatnonzero(outside_mask)
    if outside.size < replacement_count:
        raise ValueError("replacement-count exceeds the number of Gaussians outside the cheap baseline")
    outside_order = outside[np.lexsort((outside, -np.asarray(cheap_scores, dtype=np.float64)[outside]))]
    challenger_count = min(outside.size, int(candidate_cap) - replacement_count)
    quality_count = (challenger_count + 1) // 2
    spatial_count = challenger_count - quality_count
    quality = outside_order[:quality_count]
    spatial_pool = outside[~np.isin(outside, quality)]
    spatial = _deterministic_spatial_representatives(means, spatial_pool, cheap_scores, spatial_count)
    challengers = np.concatenate([quality, spatial])
    sources = np.concatenate([
        np.full(quality.size, CANDIDATE_SOURCE_QUALITY, dtype=np.int64),
        np.full(spatial.size, CANDIDATE_SOURCE_SPATIAL, dtype=np.int64),
    ])
    if challengers.size < challenger_count:
        present = np.zeros(means.shape[0], dtype=bool)
        present[challengers] = True
        fill = outside_order[~present[outside_order]][: challenger_count - challengers.size]
        challengers = np.concatenate([challengers, fill])
        sources = np.concatenate([sources, np.full(fill.size, CANDIDATE_SOURCE_CHEAP_FILL, dtype=np.int64)])
    if np.unique(challengers).size != challengers.size or np.any(np.isin(challengers, baseline)):
        raise AssertionError("challengers must be unique and strictly outside the cheap baseline")

    factor_ids: list[int] = []
    factor_sources: list[int] = []
    for index in range(max(removed.size, challengers.size)):
        if index < removed.size:
            factor_ids.append(int(removed[index]))
            factor_sources.append(CANDIDATE_SOURCE_REMOVED)
        if index < challengers.size:
            factor_ids.append(int(challengers[index]))
            factor_sources.append(int(sources[index]))
    return (
        removed,
        challengers,
        np.asarray(factor_ids, dtype=np.int64),
        np.asarray(factor_sources, dtype=np.int64),
    )


def _quota_select(
    means: np.ndarray,
    eligible: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.asarray(eligible, dtype=np.int64)
    if count <= 0 or eligible.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    count = min(int(count), eligible.size)
    quality_count = (count + 1) // 2
    order = eligible[np.lexsort((eligible, -np.asarray(scores, dtype=np.float64)[eligible]))]
    quality = order[:quality_count]
    spatial_pool = eligible[~np.isin(eligible, quality)]
    spatial = _deterministic_spatial_representatives(
        means, spatial_pool, scores, count - quality.size
    )
    selected = np.concatenate([quality, spatial])
    sources = np.concatenate([
        np.full(quality.size, CANDIDATE_SOURCE_QUALITY, dtype=np.int64),
        np.full(spatial.size, CANDIDATE_SOURCE_SPATIAL, dtype=np.int64),
    ])
    if selected.size < count:
        remaining = order[~np.isin(order, selected)][: count - selected.size]
        selected = np.concatenate([selected, remaining])
        sources = np.concatenate([
            sources, np.full(remaining.size, CANDIDATE_SOURCE_CHEAP_FILL, dtype=np.int64)
        ])
    return selected, sources


def select_reserve_gaussians(
    means: np.ndarray,
    cheap_scores: np.ndarray,
    selected_global_indices: np.ndarray,
    baseline_indices: np.ndarray,
    challenger_indices: np.ndarray,
    factor_global_ids: np.ndarray,
    candidate_source_codes: np.ndarray,
    candidate_trace_scores: np.ndarray | None,
    reserve_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a deterministic sidecar-only reserve without changing the base map."""
    count = min(int(reserve_count), means.shape[0] - selected_global_indices.size)
    if count <= 0:
        empty_ids = np.empty(0, dtype=np.int64)
        return empty_ids, empty_ids, np.empty(0, dtype=np.float64)

    selected_mask = np.zeros(means.shape[0], dtype=bool)
    selected_mask[np.asarray(selected_global_indices, dtype=np.int64)] = True
    trace_by_global = np.zeros(means.shape[0], dtype=np.float64)
    source_by_global = np.full(means.shape[0], CANDIDATE_SOURCE_CHEAP_FILL, dtype=np.int64)
    if candidate_trace_scores is not None:
        factor_ids = np.asarray(factor_global_ids, dtype=np.int64)
        trace_by_global[factor_ids] = np.asarray(candidate_trace_scores, dtype=np.float64)
        source_by_global[factor_ids] = np.asarray(candidate_source_codes, dtype=np.int64)

    challengers = np.asarray(challenger_indices, dtype=np.int64)
    positive = challengers[(~selected_mask[challengers]) & (trace_by_global[challengers] > 0.0)]
    reserve_ids, _ = _quota_select(means, positive, trace_by_global, min(count, positive.size))
    reserve_sources = source_by_global[reserve_ids]
    if reserve_ids.size < count:
        unavailable = selected_mask.copy()
        unavailable[np.asarray(baseline_indices, dtype=np.int64)] = True
        unavailable[reserve_ids] = True
        outside = np.flatnonzero(~unavailable)
        fill_ids, fill_sources = _quota_select(
            means, outside, cheap_scores, count - reserve_ids.size
        )
        reserve_ids = np.concatenate([reserve_ids, fill_ids])
        reserve_sources = np.concatenate([reserve_sources, fill_sources])
    if np.unique(reserve_ids).size != reserve_ids.size or np.any(selected_mask[reserve_ids]):
        raise AssertionError("reserve IDs must be unique and strictly outside the selected map")
    return reserve_ids, reserve_sources, trace_by_global[reserve_ids]


def _reserve_normals(
    params: dict[str, Any], means: np.ndarray, reserve_indices: np.ndarray, normal_k: int
) -> tuple[np.ndarray, str]:
    if reserve_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), "empty_reserve_set"
    for key in ("normals", "map_normals", "means3D_normals"):
        if key in params:
            normals = np.asarray(params[key], dtype=np.float64)
            if normals.shape != means.shape:
                raise ValueError(f"Explicit {key} must have shape {means.shape}, got {normals.shape}")
            return normals[reserve_indices], f"params:{key}"
    if reserve_indices.size < 3:
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (reserve_indices.size, 1))
        return normals, "reserve_small_set_default_z"
    normals = estimate_normals(means[reserve_indices], k=min(int(normal_k), reserve_indices.size))
    return normals, "bounded_reserve_local_pca"


def _candidate_normals(
    params: dict[str, Any], means: np.ndarray, candidate_indices: np.ndarray, normal_k: int
) -> tuple[np.ndarray, str]:
    for key in ("normals", "map_normals", "means3D_normals"):
        if key in params:
            normals = np.asarray(params[key], dtype=np.float64)
            if normals.shape != means.shape:
                raise ValueError(f"Explicit {key} must have shape {means.shape}, got {normals.shape}")
            return normals[candidate_indices], f"params:{key}"
    candidate_means = means[candidate_indices]
    if candidate_means.shape[0] < 3:
        raise ValueError("At least three candidates are required for local PCA normals")
    normals = estimate_normals(candidate_means, k=min(int(normal_k), candidate_means.shape[0]))
    return normals, "bounded_candidate_local_pca"


def _build_activation_geometry(
    params: dict[str, Any],
    selected_global_indices: np.ndarray,
    baseline_indices: np.ndarray,
    preserved_indices: np.ndarray,
    replacement_indices: np.ndarray,
    removed_indices: np.ndarray,
    removed_normals: np.ndarray,
    normal_source: str,
    reserve_indices: np.ndarray,
    reserve_normals: np.ndarray,
    reserve_source_codes: np.ndarray,
    reserve_trace_scores: np.ndarray,
    reserve_normal_source: str,
    cheap_scores: np.ndarray,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    log_scales = np.asarray(params["log_scales"], dtype=np.float64)[removed_indices]
    rotations = normalize_quaternions(np.asarray(params["unnorm_rotations"], dtype=np.float64)[removed_indices])
    map_local_by_global = {int(global_id): local_id for local_id, global_id in enumerate(selected_global_indices)}
    replacement_local_ids = np.asarray(
        [map_local_by_global[int(global_id)] for global_id in replacement_indices], dtype=np.int64
    )
    reserve_log_scales = np.asarray(params["log_scales"], dtype=np.float64)[reserve_indices]
    reserve_rotations = normalize_quaternions(
        np.asarray(params["unnorm_rotations"], dtype=np.float64)[reserve_indices]
    )
    return {
        "schema_version": np.asarray(ACTIVATION_GEOMETRY_SCHEMA_VERSION, dtype=np.int64),
        "id_space": np.asarray(ACTIVATION_GEOMETRY_ID_SPACE),
        "selected_global_ids": np.asarray(selected_global_indices, dtype=np.int64),
        "baseline_global_ids": np.asarray(baseline_indices, dtype=np.int64),
        "preserved_global_ids": np.asarray(preserved_indices, dtype=np.int64),
        "replacement_global_ids": np.asarray(replacement_indices, dtype=np.int64),
        "removed_global_ids": np.asarray(removed_indices, dtype=np.int64),
        "removed_means": np.asarray(params["means3D"], dtype=np.float64)[removed_indices].astype(np.float32),
        "removed_log_scales": log_scales.astype(np.float32),
        "removed_rotations": rotations.astype(np.float32),
        "removed_scales": np.exp(expand_log_scales(log_scales)).astype(np.float32),
        "removed_covariances": gaussians_to_covariances(log_scales, rotations).astype(np.float32),
        "removed_scores": np.asarray(cheap_scores, dtype=np.float64)[removed_indices].astype(np.float32),
        "removed_normals": np.asarray(removed_normals, dtype=np.float32),
        "normal_source": np.asarray(normal_source),
        "reserve_global_ids": np.asarray(reserve_indices, dtype=np.int64),
        "reserve_means": np.asarray(params["means3D"], dtype=np.float64)[reserve_indices].astype(np.float32),
        "reserve_normals": np.asarray(reserve_normals, dtype=np.float32),
        "reserve_scores": np.asarray(cheap_scores, dtype=np.float64)[reserve_indices].astype(np.float32),
        "reserve_covariances": gaussians_to_covariances(
            reserve_log_scales, reserve_rotations
        ).astype(np.float32),
        "reserve_scales": np.exp(expand_log_scales(reserve_log_scales)).astype(np.float32),
        "reserve_rotations": reserve_rotations.astype(np.float32),
        "reserve_source_codes": np.asarray(reserve_source_codes, dtype=np.int64),
        "reserve_source_names": np.asarray(CANDIDATE_SOURCE_NAMES),
        "reserve_trace_scores": np.asarray(reserve_trace_scores, dtype=np.float32),
        "reserve_count": np.asarray(reserve_indices.size, dtype=np.int64),
        "reserve_policy": np.asarray(RESERVE_POLICY),
        "reserve_normal_source": np.asarray(reserve_normal_source),
        "source_params_sha256": np.asarray(provenance["params_sha256"]),
        "selection_controls_sha256": np.asarray(provenance["controls_sha256"]),
        "map_selected_ids_sha256": np.asarray(selected_ids_sha256(selected_global_indices)),
        "replacement_selected_local_ids": replacement_local_ids,
    }


def build_observable_selection(
    params_path: str,
    output_path: str,
    count: int | None = None,
    ratio: float | None = 0.2,
    map_output: str | None = None,
    activation_geometry_output: str | None = None,
    reserve_count: int = 4096,
    keyframes_only: bool = True,
    max_reference_poses: int | None = 64,
    pair_cap: int | None = 200_000,
    per_pose_cap: int | None = None,
    chunk_size: int = 65_536,
    candidate_cap: int | None = 4096,
    greedy_core_cap: int | None = 128,
    replacement_count: int | None = None,
    min_greedy_fraction: float = 0.0,
    fill_strategy: str = "scale_opacity",
    normal_k: int = 20,
    regularization: float = 1e-6,
    min_eigen_every: int = 16,
    min_eigen_bonus: float = 0.1,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    save_factors: bool = False,
) -> dict[str, Any]:
    if activation_geometry_output is not None and map_output is None:
        raise ValueError("activation-geometry-output requires map-output")
    validate_controls(
        pair_cap,
        per_pose_cap,
        chunk_size,
        candidate_cap,
        greedy_core_cap,
        min_greedy_fraction,
        normal_k,
        regularization,
        min_eigen_every,
        min_eigen_bonus,
        min_depth,
        max_depth,
        replacement_count,
        reserve_count,
    )
    params = load_splatam_params(params_path)
    means = np.asarray(params["means3D"], dtype=np.float64)
    cheap_scores = _cheap_scores(params, fill_strategy)
    if cheap_scores.shape != (means.shape[0],):
        raise ValueError(f"Expected cheap scores with shape [{means.shape[0]}], got {cheap_scores.shape}")
    views = load_reference_views(params, keyframes_only=keyframes_only, max_reference_poses=max_reference_poses)
    budget = resolve_budget(means.shape[0], count=count, ratio=None if count is not None else ratio)
    replacement_count_was_default = replacement_count is None
    if replacement_count is None:
        configured_replacements = budget if greedy_core_cap is None else int(greedy_core_cap)
        replacement_count = 0 if configured_replacements == 0 else min(
            configured_replacements, budget, means.shape[0] - budget
        )
    replacement_count = int(replacement_count)
    if replacement_count > budget:
        raise ValueError("replacement-count must not exceed the cheap top budget")
    if replacement_count > means.shape[0] - budget:
        raise ValueError("replacement-count must not exceed the number of Gaussians outside the cheap top budget")
    factor_cap = means.shape[0] if candidate_cap is None else int(candidate_cap)
    if replacement_count > 0 and factor_cap < 2 * replacement_count:
        raise ValueError("candidate-cap must be at least 2 * replacement-count")
    plan = estimate_plan(
        means.shape[0], views.w2c.shape[0], budget, pair_cap, candidate_cap, greedy_core_cap, replacement_count
    )
    _enforce_safe_plan(plan)
    if plan["observable_core_fraction"] < min_greedy_fraction:
        raise ValueError(
            "observable_core_fraction="
            f"{plan['observable_core_fraction']:.6g} is below min_greedy_fraction={min_greedy_fraction:.6g}"
        )
    plan.update(
        {
            "dry_run": bool(dry_run),
            "reference_source": views.source,
            "reference_pose_role": "mapping_estimated_keyframes",
            "held_out_query": False,
            "uses_future_estimated_query_poses": True,
            "uses_query_future_ground_truth": False,
            "params_path": params_path,
            "selection_output": output_path,
            "map_output": map_output,
            "activation_geometry_output": activation_geometry_output,
            "reserve_count": int(reserve_count),
            "reserve_policy": RESERVE_POLICY,
            "pair_cap": pair_cap,
            "per_pose_cap": per_pose_cap,
            "max_reference_poses": max_reference_poses,
            "chunk_size": int(chunk_size),
            "greedy_core_cap": greedy_core_cap,
            "replacement_count": replacement_count,
            "replacement_count_default": "greedy_core_cap_unless_cheap_core_zero",
            "replacement_count_was_default": replacement_count_was_default,
            "cheap_baseline_budget": int(budget),
            "cheap_baseline_semantics": "B0_is_stable_cheap_top_budget",
            "min_greedy_fraction": float(min_greedy_fraction),
            "min_greedy_fraction_semantics": "minimum_observable_core_fraction",
            "fill_strategy": fill_strategy,
            "score_name": fill_strategy,
            "normal_k": int(normal_k),
            "regularization": float(regularization),
            "min_eigen_every": int(min_eigen_every),
            "min_eigen_bonus": float(min_eigen_bonus),
            "min_depth": float(min_depth),
            "max_depth": max_depth,
            "selection_mode": "cheap_baseline" if replacement_count == 0 else "observable_forced_replacement",
            "candidate_pool_scope": "removed_plus_outside_challengers_total_cap",
            "challenger_policy": "half_cheap_quality_frontier_half_deterministic_spatial_voxel_then_cheap_fill",
            "challenger_quality_fraction": 0.5,
            "challenger_spatial_strategy": "deterministic_voxel_representatives",
            "selection_algorithm_version": 2,
            "save_factors": bool(save_factors),
        }
    )
    controls = {
        key: plan[key]
        for key in (
            "budget", "pair_cap", "per_pose_cap", "max_reference_poses", "chunk_size", "candidate_cap",
            "greedy_core_cap", "replacement_count", "replacement_count_was_default", "min_greedy_fraction",
            "fill_strategy", "normal_k", "regularization",
            "min_eigen_every", "min_eigen_bonus", "min_depth", "max_depth", "selection_mode", "save_factors",
            "candidate_pool_scope", "challenger_policy", "challenger_quality_fraction",
            "challenger_spatial_strategy", "selection_algorithm_version", "activation_geometry_output",
            "reserve_count", "reserve_policy",
        )
    }
    provenance = _params_record(params_path, controls)
    done_path = f"{output_path}.done.json"
    if skip_existing and _verified_existing(
        output_path, map_output, activation_geometry_output, done_path, provenance
    ):
        with open(done_path) as stream:
            selected_count = int(json.load(stream)["selected_count"])
        plan.update({"skipped_existing": True, "selected_count": selected_count})
        return plan
    if dry_run:
        return plan

    baseline_indices = select_topk(cheap_scores, budget)
    preserved_indices = baseline_indices[: budget - replacement_count]
    removed_indices = baseline_indices[budget - replacement_count :]
    challenger_indices = np.empty(0, dtype=np.int64)
    factor_global_ids = np.empty(0, dtype=np.int64)
    candidate_source_codes = np.empty(0, dtype=np.int64)
    replacement_indices = np.empty(0, dtype=np.int64)
    diagnostics = None
    removed_normals = np.empty((0, 3), dtype=np.float64)
    activation_normal_source = "empty_removed_set"
    if replacement_count == 0:
        selected_global_indices = baseline_indices.copy()
        plan.update(
            {
                "actual_visibility_pairs": 0,
                "nonempty_pose_count": 0,
                "visible_pose_count": 0,
                "pose_coverage": None,
                "reference_coverage": None,
                "normal_source": "not_required_for_cheap_baseline",
                "selected_count": int(selected_global_indices.size),
                "observable_core_count": 0,
                "cheap_fill_count": int(selected_global_indices.size),
                "observable_core_fraction": 0.0,
                "replacement_guarantee": True,
                "zero_gain_replacements": 0,
            }
        )
        selection = None
        factors = None
    else:
        removed_indices, challenger_indices, factor_global_ids, candidate_source_codes = build_forced_replacement_candidates(
            means, cheap_scores, baseline_indices, replacement_count, factor_cap
        )
        candidate_means = means[factor_global_ids]
        candidate_normals, normal_source = _candidate_normals(params, means, factor_global_ids, normal_k)
        removed_normals = candidate_normals[candidate_source_codes == CANDIDATE_SOURCE_REMOVED]
        activation_normal_source = normal_source
        factors = build_frustum_center_factors(
            candidate_means,
            candidate_normals,
            views,
            pair_cap=pair_cap,
            per_pose_cap=per_pose_cap,
            chunk_size=chunk_size,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if factors.visible_pose_count > factors.nonempty_pose_count:
            raise ValueError(
                "pair/per-pose cap leaves geometrically visible reference poses without factors; increase pair_cap"
            )
        selection = select_observable_gaussians(
            factors,
            budget=replacement_count,
            candidate_cap=None,
            greedy_core_cap=None,
            regularization=regularization,
            min_eigen_every=min_eigen_every,
            min_eigen_bonus=min_eigen_bonus,
            eligible_indices=np.flatnonzero(candidate_source_codes != CANDIDATE_SOURCE_REMOVED),
        )
        replacement_indices = factor_global_ids[selection.indices]
        selected_global_indices = np.concatenate([preserved_indices, replacement_indices])
        removed_local = np.flatnonzero(candidate_source_codes == CANDIDATE_SOURCE_REMOVED)
        diagnostics = compare_replacement_information(
            factors, removed_local, selection.indices, regularization=regularization
        )
        baseline_set = set(int(value) for value in baseline_indices)
        selected_set = set(int(value) for value in selected_global_indices)
        added_set = selected_set - baseline_set
        removed_set = baseline_set - selected_set
        if (
            selected_set == baseline_set
            or len(added_set) != replacement_count
            or len(removed_set) != replacement_count
            or set(int(value) for value in replacement_indices) & baseline_set
        ):
            raise AssertionError("forced replacement must change exactly replacement-count baseline members")
        plan.update(
            {
                "actual_visibility_pairs": factors.num_pairs,
                "nonempty_pose_count": factors.nonempty_pose_count,
                "visible_pose_count": factors.visible_pose_count,
                "pose_coverage": factors.pose_coverage,
                "reference_coverage": factors.pose_coverage,
                "normal_source": normal_source,
                "normal_estimation_scope": "candidate_local_approximation",
                "selected_count": int(selected_global_indices.size),
                "observable_core_count": int(selection.indices.size),
                "cheap_fill_count": int(preserved_indices.size),
                "observable_core_fraction": float(selection.indices.size / budget),
                "objective_value": selection.objective,
                "aggregate_information_logdet": selection.logdet,
                "aggregate_information_lambda_min": selection.lambda_min,
                "replacement_guarantee": True,
                "zero_gain_replacements": int(np.count_nonzero(selection.trace_scores[selection.indices] <= 0.0)),
                "preservation_scope": diagnostics["scope"],
                "preservation_summary": diagnostics["summary"],
            }
        )
    baseline_set = set(int(value) for value in baseline_indices)
    selected_set = set(int(value) for value in selected_global_indices)
    intersection_count = len(baseline_set & selected_set)
    added_count = len(selected_set - baseline_set)
    removed_count = len(baseline_set - selected_set)
    baseline_score_sum = float(np.sum(cheap_scores[baseline_indices]))
    selected_score_sum = float(np.sum(cheap_scores[selected_global_indices]))
    candidate_trace_scores = None if selection is None else selection.trace_scores
    effective_reserve_count = int(reserve_count) if activation_geometry_output is not None else 0
    reserve_indices, reserve_source_codes, reserve_trace_scores = select_reserve_gaussians(
        means,
        cheap_scores,
        selected_global_indices,
        baseline_indices,
        challenger_indices,
        factor_global_ids,
        candidate_source_codes,
        candidate_trace_scores,
        effective_reserve_count,
    )
    reserve_normals, reserve_normal_source = _reserve_normals(
        params, means, reserve_indices, normal_k
    )
    reserve_source_coverage = {
        name: int(np.count_nonzero(reserve_source_codes == source_code))
        for source_code, name in enumerate(CANDIDATE_SOURCE_NAMES)
    }
    plan.update(
        {
            "baseline_ids": baseline_indices.tolist(),
            "preserved_ids": preserved_indices.tolist(),
            "removed_ids": removed_indices.tolist(),
            "replacement_ids": replacement_indices.tolist(),
            "challenger_ids": challenger_indices.tolist(),
            "changed_count": removed_count,
            "added_count": added_count,
            "removed_count": removed_count,
            "intersection_count": intersection_count,
            "overlap_fraction": float(intersection_count / budget),
            "jaccard": float(intersection_count / len(baseline_set | selected_set)),
            "cheap_score_baseline_sum": baseline_score_sum,
            "cheap_score_retained_sum": selected_score_sum,
            "cheap_score_retention": float(selected_score_sum / baseline_score_sum) if baseline_score_sum > 0 else 1.0,
            "reserve_requested_count": int(reserve_count),
            "reserve_enabled": activation_geometry_output is not None and reserve_count > 0,
            "reserve_count": int(reserve_indices.size),
            "reserve_policy": RESERVE_POLICY,
            "reserve_source_coverage": reserve_source_coverage,
            "reserve_positive_trace_count": int(np.count_nonzero(reserve_trace_scores > 0.0)),
            "reserve_normal_source": reserve_normal_source,
        }
    )
    if added_count != replacement_count or removed_count != replacement_count:
        raise AssertionError("changed/added/removed counts must equal replacement-count")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    artifact = {
        "indices": selected_global_indices,
        "selected_indices": selected_global_indices,
        "candidate_global_ids": factor_global_ids,
        "factor_pool_global_ids": factor_global_ids,
        "candidate_source_codes": candidate_source_codes,
        "candidate_source_names": np.asarray(CANDIDATE_SOURCE_NAMES),
        "baseline_indices": baseline_indices,
        "preserved_indices": preserved_indices,
        "removed_indices": removed_indices,
        "replacement_indices": replacement_indices,
        "challenger_indices": challenger_indices,
        "reserve_indices": reserve_indices,
        "reserve_source_codes": reserve_source_codes,
        "reserve_source_names": np.asarray(CANDIDATE_SOURCE_NAMES),
        "reserve_trace_scores": reserve_trace_scores,
        "reserve_policy": np.asarray(RESERVE_POLICY),
        "reserve_source_coverage_json": np.asarray(json.dumps(reserve_source_coverage, sort_keys=True)),
        "changed_count": np.asarray(removed_count),
        "added_count": np.asarray(added_count),
        "removed_count": np.asarray(removed_count),
        "intersection_count": np.asarray(intersection_count),
        "overlap_fraction": np.asarray(plan["overlap_fraction"]),
        "jaccard": np.asarray(plan["jaccard"]),
        "cheap_score_retention": np.asarray(plan["cheap_score_retention"]),
        "replacement_guarantee": np.asarray(True),
        "zero_gain_replacements": np.asarray(plan["zero_gain_replacements"]),
        "factor_gaussian_id_space": np.asarray("candidate_local"),
        "score_name": np.asarray(fill_strategy),
        "reference_frame_ids": views.frame_ids,
        "source_params_path": np.asarray(params_path),
        "selection_plan_json": np.asarray(json.dumps(plan, sort_keys=True, allow_nan=False)),
        "observable_core_indices": (
            np.empty(0, dtype=np.int64)
            if selection is None
            else replacement_indices
        ),
        "cheap_fill_indices": (
            selected_global_indices
            if selection is None
            else preserved_indices
        ),
    }
    if selection is not None and factors is not None:
        full_trace_scores = np.zeros(means.shape[0], dtype=np.float64)
        full_trace_scores[factor_global_ids] = selection.trace_scores
        artifact.update(
            {
                "candidate_trace_scores": selection.trace_scores,
                "information_matrix": selection.information_matrix,
                "per_pose_information": selection.per_pose_information,
                "per_pose_logdet": selection.per_pose_logdet,
                "per_pose_lambda_min": selection.per_pose_lambda_min,
                "objective": np.asarray(selection.objective),
                "preservation_scope": np.asarray(diagnostics["scope"]),
                "removed_per_pose_information": diagnostics["removed_per_pose_information"],
                "replacement_per_pose_information": diagnostics["replacement_per_pose_information"],
                "preservation_trace_ratio": diagnostics["trace_ratio"],
                "preservation_logdet_delta": diagnostics["logdet_delta"],
                "preservation_lambda_min_delta": diagnostics["lambda_min_delta"],
                "preservation_difference_lambda_min": diagnostics["difference_lambda_min"],
                "preservation_psd_preserved": diagnostics["psd_preserved"],
                "preservation_summary_json": np.asarray(json.dumps(diagnostics["summary"], sort_keys=True)),
            }
        )
        if save_factors:
            artifact.update(
                {
                    "cheap_scores": cheap_scores,
                    "full_trace_scores": full_trace_scores,
                    "pose_ptr": factors.pose_ptr,
                    "gaussian_ids": factors.gaussian_ids,
                    "jacobians": factors.jacobians,
                    "weights": factors.weights,
                }
            )

    fsm_map = None
    if map_output is not None:
        fsm_map = build_fsm_map(
            params,
            indices=selected_global_indices,
            scores=cheap_scores,
            strategy=plan["selection_mode"],
            compression_ratio=float(selected_global_indices.size / means.shape[0]),
            lambda_min=None if selection is None else float(np.min(selection.per_pose_lambda_min)),
            source_params_path=params_path,
            source_params_sha256=provenance["params_sha256"],
            selection_controls_sha256=provenance["controls_sha256"],
        )
        fsm_map["observable_selection_json"] = np.asarray(json.dumps(plan, sort_keys=True))
        fsm_map["score_name"] = np.asarray(fill_strategy)
        fsm_map["baseline_indices"] = baseline_indices
        fsm_map["preserved_indices"] = preserved_indices
        fsm_map["removed_indices"] = removed_indices
        fsm_map["replacement_indices"] = replacement_indices
        fsm_map["challenger_indices"] = challenger_indices
    activation_geometry = None
    if activation_geometry_output is not None:
        activation_geometry = _build_activation_geometry(
            params,
            selected_global_indices,
            baseline_indices,
            preserved_indices,
            replacement_indices,
            removed_indices,
            removed_normals,
            activation_normal_source,
            reserve_indices,
            reserve_normals,
            reserve_source_codes,
            reserve_trace_scores,
            reserve_normal_source,
            cheap_scores,
            provenance,
        )
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    selection_fd, selection_tmp = tempfile.mkstemp(prefix=".tmp-selection-", suffix=".npz", dir=output_dir)
    os.close(selection_fd)
    map_tmp = None
    activation_tmp = None
    try:
        np.savez(selection_tmp, **artifact)
        with np.load(selection_tmp, allow_pickle=False) as loaded:
            if np.asarray(loaded["selected_indices"]).size != selected_global_indices.size:
                raise ValueError("temporary selection validation failed")
        if map_output is not None and fsm_map is not None:
            map_dir = os.path.dirname(map_output) or "."
            os.makedirs(map_dir, exist_ok=True)
            map_fd, map_tmp = tempfile.mkstemp(prefix=".tmp-map-", suffix=".npz", dir=map_dir)
            os.close(map_fd)
            save_fsm_map(fsm_map, map_tmp)
            loaded_map = load_fsm_map(map_tmp)
            if not np.array_equal(np.asarray(loaded_map["indices"], dtype=np.int64), selected_global_indices):
                raise ValueError("temporary map validation failed")
        if activation_geometry_output is not None and activation_geometry is not None and fsm_map is not None:
            activation_dir = os.path.dirname(activation_geometry_output) or "."
            os.makedirs(activation_dir, exist_ok=True)
            activation_fd, activation_tmp = tempfile.mkstemp(
                prefix=".tmp-activation-geometry-", suffix=".npz", dir=activation_dir
            )
            os.close(activation_fd)
            np.savez(activation_tmp, **activation_geometry)
            load_activation_geometry(
                activation_tmp,
                fsm_map,
                source_params_sha256=provenance["params_sha256"],
                selection_controls_sha256=provenance["controls_sha256"],
            )
        os.replace(selection_tmp, output_path)
        if map_tmp is not None and map_output is not None:
            os.replace(map_tmp, map_output)
        if activation_tmp is not None and activation_geometry_output is not None:
            os.replace(activation_tmp, activation_geometry_output)
        _atomic_json(done_path, {"provenance": provenance, "selected_count": int(selected_global_indices.size)})
    finally:
        for temporary in (selection_tmp, map_tmp, activation_tmp):
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    return plan


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_observable_selection(
        params_path=args.params,
        output_path=args.output,
        count=args.count,
        ratio=args.ratio,
        map_output=args.map_output,
        activation_geometry_output=args.activation_geometry_output,
        reserve_count=args.reserve_count,
        keyframes_only=not args.all_estimated_poses,
        max_reference_poses=args.max_reference_poses,
        pair_cap=args.pair_cap,
        per_pose_cap=args.per_pose_cap,
        chunk_size=args.chunk_size,
        candidate_cap=args.candidate_cap,
        greedy_core_cap=args.greedy_core_cap,
        replacement_count=args.replacement_count,
        min_greedy_fraction=args.min_greedy_fraction,
        fill_strategy=args.fill_strategy,
        normal_k=args.normal_k,
        regularization=args.regularization,
        min_eigen_every=args.min_eigen_every,
        min_eigen_bonus=args.min_eigen_bonus,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        save_factors=args.save_factors,
    )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
