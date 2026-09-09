"""Strict common-valid offline comparison for FSM localization runs."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import itertools
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.fsm_metrics import align_c2w_trajectory, compute_absolute_pose_errors


SCHEMA_VERSION = 1
TIMESTAMP_ATOL = 5e-9
GT_POSE_ATOL = 1e-8
DEFAULT_FRAME_DELTAS = (1, 10, 30, 100)

SUMMARY_FIELDS = (
    "scope", "scope_methods", "method", "metric_group", "delta_frames",
    "common_frame_count", "sample_count", "translation_rmse_m",
    "translation_mean_m", "translation_median_m", "rotation_rmse_deg",
    "rotation_mean_deg", "rotation_median_deg", "alignment_status",
    "alignment_count", "alignment_scale",
)
COMPARISON_FIELDS = (
    "scope", "scope_methods", "method_a", "method_b", "metric_group",
    "delta_frames", "metric", "unit", "sample_unit", "sample_count",
    "value_a", "value_b", "delta_b_minus_a", "ci_low", "ci_high",
    "bootstrap_replicates", "bootstrap_seed", "bootstrap_method",
)
FRAME_FIELDS = (
    "scope", "scope_methods", "method", "frame_id", "timestamp",
    "present", "individual_valid", "scope_valid", "raw_translation_error_m",
    "raw_rotation_error_deg", "aligned_translation_error_m",
    "aligned_rotation_error_deg",
)
RPE_PAIR_FIELDS = (
    "scope", "scope_methods", "method", "delta_frames", "start_frame_id",
    "end_frame_id", "start_timestamp", "end_timestamp",
    "translation_error_m", "rotation_error_deg",
)


@dataclass(frozen=True)
class CommonValidConfig:
    frame_deltas: tuple[int, ...] = DEFAULT_FRAME_DELTAS
    align_with_scale: bool = False
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 0
    confidence_level: float = 0.95
    timestamp_atol: float = TIMESTAMP_ATOL
    gt_pose_atol: float = GT_POSE_ATOL

    def __post_init__(self) -> None:
        deltas = tuple(sorted(set(int(value) for value in self.frame_deltas)))
        if not deltas or any(value <= 0 for value in deltas):
            raise ValueError("frame_deltas must contain positive integers")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if not np.isfinite(self.timestamp_atol) or self.timestamp_atol < 0.0:
            raise ValueError("timestamp_atol must be non-negative and finite")
        if not np.isfinite(self.gt_pose_atol) or self.gt_pose_atol < 0.0:
            raise ValueError("gt_pose_atol must be non-negative and finite")
        object.__setattr__(self, "frame_deltas", deltas)


@dataclass(frozen=True)
class RunData:
    name: str
    run_dir: Path
    frame_ids: np.ndarray
    timestamps: np.ndarray
    est_c2w: np.ndarray
    gt_c2w: np.ndarray
    valid_mask: np.ndarray
    diagnostics: dict[str, Any]
    hashes: dict[str, str | None]


@dataclass(frozen=True)
class CommonValidResult:
    method_names: tuple[str, ...]
    pair_names: tuple[str, ...]
    frame_ids: np.ndarray
    timestamps: np.ndarray
    gt_c2w: np.ndarray
    est_c2w: np.ndarray
    present_masks: np.ndarray
    individual_masks: np.ndarray
    pair_masks: np.ndarray
    all_mask: np.ndarray
    summary_rows: tuple[dict[str, Any], ...]
    comparison_rows: tuple[dict[str, Any], ...]
    frame_rows: tuple[dict[str, Any], ...]
    rpe_pair_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _integer(value: Any, name: str) -> int:
    parsed_float = _finite_float(value, name)
    parsed = int(parsed_float)
    if parsed_float != parsed or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _fail_closed_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _quaternion_to_rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("trajectory quaternion must have finite non-zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_tum_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a strict timestamp/translation/quaternion TUM c2w trajectory."""
    trajectory_path = Path(path)
    rows: list[list[float]] = []
    with open(trajectory_path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{trajectory_path}:{line_number} must contain exactly 8 columns")
            try:
                values = [float(value) for value in parts]
            except ValueError as exc:
                raise ValueError(f"{trajectory_path}:{line_number} contains a non-numeric value") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{trajectory_path}:{line_number} contains a non-finite value")
            rows.append(values)
    if not rows:
        raise ValueError(f"{trajectory_path} must contain at least one pose")
    data = np.asarray(rows, dtype=np.float64)
    timestamps = data[:, 0]
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{trajectory_path} timestamps must be unique and strictly increasing")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], data.shape[0], axis=0)
    poses[:, :3, 3] = data[:, 1:4]
    for index, quaternion in enumerate(data[:, 4:8]):
        poses[index, :3, :3] = _quaternion_to_rotation_xyzw(quaternion)
    return timestamps, poses


def _read_frame_metrics(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        required = {"frame_id", "timestamp", "tracking_valid"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"invalid frame_metrics.csv schema; missing {missing}")
        rows: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        timestamps: list[float] = []
        held_count = 0
        parse_failed_count = 0
        for raw in reader:
            frame_id = _integer(raw["frame_id"], "frame_id")
            timestamp = _finite_float(raw["timestamp"], "timestamp")
            if frame_id in seen_ids:
                raise ValueError(f"duplicate frame_id in {path}: {frame_id}")
            if any(abs(timestamp - previous) <= TIMESTAMP_ATOL for previous in timestamps):
                raise ValueError(f"duplicate timestamp in {path}: {timestamp}")
            seen_ids.add(frame_id)
            timestamps.append(timestamp)
            raw_valid = str(raw.get("tracking_valid", "")).strip().lower()
            parsed_valid = _fail_closed_boolean(raw.get("tracking_valid"))
            if raw_valid not in {"0", "1", "true", "false", "yes", "no"}:
                parse_failed_count += 1
            status_text = " ".join(
                str(raw.get(field, "")).strip().lower()
                for field in ("tracking_status", "status", "failure_reason")
            )
            held = "hold" in status_text or "held" in status_text
            held_count += int(held)
            rows.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "valid": bool(parsed_valid and not held),
            })
    if not rows:
        raise ValueError(f"{path} must contain at least one frame row")
    rows.sort(key=lambda row: row["frame_id"])
    return rows, {"held_pose_count": held_count, "tracking_valid_parse_failure_count": parse_failed_count}


def _timestamp_match_indices(reference: np.ndarray, query: np.ndarray, atol: float, context: str) -> np.ndarray:
    indices = []
    for timestamp in query:
        matches = np.flatnonzero(np.abs(reference - timestamp) <= atol)
        if matches.size != 1:
            raise ValueError(
                f"{context}: timestamp {timestamp:.12g} matched {matches.size} rows within atol={atol}"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def load_run(name: str, run_dir: str | Path, config: CommonValidConfig = CommonValidConfig()) -> RunData:
    """Load and strictly join one localization run by frame_id and timestamp."""
    if not name or "=" in name:
        raise ValueError("method name must be non-empty and must not contain '='")
    root = Path(run_dir).resolve()
    paths = {
        "trajectory_est": root / "trajectory_est.txt",
        "trajectory_gt": root / "trajectory_gt.txt",
        "frame_metrics": root / "frame_metrics.csv",
        "summary": root / "summary.json",
    }
    required = (paths["trajectory_est"], paths["trajectory_gt"], paths["frame_metrics"])
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"method {name!r} is missing required inputs: {missing}")

    frame_rows, frame_diagnostics = _read_frame_metrics(paths["frame_metrics"])
    est_timestamps, est_poses = load_tum_trajectory(paths["trajectory_est"])
    gt_timestamps, gt_poses = load_tum_trajectory(paths["trajectory_gt"])
    if est_timestamps.shape != gt_timestamps.shape or not np.allclose(
        est_timestamps, gt_timestamps, rtol=0.0, atol=config.timestamp_atol
    ):
        raise ValueError(f"method {name!r} estimated and GT trajectory timestamps disagree")
    csv_timestamps = np.asarray([row["timestamp"] for row in frame_rows], dtype=np.float64)
    if csv_timestamps.shape[0] != est_timestamps.shape[0]:
        raise ValueError(f"method {name!r} frame_metrics and trajectories have different row counts")
    trajectory_indices = _timestamp_match_indices(
        est_timestamps, csv_timestamps, config.timestamp_atol, f"method {name!r} frame join"
    )
    est_joined = est_poses[trajectory_indices]
    gt_joined = gt_poses[trajectory_indices]
    frame_ids = np.asarray([row["frame_id"] for row in frame_rows], dtype=np.int64)
    timestamps = csv_timestamps
    valid = np.asarray([row["valid"] for row in frame_rows], dtype=bool)
    summary_valid = None
    if paths["summary"].is_file():
        with open(paths["summary"], encoding="utf-8") as stream:
            summary = json.load(stream)
        summary_valid = summary.get("num_valid_frames")
    diagnostics = {
        "method": name,
        "run_dir": str(root),
        "frame_count": int(frame_ids.size),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "invalid_frame_count": int(np.count_nonzero(~valid)),
        "frame_id_min": int(frame_ids.min()),
        "frame_id_max": int(frame_ids.max()),
        "summary_num_valid_frames": summary_valid,
        **frame_diagnostics,
    }
    hashes = {
        key: _sha256(path) if path.is_file() else None
        for key, path in paths.items()
    }
    return RunData(
        name=name, run_dir=root, frame_ids=frame_ids, timestamps=timestamps,
        est_c2w=est_joined, gt_c2w=gt_joined, valid_mask=valid,
        diagnostics=diagnostics, hashes=hashes,
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _error_stats(translation: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    if translation.size == 0:
        return {
            "sample_count": 0, "translation_rmse_m": None,
            "translation_mean_m": None, "translation_median_m": None,
            "rotation_rmse_deg": None, "rotation_mean_deg": None,
            "rotation_median_deg": None,
        }
    return {
        "sample_count": int(translation.size),
        "translation_rmse_m": float(np.sqrt(np.mean(translation * translation))),
        "translation_mean_m": float(np.mean(translation)),
        "translation_median_m": float(np.median(translation)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation * rotation))),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_median_deg": float(np.median(rotation)),
    }


def _rpe_samples(
    est: np.ndarray,
    gt: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
    delta: int,
) -> list[dict[str, Any]]:
    index_by_frame = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    samples = []
    for start_index, start_frame in enumerate(frame_ids):
        end_frame = int(start_frame) + delta
        end_index = index_by_frame.get(end_frame)
        if end_index is None:
            continue
        interval_indices = [index_by_frame.get(frame_id) for frame_id in range(int(start_frame), end_frame + 1)]
        if any(index is None for index in interval_indices):
            continue
        interval = np.asarray(interval_indices, dtype=np.int64)
        if not np.all(mask[interval]):
            continue
        est_relative = np.linalg.inv(est[start_index]) @ est[end_index]
        gt_relative = np.linalg.inv(gt[start_index]) @ gt[end_index]
        error = np.linalg.inv(gt_relative) @ est_relative
        samples.append({
            "start_index": start_index, "end_index": end_index,
            "start_frame_id": int(start_frame), "end_frame_id": end_frame,
            "start_timestamp": float(timestamps[start_index]),
            "end_timestamp": float(timestamps[end_index]),
            "translation_error_m": float(np.linalg.norm(error[:3, 3])),
            "rotation_error_deg": _rotation_angle_deg(error[:3, :3]),
        })
    return samples


def _scope_name(methods: Sequence[str], all_methods: tuple[str, ...]) -> str:
    return "all" if tuple(methods) == all_methods else "pair:" + "|".join(methods)


def _paired_rmse_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    draws: np.ndarray,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if values_a.size == 0 or draws.shape[0] == 0:
        return None, None
    indices = draws[:, : values_a.size] % np.uint64(values_a.size)
    deltas = np.sqrt(np.mean(values_b[indices] ** 2, axis=1)) - np.sqrt(
        np.mean(values_a[indices] ** 2, axis=1)
    )
    alpha = (1.0 - confidence_level) * 0.5
    bounds = np.quantile(deltas, [alpha, 1.0 - alpha], method="linear")
    return float(bounds[0]), float(bounds[1])


def _external_gt_check(
    gt_path: Path,
    canonical_timestamps: np.ndarray,
    canonical_gt: np.ndarray,
    config: CommonValidConfig,
) -> dict[str, Any]:
    timestamps, poses = load_tum_trajectory(gt_path)
    indices = _timestamp_match_indices(
        timestamps, canonical_timestamps, config.timestamp_atol, "external GT join"
    )
    if not np.allclose(poses[indices], canonical_gt, rtol=0.0, atol=config.gt_pose_atol):
        raise ValueError("external GT poses disagree with run GT poses")
    return {"path": str(gt_path.resolve()), "sha256": _sha256(gt_path), "row_count": int(timestamps.size)}


def analyze_common_valid(
    methods: Mapping[str, str | Path],
    gt: str | Path | None = None,
    config: CommonValidConfig = CommonValidConfig(),
) -> CommonValidResult:
    """Analyze all-method and pairwise common-valid scopes."""
    if len(methods) < 2:
        raise ValueError("at least two methods are required")
    method_names = tuple(methods)
    if len(set(method_names)) != len(method_names):
        raise ValueError("method names must be unique")
    runs = [load_run(name, methods[name], config) for name in method_names]

    canonical_ids = np.asarray(sorted({int(value) for run in runs for value in run.frame_ids}), dtype=np.int64)
    frame_index = {int(value): index for index, value in enumerate(canonical_ids)}
    frame_count = canonical_ids.size
    timestamps = np.full(frame_count, np.nan, dtype=np.float64)
    gt_c2w = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    est_c2w = np.repeat(np.eye(4, dtype=np.float64)[None, None], len(runs) * frame_count, axis=0).reshape(
        len(runs), frame_count, 4, 4
    )
    present = np.zeros((len(runs), frame_count), dtype=bool)
    individual = np.zeros((len(runs), frame_count), dtype=bool)
    for method_index, run in enumerate(runs):
        for local_index, frame_id in enumerate(run.frame_ids):
            index = frame_index[int(frame_id)]
            if np.isfinite(timestamps[index]) and not np.isclose(
                timestamps[index], run.timestamps[local_index], rtol=0.0, atol=config.timestamp_atol
            ):
                raise ValueError(f"frame_id {frame_id} maps to inconsistent timestamps across methods")
            if np.isfinite(gt_c2w[index]).all() and not np.allclose(
                gt_c2w[index], run.gt_c2w[local_index], rtol=0.0, atol=config.gt_pose_atol
            ):
                raise ValueError(f"frame_id {frame_id} maps to inconsistent GT poses across methods")
            timestamps[index] = run.timestamps[local_index]
            gt_c2w[index] = run.gt_c2w[local_index]
            est_c2w[method_index, index] = run.est_c2w[local_index]
            present[method_index, index] = True
            individual[method_index, index] = bool(run.valid_mask[local_index])
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(gt_c2w)):
        raise AssertionError("canonical union construction left incomplete GT data")

    pairs = tuple(itertools.combinations(range(len(method_names)), 2))
    pair_names = tuple(f"{method_names[a]}|{method_names[b]}" for a, b in pairs)
    pair_masks = np.stack([individual[a] & individual[b] for a, b in pairs])
    all_mask = np.all(individual, axis=0)
    external_gt = None
    if gt is not None:
        external_path = Path(gt)
        if not external_path.is_file():
            raise FileNotFoundError(str(external_path))
        external_gt = _external_gt_check(external_path, timestamps, gt_c2w, config)

    max_units = max(1, frame_count)
    bootstrap_draws = np.random.PCG64(config.bootstrap_seed).random_raw(
        (config.bootstrap_replicates, max_units)
    )
    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    rpe_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    scopes: list[tuple[tuple[int, ...], np.ndarray]] = [(tuple(range(len(runs))), all_mask)]
    scopes.extend(((a, b), pair_masks[index]) for index, (a, b) in enumerate(pairs))
    for scope_indices, scope_mask in scopes:
        scope_methods = tuple(method_names[index] for index in scope_indices)
        scope = _scope_name(scope_methods, method_names)
        scope_text = "|".join(scope_methods)
        method_results: dict[int, dict[str, Any]] = {}
        for method_index in scope_indices:
            raw = compute_absolute_pose_errors(est_c2w[method_index], gt_c2w, valid_mask=scope_mask)
            aligned = compute_absolute_pose_errors(
                est_c2w[method_index], gt_c2w, valid_mask=scope_mask,
                align=True, with_scale=config.align_with_scale,
            )
            alignment = aligned["alignment"]
            alignment_status = "ok" if alignment["aligned_c2w"] is not None else "insufficient_frames"
            method_result = {"raw": raw, "aligned": aligned, "rpe": {}}
            method_results[method_index] = method_result
            for group, metrics in (("raw_absolute", raw), ("aligned_absolute", aligned)):
                summary_rows.append({
                    "scope": scope, "scope_methods": scope_text,
                    "method": method_names[method_index], "metric_group": group,
                    "delta_frames": None, "common_frame_count": int(np.count_nonzero(scope_mask)),
                    "sample_count": metrics["count"],
                    "translation_rmse_m": metrics["translation_rmse_m"],
                    "translation_mean_m": metrics["translation_mean_m"],
                    "translation_median_m": metrics["translation_median_m"],
                    "rotation_rmse_deg": metrics["rotation_rmse_deg"],
                    "rotation_mean_deg": metrics["rotation_mean_deg"],
                    "rotation_median_deg": metrics["rotation_median_deg"],
                    "alignment_status": alignment_status if group == "aligned_absolute" else "not_applicable",
                    "alignment_count": alignment["count"] if group == "aligned_absolute" else None,
                    "alignment_scale": alignment["scale"] if group == "aligned_absolute" else None,
                })
            for frame_index_value, frame_id in enumerate(canonical_ids):
                frame_rows.append({
                    "scope": scope, "scope_methods": scope_text, "method": method_names[method_index],
                    "frame_id": int(frame_id), "timestamp": float(timestamps[frame_index_value]),
                    "present": bool(present[method_index, frame_index_value]),
                    "individual_valid": bool(individual[method_index, frame_index_value]),
                    "scope_valid": bool(scope_mask[frame_index_value]),
                    "raw_translation_error_m": raw["translation_errors_m"][frame_index_value],
                    "raw_rotation_error_deg": raw["rotation_errors_deg"][frame_index_value],
                    "aligned_translation_error_m": aligned["translation_errors_m"][frame_index_value],
                    "aligned_rotation_error_deg": aligned["rotation_errors_deg"][frame_index_value],
                })
            for delta in config.frame_deltas:
                samples = _rpe_samples(
                    est_c2w[method_index], gt_c2w, canonical_ids, timestamps, scope_mask, delta
                )
                translation = np.asarray([row["translation_error_m"] for row in samples], dtype=np.float64)
                rotation = np.asarray([row["rotation_error_deg"] for row in samples], dtype=np.float64)
                stats = _error_stats(translation, rotation)
                method_result["rpe"][delta] = {
                    "samples": samples, "translation": translation, "rotation": rotation, **stats,
                }
                summary_rows.append({
                    "scope": scope, "scope_methods": scope_text,
                    "method": method_names[method_index], "metric_group": "frame_rpe",
                    "delta_frames": delta, "common_frame_count": int(np.count_nonzero(scope_mask)),
                    **stats, "alignment_status": "not_applicable", "alignment_count": None,
                    "alignment_scale": None,
                })
                for sample in samples:
                    rpe_rows.append({
                        "scope": scope, "scope_methods": scope_text,
                        "method": method_names[method_index], "delta_frames": delta,
                        **{field: sample[field] for field in RPE_PAIR_FIELDS if field in sample},
                    })

        for method_a, method_b in itertools.combinations(scope_indices, 2):
            name_a, name_b = method_names[method_a], method_names[method_b]
            for group, key in (("raw_absolute", "raw"), ("aligned_absolute", "aligned")):
                result_a, result_b = method_results[method_a][key], method_results[method_b][key]
                for metric, field, errors_field, unit in (
                    ("translation_rmse", "translation_rmse_m", "translation_errors_m", "m"),
                    ("rotation_rmse", "rotation_rmse_deg", "rotation_errors_deg", "deg"),
                ):
                    value_a, value_b = result_a[field], result_b[field]
                    ci = (None, None)
                    bootstrap_method = None
                    if group == "raw_absolute" and value_a is not None:
                        values_a = np.asarray([value for value in result_a[errors_field] if value is not None])
                        values_b = np.asarray([value for value in result_b[errors_field] if value is not None])
                        ci = _paired_rmse_bootstrap(
                            values_a, values_b, bootstrap_draws, config.confidence_level
                        )
                        bootstrap_method = "IID paired percentile over frames"
                    comparison_rows.append({
                        "scope": scope, "scope_methods": scope_text, "method_a": name_a,
                        "method_b": name_b, "metric_group": group, "delta_frames": None,
                        "metric": metric, "unit": unit, "sample_unit": "frame",
                        "sample_count": int(np.count_nonzero(scope_mask)), "value_a": value_a,
                        "value_b": value_b,
                        "delta_b_minus_a": None if value_a is None else float(value_b - value_a),
                        "ci_low": ci[0], "ci_high": ci[1],
                        "bootstrap_replicates": config.bootstrap_replicates if bootstrap_method else 0,
                        "bootstrap_seed": config.bootstrap_seed if bootstrap_method else None,
                        "bootstrap_method": bootstrap_method,
                    })
            for delta in config.frame_deltas:
                rpe_a = method_results[method_a]["rpe"][delta]
                rpe_b = method_results[method_b]["rpe"][delta]
                for metric, values_key, field, unit in (
                    ("translation_rmse", "translation", "translation_rmse_m", "m"),
                    ("rotation_rmse", "rotation", "rotation_rmse_deg", "deg"),
                ):
                    value_a, value_b = rpe_a[field], rpe_b[field]
                    ci = _paired_rmse_bootstrap(
                        rpe_a[values_key], rpe_b[values_key], bootstrap_draws, config.confidence_level
                    )
                    comparison_rows.append({
                        "scope": scope, "scope_methods": scope_text, "method_a": name_a,
                        "method_b": name_b, "metric_group": "frame_rpe", "delta_frames": delta,
                        "metric": metric, "unit": unit, "sample_unit": "RPE pair",
                        "sample_count": rpe_a["sample_count"], "value_a": value_a,
                        "value_b": value_b,
                        "delta_b_minus_a": None if value_a is None else float(value_b - value_a),
                        "ci_low": ci[0], "ci_high": ci[1],
                        "bootstrap_replicates": config.bootstrap_replicates if value_a is not None else 0,
                        "bootstrap_seed": config.bootstrap_seed if value_a is not None else None,
                        "bootstrap_method": "IID paired percentile over RPE pairs" if value_a is not None else None,
                    })

    diagnostics = {
        "status": "ok", "method_count": len(method_names), "frame_count": int(frame_count),
        "all_common_valid_count": int(np.count_nonzero(all_mask)),
        "individual_valid_counts": {
            name: int(np.count_nonzero(individual[index])) for index, name in enumerate(method_names)
        },
        "pairwise_common_valid_counts": {
            pair_names[index]: int(np.count_nonzero(pair_masks[index])) for index in range(len(pair_names))
        },
        "runs": [run.diagnostics for run in runs],
        "input_hashes": {run.name: run.hashes for run in runs},
    }
    provenance = {
        "schema_version": SCHEMA_VERSION, "config": dataclasses.asdict(config),
        "method_order": list(method_names), "pair_order": list(pair_names),
        "runs": [
            {
                "method": run.name, "run_dir": str(run.run_dir),
                "diagnostics": run.diagnostics, "input_sha256": run.hashes,
            }
            for run in runs
        ],
        "canonical_frame_policy": "sorted union of frame_id",
        "validity_policy": "tracking_valid fail-closed; held poses invalid",
        "gt_policy": "all run GT poses must agree on shared frame_id",
        "alignment_fit_mask": "scope common-valid mask only; minimum 3 frames",
        "comparison_direction": "delta_b_minus_a; negative means method B is lower",
        "rpe_policy": "frame-id delta; every integer frame in interval must be scope-common-valid",
        "bootstrap": {
            "generator": "numpy.PCG64", "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed, "method": "IID paired percentile",
            "confidence_level": config.confidence_level,
            "shared_draw_matrix_across_contrasts": True,
        },
        "external_gt": external_gt, "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    return CommonValidResult(
        method_names=method_names, pair_names=pair_names, frame_ids=canonical_ids,
        timestamps=timestamps, gt_c2w=gt_c2w, est_c2w=est_c2w,
        present_masks=present, individual_masks=individual, pair_masks=pair_masks,
        all_mask=all_mask, summary_rows=tuple(summary_rows),
        comparison_rows=tuple(comparison_rows), frame_rows=tuple(frame_rows),
        rpe_pair_rows=tuple(rpe_rows), diagnostics=diagnostics, provenance=provenance,
    )


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_path(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_common_valid(result: CommonValidResult, output_dir: str | Path) -> dict[str, Path]:
    """Atomically write the common-valid output contract."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output / "summary.csv", "comparisons_csv": output / "comparisons.csv",
        "frames_csv": output / "frames.csv", "rpe_pairs_csv": output / "rpe_pairs.csv",
        "result_json": output / "result.json", "arrays_npz": output / "arrays.npz",
        "provenance_json": output / "provenance.json",
    }
    _atomic_csv(paths["summary_csv"], result.summary_rows, SUMMARY_FIELDS)
    _atomic_csv(paths["comparisons_csv"], result.comparison_rows, COMPARISON_FIELDS)
    _atomic_csv(paths["frames_csv"], result.frame_rows, FRAME_FIELDS)
    _atomic_csv(paths["rpe_pairs_csv"], result.rpe_pair_rows, RPE_PAIR_FIELDS)
    _atomic_json(paths["result_json"], {
        "schema_version": SCHEMA_VERSION, "diagnostics": result.diagnostics,
        "summary": result.summary_rows, "comparisons": result.comparison_rows,
        "outputs": {key: path.name for key, path in paths.items()},
    })
    _atomic_npz(paths["arrays_npz"], {
        "method_names": np.asarray(result.method_names, dtype=np.str_),
        "pair_names": np.asarray(result.pair_names, dtype=np.str_),
        "frame_ids": result.frame_ids.astype(np.int64), "timestamps": result.timestamps.astype(np.float64),
        "gt_c2w": result.gt_c2w.astype(np.float64), "est_c2w": result.est_c2w.astype(np.float64),
        "present_masks": result.present_masks.astype(bool),
        "individual_masks": result.individual_masks.astype(bool),
        "pair_masks": result.pair_masks.astype(bool), "all_mask": result.all_mask.astype(bool),
    })
    _atomic_json(paths["provenance_json"], result.provenance)
    return paths


def run_common_valid(
    methods: Mapping[str, str | Path],
    output_dir: str | Path,
    gt: str | Path | None = None,
    config: CommonValidConfig = CommonValidConfig(),
) -> dict[str, Any]:
    result = analyze_common_valid(methods, gt=gt, config=config)
    paths = write_common_valid(result, output_dir)
    return {
        "status": "ok", "output_dir": str(Path(output_dir).resolve()),
        "method_count": len(result.method_names), "frame_count": int(result.frame_ids.size),
        "all_common_valid_count": int(np.count_nonzero(result.all_mask)),
        "outputs": {key: str(path) for key, path in sorted(paths.items())},
    }


def parse_method_specs(specs: Sequence[str]) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --method {spec!r}; expected NAME=RUN_DIR")
        name, run_dir = spec.split("=", 1)
        name = name.strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid method name {name!r}")
        if name in methods:
            raise ValueError(f"duplicate method name {name!r}")
        if not run_dir.strip():
            raise ValueError(f"method {name!r} has an empty run directory")
        methods[name] = Path(run_dir).expanduser()
    return methods
