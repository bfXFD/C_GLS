"""Preflight or explicitly build E3 reserve-sidecar controls; never run localization."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_build_observable_map import RESERVE_POLICY, _cheap_scores, _reserve_normals
from utils.fsm_map import (
    expand_log_scales, gaussians_to_covariances, load_fsm_map, load_runtime_reserve_geometry,
    normalize_quaternions, selected_ids_sha256, validate_activation_geometry,
)
from utils.fsm_observability import load_reference_views


DEFAULT_SCENES = ("room2", "office4", "fr3")
DEFAULT_RANDOM_SEEDS = (0, 1, 2)
RESERVE_COUNT = 4096
NORMAL_K = 20
VOXEL_SIZE_M = 0.5
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
DEFAULT_OUTPUT_ROOT = MVP / "cn_supplement_e3_pools_v1"
REFERENCE_FIELDS = (
    "cam_unnorm_rots", "cam_trans", "intrinsics", "org_width", "org_height", "keyframe_time_indices",
)
PARAM_FIELDS = ("means3D", "log_scales", "unnorm_rotations", "logit_opacities")
NORMAL_FIELDS = ("normals", "map_normals", "means3D_normals")


def scene_inputs(scene):
    """Return the known frozen inputs; the optional full FSM export is not used to build pools."""
    if scene not in DEFAULT_SCENES:
        raise ValueError(f"Unknown E3 scene: {scene}")
    dataset = "tum" if scene == "fr3" else "replica"
    sequence = "freiburg3_long_office_household" if scene == "fr3" else scene
    key = f"tum_{sequence}" if dataset == "tum" else f"replica_{sequence}_0"
    suite = "replica_office1_4_untouched_v1" if scene == "office4" else "runtime_reserve_v1"
    archive = MVP / suite / "maps" / key / "logdet"
    params = (ROOT / "experiments/TUM" / f"{sequence}_seed0/params.npz" if dataset == "tum" else
              ROOT.parent / "SplaTAM/experiments/Replica" / f"{sequence}_0/params.npz")
    audit = ROOT / "experiments/FSM/audit_v2_full_rerun/main"
    config_root = ROOT / "configs/fsm/mvp_observable"
    return dict(
        dataset=dataset, sequence=sequence, params=params, map_path=archive / "fsm_map.npz",
        geometry_path=archive / "activation_geometry.npz", selection_path=archive / "selection.npz",
        full_map=audit / key / "full/maps/full_ratio_1/fsm_map.npz",
        base_config=config_root / f"{dataset}_{sequence}_corr_weak128_info_v4.py",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _outside(selected, baseline, count):
    mask = np.ones(count, dtype=bool)
    for ids in (selected, baseline):
        if np.any((ids < 0) | (ids >= count)):
            raise ValueError("Archived map/baseline IDs are outside original params")
        mask[ids] = False
    return np.flatnonzero(mask)


def _strata(means, eligible, count):
    points = means[eligible]
    grid = np.floor(points / VOXEL_SIZE_M).astype(np.int64)
    order = np.lexsort((eligible, points[:, 2], points[:, 1], points[:, 0], grid[:, 2], grid[:, 1], grid[:, 0]))
    start = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(grid[order], axis=0), axis=1))]
    sizes = np.diff(np.r_[start, eligible.size])
    lo, hi = 0, count
    while lo < hi:
        level = (lo + hi + 1) // 2
        if np.minimum(sizes, level).sum() <= count:
            lo = level
        else:
            hi = level - 1
    quotas = np.minimum(sizes, lo)
    spare = np.flatnonzero(sizes > quotas)
    extra = count - int(quotas.sum())
    if extra:
        quotas[spare[np.linspace(0, spare.size - 1, extra, dtype=np.int64)]] += 1
    return eligible[order], start, sizes, quotas, grid[order[start]]


def _select(strata, scores, *, seed):
    ordered, starts, sizes, quotas, _ = strata
    rng = np.random.default_rng(seed) if seed is not None else None
    parts, sources = [], []
    used = 0
    for start, size, quota in zip(starts, sizes, quotas):
        if not quota:
            continue
        ids = ordered[start:start + size]
        if rng is not None:
            parts.append(rng.choice(ids, size=quota, replace=False))
            sources.append(np.full(quota, 2, dtype=np.int64))
            continue
        # Alternate the odd quota remainders to preserve an exact global half quota.
        quality_count = (used + quota + 1) // 2 - (used + 1) // 2
        used += quota
        best = np.lexsort((ids, -scores[ids]))[:quality_count]
        remaining = np.ones(size, dtype=bool)
        remaining[best] = False
        available = ids[remaining]
        spatial_count = quota - quality_count
        spatial = available[np.linspace(0, available.size - 1, spatial_count, dtype=np.int64)]
        parts.extend((ids[best], spatial))
        sources.extend((np.full(quality_count, 1, dtype=np.int64), np.full(spatial_count, 2, dtype=np.int64)))
    return np.concatenate(parts), np.concatenate(sources)


def _plan(scene, output_root, seeds, inputs):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Provide at least two unique non-negative integer pool seeds")
    paths = scene_inputs(scene)
    if inputs:
        if set(inputs) - {key for key, value in paths.items() if isinstance(value, Path)}:
            raise ValueError("Unknown E3 input override")
        paths.update({key: Path(value).expanduser().resolve() for key, value in inputs.items()})
    dest = Path(os.path.abspath(Path(output_root).expanduser())) / scene
    files = {key: dict(path=str(path), present=path.is_file()) for key, path in paths.items() if isinstance(path, Path)}
    pools = []
    for arm, seed in [*((f"spatial_random_s{seed}", seed) for seed in seeds), ("quality_spatial", None), ("geometry", None)]:
        geometry_path = paths["geometry_path"] if arm == "geometry" else dest / arm / "activation_geometry.npz"
        pools.append(dict(arm=arm, seed=seed, count=RESERVE_COUNT, status="planned", map_path=str(paths["map_path"]),
                          geometry_path=str(geometry_path)))
    return paths, dict(schema_version=1, scene=scene, dataset=paths["dataset"], sequence=paths["sequence"],
                       manifest_path=str(dest / "manifest.json"), output_root=str(dest.parent),
                       inputs=files, pools=pools)


def _inspect(scene, output_root, seeds, inputs):
    paths, manifest = _plan(scene, output_root, seeds, inputs)
    files = manifest["inputs"]
    controls = manifest["pools"][:-1]
    original = manifest["pools"][-1]
    base = geometry = params = eligible = None
    try:
        base = load_fsm_map(paths["map_path"])
        geometry = load_runtime_reserve_geometry(paths["geometry_path"], base)
        if (geometry is None or "baseline_global_ids" not in geometry or
                str(geometry.get("reserve_policy")) != RESERVE_POLICY or int(geometry["reserve_count"]) != RESERVE_COUNT):
            raise ValueError("Need the complete archived trajectory-trace geometry policy and exact reserve count")
        reserve = geometry["reserve_global_ids"]
        positive = int(np.count_nonzero(geometry["reserve_trace_scores"] > 0))
        if positive == 0 or np.isin(reserve, geometry["baseline_global_ids"]).any():
            raise ValueError("Invalid archived trajectory-trace eligibility or no positive trace support")
        files["map_path"]["sha256"] = _sha256(paths["map_path"])
        files["geometry_path"]["sha256"] = _sha256(paths["geometry_path"])
        original.update(status="ready", policy=RESERVE_POLICY, positive_trace_count=positive,
                        eligible_scope="outside_selected_and_cheap_baseline; positive_reference_trace_priority_then_fill",
                        point_ids_sha256=selected_ids_sha256(reserve),
                        geometry_sha256=files["geometry_path"]["sha256"],
                        normal_source=str(geometry["reserve_normal_source"]),
                        quotas="archived_trace_and_fill; not matched to control voxel quotas")
    except (OSError, KeyError, ValueError) as exc:
        original.update(status="blocked", reason=str(exc))

    try:
        if original["status"] != "ready":
            raise ValueError(f"Missing verified fixed map/activation archive: {original['reason']}")
        params_sha = _sha256(paths["params"])
        files["params"]["sha256"] = params_sha
        if params_sha != str(geometry["source_params_sha256"]):
            raise ValueError("Original params SHA-256 does not match the archived map/sidecar source")
        with np.load(paths["params"], allow_pickle=False) as archive:
            params = {key: archive[key] for key in PARAM_FIELDS}
            params.update({key: archive[key] for key in NORMAL_FIELDS if key in archive})
            try:
                views = load_reference_views({key: archive[key] for key in REFERENCE_FIELDS})
                ref = dict(status="verified", count=int(views.frame_ids.size), role="mapping_estimated_keyframes")
                if files["selection_path"]["present"]:
                    with np.load(paths["selection_path"], allow_pickle=False) as selection:
                        archived = selection["reference_frame_ids"]
                    expected = views.frame_ids[np.linspace(0, views.frame_ids.size - 1,
                                                           min(64, views.frame_ids.size), dtype=np.int64)]
                    ref.update(archived_count=int(archived.size), archived_ids_match=np.array_equal(archived, expected))
            except (OSError, IndexError, KeyError, ValueError) as exc:
                ref = dict(status="unverified", reason=str(exc))
            manifest["reference_views"] = ref
        means = params["means3D"]
        if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
            raise ValueError("Full source means3D must have finite shape [N, 3]")
        source_count = means.shape[0]
        if (params["log_scales"].shape not in ((source_count, 1), (source_count, 3)) or
                params["unnorm_rotations"].shape != (source_count, 4) or
                params["logit_opacities"].shape not in ((source_count,), (source_count, 1))):
            raise ValueError("Original full geometry and quality arrays have inconsistent shapes")
        eligible = _outside(base["indices"], geometry["baseline_global_ids"], source_count)
        if eligible.size < RESERVE_COUNT or not np.isin(reserve, eligible).all():
            raise ValueError("Original candidate universe cannot cover the requested count or archived reserve IDs")
        manifest["candidate_universe"] = dict(
            source="verified_original_params_full_universe", source_count=source_count,
            excluded_scope="union_of_frozen_selected_ids_and_archived_cheap_baseline_ids",
            eligible_count=int(eligible.size), eligible_ids_sha256=selected_ids_sha256(eligible),
            archived_reserve_subset_verified=True,
        )
        for row in controls:
            row.update(eligible_scope="candidate_universe", policy=row["arm"] if row["seed"] is None else "spatial_random",
                       quotas="shared_balanced_voxel_quota", voxel_size_m=VOXEL_SIZE_M)
    except (OSError, KeyError, ValueError) as exc:
        eligible = None
        manifest.setdefault("reference_views", dict(status="unverified", reason=str(exc)))
        for row in controls:
            row.update(status="blocked", reason=str(exc))
    if files["full_map"]["present"]:
        try:
            with np.load(paths["full_map"], allow_pickle=False) as full:
                ids = full["indices"]
            source_count = (manifest.get("candidate_universe") or {}).get("source_count")
            covers_source = None
            if source_count is not None:
                covers_source = bool(ids.shape == (source_count,) and ids.dtype.kind in "iu" and
                                     ids.min() == 0 and ids.max() == source_count - 1 and
                                     np.unique(ids).size == source_count)
            files["full_map"].update(count=int(ids.size), covers_source_ids=covers_source)
        except (OSError, KeyError, ValueError) as exc:
            files["full_map"]["error"] = str(exc)
    return manifest, base, geometry, params, eligible


def pool_manifest(scene, output_root=DEFAULT_OUTPUT_ROOT, *, seeds=DEFAULT_RANDOM_SEEDS, inputs=None):
    """Read-only preflight/lookup. Generated geometry is ready only with matching persisted proof."""
    manifest, _, _, _, _ = _inspect(scene, output_root, seeds, inputs)
    proof_path = Path(manifest["manifest_path"])
    if proof_path.is_file():
        try:
            proof = json.loads(proof_path.read_text())
            if proof["preflight_sha256"] != _json_digest(manifest):
                raise ValueError(f"Existing E3 manifest conflicts with current preflight: {proof_path}")
            if proof["pools"][-1] != manifest["pools"][-1]:
                raise ValueError(f"Archived geometry pool differs from preflight: {proof_path}")
            for row, stored in zip(manifest["pools"][:-1], proof["pools"][:-1], strict=True):
                path = Path(row["geometry_path"])
                if (any(stored.get(key) != value for key, value in row.items() if key != "status") or
                        stored["status"] != "ready" or _sha256(path) != stored["geometry_sha256"]):
                    raise ValueError(f"Pool sidecar differs from persisted proof: {path}")
                with np.load(path, allow_pickle=False) as sidecar:
                    ids = sidecar["reserve_global_ids"]
                    if ids.size != row["count"] or selected_ids_sha256(ids) != stored["point_ids_sha256"]:
                        raise ValueError(f"Pool point IDs differ from persisted proof: {path}")
        except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
            manifest["lookup_error"] = str(exc)
            for row in manifest["pools"][:-1]:
                row.update(status="blocked", reason=str(exc))
        else:
            manifest = proof
    return manifest


def build_pools(scene, output_root=DEFAULT_OUTPUT_ROOT, *, seeds=DEFAULT_RANDOM_SEEDS, inputs=None):
    """Write only new control sidecars and their manifest; do not rebuild any FSM map."""
    manifest, base, template, params, eligible = _inspect(scene, output_root, seeds, inputs)
    if eligible is None:
        raise ValueError(f"Cannot build E3 pools: {manifest['pools'][0]['reason']}")
    manifest_path = Path(manifest["manifest_path"])
    controls = manifest["pools"][:-1]
    destinations = [Path(row["geometry_path"]) for row in controls]
    historical = ROOT / "experiments/FSM"
    for destination in (*destinations, manifest_path):
        under_study = destination.is_relative_to(MVP) and destination.relative_to(MVP).parts[0].startswith("cn_supplement_")
        if (destination.exists() or any(part.is_symlink() for part in (destination, *destination.parents)) or
                (destination.is_relative_to(historical) and not under_study) or
                any(part in {"audit_v2_full_rerun", "scripts", "tests", "configs", "data", "utils"} for part in destination.parts)):
            raise ValueError(f"Refusing existing, linked, or protected output: {destination}")
    manifest["preflight_sha256"] = _json_digest(manifest)
    means = params["means3D"]
    if not all(np.isfinite(params[key]).all() for key in PARAM_FIELDS[1:]):
        raise ValueError("Full source geometry and quality inputs must be finite")
    scores = _cheap_scores(params, "scale_opacity")
    if not np.isfinite(scores[eligible]).all():
        raise ValueError("scale_opacity scores must be finite throughout the eligible universe")
    strata = _strata(means, eligible, RESERVE_COUNT)
    quotas, voxels = strata[3:]
    manifest["control_quota"] = dict(occupied_voxels=int(quotas.size), used_voxels=int(np.count_nonzero(quotas)),
                                     voxel_origin_m=[0.0, 0.0, 0.0],
                                     voxel_quotas=np.column_stack((voxels[quotas > 0], quotas[quotas > 0])).tolist())
    written = []
    try:
        for row, destination in zip(controls, destinations):
            ids, sources = _select(strata, scores, seed=row["seed"])
            if ids.size != RESERVE_COUNT or np.unique(ids).size != RESERVE_COUNT:
                raise ValueError("Expected exactly the requested number of unique reserve IDs")
            log_scales = np.asarray(params["log_scales"][ids], dtype=np.float64)
            rotations = normalize_quaternions(params["unnorm_rotations"][ids])
            normals, normal_source = _reserve_normals(params, means, ids, NORMAL_K)
            point_digest = selected_ids_sha256(ids)
            geometry = {key: value for key, value in template.items() if not key.startswith("reserve_")}
            geometry.update(
                reserve_count=np.asarray(ids.size), reserve_global_ids=ids, reserve_means=means[ids].astype(np.float32),
                reserve_normals=normals.astype(np.float32), reserve_scores=scores[ids].astype(np.float32),
                reserve_covariances=gaussians_to_covariances(log_scales, rotations).astype(np.float32),
                reserve_scales=np.exp(expand_log_scales(log_scales)).astype(np.float32),
                reserve_rotations=rotations.astype(np.float32), reserve_source_codes=sources,
                reserve_source_names=np.asarray(("unused", "quality_frontier", "spatial_quota")),
                reserve_trace_scores=np.zeros(ids.size, dtype=np.float32), reserve_policy=np.asarray(row["policy"]),
                reserve_normal_source=np.asarray(normal_source), e3_point_ids_sha256=np.asarray(point_digest),
                e3_eligible_ids_sha256=np.asarray(manifest["candidate_universe"]["eligible_ids_sha256"]),
            )
            validate_activation_geometry(geometry, base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                written.append(destination)
                np.savez(stream, **geometry)
            row.update(status="ready", point_ids_sha256=point_digest, geometry_sha256=_sha256(destination),
                       normal_source=normal_source, quality_count=int(np.count_nonzero(sources == 1)),
                       spatial_count=int(np.count_nonzero(sources == 2)))
        with manifest_path.open("x") as stream:
            written.append(manifest_path)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        for path in written:
            path.unlink()
        raise
    return manifest


def make_config(base_config, pool, *, mode):
    """Return a config for the external runner; replay needs its no-commit candidate probe."""
    if mode not in ("closed_loop", "replay", "baseline", "shadow"):
        raise ValueError(f"Unknown E3 run mode: {mode}")
    if mode != "baseline" and pool.get("status") == "blocked":
        raise ValueError(f"Cannot configure blocked E3 pool: {pool.get('reason')}")
    config = deepcopy(base_config)
    active = mode in ("closed_loop", "replay")
    path = pool.get("geometry_path")
    if mode != "baseline" and not path:
        raise ValueError("E3 reserve configuration requires a geometry path")
    reserve = config.setdefault("observable_reserve", {})
    reserve.update(enabled=active, mode="active" if active else "off", geometry_path=path if active else None,
                   candidate_target_mode="append", attempt_without_selection=False)
    reserve.setdefault("selection", "runtime_corr_weak")
    reserve.setdefault("budget", 128)
    reserve.setdefault("max_points", reserve["budget"])
    paired = config.setdefault("paired_reserve_evaluation", {})
    paired.update(enabled=mode == "shadow", mode="shadow", geometry_path=path if mode == "shadow" else None)
    if mode == "shadow":
        paired.update(policies=[reserve["selection"]], budgets=[reserve["budget"]],
                      random_seeds=[reserve.get("seed", 0)], eligible_pool="visible")
    if config.get("activation_diagnostics", {}).get("enabled"):
        config["activation_diagnostics"]["geometry_path"] = path
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", nargs="+", choices=DEFAULT_SCENES, default=DEFAULT_SCENES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument("--build", action="store_true", help="Write new pool sidecars; default is read-only preflight")
    args = parser.parse_args(argv)
    operation = build_pools if args.build else pool_manifest
    print(json.dumps([operation(scene, args.output_root, seeds=args.seeds) for scene in args.scene], indent=2))


if __name__ == "__main__":
    main()
