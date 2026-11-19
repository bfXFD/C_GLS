"""Benchmark the uncapped radius and voxel-block runtime query paths on Replica."""

import argparse
from dataclasses import asdict, fields, replace
import json
import os
from pathlib import Path
import platform
import runpy
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from scripts.fsm_cn_consistency_experiments import MVP, file_record, input_paths, safe_output
from utils.fsm_dataset import load_dataset_from_config
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_map import load_fsm_map, selected_ids_sha256


SCENES = ("room2", "office0")
ARMS = ("radius", "voxel_blocks")
CONFIG_DIR = ROOT / "configs/fsm/mvp_observable"
COUNTS = ("candidate_count", "projection_input_count", "frustum_count", "count")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("pilot", "large"), default="pilot")
    parser.add_argument("--frames", type=int, help="Evenly spaced frames per scene (pilot 6, large 120)")
    parser.add_argument("--repeats", type=int, help="Paired trials per frame, even (pilot 2, large 6)")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup pairs per scene")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    defaults = (6, 2) if args.scale == "pilot" else (120, 6)
    if args.frames is None:
        args.frames = defaults[0]
    if args.repeats is None:
        args.repeats = defaults[1]
    if args.frames < 2 or args.repeats < 2 or args.repeats % 2 or args.warmup < 1:
        parser.error("require >=2 frames, positive even repeats, and >=1 warmup pair")
    name = f"cn_consistency_query_{args.scale}_v1"
    args.output_root = safe_output(args.output_root or MVP / name)
    return args


def elapsed_ms(start):
    return (perf_counter() - start) * 1000.0


def distribution(values):
    p50, p95 = np.percentile(values, [50, 95])
    return dict(n=len(values), min=float(min(values)), p50=float(p50), p95=float(p95), max=float(max(values)))


def benchmark_scene(scene, args, config_path, config, map_path):
    local = config["local_submap"]
    options = {field.name: local[field.name] for field in fields(LocalSubmapConfig) if field.name in local}
    base = LocalSubmapConfig(**(options | dict(frustum_culling="radius", projection_pre_cap=None, oracle_mode="off")))
    configs = {arm: replace(base, frustum_culling=arm) for arm in ARMS}
    dataset = load_dataset_from_config(str(config_path))
    if args.frames > len(dataset):
        raise ValueError(f"{scene}: requested {args.frames} frames but only {len(dataset)} available")
    height, width = dataset.desired_height, dataset.desired_width
    if (height, width) != (dataset.orig_height, dataset.orig_width) or dataset.crop_edge or dataset.crop_size:
        raise ValueError("Native, uncropped Replica intrinsics are required")
    intrinsics = dataset.get_cam_K().numpy()
    poses = dataset.transformed_poses.numpy().astype(np.float64)
    frame_ids = np.linspace(0, len(dataset) - 1, args.frames, dtype=int).tolist()
    warmup_ids = [frame_ids[(i + 1) % len(frame_ids)] for i in range(args.warmup)]

    start = perf_counter()
    fsm_map = load_fsm_map(str(map_path))
    points = np.asarray(fsm_map["means"], dtype=np.float64)
    scores = np.asarray(fsm_map["scores"], dtype=np.float64)
    global_ids = np.asarray(fsm_map["indices"], dtype=np.int64)
    map_load_ms = elapsed_ms(start)
    del fsm_map
    if scores.shape != global_ids.shape or not np.all(np.isfinite(scores)):
        raise ValueError(f"{scene}: expected finite retained-map scores for all global IDs")

    def query(arm, frame, indexes, score_kwargs):
        return indexes[arm].query_local_submap(poses[frame], intrinsics, height, width,
                                               configs[arm], **score_kwargs[arm])

    def compare(results, wall_ms):
        ids = {arm: global_ids[result["indices"]] for arm, result in results.items()}
        sorted_ids = {arm: np.unique(values) for arm, values in ids.items()}
        summary = dict(
            same_set=bool(np.array_equal(*sorted_ids.values())),
            same_order=bool(np.array_equal(*ids.values())),
        )
        for arm, other in (ARMS, ARMS[::-1]):
            result = results[arm]
            if (result["status"] != "ok" or result["oracle_reason"] != "off"
                    or result["candidate_count"] != result["projection_input_count"]
                    or not sorted_ids[arm].size or sorted_ids[arm].size != ids[arm].size):
                raise ValueError(f"{scene}: invalid, empty or duplicate query results for {arm}")
            summary[arm] = dict(
                wall_ms=wall_ms[arm], internal_ms=result["query_latency_ms"],
                ordered_ids_sha256=selected_ids_sha256(ids[arm]),
                sorted_ids_sha256=selected_ids_sha256(sorted_ids[arm]),
                only_count=int(np.setdiff1d(sorted_ids[arm], sorted_ids[other], assume_unique=True).size),
                **{key: result[key] for key in COUNTS},
            )
        return summary

    cold = []
    for order in (ARMS, ARMS[::-1]):
        indexes, score_kwargs, startup, results, wall = {}, {}, {}, {}, {}
        for arm in order:
            start = perf_counter()
            indexes[arm] = MapSpatialIndex(points, backend=local["spatial_backend"], voxel_size=local["grid_voxel"])
            index_ms = elapsed_ms(start)
            start = perf_counter()
            block = None
            if arm == "voxel_blocks":
                block = indexes[arm].prebuild_blocks(base.block_size_m, scores)
                score_kwargs[arm] = {"block_score_token": block["block_score_token"]}
            else:
                score_kwargs[arm] = {"scores": scores}
            prebuild_ms = elapsed_ms(start) if block is not None else 0.0
            start = perf_counter()
            results[arm] = query(arm, frame_ids[0], indexes, score_kwargs)
            wall[arm] = elapsed_ms(start)
            startup[arm] = dict(
                spatial_index_ms=index_ms, block_prebuild_ms=prebuild_ms, first_query_ms=wall[arm],
                build_plus_first_query_ms=index_ms + prebuild_ms + wall[arm],
                block_metrics=None if block is None else {k: v for k, v in block.items() if k != "block_score_token"},
            )
        cold.append(dict(frame_id=frame_ids[0], order=order, startup=startup, **compare(results, wall)))

    for i, frame in enumerate(warmup_ids):
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
            query(arm, frame, indexes, score_kwargs)

    rows = []
    for frame in frame_ids:
        for repeat in range(args.repeats):
            order = ARMS if len(rows) % 2 == 0 else ARMS[::-1]
            results, wall = {}, {}
            for arm in order:
                start = perf_counter()
                results[arm] = query(arm, frame, indexes, score_kwargs)
                wall[arm] = elapsed_ms(start)
            rows.append(dict(frame_id=frame, repeat=repeat, order=order, **compare(results, wall)))

    summary = {arm: {field: distribution([row[arm][field] for row in rows])
                     for field in ("wall_ms", "internal_ms", *COUNTS)} for arm in ARMS}
    summary.update(
        same_set_pairs=sum(row["same_set"] for row in rows),
        same_order_pairs=sum(row["same_order"] for row in rows),
        paired_wall_speedup_radius_over_blocks=distribution(
            [row["radius"]["wall_ms"] / row["voxel_blocks"]["wall_ms"] for row in rows]),
    )
    return dict(scene=scene, frame_ids=frame_ids, warmup_frame_ids=warmup_ids, image_hw=[height, width],
                intrinsics=intrinsics.tolist(), pose_count=len(dataset), pose_is_gt=True,
                pose_coordinates="Replica dataset c2w relative to sequence frame 0",
                frame_c2w=poses[frame_ids].tolist(), map_points=len(points), map_load_ms=map_load_ms,
                map_global_ids_sha256=selected_ids_sha256(global_ids), index_backend=indexes[ARMS[0]].backend,
                grid_voxel=local["grid_voxel"], query_configs={arm: asdict(cfg) for arm, cfg in configs.items()},
                cold=cold, summary=summary, pairs=rows)


def main():
    args = parse_args()
    paths = {Path(__file__).resolve(), CONFIG_DIR / "common.py",
             ROOT / "scripts/fsm_cn_consistency_experiments.py", ROOT / "scripts/splatam.py",
             *(ROOT / f"utils/{name}.py" for name in ("fsm_dataset", "fsm_local_map", "fsm_map")),
             *(ROOT / f"datasets/gradslam_datasets/{name}.py"
               for name in ("replica", "basedataset", "dataconfig", "datautils"))}
    cases = []
    for scene in SCENES:
        config_path = CONFIG_DIR / f"replica_{scene}.py"
        config = runpy.run_path(str(config_path))["config"]
        data = config["data"]
        map_path, _ = input_paths("replica", scene)
        paths.update((config_path, map_path, Path(data["gradslam_data_cfg"]),
                      Path(data["basedir"]) / data["sequence"] / "traj.txt"))
        cases.append((scene, config_path, config, map_path))
    frozen = [file_record(path) for path in sorted(paths)]
    results = [benchmark_scene(scene, args, config_path, config, map_path)
               for scene, config_path, config, map_path in cases]
    if [file_record(path) for path in sorted(paths)] != frozen:
        raise RuntimeError("A benchmark input or source changed during the run")

    passed = all(row["same_set"] and row["same_order"]
                 for scene in results for stage in ("cold", "pairs") for row in scene[stage])
    report = dict(
        schema_version=1, scale=args.scale, frames_per_scene=args.frames, repeats=args.repeats,
        warmup_pairs_per_scene=args.warmup, runtime=dict(
            python=sys.executable, python_version=platform.python_version(), machine=platform.machine(),
            numpy=np.__version__, scipy=scipy.__version__, affinity=sorted(os.sched_getaffinity(0)),
            thread_env={name: os.getenv(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        ),
        timing="Wall time for query_local_submap only; no warmup or ID checks. Fresh indexes in both cold orders; "
               "cached-map/OS state, not fresh process or disk. Warm queries reuse second cold round indexes.",
        equality="Final returned global IDs only; candidate counts can differ. No oracle or projection pre-cap.",
        input_and_source_hashes=frozen, all_final_id_sets_and_orders_match=passed, scenes=results,
    )
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    safe_output(root)
    path = root / "query_paths.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(file_record(path), sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
