"""Paired score on/off diagnostics and closed-loop runs on the same compact maps."""

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_closed_loop_queries import arm_config, run_metrics
from scripts.fsm_bench_compact_convergence import samples
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex


WINDOWS = (("room2", 0, 80), ("office0", 0, 80), ("office0", 1700, 1850))
ARMS = ("scored", "unscored")


def diagnostic(dataset, map_path, config, frame_ids, arms):
    with np.load(map_path, allow_pickle=False) as saved:
        points, scores = saved["means"], saved["scores"]
    options = config["local_submap"]
    local = LocalSubmapConfig(**{f.name: options[f.name] for f in fields(LocalSubmapConfig) if f.name in options})
    xyz = MapSpatialIndex(points, backend=options["spatial_backend"])
    rows = []
    for frame_id in frame_ids:
        frame = get_frame(dataset, frame_id)
        pose = np.asarray(frame["c2w"], dtype=np.float64)
        height, width = frame["depth"].shape[:2]
        cfg = config["dataset"]
        source = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"], depth_min=cfg["depth_min"],
                                    depth_max=cfg["depth_max"], max_points=cfg["max_points"], seed=frame_id)["points"]
        world = source @ pose[:3, :3].T + pose[:3, 3]
        result = {"frame": frame_id}
        ids = {}
        for arm in arms:
            query_config = replace(local, coverage_representatives="spatial_strata") if arm == "spatial" else local
            query = xyz.query_local_submap(pose, frame["intrinsics"], height, width, query_config,
                                           scores=scores if arm == "scored" else None)
            ids[arm] = query["indices"]
            target = points[ids[arm]]
            if not len(target):
                raise ValueError("The fixed query returned an empty target")
            target_depth = (target - pose[:3, 3]) @ pose[:3, 2]
            distances, nearest = cKDTree(target).query(world)
            valid = distances <= config["registration"]["max_correspondence_distance"]
            result[arm] = {k: int(query[k]) for k in ("selected_tile_count", "selected_depth_bin_count",
                                                      "selected_tile_depth_count", "frustum_count")}
            result[arm].update(target_count=len(target), target_depth_p50_m=float(np.median(target_depth)),
                               correspondence_fraction=float(np.mean(valid)),
                               unique_matched_targets=int(np.unique(nearest[valid]).size),
                               source_depth_match_fraction=[float(np.mean(part)) for part in np.array_split(
                                   valid[np.argsort(source[:, 2], kind="stable")], 4)],
                               nearest_p50_m=float(np.median(distances)))
        result["overlap_fraction"] = len(np.intersect1d(ids["scored"], ids["unscored"])) / max(
            len(ids["scored"]), len(ids["unscored"]))
        if "spatial" in ids:
            result["spatial_overlap"] = {arm: len(np.intersect1d(ids[arm], ids["spatial"])) /
                                         max(len(ids[arm]), len(ids["spatial"])) for arm in ARMS}
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--spatial", action="store_true", help="Add within-bucket spatial strata")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = safe_output(args.output_root)
    windows = (("room2", 0, None), ("office0", 0, None)) if args.full else WINDOWS
    arms = ARMS + ("spatial",) if args.spatial else ARMS
    configs = {s: {a: arm_config(s, "voxel_blocks") for a in arms} for s, _, _ in windows}
    for pair in configs.values():
        for arm, cfg in pair.items():
            cfg["local_submap"]["use_map_scores"] = arm == "scored"
            if arm == "spatial":
                cfg["local_submap"]["coverage_representatives"] = "spatial_strata"
    protocol = dict(windows=windows, configs=configs,
                    advancement="candidate vs scored: JTS not lower and lost not higher in each window; at least one window improves JTS or lost",
                    limitation="hard window restarts with historical estimate, not complete tracker state")
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    paths = {Path(__file__), CASE, CASE.with_name("common.py"), ROOT / "scripts/fsm_localize.py",
             ROOT / "scripts/fsm_bench_closed_loop_queries.py", ROOT / "scripts/fsm_bench_compact_convergence.py",
             ROOT / "scripts/fsm_cn_consistency_experiments.py", ROOT / "scripts/splatam.py",
             *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    old = MVP / "cn_consistency_query_closed_loop_full_v1"
    maps = {s: input_paths("replica", s)[0] for s in configs}
    for s, pair in configs.items():
        data = pair["scored"]["data"]
        paths.update((maps[s], Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
        name = f"replica_{s}_baseline"
        paths.update((MVP / "cn_consistency_dev_v1/runs" / name / "localize" / name).glob("trajectory_*.txt"))
        paths.add(MVP / "cn_consistency_dev_v1/runs" / name / "localize" / name / "frame_metrics.csv")
    for s, start, _ in windows:
        if start:
            paths.add(old / f"{s}_r1/voxel_blocks/trajectory_est.txt")
    hashes = [file_record(p) for p in sorted(paths)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protocol.json").open("x") as f:
        json.dump(dict(protocol, source_and_input_hashes=hashes), f, indent=2, allow_nan=False)
    results, diagnostics = {}, {}
    for i, (scene, start, end) in enumerate(windows):
        os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
        dataset = load_dataset_from_config(str(CASE))
        scored = configs[scene]["scored"]
        if scene not in diagnostics:
            panel, _ = samples(scene, scored["registration"], "pilot")
            diagnostics[scene] = diagnostic(dataset, maps[scene], scored, [frame for _, frame in panel], arms)
        initial_args = {}
        if start:
            row = np.loadtxt(old / f"{scene}_r1/voxel_blocks/trajectory_est.txt")[start]
            if row[0] != start:
                raise ValueError("Wrong restart frame")
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(row[4:]).as_matrix()
            pose[:3, 3] = row[1:4]
            initial_args = dict(first_pose_protocol_override="explicit", explicit_initial_pose=pose)
        key = f"{scene}_{start}"
        metrics, identities = {}, {}
        for arm in (arms[i % len(arms):] + arms[:i % len(arms)]):
            target = output / key / arm
            print(f"{key}: {arm} to {end}", flush=True)
            summary = run_localization(configs[scene][arm], str(maps[scene]), str(target), dataset=dataset,
                                       start=start, end=end, **initial_args)
            metrics[arm], identities[arm], _ = run_metrics(target, summary, 100)
        gt = (output / key / "scored/trajectory_gt.txt").read_bytes()
        if any(identities["scored"] != identities[arm] or
               gt != (output / key / arm / "trajectory_gt.txt").read_bytes() for arm in arms[1:]):
            raise ValueError("Closed-loop arms ran different frames")
        results[key] = metrics
        del dataset
    eligible = [arm for arm in arms[1:] if all(r[arm]["jts_percent"] >= r["scored"]["jts_percent"]
                  and r[arm]["lost"] <= r["scored"]["lost"] for r in results.values()) and any(
                      r[arm]["jts_percent"] > r["scored"]["jts_percent"]
                      or r[arm]["lost"] < r["scored"]["lost"] for r in results.values())]
    if hashes != [file_record(p) for p in sorted(paths)]:
        raise RuntimeError("Recorded input or source changed during experiment")
    with (output / "summary.json").open("x") as f:
        json.dump(dict(windows=results, fixed_queries=diagnostics, advance=bool(eligible), eligible=eligible),
                  f, indent=2, allow_nan=False)
    print("Eligible for full validation:", eligible, flush=True)


if __name__ == "__main__":
    main()
