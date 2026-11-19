"""Test local-sampling prevention and held-out-depth recovery separately."""

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_closed_loop_queries import arm_config
from scripts.fsm_bench_compact_convergence import errors, poses, run_solver
from scripts.fsm_bench_compression_loss import score_run
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, safe_output
from scripts.fsm_localize import _tracking_acceptance, run_localization
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals


PILOT_FRAMES = (1660, 1690, 1700, 1710, 1718, 1728, 1740)
STRIDE_M = .45
BLOCK = 8
TOLERANCE_M = .08
MIN_SUPPORT = .15
MIN_MARGIN = .02


def depth_score(points, pose, intrinsics, depth):
    """Project onto held-out checkerboard tiles, with a fixed observed-tile denominator."""
    h, w = depth.shape
    h, w = h // BLOCK, w // BLOCK
    valid_depth = np.where(np.isfinite(depth) & (depth > .1) & (depth < 8), depth, np.inf)
    observed = valid_depth[:h*BLOCK, :w*BLOCK].reshape(h, BLOCK, w, BLOCK).min(axis=(1, 3))
    holdout = (np.indices((h, w)).sum(axis=0) % 2 == 0) & np.isfinite(observed)
    if not holdout.any():
        return None
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    camera = camera[(camera[:, 2] > .1) & (camera[:, 2] < 8)]
    u = np.floor((intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]) / BLOCK).astype(int)
    v = np.floor((intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]) / BLOCK).astype(int)
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    predicted = np.full((h, w), np.inf)
    np.minimum.at(predicted, (v[mask], u[mask]), camera[mask, 2])
    delta = predicted[holdout] - observed[holdout]
    agree = float(np.mean(np.abs(delta) <= TOLERANCE_M))
    contradiction = float(np.mean(delta < -TOLERANCE_M))
    return dict(agree=agree, contradiction=contradiction, coverage=float(np.isfinite(delta).mean()),
                score=agree - contradiction)


def choose(hypotheses):
    reference = hypotheses[0]
    best = max((h for h in hypotheses[1:] if h["valid"] and h["depth"] is not None
                and h["depth"]["agree"] >= MIN_SUPPORT), key=lambda h: h["depth"]["score"], default=None)
    if best is None or reference["depth"] is None:
        return reference
    return best if best["depth"]["score"] >= reference["depth"]["score"] + MIN_MARGIN else reference


def recovery(map_path, dataset, cfg, frame_ids, histories):
    loc, reg, data = (cfg[k] for k in ("local_submap", "registration", "dataset"))
    query = LocalSubmapConfig(**{f.name: loc[f.name] for f in fields(LocalSubmapConfig) if f.name in loc})
    query = replace(query, coverage_representatives="spatial_strata")
    with np.load(map_path, allow_pickle=False) as saved:
        index = MapSpatialIndex(saved["means"], backend=loc["spatial_backend"])
    token = index.prebuild_blocks(query.block_size_m, None)["block_score_token"]
    trajectories = {name: poses(path)[1] for name, path in histories.items()}
    rows = []
    for i in frame_ids:
        frame = get_frame(dataset, i)
        depth = np.asarray(frame["depth"]).squeeze()
        k, gt = frame["intrinsics"], frame["c2w"]
        v, u = np.ogrid[:depth.shape[0], :depth.shape[1]]
        training_depth = np.where((v//BLOCK + u//BLOCK) % 2 == 1, depth, 0.)
        source = rgbd_to_pointcloud(None, training_depth, k, depth_min=data["depth_min"],
                                    depth_max=data["depth_max"], max_points=data["max_points"], seed=i)["points"]
        for name, history in trajectories.items():
            seed = history[i-1]
            # Keep verifier geometry identical across every hypothesis, including the reference.
            verify_points = index.points[index.query_radius(seed[:3, 3], min(query.far + 2., 9.))]
            hypotheses, solved = [], []
            for j, offset in enumerate(np.vstack((np.zeros(3), STRIDE_M*np.eye(3), -STRIDE_M*np.eye(3)))):
                initial = seed.copy()
                initial[:3, 3] += seed[:3, :3] @ offset
                ids = index.query_local_submap(initial, k, *depth.shape, query, block_score_token=token)["indices"]
                target = index.points[ids]
                valid = False
                pose = initial
                if len(target) >= reg["min_correspondences"] and len(source) >= reg["min_correspondences"]:
                    tree = cKDTree(target)
                    normals = estimate_normals(target, k=reg["normal_max_nn"], target_tree=tree)
                    result = run_solver(source, target, normals, tree, initial, reg, 30)
                    valid, _ = _tracking_acceptance(result, reg["tracking_acceptance"], reg["backend"])
                    if np.isfinite(result["transform"]).all():
                        pose = result["transform"]
                    else:
                        valid = False
                hypotheses.append(dict(index=j, valid=bool(valid), depth=depth_score(verify_points, pose, k, depth)))
                solved.append(pose)
            # Select before computing any GT errors. This is not an online trigger or a closed-loop commit.
            selection = choose(hypotheses)["index"]
            for hyp, pose in zip(hypotheses, solved):
                t, r = errors(gt, pose)
                hyp.update(translation_m=t, rotation_deg=r, success=bool(hyp["valid"] and t <= .05 and r <= 2))
            rows.append(dict(frame=i, history=name, initial_translation_m=errors(gt, seed)[0],
                             selected=selection, hypotheses=hypotheses))
        print(f"recovery frame {i}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prevention", "recovery"), required=True)
    parser.add_argument("--scene", choices=("room2", "office0", "office4"), default="office4")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scene = "office0" if args.stage == "recovery" else args.scene
    out = safe_output(args.output_root)
    mp, _ = input_paths("replica", scene)
    cfg = arm_config(scene, "voxel_blocks")
    old = MVP / "reviewer_ablation_suite_v2/runs"
    histories = {a: old / k / "localize" / k / "trajectory_est.txt"
                 for a in ("corrweak128", "random128_s2") for k in (f"replica_office0_{a}",)}
    frames = tuple(range(1640, 1751, 2)) if args.full else PILOT_FRAMES
    plan = dict(stage=args.stage, scene=scene, end=None if args.full else 80, map=str(mp), config=cfg,
                recovery_frames=frames if args.stage == "recovery" else None, recovery_stride_m=STRIDE_M,
                recovery_score=dict(block=BLOCK, tolerance_m=TOLERANCE_M, minimum_support=MIN_SUPPORT,
                                    margin=MIN_MARGIN), scope="historically exposed research diagnostic")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    inputs = [mp, Path(__file__), CASE, CASE.with_name("common.py"), ROOT / "scripts/fsm_localize.py",
              ROOT / "scripts/fsm_bench_compact_convergence.py", ROOT / "scripts/fsm_bench_closed_loop_queries.py",
              ROOT / "scripts/fsm_bench_compression_loss.py", ROOT / "scripts/fsm_cn_consistency_experiments.py",
              *sorted(ROOT.glob("utils/fsm*.py"))]
    if args.stage == "recovery":
        inputs.extend(histories.values())
    out.mkdir(exist_ok=False)
    fingerprints = [file_record(p) for p in inputs]
    with (out / "protocol.json").open("x") as f:
        json.dump(dict(plan, fingerprints=fingerprints), f, indent=2, allow_nan=False)
    if args.stage == "recovery":
        rows = recovery(mp, dataset, cfg, frames, histories)
        def aggregate(group):
            return dict(n=len(group), base_pass=sum(r["hypotheses"][0]["success"] for r in group),
                        oracle_pass=sum(any(h["success"] for h in r["hypotheses"]) for r in group),
                        selected_pass=sum(r["hypotheses"][r["selected"]]["success"] for r in group),
                        selected_regressions=sum(r["hypotheses"][0]["success"] and
                                                 not r["hypotheses"][r["selected"]]["success"] for r in group),
                        switches=sum(r["selected"] != 0 for r in group))
        result = dict(all=aggregate(rows), by_history={k: aggregate([r for r in rows if r["history"] == k])
                                                      for k in histories}, rows=rows)
        print(result["all"], flush=True)
    else:
        result = {}
        identity = None
        for arm in ("scored", "spatial"):
            cfg = arm_config(scene, "voxel_blocks")
            cfg["local_submap"].update(use_map_scores=arm == "scored",
                                       coverage_representatives="ranked" if arm == "scored" else "spatial_strata")
            dest = out / arm
            print(scene, arm, flush=True)
            raw = run_localization(cfg, str(mp), str(dest), dataset=dataset, start=0,
                                   end=len(dataset) if args.full else 80)
            metrics, ids, _ = score_run(dest, raw)
            paired = (ids, (dest / "trajectory_gt.txt").read_bytes())
            if identity is not None and paired != identity:
                raise ValueError("A/B frame identity or ground truth mismatch")
            identity = paired
            result[arm] = metrics
    if fingerprints != [file_record(p) for p in inputs]:
        raise RuntimeError("Recorded inputs changed during the experiment")
    with (out / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
