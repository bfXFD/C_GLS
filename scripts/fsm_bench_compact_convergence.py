"""Paired multistep ICP convergence on archived Replica maps, without writeback."""

import argparse
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_check_pose_update_preservation import SCENES, scene_paths
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, resolved_config, safe_output
from scripts.fsm_localize import _strict_motion_initialization, _tracking_acceptance
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals, register_robust_point_to_plane


def poses(path):
    rows = np.loadtxt(path, ndmin=2)
    transforms = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
    transforms[:, :3, :3] = Rotation.from_quat(rows[:, 4:]).as_matrix()
    transforms[:, :3, 3] = rows[:, 1:4]
    return rows[:, 0], transforms


def errors(truth, estimate):
    return (float(np.linalg.norm(truth[:3, 3] - estimate[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude())))


def history_inputs(scene):
    name = f"replica_{scene}_baseline"
    base = MVP / "cn_consistency_dev_v1/runs" / name / "localize" / name
    return {key: base / file for key, file in (("est", "trajectory_est.txt"),
                                               ("truth", "trajectory_gt.txt"), ("frames", "frame_metrics.csv"))}


def samples(scene, reg, scale):
    paths = history_inputs(scene)
    est_ts, est = poses(paths["est"])
    gt_ts, gt = poses(paths["truth"])
    with paths["frames"].open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    if len(est_ts) != len(frames) or not np.array_equal(est_ts, gt_ts):
        raise ValueError("trajectory identity mismatch")
    history, deltas = [], {}
    for i, row in enumerate(frames):
        if int(row["frame_id"]) != i or float(row["timestamp"]) != est_ts[i]:
            raise ValueError("CSV frame identity mismatch")
        if history:
            initial, source, translation, _, guard = _strict_motion_initialization(
                reg["motion_model"], history[-2:], i, reg["prediction_max_translation_m"],
                reg["prediction_max_rotation_deg"], reg.get("prediction_max_total_translation_m", float("inf")),
                reg.get("prediction_max_total_rotation_deg", float("inf")), reg["prediction_max_horizon_frames"],
            )
            if (source != row["init_source"] or guard != row["prediction_guard"]
                    or not np.isclose(translation, float(row["prediction_translation_m"]), atol=1e-7)):
                raise ValueError(f"{scene}:{i}: reconstructed motion initialization disagrees with logs")
            if row["tracking_valid"] == "True" and row["recovered"] == "False":
                deltas[i] = np.linalg.inv(gt[i]) @ initial
        if row["tracking_valid"] == "True":
            history.append((i, est[i], row["recovered"] == "True"))

    ranked = sorted(deltas, key=lambda i: sum(np.array(errors(np.eye(4), deltas[i])) / (0.05, 2)))
    quantiles = (0.5, 0.9) if scale == "pilot" else (0.25, 0.5, 0.9, 0.95)
    uniform, difficult = (4, 2) if scale == "pilot" else (16, 8)
    donors = [(f"q{q:g}", ranked[round(q*(len(ranked)-1))]) for q in quantiles]
    uniform_ids = [int(x) for x in np.linspace(1, len(gt)-1, uniform+2)[1:-1]]
    difficult_ids = []
    for i in reversed(ranked):
        if i not in uniform_ids and all(abs(i-j) >= 80 for j in difficult_ids):
            difficult_ids.append(i)
            if len(difficult_ids) == difficult:
                break
    if len(difficult_ids) != difficult:
        raise ValueError("could not find the requested number of separated difficult views")
    return ([("coverage", i) for i in uniform_ids] + [("difficult", i) for i in sorted(difficult_ids)],
            [("zero", None, np.eye(4))] + [(label, i, deltas[i]) for label, i in donors])


def map_paths(scene):
    return dict(scene_paths(scene), coverage=MVP / f"maps/replica_{scene}_0/cheap_baseline/fsm_map.npz")


def run_solver(source, target, normals, tree, initial, reg, budget):
    return register_robust_point_to_plane(
        source, target, init_transform=initial, target_normals=normals, target_tree=tree,
        max_iterations=budget, min_iterations=reg["min_iterations"],
        max_correspondence_distance=reg["max_correspondence_distance"], loss=reg["robust_loss"],
        loss_scale=reg["loss_scale"], lm_damping=reg["lm_damping"], min_correspondences=reg["min_correspondences"],
        rank_tolerance=reg["rank_tolerance"], rotation_step_tolerance_deg=reg["rotation_step_tolerance_deg"],
        translation_step_tolerance_m=reg["translation_step_tolerance_m"],
        relative_cost_tolerance=reg["relative_cost_tolerance"], convergence_patience=reg["convergence_patience"],
    )


def evaluate(scene, panel, perturbations, cfg):
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    data, loc, reg = (cfg[key] for key in ("dataset", "local_submap", "registration"))
    query = LocalSubmapConfig(**{f.name: loc[f.name] for f in fields(LocalSubmapConfig) if f.name in loc})
    maps, ids = {}, {}
    for arm, path in map_paths(scene).items():
        with np.load(path, allow_pickle=False) as saved:
            maps[arm] = MapSpatialIndex(saved["means"], backend=loc["spatial_backend"])
            ids[arm] = saved["indices"]
    if len(ids["coverage"]) != len(ids["compact"]):
        raise ValueError("compact counts differ")
    order = np.argsort(ids["full"])
    for arm in ("compact", "coverage"):
        if np.setdiff1d(ids[arm], ids["full"]).size:
            raise ValueError(f"{arm} is not a subset of full map")
        matched = order[np.searchsorted(ids["full"][order], ids[arm])]
        np.testing.assert_allclose(maps[arm].points, maps["full"].points[matched], rtol=0, atol=1e-7)

    for group, i in panel:
        frame = get_frame(dataset, i)
        truth = np.array(frame["c2w"], dtype=np.float64, copy=True)
        truth[:3, :3] = Rotation.from_matrix(truth[:3, :3]).as_matrix()
        source = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"], depth_min=data["depth_min"],
                                    depth_max=data["depth_max"], max_points=data["max_points"], seed=i)["points"]
        height, width = frame["depth"].shape[:2]
        for label, donor, delta in perturbations:
            initial = truth @ delta
            initial_t, initial_r = errors(truth, initial)
            row = dict(scene=scene, frame=i, sample_group=group, perturbation=label, donor_frame=donor,
                       initial_translation_m=initial_t, initial_rotation_deg=initial_r)
            for arm, index in maps.items():
                start = perf_counter()
                target_ids = index.query_local_submap(initial, frame["intrinsics"], height, width, query)["indices"]
                query_ms = (perf_counter()-start)*1000
                target = index.points[target_ids]
                start = perf_counter()
                if min(len(target), len(source)) < reg["min_correspondences"]:
                    tree = normals = None
                else:
                    tree = cKDTree(target)
                    normals = estimate_normals(target, k=reg.get("normal_k", reg["normal_max_nn"]), target_tree=tree)
                prepare_ms = (perf_counter()-start)*1000
                state = initial
                cumulative_ms = query_ms + prepare_ms
                for budget in ((30,) if arm == "full" else (30, 15)):
                    start = perf_counter()
                    if tree is None:
                        result = dict(transform=state, optimizer_valid=False, failure_reason="too_few_points")
                    else:
                        result = run_solver(source, target, normals, tree, state, reg, budget)
                    solve_ms = (perf_counter()-start)*1000
                    cumulative_ms += solve_ms
                    valid, reason = _tracking_acceptance(result, reg["tracking_acceptance"], reg["backend"])
                    translation, rotation = errors(truth, result["transform"])
                    state = result["transform"]
                    key = f"{arm}_{'30' if budget == 30 else '30plus15'}"
                    row.update({f"{key}_{k}": v for k,v in dict(
                        success=bool(valid and translation <= .05 and rotation <= 2),
                        strict=bool(valid and translation <= .02 and rotation <= 1),
                        translation_m=translation, rotation_deg=rotation, valid=valid, reason=reason,
                        termination=result.get("termination_reason"), correspondences=result.get("num_correspondences"),
                        iterations=result.get("iterations"), target_count=len(target), query_ms=query_ms,
                        prepare_ms=prepare_ms, solve_ms=solve_ms, cumulative_ms=cumulative_ms,
                    ).items()})
            yield row
        print(f"{scene}: completed {group} frame {i}", flush=True)


ARMS = ("full_30", "compact_30", "compact_30plus15", "coverage_30", "coverage_30plus15")


def summarize(rows):
    tested = [r for r in rows if r["perturbation"] != "zero"]
    result = {"cases": len(tested), "frames": len({r["frame"] for r in tested})}
    result["success"] = {a: sum(r[f"{a}_success"] for r in tested) for a in ARMS}
    result["strict"] = {a: sum(r[f"{a}_strict"] for r in tested) for a in ARMS}
    result["median_total_ms"] = {a: float(np.median([r[f"{a}_cumulative_ms"] for r in tested])) for a in ARMS}
    for arm in ("compact", "coverage"):
        failure = [r for r in tested if r["full_30_success"] and not r[f"{arm}_30_success"]]
        persistent = [r for r in failure if not r[f"{arm}_30plus15_success"]]
        frames = sorted({r["frame"] for r in persistent})
        result[arm] = dict(full_only=len(failure), rescued_by_restart=len(failure)-len(persistent),
                           persistent=len(persistent), persistent_frames=frames,
                           persistent_separated=(len(frames) >= 3 and frames[-1]-frames[0] >= 250),
                           restart_harms=sum(r[f"{arm}_30_success"] and not r[f"{arm}_30plus15_success"] for r in tested),
                           both_fail=sum(not r["full_30_success"] and not r[f"{arm}_30_success"] for r in tested),
                           compact_only=sum(not r["full_30_success"] and r[f"{arm}_30_success"] for r in tested))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("pilot", "large"), default="pilot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    dest = safe_output(args.output_root or MVP / f"cn_consistency_convergence_{args.scale}_v1")
    cfgs = {scene: resolved_config("replica", scene, "baseline") for scene in SCENES}
    plans = {scene: samples(scene, cfgs[scene]["registration"], args.scale) for scene in SCENES}
    manifest = dict(scale=args.scale, data="historically exposed checks; offline pose initialization only",
                    timing="query + index/PCA + registration; excludes decode and offline analysis",
                    controls="same unscored coverage sampler, same source and point budget; restart reuses target and PCA",
                    continuance="compact >=3 persistent views spanning >=250 frame indices in each scene",
                    thresholds=[[.05, 2], [.02, 1]],
                    plans={s: {"views": v, "perturbations": [(l,i,t.tolist()) for l,i,t in p]} for s,(v,p) in plans.items()})
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    manifest["inputs"] = [file_record(p) for s in SCENES for p in [*map_paths(s).values(), *history_inputs(s).values()]]
    manifest["sources"] = [file_record(p) for p in (Path(__file__), ROOT / "utils/fsm_robust_registration.py",
                           ROOT / "utils/fsm_local_map.py", ROOT / "scripts/fsm_localize.py", CASE, CASE.with_name("common.py"))]
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "protocol.json").open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    summary = {}
    with (dest / "paired.csv").open("x", newline="") as file:
        writer = None
        for scene, (panel, perturbations) in plans.items():
            rows = []
            for row in evaluate(scene, panel, perturbations, cfgs[scene]):
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            file.flush()
            summary[scene] = {group: summarize([r for r in rows if group == "all" or r["sample_group"] == group])
                              for group in ("all", "coverage", "difficult")}
            summary[scene]["zero_median_translation_m"] = {a: float(np.median([
                r[f"{a}_translation_m"] for r in rows if r["perturbation"] == "zero"])) for a in ARMS}
    summary["advance"] = all(summary[s]["all"]["compact"]["persistent_separated"] for s in SCENES)
    with (dest / "summary.json").open("x") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    print(json.dumps({s: summary[s]["all"] for s in SCENES} | {"advance": summary["advance"]}, indent=2))


if __name__ == "__main__":
    main()
