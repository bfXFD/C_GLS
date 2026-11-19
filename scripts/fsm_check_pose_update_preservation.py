"""Paired one-step ICP diagnostic on existing Replica maps; no map is exported."""

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, resolved_config, safe_output
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals, register_robust_point_to_plane, se3_exp, transform_points


SCENES = ("room2", "office0")
FRAME_IDS = tuple(range(125, 2000, 250))
TRANSLATION_M = 0.03
ROTATION_DEG = 2.0
MEANINGFUL_GAIN_MM = 1.0


def perturbations():
    yield "zero", np.zeros(6)
    for index, axis in enumerate(("rx", "ry", "rz", "tx", "ty", "tz")):
        scale = np.deg2rad(ROTATION_DEG) if index < 3 else TRANSLATION_M
        for sign in (-1, 1):
            twist = np.zeros(6)
            twist[index] = sign * scale
            yield f"{axis}{'+' if sign > 0 else '-'}", twist


def displacement_mm(first, second, source):
    delta = transform_points(source, first) - transform_points(source, second)
    return float(1000 * np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def solve(source, target, initial, cfg):
    minimum = cfg["min_correspondences"]
    if len(source) < minimum or len(target) < minimum:
        return {"transform": initial, "optimizer_valid": False, "failure_reason": "too_few_points"}
    tree = cKDTree(target)
    normals = estimate_normals(target, k=cfg.get("normal_k", cfg["normal_max_nn"]), target_tree=tree)
    return register_robust_point_to_plane(
        source, target, init_transform=initial, target_normals=normals, target_tree=tree,
        max_iterations=1, min_iterations=1, max_correspondence_distance=cfg["max_correspondence_distance"],
        loss=cfg["robust_loss"], loss_scale=cfg["loss_scale"], lm_damping=cfg["lm_damping"],
        min_correspondences=minimum, rank_tolerance=cfg["rank_tolerance"],
        rotation_step_tolerance_deg=cfg["rotation_step_tolerance_deg"],
        translation_step_tolerance_m=cfg["translation_step_tolerance_m"],
        relative_cost_tolerance=cfg["relative_cost_tolerance"], convergence_patience=cfg["convergence_patience"],
    )


def scene_paths(scene):
    compact, _ = input_paths("replica", scene)
    full = ROOT / (f"experiments/FSM/audit_v2_full_rerun/main/replica_{scene}_0/"
                   "full/maps/full_ratio_1/fsm_map.npz")
    return {"full": full, "compact": compact}


def evaluate_scene(scene):
    cfg = resolved_config("replica", scene, "baseline")
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    data_cfg = cfg["dataset"]
    loc = cfg["local_submap"]
    query = LocalSubmapConfig(**{field.name: loc[field.name] for field in fields(LocalSubmapConfig)
                                 if field.name in loc})
    maps, ids = {}, {}
    for name, path in scene_paths(scene).items():
        with np.load(path, allow_pickle=False) as saved:
            maps[name] = MapSpatialIndex(saved["means"], backend=loc["spatial_backend"])
            ids[name] = saved["indices"]
    if np.setdiff1d(ids["compact"], ids["full"]).size:
        raise ValueError("compact indices must be a subset of full map")
    order = np.argsort(ids["full"])
    matched = order[np.searchsorted(ids["full"][order], ids["compact"])]
    np.testing.assert_allclose(maps["compact"].points, maps["full"].points[matched], rtol=0, atol=1e-7)
    for frame_id in FRAME_IDS:
        frame = get_frame(dataset, frame_id)
        source = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"],
                                    depth_min=data_cfg["depth_min"], depth_max=data_cfg["depth_max"],
                                    max_points=data_cfg["max_points"], seed=frame_id)["points"].astype(np.float64)
        truth = np.array(frame["c2w"], dtype=np.float64, copy=True)
        truth[:3, :3] = Rotation.from_matrix(truth[:3, :3]).as_matrix()
        height, width = frame["depth"].shape[:2]
        for label, twist in perturbations():
            initial = truth @ se3_exp(twist)
            before = displacement_mm(initial, truth, source)
            row = dict(scene=scene, frame=frame_id, perturbation=label, initial_mm=before)
            poses = {}
            for name, index in maps.items():
                local = index.query_local_submap(initial, frame["intrinsics"], height, width, query)
                target = index.points[local["indices"]]
                solution = solve(source, target, initial, cfg["registration"])
                valid = bool(solution["optimizer_valid"])
                pose = solution["transform"] if valid else initial
                if valid:
                    poses[name] = pose
                row.update({f"{name}_{key}": value for key, value in dict(
                    valid=valid, failure=solution.get("failure_reason"), points=len(target),
                    correspondences=solution.get("num_correspondences"),
                    accepted_steps=solution.get("accepted_step_count", 0),
                    error_mm=displacement_mm(pose, truth, source),
                    translation_mm=float(1000 * np.linalg.norm(pose[:3, 3] - truth[:3, 3])),
                    rotation_deg=float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ pose[:3, :3]).magnitude())),
                ).items()})
            row["step_disagreement_mm"] = (displacement_mm(poses["full"], poses["compact"], source)
                                            if len(poses) == 2 else None)
            yield row
        print(f"{scene}: completed frame {frame_id}", flush=True)


def summarize(rows):
    perturbed = [r for r in rows if r["perturbation"] != "zero"]
    both = [r for r in perturbed if r["step_disagreement_mm"] is not None]

    def gain(row, arm):
        return row["initial_mm"] - row[f"{arm}_error_mm"]

    teacher_good = [r for r in perturbed if r["full_valid"] and gain(r, "full") > MEANINGFUL_GAIN_MM]
    eligible = [r for r in teacher_good if r["step_disagreement_mm"] is not None]
    # Remove per-perturbation medians before correlating across pooled directions.
    residuals = []
    for label, _ in perturbations():
        cluster = [r for r in eligible if r["perturbation"] == label]
        if len(cluster) > 1:
            gap = np.array([r["step_disagreement_mm"] for r in cluster])
            loss = np.array([-gain(r, "compact") for r in cluster])
            residuals.extend(zip(gap - np.median(gap), loss - np.median(loss)))
    rho = None
    if len(residuals) >= 3:
        x, y = np.array(residuals).T
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            rho = float(spearmanr(x, y).statistic)
    return dict(
        perturbed=len(perturbed), both_valid=len(both), full_good=len(teacher_good),
        compact_miss_frames=len({r["frame"] for r in teacher_good if gain(r, "compact") <= MEANINGFUL_GAIN_MM}),
        full_good_compact_invalid=sum(not r["compact_valid"] for r in teacher_good),
        full_good_compact_worse=sum(gain(r, "compact") < -MEANINGFUL_GAIN_MM for r in teacher_good),
        full_good_compact_not_meaningfully_improved=sum(gain(r, "compact") <= MEANINGFUL_GAIN_MM for r in teacher_good),
        directional_centered_rho_gap_vs_compact_negative_gain=rho, rho_rows=len(residuals),
        full_gain_median_mm=float(np.median([gain(r, "full") for r in perturbed])),
        compact_gain_median_mm=float(np.median([gain(r, "compact") for r in perturbed])),
        disagreement_median_mm=float(np.median([r["step_disagreement_mm"] for r in both])) if both else None,
        zero_drift_median_mm={arm: float(np.median([r[f"{arm}_error_mm"] for r in rows
                                                   if r["perturbation"] == "zero"])) for arm in ("full", "compact")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=MVP / "cn_consistency_update_probe_v1")
    args = parser.parse_args()
    output = safe_output(args.output_root)
    manifest = dict(scenes=SCENES, frame_ids=FRAME_IDS, translation_m=TRANSLATION_M, rotation_deg=ROTATION_DEG,
                    gain_threshold_mm=MEANINGFUL_GAIN_MM, histories="exposed development frames, no independence claim",
                    metric="RMS displacement of same query points under poses; diagnostic, not physical ATE",
                    control="both use identical unscored coverage cap, local PCA and one LM iteration; no fallback or reserve",
                    screening="proceed only if each scene has compact misses on >=2 frames with a good full teacher and centered rho >=0.3",
                    map_paths={scene: {key: str(p) for key, p in scene_paths(scene).items()} for scene in SCENES},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    manifest["inputs"] = [file_record(path) for scene in SCENES for path in scene_paths(scene).values()]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protocol.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    summaries = {}
    with (output / "paired.csv").open("x", newline="") as stream:
        writer = None
        for scene in SCENES:
            rows = []
            for row in evaluate_scene(scene):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            stream.flush()
            summaries[scene] = summarize(rows)
    with (output / "summary.json").open("x") as stream:
        json.dump(summaries, stream, indent=2, allow_nan=False)
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
