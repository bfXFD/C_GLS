"""Fixed-target normal ablation; full geometry supplies normals offline, never matches."""

import argparse
import csv
from dataclasses import fields
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

from scripts.fsm_bench_compact_convergence import errors, history_inputs, map_paths, run_solver, samples
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, resolved_config, safe_output
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals, find_correspondences, transform_points


SCENES = ("room2", "office0")
ARMS = ("local", "compact_support", "full_support")


def support_normals(target, support, tree, k):
    distances, ids = tree.query(target, k=k, workers=-1)
    neighbors = support[ids]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered)
    _, vectors = np.linalg.eigh(covariances)
    normals = vectors[:, :, 0]
    dominant = np.argmax(np.abs(normals), axis=1)
    normals *= np.where(normals[np.arange(len(normals)), dominant] < 0, -1., 1.)[:, None]
    return normals, distances[:, -1]


def evaluate(scene, cfg, panel, unscored):
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    loc, data, reg = (cfg[key] for key in ("local_submap", "dataset", "registration"))
    query = LocalSubmapConfig(**{f.name: loc[f.name] for f in fields(LocalSubmapConfig) if f.name in loc})
    with np.load(map_paths(scene)["compact"], allow_pickle=False) as saved:
        compact = np.asarray(saved["means"], dtype=np.float64)
        ids = saved["indices"]
        scores = saved["scores"]
    with np.load(map_paths(scene)["full"], allow_pickle=False) as saved:
        full = saved["means"]
        full_ids = saved["indices"]
    order = np.argsort(full_ids)
    offsets = np.searchsorted(full_ids[order], ids)
    if np.any(offsets >= len(order)) or not np.array_equal(full_ids[order[offsets]], ids):
        raise ValueError("Compact IDs are not a subset of the full map")
    np.testing.assert_allclose(compact, full[order[offsets]], rtol=0, atol=1e-7)
    index = MapSpatialIndex(compact, backend=loc["spatial_backend"])
    support = {"compact_support": compact, "full_support": full}
    trees = {name: cKDTree(points) for name, points in support.items()}
    k = reg.get("normal_k", reg["normal_max_nn"])
    for group, i in panel[0]:
        frame = get_frame(dataset, i)
        gt = np.array(frame["c2w"], dtype=np.float64, copy=True)
        gt[:3, :3] = Rotation.from_matrix(gt[:3, :3]).as_matrix()
        source = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"], depth_min=data["depth_min"],
                                    depth_max=data["depth_max"], max_points=data["max_points"], seed=i)["points"]
        height, width = frame["depth"].shape[:2]
        for tag, donor, delta in panel[1]:
            initial = gt @ delta
            local_ids = index.query_local_submap(initial, frame["intrinsics"], height, width, query,
                                                  scores=None if unscored else scores)["indices"]
            target = compact[local_ids]
            tree = cKDTree(target)
            local_normals = estimate_normals(target, k=k, target_tree=tree)
            src_corr, tgt_corr, _ = find_correspondences(transform_points(source, initial), target,
                                                         reg["max_correspondence_distance"], tree)
            if not len(tgt_corr):
                raise ValueError("No common initial correspondences")
            row = dict(scene=scene, frame=i, group=group, perturbation=tag, donor=donor,
                       target_count=len(target), initial_correspondences=len(src_corr))
            for arm in ARMS:
                if arm == "local":
                    normals = local_normals
                    radius = tree.query(target, k=k, workers=-1)[0][:, -1]
                else:
                    normals, radius = support_normals(target, support[arm], trees[arm], k)
                dot = np.abs(np.sum(normals[tgt_corr] * local_normals[tgt_corr], axis=1))
                angle = np.rad2deg(np.arccos(np.clip(dot, 0, 1)))
                result = run_solver(source, target, normals, tree, initial, reg, 30)
                pose = result["transform"]
                translation, rotation = errors(gt, pose)
                valid = bool(result["optimizer_valid"])
                shift = gt[:3, :3].T @ (pose[:3, 3] - gt[:3, 3])
                row.update({f"{arm}_{key}": value for key, value in dict(
                    valid=valid, translation_m=translation, rotation_deg=rotation,
                    tx_m=float(shift[0]), ty_m=float(shift[1]), tz_m=float(shift[2]),
                    strict=bool(valid and translation <= .02 and rotation <= 1),
                    coarse=bool(valid and translation <= .05 and rotation <= 2),
                    normal_angle_p50_deg=float(np.median(angle)),
                    normal_angle_p95_deg=float(np.percentile(angle, 95)),
                    neighbor_radius_p50_m=float(np.median(radius[tgt_corr])),
                    iterations=result.get("iterations"), failure=result.get("failure_reason"),
                ).items()})
            yield row
        print(f"{scene}: frame {i}", flush=True)


def summarize(rows):
    perturbed = [r for r in rows if r["perturbation"] != "zero"]
    zero = [r for r in rows if r["perturbation"] == "zero"]
    summary = {}
    for arm in ARMS:
        values = {}
        for group, subset in (("perturbed", perturbed), ("zero", zero)):
            values[group] = dict(
                n=len(subset), coarse=sum(r[f"{arm}_coarse"] for r in subset),
                strict=sum(r[f"{arm}_strict"] for r in subset), valid=sum(r[f"{arm}_valid"] for r in subset),
                translation_m_p50=float(np.median([r[f"{arm}_translation_m"] for r in subset])),
                rotation_deg_p50=float(np.median([r[f"{arm}_rotation_deg"] for r in subset])),
                coarse_regressions=sum(r["local_coarse"] and not r[f"{arm}_coarse"] for r in subset),
                strict_gains=sum(not r["local_strict"] and r[f"{arm}_strict"] for r in subset),
                strict_losses=sum(r["local_strict"] and not r[f"{arm}_strict"] for r in subset),
                normal_angle_p95_deg_p50=float(np.median([r[f"{arm}_normal_angle_p95_deg"] for r in subset])),
                neighbor_radius_p50_m_p50=float(np.median([r[f"{arm}_neighbor_radius_p50_m"] for r in subset])),
            )
        summary[arm] = values
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scale", choices=("pilot", "large"), default="pilot")
    parser.add_argument("--unscored", action="store_true", help="Use coverage sampling without map scores")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = safe_output(args.output_root)
    configs = {s: resolved_config("replica", s, "baseline") for s in SCENES}
    panels = {s: samples(s, cfg["registration"], args.scale) for s, cfg in configs.items()}
    protocol = dict(scale=args.scale, arms=ARMS, same_target_ids=True, map_scores=not args.unscored,
                    scope="exposed Replica development, independent single-frame solves, no online full map",
                    criteria="same normal arm in both scenes: >=2 net strict gains, >=20% smaller zero median translation, no coarse regressions, all valid",
                    panels={s: [frames, [(tag, donor, delta.tolist()) for tag, donor, delta in offsets]]
                            for s, (frames, offsets) in panels.items()})
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    paths = {Path(__file__).resolve(), CASE, CASE.with_name("common.py"), ROOT / "scripts/splatam.py",
             *ROOT.glob("scripts/fsm_bench*convergence.py"), ROOT / "scripts/fsm_check_pose_update_preservation.py",
             ROOT / "scripts/fsm_localize.py", ROOT / "scripts/fsm_cn_consistency_experiments.py",
             *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    for scene, cfg in configs.items():
        paths.update(map_paths(scene)[arm] for arm in ("compact", "full"))
        paths.update(history_inputs(scene).values())
        data = cfg["data"]
        paths.update((Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    hashes = [file_record(p) for p in sorted(paths)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(protocol, source_and_input_hashes=hashes), stream, indent=2, allow_nan=False)
    rows = []
    with (output / "paired.csv").open("x", newline="") as stream:
        writer = None
        for scene in SCENES:
            for row in evaluate(scene, configs[scene], panels[scene], args.unscored):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
    summary = {s: summarize([r for r in rows if r["scene"] == s]) for s in SCENES}
    eligible = []
    for arm in ARMS[1:]:
        if all(s[arm]["perturbed"]["strict_gains"] - s[arm]["perturbed"]["strict_losses"] >= 2
               and s[arm]["zero"]["translation_m_p50"] <= .8 * s["local"]["zero"]["translation_m_p50"]
               and all(v["coarse_regressions"] == 0 and v["valid"] == v["n"] for v in s[arm].values())
               for s in summary.values()):
            eligible.append(arm)
    if hashes != [file_record(p) for p in sorted(paths)]:
        raise RuntimeError("Recorded input or code changed during experiment")
    with (output / "summary.json").open("x") as stream:
        json.dump(dict(scenes=summary, eligible=eligible), stream, indent=2, allow_nan=False)
    print("Eligible for closed-loop validation:", eligible, flush=True)


if __name__ == "__main__":
    main()
