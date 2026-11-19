"""Offline nomination and held-out tests of additive compact-map patches."""

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

from scripts.fsm_bench_closed_loop_queries import arm_config
from scripts.fsm_bench_compact_convergence import errors, history_inputs, map_paths, run_solver, samples
from scripts.fsm_cn_consistency_experiments import CASE, file_record, safe_output
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals, robust_weights, transform_points


SCENES = ("room2", "office0")
NOMINATION = (250, 1000, 1500)
SIZE = 2048
PATCH_COUNTS = (2, 4)
ARMS = ("base", "patch2", "scatter2", "patch4", "scatter4", "full")


def read_frame(dataset, index, cfg):
    frame = get_frame(dataset, index)
    data = cfg["dataset"]
    source = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"], depth_min=data["depth_min"],
                                depth_max=data["depth_max"], max_points=data["max_points"], seed=index)["points"]
    pose = np.array(frame["c2w"], dtype=np.float64, copy=True)
    pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    return source, pose, (frame["intrinsics"], *frame["depth"].shape[:2])


def local_target(index, pose, camera, query, token):
    ids = index.query_local_submap(pose, *camera, query, block_score_token=token)["indices"]
    target = index.points[ids]
    return ids, target, cKDTree(target)


def nominate(dataset, base, missing, missing_ids, query, base_token, cfg):
    # Votes are hints, not effects: unmatched source or a large robust residual at reference pose.
    seeds, votes = [], []
    reg = cfg["registration"]
    gate = reg["max_correspondence_distance"]
    for i in NOMINATION:
        source, pose, camera = read_frame(dataset, i, cfg)
        _, target, tree = local_target(base, pose, camera, query, base_token)
        world = transform_points(source, pose)
        distance, nearest = tree.query(world)
        normals = estimate_normals(target, k=reg["normal_max_nn"], target_tree=tree)
        residual = np.einsum("ij,ij->i", world - target[nearest], normals[nearest])
        votes.append(np.where(distance > gate, reg["loss_scale"],
                              np.abs(residual) * robust_weights(residual, reg["robust_loss"], reg["loss_scale"])))
        seeds.append(world)
    seeds, votes = np.concatenate(seeds), np.concatenate(votes)
    voxels = np.floor(seeds / .30).astype(np.int64)
    cells, inverse = np.unique(voxels, axis=0, return_inverse=True)
    strength = np.bincount(inverse, weights=votes)
    tree = cKDTree(missing)
    proposals, occupied = [], set()
    for cell in np.argsort(-strength, kind="stable"):
        in_cell = inverse == cell
        center = np.average(seeds[in_cell], axis=0, weights=votes[in_cell]) if strength[cell] > 0 else None
        if center is None or any(np.linalg.norm(center - p["center"]) < .55 for p in proposals):
            continue
        pool = np.array(sorted(tree.query_ball_point(center, .25)), dtype=np.int64)
        if occupied:
            pool = pool[~np.isin(pool, list(occupied))]
        if len(pool) < 2 * SIZE:
            continue
        # Fit a tangent at the closest missing point, then prefer a compact tangent patch.
        _, anchor = tree.query(center)
        anchor = missing[anchor]
        _, neighbors = tree.query(anchor, k=128)
        centered = missing[neighbors] - np.mean(missing[neighbors], axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        delta = missing[pool] - anchor
        plane = np.abs(delta @ vectors[:, 0])
        local = np.flatnonzero(plane <= .025)
        if len(local) < SIZE:
            continue
        distance = np.einsum("ij,ij->i", delta, delta)
        order = local[np.lexsort((missing_ids[pool[local]], distance[local]))]
        patch = pool[order[:SIZE]]
        # Equal count and same parent region; control is unrestricted in the region.
        rng = np.random.default_rng(len(proposals))
        scatter = np.sort(rng.choice(pool, SIZE, replace=False))
        occupied.update(patch.tolist() + scatter.tolist())
        proposals.append(dict(center=center, vote=float(strength[cell]), pool=len(pool), patch=patch, scatter=scatter,
                              patch_plane_m=float(np.median(plane[order[:SIZE]])),
                              scatter_plane_m=float(np.median(plane[np.searchsorted(pool, scatter)]))))
        if len(proposals) == max(PATCH_COUNTS):
            break
    if len(proposals) != max(PATCH_COUNTS):
        raise ValueError(f"only nominated {len(proposals)} disjoint patch regions")
    return proposals


def evaluate(scene, panel, cfg):
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    loc, reg = cfg["local_submap"], cfg["registration"]
    query = LocalSubmapConfig(**{f.name: loc[f.name] for f in fields(LocalSubmapConfig) if f.name in loc})
    with np.load(map_paths(scene)["full"], allow_pickle=False) as f:
        full, full_ids = f["means"], f["indices"]
    with np.load(map_paths(scene)["compact"], allow_pickle=False) as f:
        base_points, base_ids = f["means"], f["indices"]
    removed = ~np.isin(full_ids, base_ids)
    missing, missing_ids = full[removed], full_ids[removed]
    order = np.argsort(full_ids)
    positions = np.searchsorted(full_ids[order], base_ids)
    if np.any(positions >= len(order)) or not np.array_equal(full_ids[order[positions]], base_ids):
        raise ValueError("Compact map not a subset of full map")
    np.testing.assert_allclose(full[order[positions]], base_points, atol=1e-7, rtol=0)
    base = MapSpatialIndex(base_points, backend=loc["spatial_backend"])
    token = base.prebuild_blocks(query.block_size_m, None)["block_score_token"]
    proposals = nominate(dataset, base, missing, missing_ids, query, token, cfg)
    chosen = {kind: [p[kind] for p in proposals] for kind in ("patch", "scatter")}
    maps = {"base": (base, token)}
    for arm in ARMS[1:]:
        if arm == "full":
            points = full
        else:
            indices = np.concatenate(chosen[arm[:-1]][:int(arm[-1])])
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"duplicate additions in {arm}")
            points = np.concatenate((base_points, missing[indices]))
        index = MapSpatialIndex(points, backend=loc["spatial_backend"])
        maps[arm] = index, index.prebuild_blocks(query.block_size_m, None)["block_score_token"]
    details = [{k: (v.tolist() if k == "center" else v) for k, v in proposal.items() if k not in chosen}
               | {f"{k}_global_ids": missing_ids[proposal[k]].tolist() for k in chosen} for proposal in proposals]
    rows = []
    for group, i in panel[0]:
        source, truth, camera = read_frame(dataset, i, cfg)
        for label, donor, delta in panel[1]:
            init = truth @ delta
            row = dict(scene=scene, frame=i, group=group, perturbation=label, donor=donor)
            for arm, (index, token) in maps.items():
                ids, target, tree = local_target(index, init, camera, query, token)
                normals = estimate_normals(target, k=reg["normal_max_nn"], target_tree=tree)
                result = run_solver(source, target, normals, tree, init, reg, 30)
                pose = result["transform"]
                translation, rotation = errors(truth, pose)
                valid = bool(result["optimizer_valid"])
                visible = matched = None
                if arm != "full":
                    added = ids >= len(base_points)
                    distances, nearest = tree.query(transform_points(source, pose))
                    visible = int(added.sum())
                    matched = int(np.count_nonzero(added[nearest] & (distances <= reg["max_correspondence_distance"])))
                row.update({f"{arm}_{key}": value for key, value in dict(
                    optimizer_valid=valid, translation_m=translation, rotation_deg=rotation,
                    strict=bool(valid and translation <= .02 and rotation <= 1),
                    coarse=bool(valid and translation <= .05 and rotation <= 2),
                    target_count=len(target), added_in_target=visible, source_matches_to_added=matched,
                ).items()})
            rows.append(row)
        print(f"{scene}: holdout frame {i}", flush=True)
    return rows, dict(base_count=len(base_points), full_count=len(full), proposals=details)


def summarize(rows):
    answer = {}
    for label in ("zero", "perturbed"):
        subset = [r for r in rows if (r["perturbation"] == "zero") == (label == "zero")]
        answer[label] = {}
        for arm in ARMS:
            def col(key):
                return [r[f"{arm}_{key}"] for r in subset]

            answer[label][arm] = dict(
                n=len(subset), valid=sum(col("optimizer_valid")), strict=sum(col("strict")), coarse=sum(col("coarse")),
                translation_mm=float(np.median(col("translation_m")) * 1000),
                translation_gain_mm=float(np.median([r["base_translation_m"] - r[f"{arm}_translation_m"] for r in subset]) * 1000),
                rotation_gain_deg=float(np.median([r["base_rotation_deg"] - r[f"{arm}_rotation_deg"] for r in subset])),
                strict_rescued=sum(not r["base_strict"] and r[f"{arm}_strict"] for r in subset),
                strict_broken=sum(r["base_strict"] and not r[f"{arm}_strict"] for r in subset),
                coarse_broken=sum(r["base_coarse"] and not r[f"{arm}_coarse"] for r in subset),
                added_target_p50=None if arm == "full" else float(np.median(col("added_in_target"))),
                added_matches_p50=None if arm == "full" else float(np.median(col("source_matches_to_added"))),
            )
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = safe_output(args.output_root)
    cfgs = {s: arm_config(s, "voxel_blocks") for s in SCENES}
    for cfg in cfgs.values():
        cfg["local_submap"].update(use_map_scores=False, coverage_representatives="spatial_strata")
    panels = {s: samples(s, cfgs[s]["registration"], "pilot") for s in SCENES}
    plan = dict(nomination=NOMINATION, patch_counts=PATCH_COUNTS, points_per_patch=SIZE, configs=cfgs,
                holdout={s: [views, [(l, i, d.tolist()) for l,i,d in offsets]] for s,(views,offsets) in panels.items()},
                evaluation="same-sequence held-out frames; independent solves, optimizer validity only",
                gate="same dose in both scenes: zero median translation <=90% of base, >=2 net strict rescues on perturbed, no coarse regressions, strict count above scatter on perturbed")
    if any(set(NOMINATION) & {i for _, i in views} for views, _ in panels.values()):
        raise ValueError("nomination and holdout overlap")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    paths = {Path(__file__).resolve(), CASE, CASE.with_name("common.py"), ROOT / "scripts/splatam.py",
             *ROOT.glob("scripts/fsm_bench*.py"), ROOT / "scripts/fsm_check_pose_update_preservation.py",
             ROOT / "scripts/fsm_cn_consistency_experiments.py", ROOT / "scripts/fsm_localize.py",
             *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    for s, cfg in cfgs.items():
        paths.update(map_paths(s)[k] for k in ("full", "compact"))
        paths.update(history_inputs(s).values())
        data = cfg["data"]
        paths.update((Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    hashes = [file_record(path) for path in sorted(paths)]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol.json").open("x") as stream:
        json.dump(dict(plan, source_and_input_hashes=hashes), stream, indent=2, allow_nan=False)
    summaries, selected = {}, {}
    with (root / "paired.csv").open("x", newline="") as stream:
        writer = None
        for s in SCENES:
            rows, selected[s] = evaluate(s, panels[s], cfgs[s])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            summaries[s] = summarize(rows)
    eligible = []
    for dose in PATCH_COUNTS:
        arm, control = f"patch{dose}", f"scatter{dose}"
        if all(s["zero"][arm]["translation_mm"] <= .9 * s["zero"]["base"]["translation_mm"]
               and s["perturbed"][arm]["strict_rescued"] - s["perturbed"][arm]["strict_broken"] >= 2
               and s["perturbed"][arm]["strict"] > s["perturbed"][control]["strict"]
               and not any(s[g][arm]["coarse_broken"] for g in s) for s in summaries.values()):
            eligible.append(arm)
    if hashes != [file_record(path) for path in sorted(paths)]:
        raise RuntimeError("Recorded input or code changed during experiment")
    with (root / "patches.json").open("x") as stream:
        json.dump(selected, stream, allow_nan=False)
    with (root / "summary.json").open("x") as stream:
        json.dump(dict(scenes=summaries, eligible=eligible), stream, indent=2, allow_nan=False)
    print("Eligible:", eligible, flush=True)


if __name__ == "__main__":
    main()
