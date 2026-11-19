"""Delete missing geometry from full maps and add the identical IDs to compact maps."""

import argparse
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_additive_patches import SCENES, NOMINATION, read_frame, local_target
from scripts.fsm_bench_closed_loop_queries import arm_config
from scripts.fsm_bench_compact_convergence import errors, history_inputs, map_paths, run_solver, samples
from scripts.fsm_cn_consistency_experiments import CASE, file_record, safe_output
from utils.fsm_dataset import load_dataset_from_config
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_robust_registration import estimate_normals, transform_points


CELL_M = .50
REGIONS = 4
ARMS = ("full", "base", *(f"{side}_{g}" for g in (*map(str, range(REGIONS)), "union")
                          for side in ("drop", "add")))


def make_index(points, cfg, query):
    index = MapSpatialIndex(points, backend=cfg["local_submap"]["spatial_backend"])
    return index, index.prebuild_blocks(query.block_size_m, None)["block_score_token"]


def nominate(dataset, maps, is_missing, cfg, query):
    # Rank by where the full target has closer Euclidean support at the reference pose.
    # This only proposes regions; the interventions below measure the pose effect.
    xyz, masses = [], []
    for frame in NOMINATION:
        source, pose, camera = read_frame(dataset, frame, cfg)
        world = transform_points(source, pose)
        distances, target_ids = {}, {}
        for arm, (index, token) in maps.items():
            ids, _, tree = local_target(index, pose, camera, query, token)
            distances[arm], local = tree.query(world)
            target_ids[arm] = ids[local]
        take = is_missing[target_ids["full"]] & (distances["full"] <= cfg["registration"]["max_correspondence_distance"])
        weight = np.clip(distances["base"] - distances["full"], 0, .07)
        take &= weight > 0
        xyz.append(maps["full"][0].points[target_ids["full"][take]])
        masses.append(weight[take])
    cells, inverse = np.unique(np.floor(np.concatenate(xyz) / CELL_M).astype(np.int64), axis=0, return_inverse=True)
    votes = np.bincount(inverse, weights=np.concatenate(masses))
    if len(cells) < REGIONS:
        raise ValueError("Too few nomination regions")
    ranked = np.argsort(-votes, kind="stable")[:REGIONS]
    return cells[ranked], votes[ranked]


def evaluate(scene, cfg, panel):
    os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
    dataset = load_dataset_from_config(str(CASE))
    loc, reg = cfg["local_submap"], cfg["registration"]
    query = LocalSubmapConfig(**{f.name: loc[f.name] for f in fields(LocalSubmapConfig) if f.name in loc})
    with np.load(map_paths(scene)["full"], allow_pickle=False) as saved:
        full, full_ids = saved["means"], saved["indices"]
    with np.load(map_paths(scene)["compact"], allow_pickle=False) as saved:
        base, base_ids = saved["means"], saved["indices"]
    order = np.argsort(full_ids)
    offsets = np.searchsorted(full_ids[order], base_ids)
    if np.any(offsets >= len(order)) or not np.array_equal(full_ids[order[offsets]], base_ids):
        raise ValueError("Compact map IDs do not belong to full map")
    np.testing.assert_allclose(full[order[offsets]], base, atol=1e-7, rtol=0)
    is_missing = ~np.isin(full_ids, base_ids)
    maps = {arm: make_index(points, cfg, query) for arm, points in (("full", full), ("base", base))}
    cells, votes = nominate(dataset, maps, is_missing, cfg, query)
    point_cells = np.floor(maps["full"][0].points / CELL_M).astype(np.int64)
    groups = {str(i): np.flatnonzero(is_missing & np.all(point_cells == cell, axis=1))
              for i, cell in enumerate(cells)}
    del point_cells
    groups["union"] = np.concatenate(list(groups.values()))
    if any(not len(group) for group in groups.values()) or len(np.unique(groups["union"])) != len(groups["union"]):
        raise ValueError("Invalid or overlapping interventions")
    frames = {i: read_frame(dataset, i, cfg) for _, i in panel[0]}
    rows = []
    for arm in ARMS:
        exposure = None
        if arm in maps:
            index, token = maps.pop(arm)
            if arm == "full":
                exposure = np.full(len(full), -1, dtype=np.int8)
                for i in range(REGIONS):
                    exposure[groups[str(i)]] = i
        else:
            side, group = arm.split("_", 1)
            if side == "drop":
                keep = np.ones(len(full), dtype=bool)
                keep[groups[group]] = False
                points = full[keep]
            else:
                points = np.concatenate((base, full[groups[group]]))
            index, token = make_index(points, cfg, query)
            del points
        for label, i in panel[0]:
            source, truth, camera = frames[i]
            for perturbation, donor, delta in panel[1]:
                initial = truth @ delta
                ids, target, tree = local_target(index, initial, camera, query, token)
                normals = estimate_normals(target, k=reg["normal_max_nn"], target_tree=tree)
                result = run_solver(source, target, normals, tree, initial, reg, 30)
                pose = result["transform"]
                t, r = errors(truth, pose)
                valid = bool(result["optimizer_valid"])
                added = matches = None
                if arm.startswith("add_"):
                    added = int(np.count_nonzero(ids >= len(base)))
                    distance, nearest = tree.query(transform_points(source, pose))
                    matches = int(np.count_nonzero((ids[nearest] >= len(base)) &
                                                   (distance <= reg["max_correspondence_distance"])))
                rows.append(dict(scene=scene, frame=i, view_group=label, perturbation=perturbation, donor=donor,
                                 arm=arm, valid=valid, translation_mm=1000*t, rotation_deg=r,
                                 strict=valid and t <= .02 and r <= 1, coarse=valid and t <= .05 and r <= 2,
                                 target_count=len(ids), added_in_target=added, matches_to_added=matches,
                                 full_target_by_region=None if exposure is None else json.dumps(
                                     [int(np.count_nonzero(exposure[ids] == g)) for g in range(REGIONS)])))
        del index
        print(f"{scene}: completed {arm}", flush=True)
    details = dict(base=len(base), full=len(full), cell_m=CELL_M, cells=cells.tolist(), votes=votes.tolist(),
                   counts={k: len(v) for k, v in groups.items()},
                   global_ids={k: full_ids[groups[k]].tolist() for k in groups if k != "union"})
    return rows, details


def summarize(rows):
    trials = {}
    for row in rows:
        key = (row["frame"], row["perturbation"])
        arm = row["arm"]
        if arm in trials.setdefault(key, {}):
            raise ValueError("Duplicate trial arm")
        trials[key][arm] = row
    if any(set(trial) != set(ARMS) for trial in trials.values()):
        raise ValueError("Incomplete trial")
    summary = {}
    for label in ("zero", "perturbed"):
        pairs = [(key, r) for key, r in trials.items() if (key[1] == "zero") == (label == "zero")]
        section = summary[label] = {}
        section["reference"] = {arm: dict(strict=sum(r[arm]["strict"] for _, r in pairs),
                                         coarse=sum(r[arm]["coarse"] for _, r in pairs),
                                         median_mm=float(np.median([r[arm]["translation_mm"] for _, r in pairs])))
                                for arm in ("base", "full")}
        for group in (*map(str, range(REGIONS)), "union"):
            add, drop = f"add_{group}", f"drop_{group}"
            full_loss, base_gain, rotations, add_rotations, jointly_helpful = [], [], [], [], set()
            for (frame, _), r in pairs:
                loss = r[drop]["translation_mm"] - r["full"]["translation_mm"]
                gain = r["base"]["translation_mm"] - r[add]["translation_mm"]
                full_loss.append(loss)
                base_gain.append(gain)
                rotations.append(r[drop]["rotation_deg"] - r["full"]["rotation_deg"])
                add_rotations.append(r["base"]["rotation_deg"] - r[add]["rotation_deg"])
                if all(r[a]["valid"] for a in (add, drop, "base", "full")) and loss >= 2 and gain >= 2:
                    jointly_helpful.add(frame)
            section[group] = dict(n=len(pairs), drop_harm_mm=float(np.median(full_loss)),
                                  add_gain_mm=float(np.median(base_gain)),
                                  drop_harm_deg=float(np.median(rotations)), add_gain_deg=float(np.median(add_rotations)),
                                  joint_ge2mm_frames=sorted(jointly_helpful),
                                  add_strict=sum(r[add]["strict"] for _, r in pairs),
                                  drop_strict=sum(r[drop]["strict"] for _, r in pairs),
                                  add_coarse_regressions=sum(r["base"]["coarse"] and not r[add]["coarse"] for _, r in pairs),
                                  add_valid=sum(r[add]["valid"] for _, r in pairs),
                                  drop_valid=sum(r[drop]["valid"] for _, r in pairs),
                                  exposed=sum(r[add]["matches_to_added"] > 0 for _, r in pairs),
                                  add_target_median=float(np.median([r[add]["added_in_target"] for _, r in pairs])))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = safe_output(args.output_root)
    configs = {scene: arm_config(scene, "voxel_blocks") for scene in SCENES}
    panels = {}
    for scene, cfg in configs.items():
        cfg["local_submap"].update(use_map_scores=False, coverage_representatives="spatial_strata")
        panels[scene] = samples(scene, cfg["registration"], "pilot")
        if set(NOMINATION) & {i for _, i in panels[scene][0]}:
            raise ValueError("Nomination frames overlap held-out views")
    manifest = dict(arms=ARMS, nomination_frames=NOMINATION, cell_m=CELL_M, configs=configs,
                    holdout={s: [v, [(l, i, d.tolist()) for l, i, d in offsets]] for s, (v, offsets) in panels.items()},
                    gate="in each scene find >=1 group with same-sample drop harm and add gain >=2mm on >=2 perturbed views, and no coarse regressions in either panel",
                    scope="same-sequence holdout, target capped at 10000, delete only previously missing points")
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    paths = {Path(__file__).resolve(), CASE, CASE.with_name("common.py"), ROOT / "scripts/fsm_localize.py",
             ROOT / "scripts/fsm_cn_consistency_experiments.py", ROOT / "scripts/splatam.py",
             ROOT / "scripts/fsm_check_pose_update_preservation.py", *ROOT.glob("scripts/fsm_bench*.py"),
             *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    for scene, cfg in configs.items():
        paths.update(map_paths(scene)[a] for a in ("compact", "full"))
        paths.update(history_inputs(scene).values())
        data = cfg["data"]
        paths.update((Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    frozen = [file_record(p) for p in sorted(paths)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(manifest, source_and_input_hashes=frozen), stream, indent=2, allow_nan=False)
    details, summary = {}, {}
    with (output / "paired.csv").open("x", newline="") as stream:
        writer = None
        for scene in SCENES:
            rows, details[scene] = evaluate(scene, configs[scene], panels[scene])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            summary[scene] = summarize(rows)
    eligible = {scene: [group for group, r in x["perturbed"].items() if group != "reference"
                        and len(r["joint_ge2mm_frames"]) >= 2 and not r["add_coarse_regressions"]
                        and not x["zero"][group]["add_coarse_regressions"]] for scene, x in summary.items()}
    if frozen != [file_record(p) for p in sorted(paths)]:
        raise RuntimeError("Recorded input or source changed during experiment")
    for name, content in (("regions", details), ("summary", dict(scenes=summary, eligible=eligible, advance=all(eligible.values())))):
        with (output / f"{name}.json").open("x") as stream:
            json.dump(content, stream, indent=2, allow_nan=False)
    print(f"Eligible: {eligible}", flush=True)


if __name__ == "__main__":
    main()
