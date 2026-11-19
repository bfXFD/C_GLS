"""Run missing E3-E5 controls on frozen compact maps; default is a read-only plan."""

import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_compression_loss import score_run
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, resolved_config, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import load_dataset_from_config
from utils.fsm_map import export_runtime_reserve_geometry, load_fsm_map


SCENES = ("room2", "office4", "fr3")
SEEDS = (0, 1, 2)
STAGES = ("e3-replay", "e3-closed", "e4-replay", "e4-closed", "e5-closed")


def scene_key(scene):
    return ("tum", "freiburg3_long_office_household") if scene == "fr3" else ("replica", scene)


def arm_config(scene, arm, geometry):
    dataset, sequence = scene_key(scene)
    template = {"baseline": "baseline", "static": "baseline", "rerun": "rerun_frozen"}.get(arm, "corr_pca")
    cfg = resolved_config(dataset, sequence, template)
    cfg["local_submap"].update(frustum_culling="voxel_blocks", projection_pre_cap=None, oracle_mode="off",
                               use_map_scores=False, coverage_representatives="spatial_strata")
    reserve = cfg["observable_reserve"]
    cfg["activation_diagnostics"]["enabled"] = False
    cfg["paired_reserve_evaluation"]["enabled"] = False
    reserve["geometry_path"] = str(geometry)
    if arm in ("weak", "random"):
        reserve["corr_rank_control"] = arm
    if arm == "ungated":
        reserve["bypass_candidate_gate"] = True
    return cfg


def make_plan(stage, scenes, pools, out):
    runs = []
    deploy_paths = {}
    for scene in scenes:
        if stage in ("e3-closed",) and scene == "office4":
            continue
        base, geometry = input_paths(*scene_key(scene))
        if scene == "fr3":
            # Keep the same geometry-guided base across E3-E5, not the historical FPS hybrid.
            base = MVP / "runtime_reserve_v1/maps/tum_freiburg3_long_office_household/logdet/fsm_map.npz"
            geometry = base.with_name("activation_geometry.npz")
        entries = {}
        pool_proof = None
        if stage.startswith("e3"):
            from scripts.fsm_supplement_reserve_pools import pool_manifest
            pool_plan = pool_manifest(scene, pools)
            arms = e3_arms(stage, pool_plan)
            bases = {str(Path(x["map_path"]).resolve()) for x in pool_plan["pools"]}
            if bases != {str(base.resolve())}:
                raise ValueError(f"Unexpected E3 base for {scene}: {bases}")
            entries = {entry["arm"]: entry for entry in pool_plan["pools"]}
            pool_proof = pool_plan["manifest_path"]
        elif stage == "e4-replay":
            arms = [(name, geometry) for name in ("dynamic", "rerun", "weak", "random")]
            if scene != "office4":
                arms.extend((f"trigger_random_{seed}", geometry) for seed in SEEDS)
        elif stage == "e4-closed":
            arms = [(name, geometry) for name in ("baseline", "rerun", "static", "dynamic")]
        else:
            arms = [("ungated", geometry)]
        for arm, pool in arms:
            mode = "dynamic" if arm.startswith("trigger_random_") else arm
            cfg = arm_config(scene, mode, pool)
            deploy_pool = (deploy_paths.setdefault((scene, str(pool)), out / "deployment_pools" / scene / arm / "reserve.npz")
                           if cfg["observable_reserve"]["enabled"] else None)
            # The same archived geometry must resolve to the same logical configuration in E4 and E5.
            runs.append(dict(scene=scene, arm=arm, base=str(base), pool=str(pool),
                             pool_status=entries.get(arm, {}).get("status"), pool_proof=pool_proof,
                             runtime_pool=str(deploy_pool) if deploy_pool else None, config=cfg,
                             output=str(out / scene / arm)))
    return runs


def e3_arms(stage, pools):
    return [(p["arm"], p["geometry_path"]) for p in pools["pools"]
            if stage == "e3-replay" or p["arm"] in ("quality_spatial", "geometry")]


def valid_pose_error(pose, gt):
    if pose is None or gt is None:
        return None
    pose, gt = np.asarray(pose), np.asarray(gt)
    if pose.shape != (4, 4) or gt.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(gt).all():
        return None
    return (float(np.linalg.norm(pose[:3, 3] - gt[:3, 3])),
            float(Rotation.from_matrix(gt[:3, :3].T @ pose[:3, :3]).magnitude() * 180 / np.pi))


def success(error):
    return error is not None and error[0] <= .05 and error[1] <= 2


def probe_record(row, forced_schedule=False):
    record = {k: row[k] for k in ("frame_id", "timestamp")}
    gt = row.get("gt_pose")
    reserve = row["reserve_metrics"]
    record.update(triggered=bool(row["force_requested"] if forced_schedule else row["natural_triggered"]),
                  natural_triggered=bool(row["natural_triggered"]), force_requested=bool(row["force_requested"]),
                  attempted=bool(row["candidate_attempted"]),
                  candidate_basic_valid=bool(row.get("candidate_basic_valid")),
                  added=row.get("candidate_reserve_count"), reason=reserve.get("reserve_adoption_reason"),
                  candidate_ms=row.get("candidate_latency_ms"),
                  probe_ms=row.get("probe_latency_ms"),
                  selected=int(reserve.get("reserve_selected", 0)), visible=int(reserve.get("reserve_visible", 0)),
                  eligible=bool(row["force_eligible"]), has_gt=bool(gt is not None and np.isfinite(gt).all()),
                  submitted=bool(row["gated_adopted"]), baseline_source=row["baseline_source"],
                  gated_source=row["gated_source"], ungated_source=row["ungated_source"])
    if row["candidate_submitted"]:
        raise ValueError("Probe committed a candidate to the baseline")
    if not record["submitted"] and (row["baseline_source"] != row["gated_source"] or
            row["baseline_tracking_valid"] != row["gated_tracking_valid"] or
            not np.array_equal(row["baseline_pose"], row["gated_pose"])):
        raise ValueError("Rejected probe did not preserve baseline fallback")
    for branch in ("primary", "candidate", "baseline", "gated", "ungated"):
        valid = bool(row.get("candidate_basic_valid" if branch == "candidate" else f"{branch}_tracking_valid"))
        record[f"{branch}_valid"] = valid
        record[f"{branch}_error"] = valid_pose_error(row.get(f"{branch}_pose"), gt) if valid else None
    return record


def probe_summary(rows):
    cells = {f"primary_{p}_candidate_{c}": {"submitted": 0, "not_submitted": 0} for p in (0, 1) for c in (0, 1)}
    flow = Counter(dict.fromkeys(("frames", "triggered", "empty_proposal", "partial_addition_1_to_127",
                                  "candidate_basic_valid", "attempted_without_valid_candidate",
                                  "primary_invalid", "missing_gt", "valid_pair_with_gt"), 0))
    final_counts = Counter(dict.fromkeys(("baseline", "gated", "ungated", "candidate"), 0))
    candidate_delta = 0
    paired_deltas = []
    visible, added = [], []
    for row in rows:
        flow["frames"] += 1
        if not row["triggered"]:
            continue
        flow["triggered"] += 1
        visible.append(row["visible"])
        if row["added"] is not None:
            added.append(row["added"])
        flow["empty_proposal"] += not row["attempted"]
        flow["partial_addition_1_to_127"] += row["added"] is not None and 0 < row["added"] < 128
        flow["candidate_basic_valid"] += row["candidate_basic_valid"]
        flow["attempted_without_valid_candidate"] += row["attempted"] and not row["candidate_basic_valid"]
        flow["primary_invalid"] += not row["primary_valid"]
        flow["missing_gt"] += not row["has_gt"]
        for arm in ("baseline", "gated", "ungated", "candidate"):
            final_counts[arm] += success(row[f"{arm}_error"])
        if row["candidate_error"] is not None:
            candidate_delta += int(success(row["candidate_error"])) - int(success(row["baseline_error"]))
        if row["primary_error"] is not None and row["candidate_error"] is not None:
            paired_deltas.append(np.subtract(row["candidate_error"], row["primary_error"]))
            p, c = int(success(row["primary_error"])), int(success(row["candidate_error"]))
            cells[f"primary_{p}_candidate_{c}"]["submitted" if row["submitted"] else "not_submitted"] += 1
            flow["valid_pair_with_gt"] += 1
    for cell in cells.values():
        count = sum(cell.values())
        cell.update(n=count, submission_rate=cell["submitted"] / count if count else None)
    delta = final_counts["gated"] - final_counts["baseline"]
    n = flow["triggered"]
    timing = [r["candidate_ms"] for r in rows if r["triggered"] and r["candidate_ms"] is not None]
    return dict(flow=dict(flow), flow_flags_may_overlap=True, cells=cells, final_pass=dict(final_counts),
                final_pass_rate={k: v/n if n else None for k, v in final_counts.items()},
                candidate_valid_rate=flow["candidate_basic_valid"]/n if n else None,
                visible_p50=float(np.median(visible)) if visible else None,
                added_given_solve_p50=float(np.median(added)) if added else None,
                primary_to_candidate_delta_p50=np.median(paired_deltas, axis=0).tolist() if paired_deltas else None,
                attempt_candidate_else_baseline_net=candidate_delta,
                attempt_candidate_else_baseline_net_per_scheduled_frame=candidate_delta/n if n else None,
                gated_minus_baseline=delta, net_per_trigger=delta/n if n else None,
                candidate_ms_total=float(sum(timing)),
                probe_ms_total=float(sum(r["probe_ms"] or 0 for r in rows if r["triggered"])),
                joint_threshold=[.05, 2])


def trigger_sample(rows, seed):
    """Match attempt counts by sequence-length quartile, without looking at truth or score."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[i * 4 // len(rows)].append(row)
    rng = np.random.default_rng(seed)
    chosen = []
    for group in groups.values():
        count = sum(r["triggered"] for r in group)
        eligible = [r["frame_id"] for r in group if r["eligible"]]
        if len(eligible) < count:
            raise ValueError("Insufficient eligible frames for stratified trigger trial")
        chosen.extend(rng.choice(eligible, size=count, replace=False).tolist())
    return sorted(chosen)


def run_identity(rows):
    # No candidate is committed in a probe; lock the baseline result and frame identity across arms.
    return [(r["frame_id"], r["timestamp"], r["baseline_tracking_valid"],
             None if r.get("baseline_pose") is None else np.asarray(r["baseline_pose"]).tobytes(),
             None if r.get("primary_pose") is None else np.asarray(r["primary_pose"]).tobytes(),
             None if r.get("init_pose") is None else np.asarray(r["init_pose"]).tobytes()) for r in rows]


def aligned_subset(records):
    if not records:
        return {}
    groups = {arm: {(r["frame_id"], r["timestamp"]): r for r in seq if r["triggered"]}
              for arm, seq in records.items()}
    common = set.intersection(*(set(rows) for rows in groups.values()))
    kept = [key for key in sorted(common) if all(rows[key]["attempted"] for rows in groups.values())
            and len({rows[key]["added"] for rows in groups.values()}) == 1 and
            next(iter(groups.values()))[key]["added"] not in (0, None)]
    return {arm: probe_summary([rows[k] for k in kept]) for arm, rows in groups.items()}


def dump(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--pool-root", type=Path)
    parser.add_argument("--gated-root", type=Path, help="Completed E4 closed-loop root, required for E5")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.scenes)) != len(args.scenes):
        parser.error("duplicate scenes")
    if args.stage.startswith("e3") and args.pool_root is None:
        parser.error("E3 requires --pool-root")
    if args.stage == "e5-closed" and args.gated_root is None:
        parser.error("E5 requires --gated-root")
    out = safe_output(args.output_root)
    replay = args.stage.endswith("replay")
    runs = make_plan(args.stage, args.scenes, args.pool_root, out)
    if not runs:
        parser.error("no eligible scenes for the selected stage")
    plan = dict(stage=args.stage, runs=runs, research_only=True, frozen_replay=replay,
                scenes_historically_exposed=True, gate_definition="fixed candidate verifier, not a safety claim",
                e3_description="whole pool strategies if eligible sets differ",
                environment={key: os.getenv(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    assets = {Path(run[k]) for run in runs for k in ("base", "pool")}
    assets.update(Path(run["pool_proof"]) for run in runs if run["pool_proof"])
    for run in runs:
        data = run["config"]["data"]
        assets.add(Path(data["gradslam_data_cfg"]))
        sequence = Path(data["basedir"]) / data["sequence"]
        assets.update(sequence / f for f in (("traj.txt",) if run["scene"] != "fr3" else
                                              ("groundtruth.txt", "rgb.txt", "depth.txt")))
    reusable = {}
    if args.stage == "e5-closed" and not args.execute and not (args.gated_root / "summary.json").is_file():
        assets.add(args.gated_root / "summary.json")
    elif args.stage == "e5-closed":
        old = args.gated_root.resolve()
        prior = json.loads((old / "protocol.json").read_text())
        prior_summary = json.loads((old / "summary.json").read_text())
        for item in prior["source_and_input_hashes"]:
            if file_record(item["path"]) != item:
                raise ValueError(f"Gated reference source/input drift: {item['path']}")
        for run in runs:
            s = run["scene"]
            cfg = copy.deepcopy(run["config"])
            cfg["observable_reserve"].pop("bypass_candidate_gate")
            matches = [x for x in prior["runs"] if x["scene"] == s and x["arm"] == "dynamic"]
            if prior["stage"] != "e4-closed" or len(matches) != 1 or matches[0]["config"] != cfg:
                raise ValueError(f"Incompatible gated result for {s}")
            assets.update((Path(matches[0]["output"]) / k) for k in ("summary.json", "trajectory_gt.txt", "frame_metrics.csv"))
            reusable[s] = (matches[0], prior_summary[s]["dynamic"])
        plan["gated_reference"] = str(old)
        assets.update((old / "protocol.json", old / "summary.json"))
    sources = {Path(__file__).resolve(), CASE, CASE.with_name("common.py"),
               *ROOT.glob("scripts/fsm*.py"), *ROOT.glob("utils/fsm*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    plan["missing_inputs"] = [str(p) for p in sorted(assets) if not p.is_file()]
    unverified = [f"{r['scene']}/{r['arm']}" for r in runs if r["pool_status"] not in (None, "ready")]
    plan["unverified_pools"] = unverified
    if not args.execute:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    if plan["missing_inputs"]:
        raise FileNotFoundError("Missing inputs: " + ", ".join(plan["missing_inputs"]))
    if unverified:
        raise ValueError("Unverified pool proofs: " + ", ".join(unverified))
    if any(value != "1" for value in plan["environment"].values()):
        raise ValueError("Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1")
    frozen = [file_record(p) for p in sorted(assets | sources)]
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "protocol.json", dict(plan, source_and_input_hashes=frozen))
    summaries, initial, grouped = {}, {}, defaultdict(dict)
    identities, groundtruth, dataset, base_map, deployed = {}, {}, None, None, {}
    current_scene = None
    for run in runs:
        scene, arm = run["scene"], run["arm"]
        if current_scene != scene:
            if replay and current_scene is not None:
                initial.pop(current_scene, None)
            del dataset
            d, s = scene_key(scene)
            os.environ.update(CGLS_CN_DATASET=d, CGLS_CN_SCENE=s, CGLS_CN_ARM="baseline")
            dataset = load_dataset_from_config(str(CASE))
            base_map = load_fsm_map(run["base"])
            deployed = {}
            current_scene = scene
        dest = Path(run["output"])
        path = Path(run["base"])
        if arm == "static":
            path = out / scene / "static_map.npz"
            subprocess.run([sys.executable, str(ROOT / "scripts/fsm_materialize_static_reserve_map.py"),
                            "--base-map", run["base"], "--activation-geometry", run["pool"], "--output", str(path)],
                           cwd=ROOT, check=True)
        effective = copy.deepcopy(run["config"])
        reserve_enabled = effective["observable_reserve"]["enabled"]
        deployed_pool = None
        if reserve_enabled:
            if run["pool"] not in deployed:
                target = Path(run["runtime_pool"])
                target.parent.mkdir(parents=True, exist_ok=True)
                export_runtime_reserve_geometry(run["pool"], base_map, str(target))
                deployed[run["pool"]] = target
            deployed_pool = deployed[run["pool"]]
            effective["observable_reserve"]["geometry_path"] = str(deployed_pool)
        records, states = [], []
        force = None
        if arm.startswith("trigger_random_"):
            force = trigger_sample(grouped[scene]["dynamic"], int(arm.rsplit("_", 1)[1]))
        print(f"{args.stage} {scene} {arm}", flush=True)
        def collect(row):
            states.append({key: row.get(key) for key in ("frame_id", "timestamp", "baseline_tracking_valid",
                                                         "baseline_pose", "primary_pose", "init_pose")})
            records.append(probe_record(row, forced_schedule=force is not None))

        result = run_localization(effective, str(path), str(dest), dataset=dataset, start=0, end=len(dataset),
                                  **(dict(candidate_probe=collect, candidate_force_frames=force) if replay else {}))
        metrics, ids, _ = score_run(dest, result)
        gt = (dest / "trajectory_gt.txt").read_bytes()
        if len(ids) != len(dataset) or identities.setdefault(scene, ids) != ids or groundtruth.setdefault(scene, gt) != gt:
            raise ValueError(f"Frame identities or ground truth differ: {scene}/{arm}")
        with (dest / "frame_metrics.csv").open(newline="") as stream:
            frame_rows = list(csv.DictReader(stream))
        metrics.update(map_bytes=path.stat().st_size, reserve_bytes=deployed_pool.stat().st_size if deployed_pool else 0,
                       runtime_reserve_file=file_record(deployed_pool) if deployed_pool else None,
                       attempted=sum(r["reserve_attempted"] == "True" for r in frame_rows),
                       triggered=sum(r["reserve_triggered"] == "True" for r in frame_rows),
                       recovery_attempts=sum(int(r["recovery_count"] or 0) for r in frame_rows))
        metrics["deployment_bytes"] = metrics["map_bytes"] + metrics["reserve_bytes"]
        if replay:
            state = run_identity(states)
            if len(state) != len(ids) or initial.setdefault(scene, state) != state:
                raise ValueError(f"Probe did not preserve the common baseline state: {scene}/{arm}")
            if force is not None and [r["frame_id"] for r in records if r["force_requested"]] != force:
                raise ValueError("Random trigger trial did not execute exactly the frozen frame set")
            dump(dest / "probe.json", records)
            grouped[scene][arm] = records
            metrics["probe"] = probe_summary(records)
        if args.stage == "e5-closed":
            earlier, reference = reusable[scene]
            refdir = Path(earlier["output"])
            _, refids, _ = score_run(refdir, json.loads((refdir / "summary.json").read_text()))
            if ids != refids or gt != (refdir / "trajectory_gt.txt").read_bytes():
                raise ValueError("Gated and ungated results do not have the same frames")
            summaries.setdefault(scene, {})["gated_reused"] = reference
        summaries.setdefault(scene, {})[arm] = metrics
    if args.stage in ("e3-replay", "e4-replay"):
        for scene, arms in grouped.items():
            schedules = [[r["frame_id"] for r in rows if r["triggered"]]
                         for arm, rows in arms.items() if not arm.startswith("trigger_random_")]
            if any(schedule != schedules[0] for schedule in schedules[1:]):
                raise ValueError(f"Natural trigger sets differ across probes: {scene}")
    if args.stage == "e4-replay":
        for s, arms in grouped.items():
            summaries[s]["equal_added_selector_subset"] = aligned_subset({k: arms[k] for k in ("dynamic", "random", "weak")})
    if [file_record(p) for p in sorted(assets | sources)] != frozen:
        raise RuntimeError("Recorded source or inputs changed during the experiment")
    dump(out / "summary.json", summaries)


if __name__ == "__main__":
    main()
