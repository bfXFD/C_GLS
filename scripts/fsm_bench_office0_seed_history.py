"""Office0 history intervention: disable reserve without restarting the tracker."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.fsm.mvp_observable.common import make_reviewer_ablation
from scripts.fsm_cn_consistency_experiments import MVP, file_record, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_map import load_fsm_map


STOP = 1700
END = 1741
ARMS = (("corrweak128", "runtime_corr_weak", 0), ("random128_s2", "random", 2))


def read_run(path):
    with (path / "frame_metrics.csv").open(newline="") as f:
        states = list(csv.DictReader(f))
    with (path / "per_frame_errors.csv").open(newline="") as f:
        errors = list(csv.DictReader(f))
    keys = ("frame_id", "timestamp", "tracking_valid")
    assert len(states) == len(errors) and all(tuple(a[k] for k in keys) == tuple(b[k] for k in keys)
                                               for a, b in zip(states, errors))
    return np.loadtxt(path / "trajectory_est.txt", ndmin=2), states, errors


def error(row):
    return float(row["raw_translation_error_m"]) if row["tracking_valid"] == "True" else None


def alignment(source, points, pose, gate):
    world = source @ pose[:3, :3].T + pose[:3, 3]
    distance, idx = cKDTree(points).query(world)
    return dict(in_gate=float(np.mean(distance <= gate)), median_mm=float(np.median(distance) * 1000),
                clipped_mean_mm=float(np.minimum(distance, gate).mean() * 1000)), idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = safe_output(args.output_root)
    base = MVP / "runtime_reserve_v1/maps/replica_office0_0/logdet"
    map_path, pool = (base / n for n in ("fsm_map.npz", "activation_geometry.npz"))
    case = ROOT / "configs/fsm/mvp_observable/reviewer_ablation_case.py"
    old = MVP / "reviewer_ablation_suite_v2/runs"
    configs = {}
    for arm, selection, seed in ARMS:
        cfg = make_reviewer_ablation("replica", "office0", selection=selection, budget=128)
        cfg["observable_reserve"].update(geometry_path=str(pool), seed=seed)
        configs[arm] = cfg
    plan = dict(stop_source_idx=STOP, end_exclusive=END, configs=configs, map=str(map_path),
                treatment="only disable reserve at cutoff; preserve live tracker state",
                historical_comparison="current source re-run, not frozen-source reproduction")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    os.environ.update(CGLS_REVIEW_DATASET="replica", CGLS_REVIEW_SCENE="office0", CGLS_REVIEW_METHOD="corrweak")
    dataset = load_dataset_from_config(str(case))
    if len(dataset) != 2000:
        raise ValueError("Expected original Office0 query sequence")
    paths = [map_path, pool, case, case.with_name("common.py"), Path(__file__), ROOT / "scripts/fsm_localize.py",
             *sorted(ROOT.glob("utils/fsm*.py"))]
    out.mkdir(exist_ok=False)
    with (out / "protocol.json").open("x") as f:
        json.dump(dict(plan, fingerprints=[file_record(p) for p in paths]), f, indent=2, allow_nan=False)
    captures, summaries = {}, {}
    for arm, _, _ in ARMS:
        control = None
        for variant in ("continued", "cutoff"):
            observed = {}

            def observe(row):
                if row["source_idx"] >= STOP:
                    observed[row["source_idx"]] = row

            dest = out / arm / variant
            print(arm, variant, flush=True)
            run_localization(configs[arm], str(map_path), str(dest), dataset=dataset, start=0, end=END,
                             reserve_stop_source_idx=STOP if variant == "cutoff" else None,
                             primary_observer=observe)
            traj, states, errors = read_run(dest)
            if len(traj) != END or list(observed) != list(range(STOP, END)):
                raise ValueError("Incomplete run")
            if variant == "continued":
                key = f"replica_office0_{arm}"
                historical, _, _ = read_run(old / key / "localize" / key)
                control = traj
                good = np.all(historical[:END] == traj, axis=1)
                historical_first_difference = None if good.all() else int(np.flatnonzero(~good)[0])
            else:
                np.testing.assert_array_equal(traj[:STOP], control[:STOP])
                assert not any(row["reserve_attempted"] == "True" or row["reserve_used"] == "True" for row in states[STOP:])
                captures[arm] = observed
            summaries[f"{arm}/{variant}"] = dict(
                first_difference_from_historical=historical_first_difference if variant == "continued" else None,
                control_prefix_equal=True if variant == "cutoff" else None,
                late_adoptions=[i for i in range(STOP, END) if states[i]["reserve_used"] == "True"],
                late_recovery=[i for i in range(STOP, END) if states[i]["recovery_attempted"] == "True"],
                late_errors_m={str(i): error(errors[i]) for i in range(STOP, END)},
            )
    saved = load_fsm_map(str(map_path))
    order = np.argsort(saved["indices"])
    ids = saved["indices"][order]
    comparison = []
    arm_a, arm_b = (x[0] for x in ARMS)
    for frame_id in (1700, 1710, 1718, 1728, 1729, 1740):
        frame = get_frame(dataset, frame_id)
        cfg = configs[arm_a]
        data = cfg["dataset"]
        src = rgbd_to_pointcloud(None, frame["depth"], frame["intrinsics"], depth_min=data["depth_min"],
                                depth_max=data["depth_max"], max_points=data["max_points"], seed=frame_id)["points"]
        a, b = (captures[arm][frame_id] for arm in (arm_a, arm_b))
        ida, idb = (r["target_ids"] for r in (a, b))
        row = dict(frame=frame_id, target_jaccard=len(np.intersect1d(ida, idb))/len(np.union1d(ida, idb)),
                   prediction_separation_m=float(np.linalg.norm(a["initial_pose"][:3, 3] - b["initial_pose"][:3, 3])))
        for name, selected in zip((arm_a, arm_b), (ida, idb)):
            positions = np.searchsorted(ids, selected)
            np.testing.assert_array_equal(ids[positions], selected)
            target = saved["means"][order[positions]]
            row[name] = {label: alignment(src, target, pose, cfg["registration"]["max_correspondence_distance"])[0]
                         for label, pose in (("corrweak_pose", a["primary_pose"]), ("random_pose", b["primary_pose"]),
                                             ("gt_pose", frame["c2w"]))}
        comparison.append(row)
    with (out / "summary.json").open("x") as f:
        json.dump(dict(runs=summaries, cross_target_alignment=comparison), f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
