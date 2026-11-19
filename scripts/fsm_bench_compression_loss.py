"""Measure fixed-map localization loss against a same-protocol full-map reference."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_closed_loop_queries import SCENES, arm_config, run_metrics
from scripts.fsm_check_pose_update_preservation import scene_paths
from scripts.fsm_cn_consistency_experiments import CASE, file_record, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import load_dataset_from_config


THRESHOLDS = {"2cm_1deg": (.02, 1), "5cm_2deg": (.05, 2)}
REFERENCE_INCREASE = .50


def score_run(output, summary):
    metrics, ids, state = run_metrics(output, summary, 100.0)
    with (output / "per_frame_errors.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    is_valid = np.array([r[0] == "True" for r in state])
    error = np.array([[float(r["raw_translation_error_m"]), float(r["raw_rotation_error_deg"])]
                      if valid else [np.nan, np.nan] for r, valid in zip(rows, is_valid)])
    if not np.all(np.isfinite(error[is_valid]) & (error[is_valid] >= 0)):
        raise ValueError(f"{output}: missing or invalid raw error on valid frame")
    ratios = {key: float(np.mean(is_valid & (error[:, 0] <= t) & (error[:, 1] <= r)))
              for key, (t, r) in THRESHOLDS.items()}
    missing = np.flatnonzero(~is_valid)
    splits = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
    metrics.update(
        all_frame_joint_success=ratios,
        translation_p50_m=float(np.median(error[is_valid, 0])) if is_valid.any() else None,
        translation_p95_m=float(np.percentile(error[is_valid, 0], 95)) if is_valid.any() else None,
        rotation_p95_deg=float(np.percentile(error[is_valid, 1], 95)) if is_valid.any() else None,
        longest_lost_run=int(max(map(len, splits))),
        **{k: summary.get(k) for k in ("raw_rotation_abs_rmse_deg", "aligned_ate_rmse_m",
                                       "rpe_trans_rmse", "rpe_rot_rmse_deg")},
    )
    return metrics, ids, error


def compare(full, compact, full_errors, compact_errors):
    a, b = full["raw_ate_rmse_m"], compact["raw_ate_rmse_m"]
    available = all(x is not None and np.isfinite(x) for x in (a, b))
    increase = b / a - 1 if available and a > 0 else None
    full_valid = np.isfinite(full_errors[:, 0])
    compact_valid = np.isfinite(compact_errors[:, 0])
    common = full_valid & compact_valid
    delta = compact_errors[common, 0] - full_errors[common, 0]
    return dict(
        raw_ate_rmse_increase_m=b - a if available else None,
        raw_ate_rmse_relative_increase=increase,
        within_tentative_50pct_reference=increase <= REFERENCE_INCREASE if increase is not None else None,
        common_valid_frames=int(common.sum()),
        paired_translation_delta_p50_m=float(np.median(delta)) if len(delta) else None,
        paired_translation_delta_p95_m=float(np.percentile(delta, 95)) if len(delta) else None,
        compact_only_lost=int(np.count_nonzero(full_valid & ~compact_valid)),
        full_only_lost=int(np.count_nonzero(compact_valid & ~full_valid)),
        joint_success_delta_pp={k: 100 * (compact["all_frame_joint_success"][k] - full["all_frame_joint_success"][k])
                                for k in THRESHOLDS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--end", type=int, help="Exclusive stop; omit for complete trajectories")
    args = parser.parse_args()
    if args.end is not None and args.end < 2:
        parser.error("--end must be at least 2")
    root = safe_output(args.output_root)
    configs, paths, maps = {}, set(), {}
    for scene in SCENES:
        cfg = arm_config(scene, "voxel_blocks")
        cfg["local_submap"].update(use_map_scores=False, coverage_representatives="spatial_strata")
        configs[scene] = cfg
        maps[scene] = scene_paths(scene)
        data = cfg["data"]
        paths.update((*maps[scene].values(), Path(data["gradslam_data_cfg"]),
                      Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    plan = dict(configs=configs, map_paths={s: {k: str(p) for k, p in m.items()} for s, m in maps.items()},
                end_exclusive=args.end, thresholds=THRESHOLDS, raw_ate_reference_increase=REFERENCE_INCREASE,
                scope="historically exposed Replica development; reference line is descriptive, not application acceptance",
                includes_initialized_frame_in_accuracy=True, lost_counts_as_joint_failure=True,
                thread_env={k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    paths.update((Path(__file__).resolve(), CASE, CASE.with_name("common.py"), ROOT / "scripts/splatam.py",
                  ROOT / "scripts/fsm_localize.py", ROOT / "scripts/fsm_cn_consistency_experiments.py",
                  ROOT / "scripts/fsm_check_pose_update_preservation.py", ROOT / "scripts/fsm_bench_closed_loop_queries.py",
                  *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")))
    hashes = [file_record(p) for p in sorted(paths)]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol.json").open("x") as f:
        json.dump(dict(plan, source_and_input_hashes=hashes), f, indent=2, allow_nan=False)
    results = {}
    for i, scene in enumerate(SCENES):
        os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
        dataset = load_dataset_from_config(str(CASE))
        end = len(dataset) if args.end is None else args.end
        if end > len(dataset):
            raise ValueError(f"{scene}: requested end exceeds dataset")
        results[scene], identities, errors, gt_bytes = {}, {}, {}, {}
        with np.load(maps[scene]["full"], allow_pickle=False) as f, np.load(maps[scene]["compact"], allow_pickle=False) as c:
            full_ids, compact_ids = f["indices"], c["indices"]
            order = np.argsort(full_ids)
            offsets = np.searchsorted(full_ids[order], compact_ids)
            if np.any(offsets >= len(order)) or not np.array_equal(full_ids[order[offsets]], compact_ids):
                raise ValueError(f"{scene}: compact IDs are not a subset of full IDs")
            np.testing.assert_allclose(f["means"][order[offsets]], c["means"], atol=1e-7, rtol=0)
            counts = {"full": len(full_ids), "compact": len(compact_ids)}
        del full_ids, compact_ids, order, offsets
        order = ("full", "compact") if i == 0 else ("compact", "full")
        for arm in order:
            dest = root / scene / arm
            path = maps[scene][arm]
            print(f"{scene}: {arm}, {end} frames", flush=True)
            run = run_localization(configs[scene], str(path), str(dest), dataset=dataset, start=0, end=end)
            metrics, identities[arm], errors[arm] = score_run(dest, run)
            gt_bytes[arm] = (dest / "trajectory_gt.txt").read_bytes()
            results[scene][arm] = dict(metrics, map_bytes=path.stat().st_size, global_points=counts[arm])
        if (identities["full"] != identities["compact"] or gt_bytes["full"] != gt_bytes["compact"]
                or [idx for idx, _ in identities["full"]] != list(range(end))):
            raise ValueError(f"{scene}: arms do not have identical complete frame sets and ground truth")
        results[scene]["comparison"] = compare(results[scene]["full"], results[scene]["compact"],
                                                errors["full"], errors["compact"])
        print(json.dumps(results[scene]["comparison"], allow_nan=False), flush=True)
        del dataset
    if hashes != [file_record(p) for p in sorted(paths)]:
        raise RuntimeError("Recorded source or inputs changed during experiment")
    with (root / "summary.json").open("x") as f:
        json.dump(results, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
