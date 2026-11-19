import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("full_map", "compact_fps10", "corr_weak128_v4", "replace128_v4")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(root, arm):
    localize = root / arm / "localize" / arm
    summary_path = localize / "summary.json"
    errors_path = localize / "per_frame_errors.csv"
    summary = json.loads(summary_path.read_text())
    errors = np.genfromtxt(errors_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {"summary": summary, "errors": errors, "summary_path": summary_path, "errors_path": errors_path}


def jts(errors):
    valid = errors["tracking_valid"].astype(bool)
    translation = errors["raw_translation_error_m"].astype(float)
    rotation = errors["raw_rotation_error_deg"].astype(float)
    threshold = np.arange(1, 101)
    return 100.0 * float(np.mean(
        valid[:, None]
        & (translation[:, None] < 0.005 * threshold)
        & (rotation[:, None] < 0.1 * threshold)
    ))


def main():
    parser = argparse.ArgumentParser(description="Build the frozen desk-to-desk2 independent-revisit table.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.root, args.output):
        if "audit_v2_full_rerun" in str(Path(path)):
            parser.error(f"refusing frozen audit output path: {path}")
    root = Path(args.root).resolve()
    runs = {arm: load_arm(root, arm) for arm in ARMS}
    lengths = {len(run["errors"]) for run in runs.values()}
    if len(lengths) != 1:
        raise ValueError(f"arm lengths differ: {sorted(lengths)}")
    common = np.logical_and.reduce([
        run["errors"]["tracking_valid"].astype(bool)
        & np.isfinite(run["errors"]["raw_translation_error_m"])
        & np.isfinite(run["errors"]["raw_rotation_error_deg"])
        for run in runs.values()
    ])
    rows = []
    for arm in ARMS:
        run = runs[arm]
        summary = run["summary"]
        errors = run["errors"]
        rows.append({
            "map_sequence": "rgbd_dataset_freiburg1_desk2",
            "query_sequence": "rgbd_dataset_freiburg1_desk",
            "arm": arm,
            "frames": len(errors),
            "valid": int(errors["tracking_valid"].astype(bool).sum()),
            "lost": int(len(errors) - errors["tracking_valid"].astype(bool).sum()),
            "all_four_common": int(common.sum()),
            "jts_percent": jts(errors),
            "common_translation_rmse_m": float(np.sqrt(np.mean(errors["raw_translation_error_m"][common] ** 2))),
            "common_rotation_rmse_deg": float(np.sqrt(np.mean(errors["raw_rotation_error_deg"][common] ** 2))),
            "aligned_translation_rmse_m": summary["aligned_ate_rmse_m"],
            "rpe_translation_rmse_m": summary["rpe_trans_rmse"],
            "rpe_rotation_rmse_deg": summary["rpe_rot_rmse_deg"],
            "latency_mean_ms": summary["production_equivalent_total_frame_latency_mean_ms"],
            "latency_p95_ms": summary["production_equivalent_total_frame_latency_p95_ms"],
            "attempts": summary["num_reserve_attempt_frames"],
            "adoptions": summary["num_reserve_adopted_frames"],
            "summary_sha256": sha256(run["summary_path"]),
            "per_frame_errors_sha256": sha256(run["errors_path"]),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema_version": 1,
        "root": str(root),
        "table": str(output.resolve()),
        "table_sha256": sha256(output),
        "arms": list(ARMS),
        "all_four_common": int(common.sum()),
        "acceptance_sha256": sha256(root / "acceptance.json"),
        "uses_query_ground_truth_online": False,
        "first_pose_only_gt": True,
        "map_construction_query_frame_count": 0,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "common": int(common.sum()), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
