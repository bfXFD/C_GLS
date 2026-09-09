import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_metrics import compute_ate_rmse, compute_rpe_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Compare audited full-sequence outputs with legacy 0-500 summaries.")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--legacy-replica", required=True)
    parser.add_argument("--legacy-tum", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-frames", type=int, default=500)
    return parser.parse_args()


def load_tum_poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        tx, ty, tz, qx, qy, qz, qw = row[1:8]
        norm = np.linalg.norm([qx, qy, qz, qw])
        qx, qy, qz, qw = np.asarray([qx, qy, qz, qw]) / norm
        rotation = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = [tx, ty, tz]
        poses.append(pose)
    return np.stack(poses)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def legacy_lookup(path: Path) -> dict[tuple[str, str, float], dict[str, str]]:
    rows = {}
    for row in read_csv(path):
        key = (row["scene"], row["strategy"], float(row["compression_ratio"]))
        rows.setdefault(key, row)
    return rows


def frame_latency(localize_dir: Path, limit: int) -> tuple[float, float]:
    rows = read_csv(localize_dir / "frame_metrics.csv")[:limit]
    values = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    return float(np.mean(values)), float(np.percentile(values, 95))


def main() -> None:
    args = parse_args()
    audit_root = Path(args.audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = {}
    legacy.update(legacy_lookup(Path(args.legacy_replica)))
    legacy.update(legacy_lookup(Path(args.legacy_tum)))
    rows = []
    dataset_scene = {
        "replica_room0_0": "room0_0",
        "replica_room1_0": "room1_0",
        "replica_room2_0": "room2_0",
        "replica_office0_0": "office0_0",
        "tum_freiburg1_desk": "freiburg1_desk",
        "tum_freiburg2_xyz": "freiburg2_xyz",
        "tum_freiburg3_long_office_household": "freiburg3_long_office_household",
    }
    for dataset, scene in dataset_scene.items():
        for branch in ("full", "pareto"):
            localize_root = audit_root / "main" / dataset / branch / "localize"
            if not localize_root.exists():
                continue
            for localize_dir in sorted(localize_root.iterdir()):
                summary_path = localize_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                if "_ratio_" not in localize_dir.name:
                    continue
                strategy, ratio_token = localize_dir.name.rsplit("_ratio_", 1)
                ratio = float(ratio_token.replace("p", "."))
                est = load_tum_poses(localize_dir / "trajectory_est.txt")
                gt = load_tum_poses(localize_dir / "trajectory_gt.txt")
                limit = min(args.prefix_frames, len(est))
                est_prefix = est[:limit]
                gt_prefix = gt[:limit]
                prefix_ate = compute_ate_rmse(est_prefix, gt_prefix, align=True)
                prefix_rpe = compute_rpe_rmse(est_prefix, gt_prefix)["trans_rmse"]
                prefix_latency, prefix_p95 = frame_latency(localize_dir, limit)
                old = legacy.get((scene, strategy, ratio), {})
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "strategy": strategy,
                        "ratio": ratio,
                        "prefix_frames": limit,
                        "audit_prefix_ate": prefix_ate,
                        "legacy_ate": old.get("ate_rmse", ""),
                        "audit_full_ate": summary["ate_rmse"],
                        "audit_prefix_rpe": prefix_rpe,
                        "legacy_rpe": old.get("rpe_rmse", ""),
                        "audit_full_rpe": summary.get("rpe_trans_rmse", ""),
                        "audit_prefix_latency_ms": prefix_latency,
                        "legacy_latency_ms": old.get("latency_mean_ms", ""),
                        "audit_full_latency_ms": summary["latency_mean_ms"],
                        "audit_prefix_p95_ms": prefix_p95,
                        "audit_full_p95_ms": summary["latency_p95_ms"],
                        "target_index_build_ms": summary.get("target_index_build_ms", ""),
                        "num_frames": summary["num_frames"],
                        "num_fallbacks": summary["num_fallbacks"],
                    }
                )
    output_path = output_dir / "audit_vs_legacy.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
