import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_IDENTITY_FIELDS = ("frame_id", "timestamp", "pair_id", "policy", "seed", "budget")
_V2_RESERVED_FIELDS = {
    "offsets", "global_ids", "eligible_offsets", "eligible_global_ids",
    "selected_p_corr", "selected_actual_correspondence", "primary_pose", "candidate_pose",
    "parent_trajectory_sha256", "schema_version", "metric_definition_version",
}
_GT_CSV_FIELDS = [
    "row_index", "frame_id", "timestamp", "pair_id", "candidate_valid",
    "policy", "seed", "budget", "primary_translation_error_m",
    "primary_rotation_error_deg", "candidate_translation_error_m",
    "candidate_rotation_error_deg", "translation_delta_m", "rotation_delta_deg",
    "translation_win", "rotation_win",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GT scorer for paired-reserve shadow artifacts. Never used by localization."
    )
    parser.add_argument("--paired-npz", required=True, help="Path to paired_reserve_ids.npz.")
    parser.add_argument("--trajectory-gt", required=True, help="Same-case trajectory_gt.txt.")
    parser.add_argument(
        "--trajectory-est",
        default=None,
        help="Same-case trajectory_est.txt. Defaults to the case output containing the paired_reserve directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for paired_reserve_gt outputs.")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 8 or np.any(~np.isfinite(values)):
        raise ValueError(f"{path} must contain finite TUM rows: timestamp tx ty tz qx qy qz qw")
    timestamps = values[:, 0]
    if np.unique(timestamps).size != timestamps.size:
        raise ValueError(f"{path} contains duplicate timestamps")
    quaternions = values[:, 4:8]
    lengths = np.linalg.norm(quaternions, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError(f"{path} contains zero quaternions")
    quaternions = quaternions / lengths[:, None]
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], values.shape[0], axis=0)
    poses[:, :3, 3] = values[:, 1:4]
    x, y, z, w = quaternions.T
    poses[:, :3, :3] = np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)
    return timestamps, poses


def _errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    relative = truth[:3, :3].T @ estimate[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cosine)))


def score(
    paired_npz: str | Path,
    trajectory_gt: str | Path,
    output_dir: str | Path,
    trajectory_est: str | Path | None = None,
) -> dict[str, Any]:
    paired_path = Path(paired_npz).resolve()
    gt_path = Path(trajectory_gt).resolve()
    output = Path(output_dir).resolve()
    est_path = Path(trajectory_est).resolve() if trajectory_est else paired_path.parent.parent / "trajectory_est.txt"
    if not paired_path.is_file() or not gt_path.is_file() or not est_path.is_file():
        raise FileNotFoundError("paired NPZ and same-case GT/estimated trajectories are required")
    with np.load(paired_path, allow_pickle=False) as archive:
        required = (
            "frame_id", "timestamp", "pair_id", "policy", "seed", "budget",
            "primary_pose", "candidate_pose", "parent_trajectory_sha256",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"paired artifact missing fields: {missing}")
        frame_id = np.asarray(archive["frame_id"], dtype=np.int64)
        timestamp = np.asarray(archive["timestamp"], dtype=np.float64)
        pair_id = np.asarray(archive["pair_id"], dtype="U64")
        policy = np.asarray(archive["policy"], dtype="U32")
        seed = np.asarray(archive["seed"], dtype=np.int64)
        budget = np.asarray(archive["budget"], dtype=np.int64)
        primary = np.asarray(archive["primary_pose"], dtype=np.float64)
        candidate = np.asarray(archive["candidate_pose"], dtype=np.float64)
        parent_hash = str(np.asarray(archive["parent_trajectory_sha256"]).item())
        schema_version = int(np.asarray(archive["schema_version"]).item()) if "schema_version" in archive else 1
        if schema_version not in (1, 2):
            raise ValueError(f"unsupported paired artifact schema_version: {schema_version}")
        if schema_version >= 2 and "row_index" not in archive:
            raise KeyError("paired artifact schema v2 missing row_index")
        if schema_version >= 2 and "metric_definition_version" not in archive:
            raise KeyError("paired artifact schema v2 missing metric_definition_version")
        metric_definition_version = (
            str(np.asarray(archive["metric_definition_version"]).item())
            if "metric_definition_version" in archive else ""
        )
        row_index = (
            np.asarray(archive["row_index"], dtype=np.int64)
            if "row_index" in archive else np.arange(frame_id.size, dtype=np.int64)
        )
        diagnostic_arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if schema_version >= 2 and name not in _V2_RESERVED_FIELDS and name not in _IDENTITY_FIELDS
            and name != "row_index" and np.asarray(archive[name]).ndim > 0
            and np.asarray(archive[name]).shape[0] == frame_id.size
        }
    count = frame_id.size
    if any(value.shape != (count,) for value in (timestamp, pair_id, policy, seed, budget)) or primary.shape != (count, 4, 4) or candidate.shape != (count, 4, 4):
        raise ValueError("paired artifact row arrays have inconsistent shapes")
    if row_index.shape != (count,) or not np.array_equal(row_index, np.arange(count, dtype=np.int64)):
        raise ValueError("paired artifact row_index must be the identity row order")
    if not parent_hash or _sha256(est_path) != parent_hash:
        raise ValueError("paired artifact parent trajectory hash mismatch")
    gt_timestamps, gt_poses = _load_tum(gt_path)
    lookup = {float(value): index for index, value in enumerate(gt_timestamps)}
    gt_indices = []
    for value in timestamp:
        matches = np.flatnonzero(np.isclose(gt_timestamps, value, rtol=0.0, atol=5e-9))
        if matches.size != 1:
            raise ValueError(f"paired timestamp {value:.9f} has {matches.size} GT matches")
        gt_indices.append(int(matches[0]))
    if np.any(frame_id < 0):
        raise ValueError("paired frame IDs must be non-negative")

    rows = []
    arrays = {name: np.full(count, np.nan, dtype=np.float64) for name in (
        "primary_translation_error_m", "primary_rotation_error_deg",
        "candidate_translation_error_m", "candidate_rotation_error_deg",
        "translation_delta_m", "rotation_delta_deg",
    )}
    valid = np.all(np.isfinite(candidate), axis=(1, 2))
    for index, gt_index in enumerate(gt_indices):
        p_trans, p_rot = _errors(primary[index], gt_poses[gt_index])
        arrays["primary_translation_error_m"][index] = p_trans
        arrays["primary_rotation_error_deg"][index] = p_rot
        c_trans = c_rot = np.nan
        if valid[index]:
            c_trans, c_rot = _errors(candidate[index], gt_poses[gt_index])
            arrays["candidate_translation_error_m"][index] = c_trans
            arrays["candidate_rotation_error_deg"][index] = c_rot
            arrays["translation_delta_m"][index] = p_trans - c_trans
            arrays["rotation_delta_deg"][index] = p_rot - c_rot
        rows.append({
            "row_index": int(row_index[index]),
            "frame_id": int(frame_id[index]), "timestamp": f"{timestamp[index]:.9f}",
            "pair_id": str(pair_id[index]), "candidate_valid": bool(valid[index]),
            "policy": str(policy[index]), "seed": "" if seed[index] < 0 else int(seed[index]),
            "budget": int(budget[index]),
            **{name: ("" if not np.isfinite(values[index]) else float(values[index])) for name, values in arrays.items()},
            "translation_win": bool(valid[index] and c_trans < p_trans),
            "rotation_win": bool(valid[index] and c_rot < p_rot),
            **{
                name: _csv_value(values[index])
                for name, values in diagnostic_arrays.items()
                if values.ndim == 1
            },
            **{
                f"{name}_{row}_{col}": _csv_value(values[index, row, col])
                for name, values in diagnostic_arrays.items()
                if values.ndim == 3 and values.shape[1:] in ((6, 6), (6, 3))
                for row in range(values.shape[1]) for col in range(values.shape[2])
            },
        })
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "paired_reserve_gt.npz",
        schema_version=np.asarray(schema_version, dtype=np.int64),
        metric_definition_version=np.asarray(metric_definition_version, dtype="U64"),
        row_index=row_index, frame_id=frame_id, timestamp=timestamp, pair_id=pair_id,
        policy=policy, seed=seed, budget=budget,
        candidate_valid=valid, **arrays, parent_trajectory_sha256=np.asarray(parent_hash, dtype="U64"),
        paired_artifact_sha256=np.asarray(_sha256(paired_path), dtype="U64"),
        trajectory_gt_sha256=np.asarray(_sha256(gt_path), dtype="U64"),
        **diagnostic_arrays,
    )
    with open(output / "paired_reserve_gt_rows.csv", "w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) if rows else _GT_CSV_FIELDS
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid_count = int(np.count_nonzero(valid))
    summary = {
        "schema_version": schema_version,
        "metric_definition_version": metric_definition_version or None,
        "row_count": count,
        "paired_no_trigger": count == 0,
        "valid_candidate_count": valid_count,
        "translation_win_count": int(sum(row["translation_win"] for row in rows)),
        "rotation_win_count": int(sum(row["rotation_win"] for row in rows)),
        "translation_delta_mean_m": float(np.nanmean(arrays["translation_delta_m"])) if valid_count else None,
        "rotation_delta_mean_deg": float(np.nanmean(arrays["rotation_delta_deg"])) if valid_count else None,
        "parent_trajectory_sha256": parent_hash,
        "paired_artifact_sha256": _sha256(paired_path),
        "trajectory_gt_sha256": _sha256(gt_path),
        "diagnostic_fields": sorted(diagnostic_arrays),
    }
    for name, values in diagnostic_arrays.items():
        if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            summary[f"{name}_mean"] = float(np.mean(finite)) if finite.size else None
    with open(output / "paired_reserve_gt_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary


def _csv_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        return ""
    scalar = array.item()
    if isinstance(scalar, (float, np.floating)) and not np.isfinite(scalar):
        return ""
    if isinstance(scalar, np.generic):
        return scalar.item()
    return scalar


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(score(args.paired_npz, args.trajectory_gt, args.output_dir, args.trajectory_est), sort_keys=True))


if __name__ == "__main__":
    main()
