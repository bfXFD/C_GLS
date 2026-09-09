"""Offline critical-adoption analysis for closed-loop FSM localization runs."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_FILES = (
    "frame_metrics.csv", "trajectory_est.txt", "trajectory_gt.txt", "per_frame_errors.csv",
)
FRAME_REQUIRED = {
    "frame_id", "timestamp", "tracking_valid", "reserve_attempted", "reserve_used",
}
ERROR_REQUIRED = {
    "frame_id", "timestamp", "tracking_valid", "raw_translation_error_m",
    "raw_rotation_error_deg", "aligned_translation_error_m", "aligned_rotation_error_deg",
}
PASSTHROUGH_PREFIXES = ("reserve_information_", "reserve_refinement_", "reserve_corr_")
PASSTHROUGH_FIELDS = (
    "reserve_triggered", "reserve_trigger_reasons", "reserve_visible", "reserve_selected",
    "reserve_correspondence", "reserve_selection_score", "reserve_primary_condition",
    "reserve_augmented_condition", "reserve_primary_lambda_min", "reserve_augmented_lambda_min",
    "reserve_relative_condition_ratio", "reserve_relative_lambda_ratio", "reserve_adoption_reason",
)


@dataclass(frozen=True)
class CriticalAdoptionConfig:
    horizons: tuple[int, ...] = (10, 30, 100)
    rpe_deltas: tuple[int, ...] = (1, 5, 10)
    burst_gaps: tuple[int, ...] = (1, 2, 5, 10)
    temporal_windows: tuple[int, ...] = (1, 2, 5, 10, 30, 100)
    timestamp_atol: float = 5e-9
    gt_atol: float = 1e-9

    def __post_init__(self) -> None:
        for name in ("horizons", "rpe_deltas", "burst_gaps", "temporal_windows"):
            values = tuple(sorted(set(int(value) for value in getattr(self, name))))
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            object.__setattr__(self, name, values)
        if self.timestamp_atol < 0 or self.gt_atol < 0:
            raise ValueError("tolerances must be non-negative")


@dataclass
class RunData:
    name: str
    directory: Path
    frame_ids: np.ndarray
    timestamps: np.ndarray
    frame_rows: list[dict[str, str]]
    valid: np.ndarray
    attempted: np.ndarray
    adopted: np.ndarray
    raw_point_error: np.ndarray
    aligned_point_error: np.ndarray
    raw_rotation_error: np.ndarray
    est: np.ndarray
    gt: np.ndarray
    file_hashes: dict[str, str]
    fields: set[str]


@dataclass(frozen=True)
class CriticalAdoptionResult:
    inventory_rows: tuple[dict[str, Any], ...]
    attempt_rows: tuple[dict[str, Any], ...]
    adoption_rows: tuple[dict[str, Any], ...]
    burst_rows: tuple[dict[str, Any], ...]
    temporal_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    availability: dict[str, Any]
    provenance: dict[str, Any]


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be boolean")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    try:
        number = float(str(value).strip())
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(number) or number != result or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _finite(value: Any, name: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: Any, name: str) -> float:
    text = str(value).strip()
    if text == "":
        return float("nan")
    return _finite(text, name)


def _json_number(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _optional_number(value: Any) -> Any:
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], set[str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        rows = list(reader)
    return rows, fields


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("trajectory quaternion must be finite and nonzero")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps, poses = [], []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_number} must contain 8 TUM fields")
            values = np.asarray([float(value) for value in parts], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}:{line_number} contains non-finite values")
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = _quat_to_rotation(*values[4:8])
            pose[:3, 3] = values[1:4]
            timestamps.append(values[0]); poses.append(pose)
    return np.asarray(timestamps, dtype=np.float64), np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(name: str, run_dir: str | Path, config: CriticalAdoptionConfig) -> RunData:
    if not name or name.strip() != name:
        raise ValueError("run name must be non-empty and have no surrounding whitespace")
    directory = Path(run_dir).resolve()
    missing = [filename for filename in REQUIRED_FILES if not (directory / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"run {name!r} missing files: {missing}")
    frame_rows, fields = _read_csv(directory / "frame_metrics.csv", FRAME_REQUIRED)
    error_rows, _ = _read_csv(directory / "per_frame_errors.csv", ERROR_REQUIRED)
    if not frame_rows:
        raise ValueError(f"run {name!r} has no frame rows")

    def normalize(rows: list[dict[str, str]], label: str) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, str]]]:
        ids = np.asarray([_integer(row["frame_id"], f"{label}.frame_id") for row in rows], dtype=np.int64)
        if len(set(ids.tolist())) != ids.size:
            raise ValueError(f"run {name!r} has duplicate {label} frame_id")
        if ids.size > 1 and np.any(np.diff(ids) <= 0):
            raise ValueError(f"run {name!r} {label} frame_id must be strictly increasing")
        timestamps = np.asarray([_finite(row["timestamp"], f"{label}.timestamp") for row in rows])
        return ids, timestamps, dict(zip(ids.tolist(), rows))

    frame_ids, timestamps, frame_map = normalize(frame_rows, "frame_metrics")
    error_ids, error_times, error_map = normalize(error_rows, "per_frame_errors")
    if not np.array_equal(frame_ids, error_ids):
        raise ValueError(f"run {name!r} frame_metrics/per_frame_errors frame IDs differ")
    if not np.allclose(timestamps, error_times, rtol=0.0, atol=config.timestamp_atol):
        raise ValueError(f"run {name!r} frame_metrics/per_frame_errors timestamps differ")
    est_times, est = _read_tum(directory / "trajectory_est.txt")
    gt_times, gt = _read_tum(directory / "trajectory_gt.txt")
    if len(frame_ids) != len(est) or len(frame_ids) != len(gt):
        raise ValueError(f"run {name!r} trajectory and CSV row counts differ")
    if not np.allclose(timestamps, est_times, rtol=0.0, atol=config.timestamp_atol):
        raise ValueError(f"run {name!r} estimated trajectory timestamps differ from frame metrics")
    if not np.allclose(timestamps, gt_times, rtol=0.0, atol=config.timestamp_atol):
        raise ValueError(f"run {name!r} GT trajectory timestamps differ from frame metrics")

    valid = np.asarray([_boolean(frame_map[int(fid)]["tracking_valid"], "tracking_valid") for fid in frame_ids])
    error_valid = np.asarray([_boolean(error_map[int(fid)]["tracking_valid"], "tracking_valid") for fid in frame_ids])
    if not np.array_equal(valid, error_valid):
        raise ValueError(f"run {name!r} tracking_valid differs between joined CSVs")
    attempted = np.asarray([_boolean(frame_map[int(fid)]["reserve_attempted"], "reserve_attempted") for fid in frame_ids])
    adopted = np.asarray([_boolean(frame_map[int(fid)]["reserve_used"], "reserve_used") for fid in frame_ids])
    if np.any(adopted & ~attempted):
        bad = frame_ids[adopted & ~attempted].tolist()
        raise ValueError(f"run {name!r} adoption=reserve_used requires attempt at frames {bad[:10]}")
    raw = np.asarray([_optional_finite(error_map[int(fid)]["raw_translation_error_m"], "raw_translation_error_m") for fid in frame_ids])
    aligned = np.asarray([_optional_finite(error_map[int(fid)]["aligned_translation_error_m"], "aligned_translation_error_m") for fid in frame_ids])
    rotation = np.asarray([_optional_finite(error_map[int(fid)]["raw_rotation_error_deg"], "raw_rotation_error_deg") for fid in frame_ids])
    for label, values in (("raw", raw), ("aligned", aligned), ("rotation", rotation)):
        if np.any(valid & ~np.isfinite(values)):
            bad = frame_ids[valid & ~np.isfinite(values)].tolist()
            raise ValueError(f"run {name!r} has missing {label} error on valid frames {bad[:10]}")
    hashes = {filename: _sha256(directory / filename) for filename in REQUIRED_FILES}
    return RunData(name, directory, frame_ids, timestamps, frame_rows, valid, attempted, adopted,
                   raw, aligned, rotation, est, gt, hashes, fields)


def _validate_panel(baseline: RunData, runs: Sequence[RunData], config: CriticalAdoptionConfig) -> None:
    names = [baseline.name, *(run.name for run in runs)]
    if len(set(names)) != len(names):
        raise ValueError("baseline and run names must be unique")
    for run in runs:
        if not np.array_equal(run.frame_ids, baseline.frame_ids):
            raise ValueError(f"run {run.name!r} frame IDs differ from baseline")
        if not np.allclose(run.timestamps, baseline.timestamps, rtol=0.0, atol=config.timestamp_atol):
            raise ValueError(f"run {run.name!r} timestamps differ from baseline")
        if not np.allclose(run.gt, baseline.gt, rtol=0.0, atol=config.gt_atol):
            raise ValueError(f"run {run.name!r} GT trajectory differs from baseline")


def _auc(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    if values.size == 1:
        return float(values[0])
    return float(np.trapezoid(values, dx=1.0)) if hasattr(np, "trapezoid") else float(np.trapz(values, dx=1.0))


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rpe(run: RunData, start: int, end: int, delta: int, valid_mask: np.ndarray | None = None) -> np.ndarray:
    valid = run.valid if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    values = []
    for left in range(start, max(start, end - delta)):
        right = left + delta
        if right >= end or not np.all(valid[left:right + 1]):
            continue
        est_rel = np.linalg.inv(run.est[left]) @ run.est[right]
        gt_rel = np.linalg.inv(run.gt[left]) @ run.gt[right]
        error = np.linalg.inv(gt_rel) @ est_rel
        values.append(float(np.linalg.norm(error[:3, 3])))
    return np.asarray(values, dtype=np.float64)


def _event_passthrough(run: RunData, index: int) -> dict[str, Any]:
    row = run.frame_rows[index]
    names = sorted(field for field in run.fields if field.startswith(PASSTHROUGH_PREFIXES) or field in PASSTHROUGH_FIELDS)
    return {field: _optional_number(row.get(field, "")) for field in names}


def _event_row(run: RunData, baseline: RunData, index: int, event_type: str,
               config: CriticalAdoptionConfig) -> dict[str, Any]:
    n = len(run.frame_ids)
    row: dict[str, Any] = {
        "run": run.name, "event_type": event_type, "frame_id": int(run.frame_ids[index]),
        "timestamp": float(run.timestamps[index]), "tracking_valid": bool(run.valid[index]),
        "attempted": bool(run.attempted[index]), "adopted": bool(run.adopted[index]),
        "current_point_error_run_m": _json_number(run.raw_point_error[index]),
        "current_point_error_baseline_m": _json_number(baseline.raw_point_error[index]),
        "current_point_error_delta_m": _json_number(run.raw_point_error[index] - baseline.raw_point_error[index]),
    }
    adoption_indices = np.flatnonzero(run.adopted)
    later = adoption_indices[adoption_indices > index]
    row["next_adoption_gap_frames"] = int(later[0] - index) if later.size else None
    for horizon in config.horizons:
        stop = min(n, index + horizon + 1)
        common = run.valid[index:stop] & baseline.valid[index:stop]
        delta_all = run.raw_point_error[index:stop] - baseline.raw_point_error[index:stop]
        delta = delta_all[common & np.isfinite(delta_all)]
        row[f"future_h{horizon}_right_censored"] = stop < index + horizon + 1
        row[f"future_h{horizon}_observed_frames"] = int(delta.size)
        endpoint_valid = bool(common[-1] and np.isfinite(delta_all[-1]))
        row[f"future_h{horizon}_point_error_run_m"] = _json_number(run.raw_point_error[stop - 1]) if endpoint_valid else None
        row[f"future_h{horizon}_point_error_baseline_m"] = _json_number(baseline.raw_point_error[stop - 1]) if endpoint_valid else None
        row[f"future_h{horizon}_point_error_delta_m"] = _json_number(delta_all[-1]) if endpoint_valid else None
        row[f"future_h{horizon}_mean_delta_m"] = float(np.mean(delta)) if delta.size else None
        row[f"future_h{horizon}_max_delta_m"] = float(np.max(delta)) if delta.size else None
        row[f"future_h{horizon}_auc_delta_m_frames"] = _auc(delta)
        future_slice = slice(index + 1, stop)
        row[f"future_h{horizon}_other_adoption_count"] = int(np.sum(run.adopted[future_slice]))
        row[f"future_h{horizon}_other_attempt_count"] = int(np.sum(run.attempted[future_slice]))
        row[f"future_h{horizon}_isolated_adoption"] = bool(
            event_type == "adoption" and not np.any(run.adopted[max(0, index - horizon):index])
            and not np.any(run.adopted[index + 1:stop])
        )
        for rpe_delta in config.rpe_deltas:
            common_valid = run.valid & baseline.valid
            run_rpe = _rpe(run, index, stop, rpe_delta, common_valid)
            base_rpe = _rpe(baseline, index, stop, rpe_delta, common_valid)
            pair_count = run_rpe.size
            possible = max(0, stop - index - rpe_delta)
            prefix = f"future_h{horizon}_rpe_d{rpe_delta}"
            row[f"{prefix}_pair_count"] = int(pair_count)
            row[f"{prefix}_coverage"] = float(pair_count / possible) if possible else None
            row[f"{prefix}_run_mean_m"] = float(np.mean(run_rpe[:pair_count])) if pair_count else None
            row[f"{prefix}_baseline_mean_m"] = float(np.mean(base_rpe[:pair_count])) if pair_count else None
            row[f"{prefix}_delta_m"] = (
                float(np.mean(run_rpe - base_rpe)) if pair_count else None
            )
    row.update(_event_passthrough(run, index))
    return row


def _clusters(indices: np.ndarray, gap: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > gap) + 1
    return [part for part in np.split(indices, split) if part.size]


def _burst_rows(run: RunData, baseline: RunData, config: CriticalAdoptionConfig) -> list[dict[str, Any]]:
    output = []
    for gap in config.burst_gaps:
        for burst_index, cluster in enumerate(_clusters(np.flatnonzero(run.adopted), gap)):
            start, end = int(cluster[0]), int(cluster[-1])
            span = end - start + 1
            row: dict[str, Any] = {
                "run": run.name, "gap_threshold": gap, "burst_index": burst_index,
                "start_frame_id": int(run.frame_ids[start]), "end_frame_id": int(run.frame_ids[end]),
                "size": int(cluster.size), "duration_frames": span,
                "density": float(cluster.size / span),
                "attempt_count": int(np.sum(run.attempted[start:end + 1])),
            }
            for field in sorted(run.fields):
                if field.startswith(("reserve_information_", "reserve_refinement_")):
                    values = [_optional_number(run.frame_rows[index].get(field, "")) for index in cluster]
                    numeric = [float(value) for value in values if isinstance(value, (int, float))]
                    if numeric:
                        row[f"{field}_mean"] = float(np.mean(numeric))
            for horizon in config.horizons:
                stop = min(len(run.frame_ids), end + horizon + 1)
                common = run.valid[end:stop] & baseline.valid[end:stop]
                delta_all = run.raw_point_error[end:stop] - baseline.raw_point_error[end:stop]
                delta = delta_all[common & np.isfinite(delta_all)]
                row[f"future_h{horizon}_right_censored"] = stop < end + horizon + 1
                row[f"future_h{horizon}_observed_frames"] = int(delta.size)
                row[f"future_h{horizon}_mean_delta_m"] = float(np.mean(delta)) if delta.size else None
                row[f"future_h{horizon}_max_delta_m"] = float(np.max(delta)) if delta.size else None
                row[f"future_h{horizon}_auc_delta_m_frames"] = _auc(delta)
            output.append(row)
    return output


def _nearest_lags(source: np.ndarray, target: np.ndarray) -> list[int]:
    if target.size == 0:
        return []
    result = []
    for value in source:
        distances = np.abs(target - value)
        candidates = target[distances == np.min(distances)]
        nearest = candidates[np.argmin(np.abs(candidates - value))]
        result.append(int(nearest - value))
    return result


def _temporal_row(left: RunData, right: RunData, kind: str,
                  config: CriticalAdoptionConfig) -> dict[str, Any]:
    left_mask = left.adopted if kind == "adoption" else left.attempted
    right_mask = right.adopted if kind == "adoption" else right.attempted
    left_frames = left.frame_ids[left_mask]
    right_frames = right.frame_ids[right_mask]
    intersection = np.intersect1d(left_frames, right_frames)
    union = np.union1d(left_frames, right_frames)
    lags = _nearest_lags(left_frames, right_frames)
    row: dict[str, Any] = {
        "run": left.name, "compare_run": right.name, "event_type": kind,
        "overlap_definition": "temporal_frame_overlap_not_selected_point_set_overlap",
        "run_event_count": int(left_frames.size), "compare_event_count": int(right_frames.size),
        "frame_intersection_count": int(intersection.size), "frame_union_count": int(union.size),
        "frame_jaccard": float(intersection.size / union.size) if union.size else None,
        "nearest_signed_lag_count": len(lags),
        "nearest_signed_lag_mean": float(np.mean(lags)) if lags else None,
        "nearest_signed_lag_median": float(np.median(lags)) if lags else None,
        "nearest_signed_lag_abs_mean": float(np.mean(np.abs(lags))) if lags else None,
    }
    for window in config.temporal_windows:
        covered = sum(bool(np.any(np.abs(right_frames - frame) <= window)) for frame in left_frames)
        row[f"within_pm{window}_count"] = int(covered)
        row[f"within_pm{window}_coverage"] = float(covered / left_frames.size) if left_frames.size else None
    boundaries = np.union1d(left_frames, right_frames)
    for side, offset in (("pre", -1), ("post", 1)):
        values = []
        for frame in boundaries:
            positions = np.flatnonzero(left.frame_ids == frame)
            if not positions.size:
                continue
            index = int(positions[0]) + offset
            if 0 <= index < len(left.frame_ids):
                relative = np.linalg.inv(right.est[index]) @ left.est[index]
                values.append((np.linalg.norm(relative[:3, 3]), _rotation_angle(relative[:3, :3])))
        row[f"boundary_{side}_sample_count"] = len(values)
        row[f"boundary_{side}_translation_divergence_mean_m"] = float(np.mean([v[0] for v in values])) if values else None
        row[f"boundary_{side}_rotation_divergence_mean_deg"] = float(np.mean([v[1] for v in values])) if values else None
    return row


def _lost_metrics(run: RunData, baseline: RunData) -> dict[str, Any]:
    opportunities = np.flatnonzero(~baseline.valid)
    if opportunities.size == 0:
        return {
            "status": "not_estimable", "baseline_opportunity_count": 0,
            "avoided_count": None, "avoidance_rate": None, "recovery_delay_delta_mean_frames": None,
        }
    avoided = int(np.sum(run.valid[opportunities]))
    delays = []
    for index in opportunities:
        baseline_next = np.flatnonzero(baseline.valid[index:])
        run_next = np.flatnonzero(run.valid[index:])
        if baseline_next.size and run_next.size:
            delays.append(int(run_next[0] - baseline_next[0]))
    return {
        "status": "estimable", "baseline_opportunity_count": int(opportunities.size),
        "avoided_count": avoided, "avoidance_rate": float(avoided / opportunities.size),
        "recovery_delay_delta_mean_frames": float(np.mean(delays)) if delays else None,
    }


def analyze_runs(baseline: RunData, runs: Sequence[RunData], config: CriticalAdoptionConfig = CriticalAdoptionConfig(),
                 compare_runs: Sequence[str] = ()) -> CriticalAdoptionResult:
    _validate_panel(baseline, runs, config)
    all_runs = [baseline, *runs]
    lookup = {run.name: run for run in all_runs}
    unknown = sorted(set(compare_runs) - set(lookup))
    if unknown:
        raise ValueError(f"unknown compare runs: {unknown}")
    selected_compare = tuple(compare_runs)
    inventory, attempts, adoptions, bursts = [], [], [], []
    summary_runs: dict[str, Any] = {}
    availability_runs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "frame_id": baseline.frame_ids.astype(np.int64), "timestamp": baseline.timestamps.astype(np.float64),
        "gt_pose": baseline.gt.astype(np.float64),
    }
    for run in all_runs:
        inventory.append({
            "run": run.name, "role": "baseline" if run is baseline else "run",
            "run_dir": str(run.directory), "frame_count": len(run.frame_ids),
            "attempt_count": int(np.sum(run.attempted)), "adoption_count": int(np.sum(run.adopted)),
            "valid_count": int(np.sum(run.valid)), "lost_count": int(np.sum(~run.valid)),
            "first_frame_id": int(run.frame_ids[0]), "last_frame_id": int(run.frame_ids[-1]),
        })
        arrays[f"{run.name}__valid"] = run.valid.astype(bool)
        arrays[f"{run.name}__attempted"] = run.attempted.astype(bool)
        arrays[f"{run.name}__adopted"] = run.adopted.astype(bool)
        arrays[f"{run.name}__raw_point_error_m"] = run.raw_point_error.astype(np.float64)
        arrays[f"{run.name}__est_pose"] = run.est.astype(np.float64)
        if run is not baseline:
            attempts.extend(_event_row(run, baseline, int(index), "attempt", config) for index in np.flatnonzero(run.attempted))
            adoptions.extend(_event_row(run, baseline, int(index), "adoption", config) for index in np.flatnonzero(run.adopted))
            bursts.extend(_burst_rows(run, baseline, config))
        lost = _lost_metrics(run, baseline)
        summary_runs[run.name] = {
            "role": "baseline" if run is baseline else "run", "frame_count": len(run.frame_ids),
            "attempt_count": int(np.sum(run.attempted)), "adoption_count": int(np.sum(run.adopted)),
            "lost": lost,
        }
        availability_runs[run.name] = {
            "selected_ids": {"available": False, "reason": "not_exported_by_current_localizer_schema",
                             "future_schema_recommendation": "export per-frame selected global IDs with offsets"},
            "point_p_corr": {"available": False, "reason": "per-point p_corr not present in run artifacts",
                             "future_schema_recommendation": "export selected IDs and aligned per-point p_corr"},
            "lost_estimability": lost,
        }
    temporal = []
    for compare_name in selected_compare:
        compare = lookup[compare_name]
        for run in runs:
            if run.name == compare_name:
                continue
            temporal.append(_temporal_row(run, compare, "adoption", config))
            temporal.append(_temporal_row(run, compare, "attempt", config))
    summary = {
        "schema_version": SCHEMA_VERSION, "baseline": baseline.name,
        "runs": summary_runs, "horizons": list(config.horizons), "rpe_deltas": list(config.rpe_deltas),
        "burst_gaps": list(config.burst_gaps), "temporal_compare_runs": list(selected_compare),
        "definitions": {
            "adoption": "frame_metrics.reserve_used", "attempt": "frame_metrics.reserve_attempted",
            "point_error": "per_frame_errors.raw_translation_error_m",
            "delta_direction": "run_minus_baseline; negative is lower error",
            "future_window": "inclusive of event frame through event+horizon",
            "auc": "unit-spaced trapezoidal integral of point-error delta",
            "rpe": "raw unaligned SE3 relative translation error; pairs crossing any invalid frame are excluded",
            "temporal_overlap": "event frame overlap, not selected point-set overlap",
        },
    }
    availability = {
        "schema_version": SCHEMA_VERSION, "runs": availability_runs,
        "selected_ids_available": False, "point_p_corr_available": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "config": dataclasses.asdict(config),
        "inputs": {run.name: {"run_dir": str(run.directory), "sha256": run.file_hashes} for run in all_runs},
        "strict_join": "frame_id plus timestamp validation across frame_metrics, per_frame_errors, est and GT trajectories",
        "gt_consistency": "all run GT poses and timestamps equal baseline within configured absolute tolerances",
        "allow_pickle": False,
    }
    return CriticalAdoptionResult(tuple(inventory), tuple(attempts), tuple(adoptions), tuple(bursts),
                                  tuple(temporal), summary, arrays, availability, provenance)


def analyze_directories(baseline_name: str, baseline_dir: str | Path,
                        run_specs: Sequence[tuple[str, str | Path]],
                        config: CriticalAdoptionConfig = CriticalAdoptionConfig(),
                        compare_runs: Sequence[str] = ()) -> CriticalAdoptionResult:
    baseline = load_run(baseline_name, baseline_dir, config)
    runs = [load_run(name, directory, config) for name, directory in run_specs]
    return analyze_runs(baseline, runs, config, compare_runs)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_path(path: Path) -> tuple[int, str]:
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def _write_json(path: Path, value: Any) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _field_union(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    fields = list(preferred)
    extra = sorted({key for row in rows for key in row} - set(fields))
    return fields + extra


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> None:
    fields = _field_union(rows, preferred)
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary = _atomic_path(path)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def write_outputs(result: CriticalAdoptionResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_inventory": output / "run_inventory.csv", "attempt_events": output / "attempt_events.csv",
        "adoption_events": output / "adoption_events.csv", "bursts": output / "bursts.csv",
        "temporal_comparison": output / "temporal_comparison.csv", "summary": output / "summary.json",
        "arrays": output / "arrays.npz", "availability": output / "availability.json",
        "provenance": output / "provenance.json",
    }
    _write_csv(paths["run_inventory"], result.inventory_rows,
               ("run", "role", "run_dir", "frame_count", "attempt_count", "adoption_count", "valid_count", "lost_count"))
    event_fields = ("run", "event_type", "frame_id", "timestamp", "tracking_valid", "attempted", "adopted")
    _write_csv(paths["attempt_events"], result.attempt_rows, event_fields)
    _write_csv(paths["adoption_events"], result.adoption_rows, event_fields)
    _write_csv(paths["bursts"], result.burst_rows,
               ("run", "gap_threshold", "burst_index", "start_frame_id", "end_frame_id", "size", "duration_frames", "density"))
    _write_csv(paths["temporal_comparison"], result.temporal_rows,
               ("run", "compare_run", "event_type", "overlap_definition"))
    _write_json(paths["summary"], result.summary)
    _write_npz(paths["arrays"], result.arrays)
    _write_json(paths["availability"], result.availability)
    _write_json(paths["provenance"], result.provenance)
    return paths


def run_analysis(baseline_name: str, baseline_dir: str | Path,
                 run_specs: Sequence[tuple[str, str | Path]], output_dir: str | Path,
                 config: CriticalAdoptionConfig = CriticalAdoptionConfig(),
                 compare_runs: Sequence[str] = ()) -> dict[str, Any]:
    result = analyze_directories(baseline_name, baseline_dir, run_specs, config, compare_runs)
    paths = write_outputs(result, output_dir)
    return {
        "status": "ok", "baseline": baseline_name, "run_count": len(run_specs),
        "attempt_event_count": len(result.attempt_rows), "adoption_event_count": len(result.adoption_rows),
        "output_dir": str(Path(output_dir).resolve()),
        "outputs": {key: str(path) for key, path in sorted(paths.items())},
    }
