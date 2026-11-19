"""GT-isolated labeling and deterministic short-horizon policy evaluation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HORIZONS = (1, 3, 5, 10)
GRID = {
    "max_cost_ratio": (0.80, 0.85, 0.90, 0.95, 1.00),
    "min_fitness_retention": (0.90, 0.95, 1.00),
    "min_corr_retention": (0.90, 0.95, 1.00),
}
JOINT_RULE = "harmful_if_either_beneficial_if_both_else_neutral"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with open(output, mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_pose_artifact(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("row_index", "pair_id", "offset", "frame_id", "timestamp", "branch", "accepted", "pose")
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"branch pose artifact missing fields: {missing}")
        values = {name: np.asarray(archive[name]) for name in required}
    count = values["row_index"].size
    if not np.array_equal(values["row_index"], np.arange(count)):
        raise ValueError("branch pose row_index must be identity order")
    if values["pose"].shape != (count, 4, 4) or np.any(~np.isfinite(values["pose"])):
        raise ValueError("branch poses must be finite 4x4 matrices")
    for name in required[1:-1]:
        if values[name].shape != (count,):
            raise ValueError(f"branch pose field {name} has inconsistent shape")
    return values


def pose_errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    relative = truth[:3, :3].T @ estimate[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cosine)))


def dimension_label(delta: float, neutral_margin: float) -> str:
    if delta > neutral_margin:
        return "harmful"
    if delta < -neutral_margin:
        return "beneficial"
    return "neutral"


def joint_label(translation: str, rotation: str, rule: str = JOINT_RULE) -> str:
    if rule != JOINT_RULE:
        raise ValueError(f"unsupported joint rule: {rule}")
    if "harmful" in (translation, rotation):
        return "harmful"
    if translation == rotation == "beneficial":
        return "beneficial"
    return "neutral"


def select_candidate(row: dict[str, str], thresholds: dict[str, float]) -> tuple[bool, bool]:
    if row.get("status") != "complete":
        return False, True
    primary_lost = int(row["primary_lost_count"])
    candidate_lost = int(row["candidate_lost_count"])
    if candidate_lost != primary_lost:
        return candidate_lost < primary_lost, False
    fields = ("median_common_cost_ratio", "median_common_fitness_ratio", "median_common_corr_retention")
    if any(row.get(name, "") == "" for name in fields):
        return False, True
    selected = (
        float(row[fields[0]]) <= thresholds["max_cost_ratio"]
        and float(row[fields[1]]) >= thresholds["min_fitness_retention"]
        and float(row[fields[2]]) >= thresholds["min_corr_retention"]
    )
    return selected, False


def evaluate_rows(events: list[dict[str, str]], labels: list[dict[str, str]], thresholds: dict[str, float]) -> dict[str, Any]:
    label_by_pair = {row["pair_id"]: row for row in labels}
    if len(label_by_pair) != len(labels) or set(label_by_pair) != {row["pair_id"] for row in events}:
        raise ValueError("event and label pair_id sets must match exactly and be unique")
    counts = {
        "events": len(events), "harmful": 0, "beneficial": 0, "neutral": 0,
        "harmful_rejected": 0, "beneficial_retained": 0, "false_safe": 0,
        "false_reject": 0, "inconclusive": 0, "selected": 0,
    }
    for event in events:
        label = label_by_pair[event["pair_id"]]["joint_label"]
        if label not in ("harmful", "beneficial", "neutral"):
            raise ValueError(f"invalid joint label: {label}")
        counts[label] += 1
        selected, inconclusive = select_candidate(event, thresholds)
        counts["selected"] += int(selected)
        counts["inconclusive"] += int(inconclusive)
        counts["harmful_rejected"] += int(label == "harmful" and not selected)
        counts["beneficial_retained"] += int(label == "beneficial" and selected)
        counts["false_safe"] += int(label == "harmful" and selected)
        counts["false_reject"] += int(label == "beneficial" and not selected)
    return {
        **counts,
        "harmful_rejection_recall": _rate(counts["harmful_rejected"], counts["harmful"]),
        "beneficial_retention": _rate(counts["beneficial_retained"], counts["beneficial"]),
        "false_safe_rate": _rate(counts["false_safe"], counts["harmful"]),
        "false_reject_rate": _rate(counts["false_reject"], counts["beneficial"]),
        "inconclusive_rate": _rate(counts["inconclusive"], counts["events"]),
        "selection_rate": _rate(counts["selected"], counts["events"]),
    }


def threshold_grid() -> Iterable[dict[str, float]]:
    names = tuple(GRID)
    for values in itertools.product(*(GRID[name] for name in names)):
        yield dict(zip(names, values))


def freeze_score(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[float, float, float, float, float]:
    harmful = metrics["harmful_rejection_recall"]
    beneficial = metrics["beneficial_retention"]
    return (
        -1.0 if harmful is None else harmful,
        -1.0 if beneficial is None else beneficial,
        -thresholds["max_cost_ratio"],
        thresholds["min_fitness_retention"],
        thresholds["min_corr_retention"],
    )


def validate_label_hashes(label_provenance: dict[str, Any], events_path: Path, poses_path: Path | None = None) -> None:
    if label_provenance.get("horizon_events_sha256") != sha256_file(events_path):
        raise ValueError("label provenance horizon_events hash mismatch")
    if poses_path is not None and label_provenance.get("horizon_branch_poses_sha256") != sha256_file(poses_path):
        raise ValueError("label provenance branch pose hash mismatch")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
