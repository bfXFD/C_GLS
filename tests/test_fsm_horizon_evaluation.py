import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.fsm_evaluate_horizon_test import evaluate
from scripts.fsm_freeze_horizon_thresholds import freeze
from scripts.fsm_label_horizon_gt import label
from utils.fsm_horizon_evaluation import sha256_file, write_csv, write_json
from utils.fsm_horizon_verification import save_results


def pose(x=0.0, rotation_deg=0.0):
    value = np.eye(4)
    value[0, 3] = x
    angle = np.radians(rotation_deg)
    value[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return value


def write_tum(path, rows):
    path.write_text("".join(f"{timestamp:.9f} {x} 0 0 0 0 0 1\n" for timestamp, x in rows))


def test_output_helpers_create_parent_directories_and_preserve_exclusive_write(tmp_path):
    json_path = tmp_path / "nested" / "frozen" / "thresholds.json"
    csv_path = tmp_path / "nested" / "rows" / "labels.csv"
    write_json(json_path, {"immutable": True}, exclusive=True)
    write_csv(csv_path, [{"value": 1}], ["value"])
    assert json_path.is_file()
    assert csv_path.is_file()
    with pytest.raises(FileExistsError):
        write_json(json_path, {"immutable": False}, exclusive=True)


def make_horizon(tmp_path, horizon, labels=("harmful", "beneficial", "neutral")):
    root = tmp_path / f"h{horizon}"
    root.mkdir()
    events = []
    poses = []
    for index, expected in enumerate(labels):
        pair_id = f"h{horizon}-p{index}"
        events.append({
            "pair_id": pair_id, "frame_id": index, "policy": "runtime_corr_weak", "seed": "",
            "budget": 128, "horizon_requested": horizon, "horizon_observed": horizon,
            "status": "complete", "decision": "primary", "decision_reason": "primary_default",
            "primary_lost_count": 0, "candidate_lost_count": 0, "common_valid_count": horizon,
            "median_common_cost_ratio": 0.85 if expected == "beneficial" else 0.95,
            "median_common_fitness_ratio": 0.96, "median_common_corr_retention": 0.96,
            "primary_final_pose_sha256": "a" * 64, "candidate_final_pose_sha256": "b" * 64,
        })
        candidate_x = {"harmful": 2.0, "beneficial": 0.0, "neutral": 1.0}[expected]
        primary_rotation = 10.0 if expected == "beneficial" else 0.0
        poses.extend([
            {"pair_id": pair_id, "offset": 1, "frame_id": index, "timestamp": float(index),
             "branch": "primary", "accepted": True, "pose": pose(1.0, primary_rotation)},
            {"pair_id": pair_id, "offset": 1, "frame_id": index, "timestamp": float(index),
             "branch": "candidate", "accepted": True, "pose": pose(candidate_x)},
        ])
    save_results(root, events, [], {"uses_ground_truth": False}, poses)
    gt = tmp_path / f"gt{horizon}.txt"
    write_tum(gt, [(float(index), 0.0) for index in range(len(labels))])
    label_dir = tmp_path / f"labels{horizon}"
    label(Namespace(
        horizon_dir=str(root), trajectory_gt=str(gt), output_dir=str(label_dir),
        translation_neutral_m=0.0, rotation_neutral_deg=0.0,
        joint_rule="harmful_if_either_beneficial_if_both_else_neutral",
    ))
    return root, label_dir


def test_replay_pose_artifacts_are_npz_csv_safe_and_gt_free(tmp_path):
    root, _ = make_horizon(tmp_path, 1, ("harmful",))
    with np.load(root / "horizon_branch_poses.npz", allow_pickle=False) as archive:
        assert archive["pose"].shape == (2, 4, 4)
        assert all("gt" not in name.lower() and "truth" not in name.lower() for name in archive.files)
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    header = next(csv.reader((root / "horizon_branch_poses.csv").open()))
    assert not any("gt" in name.lower() or "truth" in name.lower() for name in header)
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["uses_ground_truth"] is False
    assert "trajectory_gt_sha256" not in provenance


def test_gt_labeler_is_separate_and_hash_bound(tmp_path):
    root, labels = make_horizon(tmp_path, 3)
    rows = list(csv.DictReader((labels / "horizon_gt_labels.csv").open()))
    assert [row["joint_label"] for row in rows] == ["harmful", "beneficial", "neutral"]
    assert not (root / "horizon_gt_labels.csv").exists()
    provenance = json.loads((labels / "horizon_gt_labels_provenance.json").read_text())
    assert provenance["mode"] == "offline_gt_only" and provenance["uses_ground_truth"] is True
    assert provenance["horizon_branch_poses_sha256"] == sha256_file(root / "horizon_branch_poses.npz")


def test_gt_labeler_keeps_right_censored_event_without_pose_rows(tmp_path):
    root, labels = make_horizon(tmp_path, 1, ("harmful",))
    events_path = root / "horizon_events.csv"
    rows = list(csv.DictReader(events_path.open()))
    extra = dict(rows[0])
    extra["pair_id"] = "right-censored"
    extra["status"] = "inconclusive"
    with events_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [extra])
    label_dir = tmp_path / "right-censored-labels"
    gt = tmp_path / "gt1.txt"
    label(Namespace(
        horizon_dir=str(root), trajectory_gt=str(gt), output_dir=str(label_dir),
        translation_neutral_m=0.0, rotation_neutral_deg=0.0,
        joint_rule="harmful_if_either_beneficial_if_both_else_neutral",
    ))
    labelled = {row["pair_id"]: row for row in csv.DictReader((label_dir / "horizon_gt_labels.csv").open())}
    assert labelled["right-censored"]["joint_label"] == "neutral"
    assert labelled["right-censored"]["label_status"] == "no_joint_evidence"


def test_freeze_and_evaluation_are_deterministic_and_validate_hashes(tmp_path):
    development = []
    tests = []
    for horizon in (1, 3, 5, 10):
        root, labels = make_horizon(tmp_path, horizon)
        development.append(f"{horizon}:{root}:{labels}")
        tests.append(f"{horizon}:{root}:{labels}")
    frozen_a = tmp_path / "frozen-a.json"
    frozen_b = tmp_path / "frozen-b.json"
    first = freeze(Namespace(development=development, output=str(frozen_a)))
    second = freeze(Namespace(development=development, output=str(frozen_b)))
    assert first == second
    assert frozen_a.read_bytes() == frozen_b.read_bytes()
    assert first["horizons"]["1"]["thresholds"] == {
        "max_cost_ratio": 0.85, "min_fitness_retention": 0.95, "min_corr_retention": 0.95,
    }
    with pytest.raises(FileExistsError):
        freeze(Namespace(development=development, output=str(frozen_a)))
    report_a = tmp_path / "report-a.json"
    report_b = tmp_path / "report-b.json"
    result_a = evaluate(Namespace(frozen_json=str(frozen_a), test=tests, output=str(report_a)))
    result_b = evaluate(Namespace(frozen_json=str(frozen_a), test=tests, output=str(report_b)))
    assert result_a == result_b
    metrics = result_a["horizons"]["1"]
    assert metrics["harmful_rejection_recall"] == 1.0
    assert metrics["beneficial_retention"] == 1.0
    assert metrics["false_safe"] == 0 and metrics["false_reject"] == 0
    assert metrics["inconclusive_rate"] == 0.0 and metrics["selection_rate"] == pytest.approx(1 / 3)

    events = Path(development[0].split(":", 2)[1]) / "horizon_events.csv"
    events.write_text(events.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        freeze(Namespace(development=development, output=str(tmp_path / "must-not-exist.json")))
