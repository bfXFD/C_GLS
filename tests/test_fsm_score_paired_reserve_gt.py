import hashlib
import csv
import json

import numpy as np
import pytest

from scripts.fsm_score_paired_reserve_gt import score


def _tum_row(timestamp, translation):
    return f"{timestamp:.9f} {translation[0]} {translation[1]} {translation[2]} 0 0 0 1\n"


def test_offline_scorer_math_and_provenance(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(1.0, (0.2, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(1.0, (0.0, 0.0, 0.0)))
    parent_hash = hashlib.sha256(trajectory_est.read_bytes()).hexdigest()
    primary = np.eye(4)[None]
    primary[0, 0, 3] = 0.2
    candidate = np.eye(4)[None]
    candidate[0, 0, 3] = 0.1
    paired_dir = tmp_path / "paired_reserve"
    paired_dir.mkdir()
    paired = paired_dir / "paired_reserve_ids.npz"
    np.savez(
        paired,
        frame_id=np.array([7]), timestamp=np.array([1.0]), pair_id=np.array(["a" * 64]),
        policy=np.array(["saliency"]), seed=np.array([-1]), budget=np.array([64]),
        primary_pose=primary, candidate_pose=candidate,
        parent_trajectory_sha256=np.asarray(parent_hash),
    )
    summary = score(paired, trajectory_gt, tmp_path / "scored", trajectory_est)
    assert summary["translation_win_count"] == 1
    assert summary["translation_delta_mean_m"] == pytest.approx(0.1)
    with np.load(tmp_path / "scored/paired_reserve_gt.npz", allow_pickle=False) as archive:
        assert archive["translation_delta_m"][0] == pytest.approx(0.1)
        assert str(archive["policy"][0]) == "saliency"
    assert json.loads((tmp_path / "scored/paired_reserve_gt_summary.json").read_text())["row_count"] == 1


def test_offline_scorer_rejects_parent_hash_mismatch(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    paired = tmp_path / "paired.npz"
    pose = np.eye(4)[None]
    np.savez(
        paired, frame_id=np.array([0]), timestamp=np.array([0.0]), pair_id=np.array(["b" * 64]),
        policy=np.array(["random"]), seed=np.array([0]), budget=np.array([64]),
        primary_pose=pose, candidate_pose=pose, parent_trajectory_sha256=np.asarray("0" * 64),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        score(paired, trajectory_gt, tmp_path / "out", trajectory_est)


def test_v2_scorer_carries_dense_diagnostics_without_rebuilding_correspondence(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(1.0, (0.0, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(1.0, (0.0, 0.0, 0.0)))
    parent_hash = hashlib.sha256(trajectory_est.read_bytes()).hexdigest()
    pose = np.eye(4)[None]
    hessian = np.arange(36, dtype=np.float64).reshape(1, 6, 6)
    paired = tmp_path / "paired_v2.npz"
    np.savez(
        paired, schema_version=np.asarray(2),
        metric_definition_version=np.asarray("paired_reserve_metrics_v2"),
        row_index=np.array([0]), frame_id=np.array([7]), timestamp=np.array([1.0]),
        pair_id=np.array(["c" * 64]), policy=np.array(["runtime_corr_weak"]),
        seed=np.array([-1]), budget=np.array([2]), primary_pose=pose, candidate_pose=pose,
        actual_reserve_hessian=hessian, actual_correspondence_count=np.array([1]),
        p_corr_mean=np.array([0.75]), offsets=np.array([0, 2]), global_ids=np.array([100, 101]),
        selected_p_corr=np.array([0.5, 1.0]),
        selected_actual_correspondence=np.array([False, True]),
        parent_trajectory_sha256=np.asarray(parent_hash),
    )
    summary = score(paired, trajectory_gt, tmp_path / "scored", trajectory_est)
    assert summary["schema_version"] == 2
    assert "actual_reserve_hessian" in summary["diagnostic_fields"]
    with np.load(tmp_path / "scored/paired_reserve_gt.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["actual_reserve_hessian"], hessian)
        assert archive["actual_correspondence_count"][0] == 1
        assert archive["row_index"][0] == 0


def test_v2_scorer_rejects_nonidentity_row_index(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    parent_hash = hashlib.sha256(trajectory_est.read_bytes()).hexdigest()
    pose = np.eye(4)[None]
    paired = tmp_path / "bad_v2.npz"
    np.savez(
        paired, schema_version=np.asarray(2), row_index=np.array([4]), frame_id=np.array([0]),
        metric_definition_version=np.asarray("paired_reserve_metrics_v2"),
        timestamp=np.array([0.0]), pair_id=np.array(["d" * 64]), policy=np.array(["saliency"]),
        seed=np.array([-1]), budget=np.array([2]), primary_pose=pose, candidate_pose=pose,
        parent_trajectory_sha256=np.asarray(parent_hash),
    )
    with pytest.raises(ValueError, match="row_index"):
        score(paired, trajectory_gt, tmp_path / "out", trajectory_est)


def test_v2_scorer_requires_metric_version_and_row_index(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    parent_hash = hashlib.sha256(trajectory_est.read_bytes()).hexdigest()
    pose = np.eye(4)[None]
    paired = tmp_path / "incomplete_v2.npz"
    np.savez(
        paired, schema_version=np.asarray(2), frame_id=np.array([0]), timestamp=np.array([0.0]),
        pair_id=np.array(["e" * 64]), policy=np.array(["saliency"]), seed=np.array([-1]),
        budget=np.array([2]), primary_pose=pose, candidate_pose=pose,
        parent_trajectory_sha256=np.asarray(parent_hash),
    )
    with pytest.raises(KeyError, match="row_index"):
        score(paired, trajectory_gt, tmp_path / "out", trajectory_est)


def test_v2_scorer_writes_full_header_for_no_trigger_rows(tmp_path):
    trajectory_est = tmp_path / "trajectory_est.txt"
    trajectory_gt = tmp_path / "trajectory_gt.txt"
    trajectory_est.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    trajectory_gt.write_text(_tum_row(0.0, (0.0, 0.0, 0.0)))
    parent_hash = hashlib.sha256(trajectory_est.read_bytes()).hexdigest()
    paired = tmp_path / "office0_empty_v2.npz"
    np.savez(
        paired, schema_version=np.asarray(2),
        metric_definition_version=np.asarray("paired_reserve_metrics_v2"),
        row_index=np.empty(0, dtype=np.int64), frame_id=np.empty(0, dtype=np.int64),
        timestamp=np.empty(0), pair_id=np.empty(0, dtype="U64"),
        policy=np.empty(0, dtype="U32"), seed=np.empty(0, dtype=np.int64),
        budget=np.empty(0, dtype=np.int64), primary_pose=np.empty((0, 4, 4)),
        candidate_pose=np.empty((0, 4, 4)), parent_trajectory_sha256=np.asarray(parent_hash),
    )
    summary = score(paired, trajectory_gt, tmp_path / "scored", trajectory_est)
    assert summary["row_count"] == 0
    assert summary["paired_no_trigger"] is True
    with open(tmp_path / "scored/paired_reserve_gt_rows.csv", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert {"policy", "seed", "budget", "candidate_valid", "translation_delta_m", "rotation_delta_deg"} <= set(reader.fieldnames)
        assert list(reader) == []
