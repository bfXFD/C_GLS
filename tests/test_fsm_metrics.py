import csv
import os

import numpy as np
import pytest

from utils.fsm_metrics import (
    align_c2w_trajectory,
    append_summary_csv,
    compute_absolute_pose_errors,
    compute_ate_rmse,
    compute_distance_delta_rpe,
    compute_endpoint_errors,
    compute_long_horizon_metrics,
    compute_multi_delta_rpe,
    compute_rpe_rmse,
    compute_segment_errors,
    compute_trajectory_drift,
    rotation_matrix_to_quaternion_xyzw,
    save_tum_trajectory,
    transform_to_tum_line,
    umeyama_align_positions,
)


def make_pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return pose


def make_trajectory(num_frames=5, offset=(0.0, 0.0, 0.0)):
    poses = []
    for idx in range(num_frames):
        poses.append(make_pose(x=idx * 0.1 + offset[0], y=offset[1], z=offset[2], yaw_deg=idx * 2.0))
    return np.stack(poses, axis=0)


def test_rotation_matrix_to_quaternion_identity_xyzw():
    quat = rotation_matrix_to_quaternion_xyzw(np.eye(3))
    np.testing.assert_allclose(quat, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8)


def test_rotation_matrix_to_quaternion_yaw_90():
    quat = rotation_matrix_to_quaternion_xyzw(make_pose(yaw_deg=90.0)[:3, :3])
    np.testing.assert_allclose(quat[:2], np.array([0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(abs(quat[2]), np.sqrt(0.5), atol=1e-8)
    np.testing.assert_allclose(abs(quat[3]), np.sqrt(0.5), atol=1e-8)


def test_transform_to_tum_line_format():
    line = transform_to_tum_line(1.5, make_pose(x=1.0, y=2.0, z=3.0))
    parts = line.split()
    assert len(parts) == 8
    assert parts[0] == "1.500000000"
    assert parts[1:4] == ["1.000000000", "2.000000000", "3.000000000"]
    assert parts[4:] == ["0.000000000", "0.000000000", "0.000000000", "1.000000000"]


def test_save_tum_trajectory(tmp_path):
    poses = make_trajectory(3)
    path = tmp_path / "traj" / "est.txt"
    save_tum_trajectory(str(path), np.array([0.0, 1.0, 2.0]), poses)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert all(len(line.split()) == 8 for line in lines)


def test_umeyama_align_positions_without_scale_removes_rigid_offset():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    dst = src + np.array([2.0, -1.0, 0.5])
    aligned, scale = umeyama_align_positions(src, dst, with_scale=False)
    np.testing.assert_allclose(aligned, dst, atol=1e-8)
    assert scale == 1.0


def test_umeyama_align_positions_with_scale():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    dst = 2.0 * src + np.array([1.0, 2.0, 3.0])
    aligned, scale = umeyama_align_positions(src, dst, with_scale=True)
    np.testing.assert_allclose(aligned, dst, atol=1e-8)
    np.testing.assert_allclose(scale, 2.0, atol=1e-8)


def test_compute_ate_identical_is_zero():
    poses = make_trajectory(5)
    assert compute_ate_rmse(poses, poses, align=False) == 0.0


def test_compute_ate_alignment_removes_translation_offset():
    gt = make_trajectory(5)
    est = make_trajectory(5, offset=(1.0, -2.0, 0.5))
    ate = compute_ate_rmse(est, gt, align=True)
    np.testing.assert_allclose(ate, 0.0, atol=1e-8)


def test_compute_ate_without_alignment_sees_translation_offset():
    gt = make_trajectory(5)
    est = make_trajectory(5, offset=(1.0, 0.0, 0.0))
    ate = compute_ate_rmse(est, gt, align=False)
    np.testing.assert_allclose(ate, 1.0, atol=1e-8)


def test_compute_rpe_identical_is_zero():
    poses = make_trajectory(5)
    rpe = compute_rpe_rmse(poses, poses, delta=1)
    np.testing.assert_allclose(rpe["trans_rmse"], 0.0, atol=1e-8)
    np.testing.assert_allclose(rpe["rot_rmse_deg"], 0.0, atol=1e-8)


def test_compute_rpe_detects_relative_translation_error():
    gt = make_trajectory(5)
    est = gt.copy()
    for idx in range(est.shape[0]):
        est[idx, 0, 3] += idx * 0.01
    rpe = compute_rpe_rmse(est, gt, delta=1)
    assert rpe["trans_rmse"] > 0.0


def test_append_summary_csv_writes_header_once(tmp_path):
    path = tmp_path / "summary.csv"
    fieldnames = ["strategy", "ate"]
    append_summary_csv(str(path), {"strategy": "full", "ate": 0.1}, fieldnames)
    append_summary_csv(str(path), {"strategy": "random", "ate": 0.2}, fieldnames)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0] == {"strategy": "full", "ate": "0.1"}
    assert rows[1] == {"strategy": "random", "ate": "0.2"}


def test_append_summary_csv_rejects_missing_field(tmp_path):
    with pytest.raises(KeyError):
        append_summary_csv(str(tmp_path / "summary.csv"), {"strategy": "full"}, ["strategy", "ate"])


def test_save_tum_trajectory_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        save_tum_trajectory(str(tmp_path / "traj.txt"), np.array([0.0]), make_trajectory(2))


def test_compute_rpe_rejects_short_trajectory():
    poses = make_trajectory(2)
    with pytest.raises(ValueError):
        compute_rpe_rmse(poses, poses, delta=2)


def test_long_horizon_identity_metrics_are_zero():
    poses = np.stack([make_pose(x=float(index)) for index in range(6)])
    metrics = compute_long_horizon_metrics(
        poses,
        poses,
        frame_deltas=(1, 3),
        distance_deltas_m=(2.0,),
        segment_lengths_m=(3.0,),
    )

    assert metrics["valid_frame_count"] == 6
    assert metrics["raw_absolute"]["count"] == 6
    assert metrics["raw_absolute"]["translation_rmse_m"] == pytest.approx(0.0)
    assert metrics["aligned_absolute"]["rotation_rmse_deg"] == pytest.approx(0.0)
    assert metrics["frame_delta_rpe"]["3"]["count"] == 3
    assert metrics["distance_delta_rpe"]["2.0"]["count"] == 4
    assert metrics["segment_errors"]["3.0"]["translation_drift_percent"] == pytest.approx(0.0)
    assert metrics["drift"]["translation_drift_percent"] == pytest.approx(0.0)
    assert metrics["drift"]["rotation_drift_deg_per_m"] == pytest.approx(0.0)
    assert metrics["endpoint"]["absolute_translation_error_m"] == pytest.approx(0.0)


def test_known_translation_drift_is_reported_in_percent():
    gt = np.stack([make_pose(x=float(index)) for index in range(6)])
    est = np.stack([make_pose(x=1.1 * index) for index in range(6)])

    drift = compute_trajectory_drift(est, gt)
    segments = compute_segment_errors(est, gt, segment_lengths_m=(2.0,))

    assert drift["count"] == 1
    assert drift["translation_drift_percent"] == pytest.approx(10.0)
    assert drift["rotation_drift_deg_per_m"] == pytest.approx(0.0)
    assert segments["2.0"]["translation_drift_percent"] == pytest.approx(10.0)


def test_known_rotation_drift_is_reported_in_degrees_per_meter():
    gt = np.stack([make_pose(x=float(index)) for index in range(6)])
    est = np.stack([make_pose(x=float(index), yaw_deg=2.0 * index) for index in range(6)])

    drift = compute_trajectory_drift(est, gt)
    segments = compute_segment_errors(est, gt, segment_lengths_m=(2.0,))

    assert drift["translation_drift_percent"] == pytest.approx(0.0)
    assert drift["rotation_drift_deg_per_m"] == pytest.approx(2.0)
    assert segments["2.0"]["rotation_drift_deg_per_m"] == pytest.approx(2.0)


def test_lost_gap_is_not_crossed_by_frame_distance_or_segment_pairs():
    gt = np.stack([make_pose(x=float(index)) for index in range(7)])
    est = gt.copy()
    valid_mask = np.array([True, True, True, False, True, True, True])

    frame_rpe = compute_multi_delta_rpe(est, gt, frame_deltas=(1, 2, 4), valid_mask=valid_mask)
    distance_rpe = compute_distance_delta_rpe(est, gt, distance_deltas_m=(2.0,), valid_mask=valid_mask)
    segments = compute_segment_errors(est, gt, segment_lengths_m=(2.0,), valid_mask=valid_mask)
    drift = compute_trajectory_drift(est, gt, valid_mask=valid_mask)

    assert frame_rpe["1"]["count"] == 4
    assert frame_rpe["2"]["count"] == 2
    assert frame_rpe["4"]["count"] == 0
    assert frame_rpe["4"]["translation_rmse_m"] is None
    assert distance_rpe["2.0"]["count"] == 2
    assert segments["2.0"]["count"] == 2
    assert all(sample["end_index"] < 3 or sample["start_index"] > 3 for sample in segments["2.0"]["samples"])
    assert drift["count"] == 2


def test_full_c2w_alignment_removes_global_rotation_and_translation():
    gt = np.stack([
        make_pose(x=0.0, y=0.0, yaw_deg=0.0),
        make_pose(x=1.0, y=0.0, yaw_deg=10.0),
        make_pose(x=0.0, y=1.0, yaw_deg=20.0),
        make_pose(x=1.0, y=1.0, yaw_deg=30.0),
    ])
    global_transform = make_pose(x=3.0, y=-2.0, yaw_deg=90.0)
    est = np.stack([global_transform @ pose for pose in gt])

    raw = compute_absolute_pose_errors(est, gt)
    aligned = compute_absolute_pose_errors(est, gt, align=True)
    alignment = align_c2w_trajectory(est, gt)

    assert raw["translation_rmse_m"] > 1.0
    assert raw["rotation_rmse_deg"] == pytest.approx(90.0)
    assert alignment["count"] == 4
    np.testing.assert_allclose(alignment["aligned_c2w"], gt, atol=1e-8)
    assert aligned["translation_rmse_m"] == pytest.approx(0.0, abs=1e-8)
    assert aligned["rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-6)


def test_c2w_alignment_optionally_estimates_scale():
    gt = np.stack([
        make_pose(x=0.0, y=0.0),
        make_pose(x=2.0, y=0.0),
        make_pose(x=0.0, y=2.0),
        make_pose(x=2.0, y=2.0),
    ])
    est = gt.copy()
    est[:, :3, 3] *= 0.5

    rigid = compute_absolute_pose_errors(est, gt, align=True, with_scale=False)
    scaled = compute_absolute_pose_errors(est, gt, align=True, with_scale=True)

    assert rigid["translation_rmse_m"] > 0.0
    assert scaled["alignment"]["scale"] == pytest.approx(2.0)
    assert scaled["translation_rmse_m"] == pytest.approx(0.0, abs=1e-8)
    assert scaled["rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-8)


def test_endpoint_and_return_to_start_metrics():
    gt = np.stack([
        make_pose(x=0.0),
        make_pose(x=1.0),
        make_pose(x=1.0, y=1.0),
        make_pose(x=0.0, y=0.0),
    ])
    est = gt.copy()
    est[-1] = make_pose(x=0.2, yaw_deg=5.0)

    endpoint = compute_endpoint_errors(est, gt, return_to_start_radius_m=0.1)

    assert endpoint["count"] == 1
    assert endpoint["is_return_to_start"] is True
    assert endpoint["absolute_translation_error_m"] == pytest.approx(0.2)
    assert endpoint["return_to_start_translation_error_m"] == pytest.approx(0.2)
    assert endpoint["return_to_start_rotation_error_deg"] == pytest.approx(5.0)


def test_invalid_final_frame_has_separate_last_valid_endpoint():
    gt = np.stack([make_pose(x=0.0), make_pose(x=1.0), make_pose(x=2.0)])
    est = gt.copy()
    est[1, 0, 3] += 0.25
    endpoint = compute_endpoint_errors(est, gt, valid_mask=np.array([True, True, False]))
    assert endpoint["final_frame_valid"] is False
    assert endpoint["endpoint_index"] == 2
    assert endpoint["absolute_translation_error_m"] is None
    assert endpoint["relative_translation_error_m"] is None
    assert endpoint["last_valid_endpoint_index"] == 1
    assert endpoint["last_valid_absolute_translation_error_m"] == pytest.approx(0.25)


def test_no_valid_data_returns_none_and_zero_counts_without_nan():
    poses = np.stack([make_pose(x=float(index)) for index in range(4)])
    metrics = compute_long_horizon_metrics(
        poses,
        poses,
        valid_mask=np.zeros(4, dtype=bool),
        frame_deltas=(1,),
        distance_deltas_m=(1.0,),
        segment_lengths_m=(1.0,),
    )

    assert metrics["raw_absolute"]["count"] == 0
    assert metrics["raw_absolute"]["translation_rmse_m"] is None
    assert metrics["aligned_absolute"]["count"] == 0
    assert metrics["frame_delta_rpe"]["1"]["count"] == 0
    assert metrics["distance_delta_rpe"]["1.0"]["count"] == 0
    assert metrics["segment_errors"]["1.0"]["count"] == 0
    assert metrics["drift"]["count"] == 0
    assert metrics["endpoint"]["count"] == 0

    def assert_no_nan(value):
        if isinstance(value, dict):
            for child in value.values():
                assert_no_nan(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_nan(child)
        elif isinstance(value, float):
            assert not np.isnan(value)

    assert_no_nan(metrics)
