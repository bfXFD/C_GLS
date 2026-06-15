import csv
import os

import numpy as np
import pytest

from utils.fsm_metrics import (
    append_summary_csv,
    compute_ate_rmse,
    compute_rpe_rmse,
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
