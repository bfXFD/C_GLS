import csv
import os
from typing import Any

import numpy as np


def _validate_poses(poses: np.ndarray, name: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected {name} with shape [T, 4, 4], got {poses.shape}")
    if poses.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(poses)):
        raise ValueError(f"{name} must contain only finite values")
    return poses


def _validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected {name} with shape [4, 4], got {transform.shape}")
    if np.any(~np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values")
    return transform


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape [3, 3], got {R.shape}")

    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Invalid rotation matrix produced a zero or non-finite quaternion")
    return quat / norm


def transform_to_tum_line(timestamp: float, c2w: np.ndarray) -> str:
    c2w = _validate_transform(c2w, "c2w")
    quat = rotation_matrix_to_quaternion_xyzw(c2w[:3, :3])
    t = c2w[:3, 3]
    values = [float(timestamp), float(t[0]), float(t[1]), float(t[2]), *quat.tolist()]
    return " ".join(f"{value:.9f}" for value in values)


def save_tum_trajectory(path: str, timestamps: np.ndarray, poses_c2w: np.ndarray) -> None:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    poses_c2w = _validate_poses(poses_c2w, "poses_c2w")
    if timestamps.ndim != 1 or timestamps.shape[0] != poses_c2w.shape[0]:
        raise ValueError(f"Expected timestamps with shape [{poses_c2w.shape[0]}], got {timestamps.shape}")
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        for timestamp, pose in zip(timestamps, poses_c2w):
            f.write(transform_to_tum_line(float(timestamp), pose) + "\n")


def umeyama_align_positions(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> tuple[np.ndarray, float]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src and dst with matching shape [N, 3], got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least three positions are required for Umeyama alignment")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        variance = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = float(np.trace(np.diag(singular_values) @ S) / variance)
    else:
        scale = 1.0
    t = dst_mean - scale * R @ src_mean
    aligned = scale * (R @ src.T).T + t
    return aligned, scale


def compute_ate_rmse(est_c2w: np.ndarray, gt_c2w: np.ndarray, align: bool = True, with_scale: bool = False) -> float:
    est_c2w = _validate_poses(est_c2w, "est_c2w")
    gt_c2w = _validate_poses(gt_c2w, "gt_c2w")
    if est_c2w.shape[0] != gt_c2w.shape[0]:
        raise ValueError("est_c2w and gt_c2w must have the same trajectory length")

    est_positions = est_c2w[:, :3, 3]
    gt_positions = gt_c2w[:, :3, 3]
    if align:
        est_positions, _ = umeyama_align_positions(est_positions, gt_positions, with_scale=with_scale)
    errors = est_positions - gt_positions
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def _relative_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a) @ b


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def compute_rpe_rmse(est_c2w: np.ndarray, gt_c2w: np.ndarray, delta: int = 1) -> dict[str, float]:
    est_c2w = _validate_poses(est_c2w, "est_c2w")
    gt_c2w = _validate_poses(gt_c2w, "gt_c2w")
    if est_c2w.shape[0] != gt_c2w.shape[0]:
        raise ValueError("est_c2w and gt_c2w must have the same trajectory length")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if est_c2w.shape[0] <= delta:
        raise ValueError("trajectory length must be greater than delta")

    trans_errors = []
    rot_errors = []
    for i in range(est_c2w.shape[0] - delta):
        est_rel = _relative_transform(est_c2w[i], est_c2w[i + delta])
        gt_rel = _relative_transform(gt_c2w[i], gt_c2w[i + delta])
        error = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(np.linalg.norm(error[:3, 3]))
        rot_errors.append(_rotation_angle_deg(error[:3, :3]))
    trans_errors = np.asarray(trans_errors, dtype=np.float64)
    rot_errors = np.asarray(rot_errors, dtype=np.float64)
    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_errors * trans_errors))),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_errors * rot_errors))),
    }


def append_summary_csv(path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    if not fieldnames:
        raise ValueError("fieldnames must not be empty")
    missing = [field for field in fieldnames if field not in row]
    if missing:
        raise KeyError(f"row is missing fields required by fieldnames: {missing}")
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fieldnames})
