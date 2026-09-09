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


def _validate_trajectory_pair(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    est_c2w = _validate_poses(est_c2w, "est_c2w")
    gt_c2w = _validate_poses(gt_c2w, "gt_c2w")
    if est_c2w.shape[0] != gt_c2w.shape[0]:
        raise ValueError("est_c2w and gt_c2w must have the same trajectory length")
    if valid_mask is None:
        valid_mask = np.ones(est_c2w.shape[0], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.shape[0] != est_c2w.shape[0]:
            raise ValueError(f"Expected valid_mask with shape [{est_c2w.shape[0]}], got {valid_mask.shape}")
    return est_c2w, gt_c2w, valid_mask


def _empty_error_stats() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "translation_rmse_m": None,
        "translation_mean_m": None,
        "translation_median_m": None,
        "rotation_rmse_deg": None,
        "rotation_mean_deg": None,
        "rotation_median_deg": None,
    }


def _summarize_pose_errors(trans_errors: list[float], rot_errors: list[float]) -> dict[str, float | int | None]:
    if not trans_errors:
        return _empty_error_stats()
    trans = np.asarray(trans_errors, dtype=np.float64)
    rot = np.asarray(rot_errors, dtype=np.float64)
    return {
        "count": int(trans.shape[0]),
        "translation_rmse_m": float(np.sqrt(np.mean(trans * trans))),
        "translation_mean_m": float(np.mean(trans)),
        "translation_median_m": float(np.median(trans)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot * rot))),
        "rotation_mean_deg": float(np.mean(rot)),
        "rotation_median_deg": float(np.median(rot)),
    }


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[np.ndarray, np.ndarray, float]:
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        correction[-1, -1] = -1.0
    rotation = U @ correction @ Vt
    scale = 1.0
    if with_scale:
        variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("Cannot estimate scale from a trajectory with zero positional variance")
        scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return rotation, translation, scale


def align_c2w_trajectory(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
    with_scale: bool = False,
) -> dict[str, Any]:
    """Align an estimated c2w trajectory to GT using one SE(3) or Sim(3) transform."""
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.shape[0] < 3:
        return {
            "count": 0,
            "aligned_c2w": None,
            "rotation": None,
            "translation": None,
            "scale": None,
        }

    rotation, translation, scale = _umeyama_alignment(
        est_c2w[valid_indices, :3, 3],
        gt_c2w[valid_indices, :3, 3],
        with_scale=with_scale,
    )
    aligned = est_c2w.copy()
    aligned[:, :3, :3] = np.einsum("ij,tjk->tik", rotation, est_c2w[:, :3, :3])
    aligned[:, :3, 3] = scale * (rotation @ est_c2w[:, :3, 3].T).T + translation
    return {
        "count": int(valid_indices.shape[0]),
        "aligned_c2w": aligned,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
    }


def compute_absolute_pose_errors(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
    align: bool = False,
    with_scale: bool = False,
) -> dict[str, Any]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    evaluated = est_c2w
    alignment = None
    if align:
        alignment = align_c2w_trajectory(est_c2w, gt_c2w, valid_mask=valid_mask, with_scale=with_scale)
        if alignment["aligned_c2w"] is None:
            result = _empty_error_stats()
            result.update({
                "translation_errors_m": [None] * est_c2w.shape[0],
                "rotation_errors_deg": [None] * est_c2w.shape[0],
                "alignment": alignment,
            })
            return result
        evaluated = alignment["aligned_c2w"]

    trans_errors: list[float] = []
    rot_errors: list[float] = []
    per_frame_trans: list[float | None] = [None] * est_c2w.shape[0]
    per_frame_rot: list[float | None] = [None] * est_c2w.shape[0]
    for index in np.flatnonzero(valid_mask):
        trans_error = float(np.linalg.norm(evaluated[index, :3, 3] - gt_c2w[index, :3, 3]))
        rot_error = _rotation_angle_deg(gt_c2w[index, :3, :3].T @ evaluated[index, :3, :3])
        trans_errors.append(trans_error)
        rot_errors.append(rot_error)
        per_frame_trans[index] = trans_error
        per_frame_rot[index] = rot_error

    result = _summarize_pose_errors(trans_errors, rot_errors)
    result.update({
        "translation_errors_m": per_frame_trans,
        "rotation_errors_deg": per_frame_rot,
        "alignment": alignment,
    })
    return result


def _interval_is_valid(valid_prefix: np.ndarray, start: int, end: int) -> bool:
    invalid_count = valid_prefix[end + 1] - valid_prefix[start]
    return bool(invalid_count == 0)


def _compute_pair_errors(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[dict[str, float | int | None], list[dict[str, float | int]]]:
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    samples: list[dict[str, float | int]] = []
    for start, end in pairs:
        est_rel = _relative_transform(est_c2w[start], est_c2w[end])
        gt_rel = _relative_transform(gt_c2w[start], gt_c2w[end])
        error = np.linalg.inv(gt_rel) @ est_rel
        trans_error = float(np.linalg.norm(error[:3, 3]))
        rot_error = _rotation_angle_deg(error[:3, :3])
        trans_errors.append(trans_error)
        rot_errors.append(rot_error)
        samples.append({
            "start_index": int(start),
            "end_index": int(end),
            "translation_error_m": trans_error,
            "rotation_error_deg": rot_error,
        })
    return _summarize_pose_errors(trans_errors, rot_errors), samples


def compute_multi_delta_rpe(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    frame_deltas: tuple[int, ...] = (1, 10, 30, 100),
    valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid_mask)))
    results: dict[str, dict[str, Any]] = {}
    for delta in frame_deltas:
        if isinstance(delta, (bool, np.bool_)) or int(delta) != delta or delta <= 0:
            raise ValueError("frame deltas must be positive integers")
        delta = int(delta)
        pairs = [
            (start, start + delta)
            for start in range(max(0, est_c2w.shape[0] - delta))
            if _interval_is_valid(invalid_prefix, start, start + delta)
        ]
        stats, samples = _compute_pair_errors(est_c2w, gt_c2w, pairs)
        stats.update({"delta_frames": delta, "samples": samples})
        results[str(delta)] = stats
    return results


def _gt_cumulative_distance(gt_c2w: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(gt_c2w[:, :3, 3], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _distance_pairs(
    cumulative_distance: np.ndarray,
    distance_m: float,
    invalid_prefix: np.ndarray,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start in range(cumulative_distance.shape[0] - 1):
        end = int(np.searchsorted(cumulative_distance, cumulative_distance[start] + distance_m, side="left"))
        if end < cumulative_distance.shape[0] and end > start and _interval_is_valid(invalid_prefix, start, end):
            pairs.append((start, end))
    return pairs


def compute_distance_delta_rpe(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    distance_deltas_m: tuple[float, ...] = (1.0, 5.0, 10.0),
    valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    cumulative_distance = _gt_cumulative_distance(gt_c2w)
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid_mask)))
    results: dict[str, dict[str, Any]] = {}
    for distance_m in distance_deltas_m:
        distance_m = float(distance_m)
        if not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("distance deltas must be positive and finite")
        pairs = _distance_pairs(cumulative_distance, distance_m, invalid_prefix)
        stats, samples = _compute_pair_errors(est_c2w, gt_c2w, pairs)
        actual_distances = [float(cumulative_distance[end] - cumulative_distance[start]) for start, end in pairs]
        for sample, actual_distance in zip(samples, actual_distances):
            sample["gt_distance_m"] = actual_distance
        stats.update({
            "delta_distance_m": distance_m,
            "mean_actual_distance_m": float(np.mean(actual_distances)) if actual_distances else None,
            "samples": samples,
        })
        results[str(distance_m)] = stats
    return results


def compute_segment_errors(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    segment_lengths_m: tuple[float, ...] = (5.0, 10.0, 20.0),
    valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    cumulative_distance = _gt_cumulative_distance(gt_c2w)
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid_mask)))
    results: dict[str, dict[str, Any]] = {}
    for length_m in segment_lengths_m:
        length_m = float(length_m)
        if not np.isfinite(length_m) or length_m <= 0:
            raise ValueError("segment lengths must be positive and finite")
        pairs = _distance_pairs(cumulative_distance, length_m, invalid_prefix)
        stats, samples = _compute_pair_errors(est_c2w, gt_c2w, pairs)
        translation_drifts: list[float] = []
        rotation_drifts: list[float] = []
        for sample, (start, end) in zip(samples, pairs):
            actual_length = float(cumulative_distance[end] - cumulative_distance[start])
            trans_drift = 100.0 * float(sample["translation_error_m"]) / actual_length
            rot_drift = float(sample["rotation_error_deg"]) / actual_length
            sample.update({
                "gt_length_m": actual_length,
                "translation_drift_percent": trans_drift,
                "rotation_drift_deg_per_m": rot_drift,
            })
            translation_drifts.append(trans_drift)
            rotation_drifts.append(rot_drift)
        stats.update({
            "segment_length_m": length_m,
            "translation_drift_percent": float(np.mean(translation_drifts)) if translation_drifts else None,
            "rotation_drift_deg_per_m": float(np.mean(rotation_drifts)) if rotation_drifts else None,
            "samples": samples,
        })
        results[str(length_m)] = stats
    return results


def _contiguous_valid_runs(valid_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, valid in enumerate(valid_mask):
        if valid and start is None:
            start = index
        if start is not None and (not valid or index == valid_mask.shape[0] - 1):
            end = index if valid else index - 1
            if end > start:
                runs.append((start, end))
            start = None
    return runs


def compute_trajectory_drift(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    cumulative_distance = _gt_cumulative_distance(gt_c2w)
    runs = []
    total_distance = 0.0
    total_trans_error = 0.0
    total_rot_error = 0.0
    for start, end in _contiguous_valid_runs(valid_mask):
        distance = float(cumulative_distance[end] - cumulative_distance[start])
        if distance <= np.finfo(np.float64).eps:
            continue
        stats, samples = _compute_pair_errors(est_c2w, gt_c2w, [(start, end)])
        sample = samples[0]
        total_distance += distance
        total_trans_error += float(sample["translation_error_m"])
        total_rot_error += float(sample["rotation_error_deg"])
        runs.append({
            "start_index": start,
            "end_index": end,
            "gt_distance_m": distance,
            "translation_error_m": sample["translation_error_m"],
            "rotation_error_deg": sample["rotation_error_deg"],
        })
    if not runs:
        return {
            "count": 0,
            "gt_distance_m": None,
            "translation_drift_percent": None,
            "rotation_drift_deg_per_m": None,
            "runs": [],
        }
    return {
        "count": len(runs),
        "gt_distance_m": total_distance,
        "translation_drift_percent": 100.0 * total_trans_error / total_distance,
        "rotation_drift_deg_per_m": total_rot_error / total_distance,
        "runs": runs,
    }


def compute_endpoint_errors(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
    return_to_start_radius_m: float = 0.5,
) -> dict[str, Any]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    return_to_start_radius_m = float(return_to_start_radius_m)
    if not np.isfinite(return_to_start_radius_m) or return_to_start_radius_m < 0:
        raise ValueError("return_to_start_radius_m must be non-negative and finite")
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.shape[0] == 0:
        return {
            "count": 0,
            "endpoint_index": None,
            "last_valid_endpoint_index": None,
            "final_frame_valid": False,
            "absolute_translation_error_m": None,
            "absolute_rotation_error_deg": None,
            "relative_translation_error_m": None,
            "relative_rotation_error_deg": None,
            "last_valid_absolute_translation_error_m": None,
            "last_valid_absolute_rotation_error_deg": None,
            "last_valid_relative_translation_error_m": None,
            "last_valid_relative_rotation_error_deg": None,
            "gt_start_to_endpoint_distance_m": None,
            "is_return_to_start": None,
            "return_to_start_translation_error_m": None,
            "return_to_start_rotation_error_deg": None,
        }
    last_valid_endpoint = int(valid_indices[-1])
    final_frame_valid = bool(valid_mask[-1])
    absolute_translation = float(
        np.linalg.norm(est_c2w[last_valid_endpoint, :3, 3] - gt_c2w[last_valid_endpoint, :3, 3])
    )
    absolute_rotation = _rotation_angle_deg(
        gt_c2w[last_valid_endpoint, :3, :3].T @ est_c2w[last_valid_endpoint, :3, :3]
    )
    gt_start_distance = float(np.linalg.norm(gt_c2w[last_valid_endpoint, :3, 3] - gt_c2w[0, :3, 3]))
    complete_interval = last_valid_endpoint > 0 and bool(np.all(valid_mask[: last_valid_endpoint + 1]))
    relative_translation = None
    relative_rotation = None
    if complete_interval:
        stats, samples = _compute_pair_errors(est_c2w, gt_c2w, [(0, last_valid_endpoint)])
        relative_translation = float(samples[0]["translation_error_m"])
        relative_rotation = float(samples[0]["rotation_error_deg"])
    is_return = complete_interval and gt_start_distance <= return_to_start_radius_m
    return {
        "count": 1,
        "endpoint_index": est_c2w.shape[0] - 1,
        "last_valid_endpoint_index": last_valid_endpoint,
        "final_frame_valid": final_frame_valid,
        "absolute_translation_error_m": absolute_translation if final_frame_valid else None,
        "absolute_rotation_error_deg": absolute_rotation if final_frame_valid else None,
        "relative_translation_error_m": relative_translation if final_frame_valid else None,
        "relative_rotation_error_deg": relative_rotation if final_frame_valid else None,
        "last_valid_absolute_translation_error_m": absolute_translation,
        "last_valid_absolute_rotation_error_deg": absolute_rotation,
        "last_valid_relative_translation_error_m": relative_translation,
        "last_valid_relative_rotation_error_deg": relative_rotation,
        "gt_start_to_endpoint_distance_m": gt_start_distance,
        "is_return_to_start": is_return,
        "return_to_start_translation_error_m": relative_translation if is_return else None,
        "return_to_start_rotation_error_deg": relative_rotation if is_return else None,
    }


def compute_long_horizon_metrics(
    est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
    frame_deltas: tuple[int, ...] = (1, 10, 30, 100),
    distance_deltas_m: tuple[float, ...] = (1.0, 5.0, 10.0),
    segment_lengths_m: tuple[float, ...] = (5.0, 10.0, 20.0),
    align_with_scale: bool = False,
    return_to_start_radius_m: float = 0.5,
) -> dict[str, Any]:
    est_c2w, gt_c2w, valid_mask = _validate_trajectory_pair(est_c2w, gt_c2w, valid_mask)
    return {
        "pose_convention": "c2w",
        "valid_frame_count": int(np.count_nonzero(valid_mask)),
        "raw_absolute": compute_absolute_pose_errors(est_c2w, gt_c2w, valid_mask=valid_mask),
        "aligned_absolute": compute_absolute_pose_errors(
            est_c2w,
            gt_c2w,
            valid_mask=valid_mask,
            align=True,
            with_scale=align_with_scale,
        ),
        "frame_delta_rpe": compute_multi_delta_rpe(
            est_c2w,
            gt_c2w,
            frame_deltas=frame_deltas,
            valid_mask=valid_mask,
        ),
        "distance_delta_rpe": compute_distance_delta_rpe(
            est_c2w,
            gt_c2w,
            distance_deltas_m=distance_deltas_m,
            valid_mask=valid_mask,
        ),
        "segment_errors": compute_segment_errors(
            est_c2w,
            gt_c2w,
            segment_lengths_m=segment_lengths_m,
            valid_mask=valid_mask,
        ),
        "drift": compute_trajectory_drift(est_c2w, gt_c2w, valid_mask=valid_mask),
        "endpoint": compute_endpoint_errors(
            est_c2w,
            gt_c2w,
            valid_mask=valid_mask,
            return_to_start_radius_m=return_to_start_radius_m,
        ),
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
