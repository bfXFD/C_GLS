import os
import time
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.fsm_correspondence_weak import (
    correspondence_aware_weak_scores,
    correspondence_probability,
    normalized_hessian_diagnostics,
    organized_depth_source_normals,
)
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_map import load_runtime_reserve_geometry
from utils.fsm_robust_registration import evaluate_point_to_plane_diagnostics


_TRIGGER_BITS = {
    "condition_number": 1,
    "normalized_lambda_min": 2,
    "iterations": 4,
    "fitness": 8,
    "effective_rank_below": 16,
    "degenerate": 32,
}


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _project_near_rotation(rotation: np.ndarray, max_defect: float = 1e-3) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or np.any(~np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    orthogonality_defect = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or max(orthogonality_defect, abs(determinant - 1.0)) > float(max_defect):
        raise ValueError("transform rotation is too far from SO(3)")
    if orthogonality_defect <= 1e-7 and abs(determinant - 1.0) <= 1e-7:
        return rotation
    left, _, right = np.linalg.svd(rotation)
    projected = left @ right
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right
    return projected


def _pose_for_relative_log(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or np.any(~np.isfinite(value)):
        raise ValueError("pose must be a finite 4x4 matrix")
    normalized = value.copy()
    normalized[:3, :3] = _project_near_rotation(value[:3, :3])
    normalized[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return normalized


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Return a signed SE(3) logarithm in [rotation, translation] order."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or np.any(~np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = _project_near_rotation(value[:3, :3])
    value = value.copy()
    value[:3, :3] = rotation
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    antisymmetric = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1e-8:
        omega = 0.5 * antisymmetric
    elif np.pi - angle < 1e-6:
        symmetric = 0.5 * (rotation + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        pivot = int(np.argmax(axis))
        if axis[pivot] <= 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != pivot:
                    axis[index] = symmetric[pivot, index] / axis[pivot]
            axis /= np.linalg.norm(axis)
        if np.dot(axis, antisymmetric) < 0.0:
            axis = -axis
        omega = angle * axis
    else:
        omega = angle * antisymmetric / (2.0 * np.sin(angle))
    omega_hat = _skew(omega)
    if angle < 1e-8:
        inverse_left_jacobian = np.eye(3) - 0.5 * omega_hat + omega_hat @ omega_hat / 12.0
    else:
        coefficient = 1.0 / angle**2 - (1.0 + np.cos(angle)) / (2.0 * angle * np.sin(angle))
        inverse_left_jacobian = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
    return np.concatenate((omega, inverse_left_jacobian @ value[:3, 3]))


def relative_se3_log(reference_pose: np.ndarray, candidate_pose: np.ndarray) -> dict[str, Any]:
    """Describe a candidate pose relative to a reference without ground truth."""
    reference = _pose_for_relative_log(reference_pose)
    candidate = _pose_for_relative_log(candidate_pose)
    twist = se3_log(np.linalg.inv(reference) @ candidate)
    rotation_norm = float(np.linalg.norm(twist[:3]))
    translation_norm = float(np.linalg.norm(twist[3:]))
    return {
        "twist": twist,
        "norm": float(np.linalg.norm(twist)),
        "rotation_norm_rad": rotation_norm,
        "rotation_norm_deg": float(np.degrees(rotation_norm)),
        "translation_norm_m": translation_norm,
    }


def evaluate_common_target_pose(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    target_tree: Any,
    pose: np.ndarray,
    *,
    max_correspondence_distance: float,
    robust_loss: str,
    loss_scale: float,
    min_correspondences: int,
    rank_tolerance: float,
) -> dict[str, Any]:
    """Evaluate a fixed pose on the primary target; nearest neighbors may vary."""
    result = evaluate_point_to_plane_diagnostics(
        source_points, target_points, target_normals, pose,
        max_correspondence_distance=max_correspondence_distance,
        target_tree=target_tree, loss=robust_loss, loss_scale=loss_scale,
        min_correspondences=min_correspondences, rank_tolerance=rank_tolerance,
    )
    count = int(result.get("num_correspondences", 0))
    robust_cost = float(result.get("robust_cost", float("inf")))
    return {
        "robust_cost_per_corr": robust_cost / count if count > 0 and np.isfinite(robust_cost) else float("inf"),
        "point_to_plane_rmse": float(result.get("point_to_plane_rmse", float("inf"))),
        "fitness": float(result.get("fitness", 0.0)),
        "corr_count": count,
    }


def _frozen_array(value: Any, dtype: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and array.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.number) and np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RuntimeReserveObservation:
    source_points_camera: np.ndarray
    source_pixel_ids: np.ndarray
    depth: np.ndarray
    primary_target_tree: Any
    primary_target_global_ids: np.ndarray
    source_normals_camera: np.ndarray | None = None
    source_normal_confidence: np.ndarray | None = None
    primary_source_target_distances: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = _frozen_array(self.source_points_camera, np.float64, (None, 3), "source_points_camera")
        pixels = _frozen_array(self.source_pixel_ids, np.int64, (points.shape[0],), "source_pixel_ids")
        raw_depth = np.asarray(self.depth)
        if raw_depth.ndim == 3 and raw_depth.shape[2] == 1:
            raw_depth = raw_depth[..., 0]
        depth = _frozen_array(raw_depth, np.float64, (None, None), "depth")
        if np.any(pixels < 0) or np.any(pixels >= depth.size):
            raise ValueError("source_pixel_ids must be valid flat depth indices")
        target_ids = _frozen_array(self.primary_target_global_ids, np.int64, (None,), "primary_target_global_ids")
        if np.unique(target_ids).size != target_ids.size:
            raise ValueError("primary_target_global_ids must be unique")
        if self.primary_target_tree is None:
            raise ValueError("primary_target_tree must be available")
        normals = None if self.source_normals_camera is None else _frozen_array(
            self.source_normals_camera, np.float64, points.shape, "source_normals_camera"
        )
        confidence = None
        if self.source_normal_confidence is not None:
            confidence = _frozen_array(
                self.source_normal_confidence, np.float64, (points.shape[0],), "source_normal_confidence"
            )
            if normals is None or np.any((confidence < 0.0) | (confidence > 1.0)):
                raise ValueError("source normal confidence requires normals and must be in [0, 1]")
        distances = None
        if self.primary_source_target_distances is not None:
            distances = np.array(self.primary_source_target_distances, dtype=np.float64, copy=True)
            if distances.shape != (points.shape[0],) or np.any(np.isnan(distances)) or np.any(distances < 0.0):
                raise ValueError("primary_source_target_distances must be non-negative with shape [N]")
            distances.setflags(write=False)
        for name, value in (
            ("source_points_camera", points), ("source_pixel_ids", pixels), ("depth", depth),
            ("primary_target_global_ids", target_ids), ("source_normals_camera", normals),
            ("source_normal_confidence", confidence), ("primary_source_target_distances", distances),
        ):
            object.__setattr__(self, name, value)


def _adjoint_c2w(c2w: np.ndarray) -> np.ndarray:
    """Map a camera-frame [rotation, translation] twist to the world left twist."""
    rotation = np.asarray(c2w, dtype=np.float64)[:3, :3]
    translation = np.asarray(c2w, dtype=np.float64)[:3, 3]
    skew = np.array(
        [[0.0, -translation[2], translation[1]],
         [translation[2], 0.0, -translation[0]],
         [-translation[1], translation[0], 0.0]],
        dtype=np.float64,
    )
    adjoint = np.zeros((6, 6), dtype=np.float64)
    adjoint[:3, :3] = rotation
    adjoint[3:, :3] = skew @ rotation
    adjoint[3:, 3:] = rotation
    return adjoint


def reserve_point_jacobians_camera(
    means_world: np.ndarray, normals_world: np.ndarray, candidate_c2w: np.ndarray
) -> np.ndarray:
    """Approximate point-to-plane Jacobians in the tracker [rotation, translation] order."""
    means_world = np.asarray(means_world, dtype=np.float64)
    normals_world = np.asarray(normals_world, dtype=np.float64)
    candidate_c2w = np.asarray(candidate_c2w, dtype=np.float64)
    if means_world.ndim != 2 or means_world.shape[1] != 3 or normals_world.shape != means_world.shape:
        raise ValueError("reserve means and normals must have matching shape [N, 3]")
    if candidate_c2w.shape != (4, 4) or np.any(~np.isfinite(candidate_c2w)):
        raise ValueError("candidate_c2w must be a finite 4x4 transform")
    rotation = candidate_c2w[:3, :3]
    points_camera = (means_world - candidate_c2w[:3, 3]) @ rotation
    normals_camera = normals_world @ rotation
    lengths = np.linalg.norm(normals_camera, axis=1)
    if np.any(~np.isfinite(points_camera)) or np.any(~np.isfinite(normals_camera)) or np.any(lengths <= 0.0):
        raise ValueError("reserve geometry must produce finite non-zero camera normals")
    normals_camera = normals_camera / lengths[:, None]
    return np.concatenate([np.cross(points_camera, normals_camera), normals_camera], axis=1)


def runtime_weak_scores(
    means_world: np.ndarray,
    normals_world: np.ndarray,
    candidate_c2w: np.ndarray,
    tracker_hessian: np.ndarray,
    weak_subspace_dim: int = 1,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score reserve rows against weak camera-frame directions of the final tracker Hessian."""
    hessian = np.asarray(tracker_hessian, dtype=np.float64)
    if hessian.shape != (6, 6) or np.any(~np.isfinite(hessian)):
        raise ValueError("tracker_hessian must be a finite 6x6 matrix")
    if weak_subspace_dim < 1 or weak_subspace_dim > 3:
        raise ValueError("weak_subspace_dim must be between 1 and 3")
    adjoint = _adjoint_c2w(candidate_c2w)
    camera_hessian = adjoint.T @ (0.5 * (hessian + hessian.T)) @ adjoint
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (camera_hessian + camera_hessian.T))
    weak_vectors = eigenvectors[:, :weak_subspace_dim]
    jacobians = reserve_point_jacobians_camera(means_world, normals_world, candidate_c2w)
    projected = jacobians @ weak_vectors
    scores = np.sum(projected * projected, axis=1)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != scores.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative with shape [N]")
        scores *= weights
    return scores, weak_vectors, eigenvalues


def hessian_block_diagnostics(tracker_hessian: np.ndarray, candidate_c2w: np.ndarray) -> dict[str, Any]:
    """Summarize rotational and translational observability in camera coordinates."""
    hessian = np.asarray(tracker_hessian, dtype=np.float64)
    pose = np.asarray(candidate_c2w, dtype=np.float64)
    empty = {
        "reserve_hessian_rotation_trace": None,
        "reserve_hessian_translation_trace": None,
        "reserve_hessian_cross_frobenius": None,
        "reserve_hessian_rotation_lambda_min": None,
        "reserve_hessian_translation_lambda_min": None,
        "reserve_weak_rotation_energy": None,
        "reserve_weak_translation_energy": None,
        "reserve_weak_dominant_block": None,
    }
    if hessian.shape != (6, 6) or pose.shape != (4, 4):
        return empty
    if np.any(~np.isfinite(hessian)) or np.any(~np.isfinite(pose)):
        return empty
    adjoint = _adjoint_c2w(pose)
    camera = adjoint.T @ (0.5 * (hessian + hessian.T)) @ adjoint
    camera = 0.5 * (camera + camera.T)
    _, eigenvectors = np.linalg.eigh(camera)
    weak = eigenvectors[:, 0]
    rotation_energy = float(np.dot(weak[:3], weak[:3]))
    translation_energy = float(np.dot(weak[3:], weak[3:]))
    return {
        "reserve_hessian_rotation_trace": float(np.trace(camera[:3, :3])),
        "reserve_hessian_translation_trace": float(np.trace(camera[3:, 3:])),
        "reserve_hessian_cross_frobenius": float(np.linalg.norm(camera[:3, 3:], ord="fro")),
        "reserve_hessian_rotation_lambda_min": float(np.linalg.eigvalsh(camera[:3, :3])[0]),
        "reserve_hessian_translation_lambda_min": float(np.linalg.eigvalsh(camera[3:, 3:])[0]),
        "reserve_weak_rotation_energy": rotation_energy,
        "reserve_weak_translation_energy": translation_energy,
        "reserve_weak_dominant_block": "rotation" if rotation_energy >= translation_energy else "translation",
    }


def rank_diverse_reserve(
    indices: np.ndarray,
    scores: np.ndarray,
    means_world: np.ndarray,
    candidate_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    tile_rows: int,
    tile_cols: int,
    depth_bins: int,
    voxel_size: float,
    tile_cap: int | None,
    depth_cap: int | None,
    voxel_cap: int | None,
    global_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complete deterministic ranking that satisfies diversity caps."""
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    means_world = np.asarray(means_world, dtype=np.float64)
    ids = indices if global_ids is None else np.asarray(global_ids, dtype=np.int64)[indices]
    if indices.size == 0:
        return indices.copy()
    rotation = np.asarray(candidate_c2w, dtype=np.float64)[:3, :3]
    camera = (means_world[indices] - np.asarray(candidate_c2w)[:3, 3]) @ rotation
    depth = camera[:, 2]
    u = intrinsics[0, 0] * camera[:, 0] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / depth + intrinsics[1, 2]
    tile_col = np.clip(np.floor(u * tile_cols / image_width).astype(np.int64), 0, tile_cols - 1)
    tile_row = np.clip(np.floor(v * tile_rows / image_height).astype(np.int64), 0, tile_rows - 1)
    tile = tile_row * tile_cols + tile_col
    positive = depth[depth > 0.0]
    near = max(float(np.min(positive)) if positive.size else np.finfo(float).tiny, np.finfo(float).tiny)
    far = max(float(np.max(depth)), near * (1.0 + 1e-9))
    edges = np.geomspace(near, far, depth_bins + 1)
    depth_bucket = np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, depth_bins - 1)
    voxel = np.floor(means_world[indices] / voxel_size).astype(np.int64)
    order = np.lexsort((ids, -scores[indices]))
    tile_counts: dict[int, int] = {}
    depth_counts: dict[int, int] = {}
    voxel_counts: dict[tuple[int, int, int], int] = {}
    selected = []
    for local in order:
        tile_key = int(tile[local])
        depth_key = int(depth_bucket[local])
        voxel_key = tuple(int(value) for value in voxel[local])
        if tile_cap is not None and tile_counts.get(tile_key, 0) >= tile_cap:
            continue
        if depth_cap is not None and depth_counts.get(depth_key, 0) >= depth_cap:
            continue
        if voxel_cap is not None and voxel_counts.get(voxel_key, 0) >= voxel_cap:
            continue
        selected.append(int(indices[local]))
        tile_counts[tile_key] = tile_counts.get(tile_key, 0) + 1
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
        voxel_counts[voxel_key] = voxel_counts.get(voxel_key, 0) + 1
    return np.asarray(selected, dtype=np.int64)


def select_diverse_reserve(
    indices: np.ndarray,
    scores: np.ndarray,
    means_world: np.ndarray,
    candidate_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    budget: int,
    tile_rows: int,
    tile_cols: int,
    depth_bins: int,
    voxel_size: float,
    tile_cap: int | None,
    depth_cap: int | None,
    voxel_cap: int | None,
    global_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible budget prefix of :func:`rank_diverse_reserve`."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    order = rank_diverse_reserve(
        indices=indices,
        scores=scores,
        means_world=means_world,
        candidate_c2w=candidate_c2w,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        depth_bins=depth_bins,
        voxel_size=voxel_size,
        tile_cap=tile_cap,
        depth_cap=depth_cap,
        voxel_cap=voxel_cap,
        global_ids=global_ids,
    )
    return order[:budget].copy()


@dataclass
class ReservePlan:
    metrics: dict[str, Any]
    selected_indices: np.ndarray
    selected_scores: np.ndarray | None = None
    selected_probabilities: np.ndarray | None = None
    selected_global_ids: np.ndarray | None = None
    selection_policy: str | None = None


class RuntimeObservableReserve:
    """Bounded runtime reserve planner backed by activation-geometry schema v2."""

    def __init__(
        self,
        config: dict[str, Any],
        fsm_map: dict[str, Any],
        map_path: str,
        local_config: LocalSubmapConfig | None,
        evaluation_protocol: str,
        backend: str,
    ) -> None:
        cfg = config.get("observable_reserve", {}) or {}
        if not isinstance(cfg, dict):
            raise ValueError("observable_reserve must be a dictionary")
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "off"))
        if self.mode not in ("off", "plan_only", "active"):
            raise ValueError("observable_reserve.mode must be off, plan_only, or active")
        self.active = self.enabled and self.mode != "off"
        self.spectral_coordinates = str(cfg.get("spectral_coordinates", "world_raw"))
        if self.spectral_coordinates not in ("world_raw", "primary_camera_depth"):
            raise ValueError("observable_reserve.spectral_coordinates must be world_raw or primary_camera_depth")
        self.attempt_without_selection = cfg.get("attempt_without_selection", False)
        if not isinstance(self.attempt_without_selection, bool):
            raise ValueError("observable_reserve.attempt_without_selection must be boolean")
        if self.attempt_without_selection and cfg.get("candidate_target_mode", "append") != "rerun_primary":
            raise ValueError("observable_reserve.attempt_without_selection requires rerun_primary")
        self.available = False
        self.unavailable_reason = "disabled"
        self.geometry_path: str | None = None
        self.records: list[dict[str, Any]] = []
        self.stability_attempts: list[dict[str, Any]] = []
        self.config = cfg
        self.local_config = local_config
        stability = cfg.get("stability_gate", {}) or {}
        if not isinstance(stability, dict):
            raise ValueError("observable_reserve.stability_gate must be a dictionary")
        self.stability_enabled = bool(stability.get("enabled", False))
        self.stability_mode = str(stability.get("mode", "off"))
        if self.stability_mode not in ("off", "report_only", "enforce"):
            raise ValueError("observable_reserve.stability_gate.mode must be off, report_only, or enforce")
        self.stability_active = self.stability_enabled and self.stability_mode != "off"
        self.stability_config = stability
        self.rolling_window_frames = int(stability.get("rolling_window_frames", 5))
        self.max_consecutive_adoptions = int(stability.get("max_consecutive_adoptions", 0))
        self.cooldown_frames = int(stability.get("cooldown_frames", 0))
        if self.rolling_window_frames <= 0 or self.max_consecutive_adoptions < 0 or self.cooldown_frames < 0:
            raise ValueError("stability rolling window must be positive; consecutive and cooldown limits non-negative")
        self.adopted_twists: deque[tuple[int, np.ndarray]] = deque()
        self.consecutive_adoptions = 0
        self.last_adoption_frame: int | None = None
        self.branch_consecutive_adoptions = 0
        self.last_adoption_branch: str | None = None
        self.branch_cooldown_until_frame: int | None = None
        self.global_consecutive_adoptions = 0
        self.global_cooldown_until_frame: int | None = None
        if not self.active:
            return
        if evaluation_protocol != "mvp_strict" or local_config is None:
            raise ValueError("observable_reserve requires mvp_strict with local_submap.enabled")
        if backend != "numpy_robust_point_to_plane":
            raise ValueError("observable_reserve requires numpy_robust_point_to_plane")
        self.budget = int(cfg.get("budget", 1000))
        self.max_points = int(cfg.get("max_points", self.budget))
        self.max_visible = int(cfg.get("max_visible", max(self.max_points, self.budget)))
        if self.budget <= 0 or self.max_points <= 0 or self.max_visible <= 0 or self.budget > self.max_points:
            raise ValueError("observable_reserve requires 0 < budget <= max_points and positive max_visible")
        self.weak_dim = int(cfg.get("weak_subspace_dim", 1))
        if self.weak_dim < 1 or self.weak_dim > 3:
            raise ValueError("observable_reserve.weak_subspace_dim must be between 1 and 3")
        self.selection = str(cfg.get("selection", "runtime_weak"))
        self.corr_rank_control = cfg.get("corr_rank_control", "corr_weak")
        if self.corr_rank_control not in ("corr_weak", "weak", "random"):
            raise ValueError("Invalid observable_reserve.corr_rank_control")
        if self.corr_rank_control != "corr_weak" and self.selection != "runtime_corr_weak":
            raise ValueError("corr_rank_control requires runtime_corr_weak")
        if self.selection not in ("runtime_weak", "runtime_corr_weak", "saliency", "random"):
            raise ValueError(
                "observable_reserve.selection must be runtime_weak, runtime_corr_weak, saliency, or random"
            )
        self.seed = int(cfg.get("seed", config.get("seed", 0)))
        self.trigger = cfg.get("trigger", {}) or {}
        if not isinstance(self.trigger, dict):
            raise ValueError("observable_reserve.trigger must be a dictionary")
        self.rerun_iterations = int(cfg.get("rerun_iterations", config.get("registration", {}).get("max_iterations", 30)))
        if self.rerun_iterations <= 0:
            raise ValueError("observable_reserve.rerun_iterations must be positive")
        self.candidate_target_mode = str(cfg.get("candidate_target_mode", "append"))
        if self.candidate_target_mode not in ("append", "replace", "rerun_primary"):
            raise ValueError(
                "observable_reserve.candidate_target_mode must be append, replace, or rerun_primary"
            )
        self.candidate_normal_mode = str(cfg.get("candidate_normal_mode", "augmented_local_pca"))
        if self.candidate_normal_mode not in ("augmented_local_pca", "frozen_concat"):
            raise ValueError(
                "observable_reserve.candidate_normal_mode must be augmented_local_pca or frozen_concat"
            )
        self.replacement_seed = int(cfg.get("replacement_seed", self.seed))
        self.accept_only_if = cfg.get("accept_only_if", {}) or {}
        if not isinstance(self.accept_only_if, dict):
            raise ValueError("observable_reserve.accept_only_if must be a dictionary")
        protocol = self.accept_only_if.get("protocol")
        if protocol not in (
            None, "condition_or_lambda_v2", "branch_aware_information_v3", "branch_aware_information_v4",
        ):
            raise ValueError("observable_reserve.accept_only_if.protocol is unsupported")
        accept_mode = str(self.accept_only_if.get("mode", "all"))
        if accept_mode not in ("all", "any"):
            raise ValueError("observable_reserve.accept_only_if.mode must be all or any")
        for key in ("condition_ratio", "lambda_ratio"):
            if self.accept_only_if.get(key) is not None:
                value = float(self.accept_only_if[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"observable_reserve.accept_only_if.{key} must be finite and positive")
        self.branch_policy = self.accept_only_if.get("branch_policy", {}) or {}
        if protocol in ("branch_aware_information_v3", "branch_aware_information_v4"):
            if not isinstance(self.branch_policy, dict):
                raise ValueError("observable_reserve.accept_only_if.branch_policy must be a dictionary")
            for branch in ("both", "condition_only", "lambda_only"):
                policy = self.branch_policy.get(branch)
                if not isinstance(policy, dict):
                    raise ValueError(f"observable_reserve.accept_only_if.branch_policy.{branch} is required")
                for key in ("condition_ratio", "lambda_ratio", "max_refinement_translation_m", "max_refinement_rotation_deg"):
                    value = float(policy.get(key, 0.0))
                    if not np.isfinite(value) or value <= 0.0:
                        raise ValueError(f"observable_reserve.accept_only_if.branch_policy.{branch}.{key} must be finite and positive")
                for key in ("max_consecutive_adoptions", "cooldown_frames"):
                    value = int(policy.get(key, -1))
                    if value < 0 or (key == "max_consecutive_adoptions" and value == 0):
                        raise ValueError(f"observable_reserve.accept_only_if.branch_policy.{branch}.{key} is invalid")
        if protocol == "branch_aware_information_v4":
            for key in (
                "max_candidate_init_translation_m", "max_candidate_init_rotation_deg",
                "joint_refinement_rotation_deg", "joint_temporal_rotation_deg",
            ):
                value = float(self.accept_only_if.get(key, 0.0))
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"observable_reserve.accept_only_if.{key} must be finite and positive")
            self.max_global_consecutive_adoptions = int(
                self.accept_only_if.get("max_global_consecutive_adoptions", 0)
            )
            self.global_cooldown_frames = int(self.accept_only_if.get("global_cooldown_frames", -1))
            if self.max_global_consecutive_adoptions <= 0 or self.global_cooldown_frames < 0:
                raise ValueError("observable_reserve v4 global adoption limits are invalid")
            require_joint_metrics = self.accept_only_if.get("require_joint_metrics", True)
            if not isinstance(require_joint_metrics, bool):
                raise ValueError("observable_reserve.accept_only_if.require_joint_metrics must be boolean")
        self.relative_unavailable_policy = str(cfg.get("relative_unavailable_policy", "reject"))
        if self.relative_unavailable_policy not in ("reject", "tracking_only"):
            raise ValueError("observable_reserve.relative_unavailable_policy must be reject or tracking_only")
        geometry_cfg = cfg.get("geometry_path", "auto")
        automatic = os.path.join(os.path.dirname(os.path.abspath(map_path)), "activation_geometry.npz")
        self.geometry_path = automatic if geometry_cfg in (None, "auto") else str(geometry_cfg)
        if not os.path.isfile(self.geometry_path):
            self.unavailable_reason = "activation_geometry_unavailable"
            return
        expected_params = str(np.asarray(fsm_map.get("source_params_sha256", "")))
        expected_controls = str(np.asarray(fsm_map.get("selection_controls_sha256", "")))
        if not expected_params or not expected_controls:
            self.unavailable_reason = "map_provenance_unavailable"
            return
        geometry = load_runtime_reserve_geometry(self.geometry_path, fsm_map)
        if geometry is None:
            self.unavailable_reason = "schema_v1_no_reserve"
            return
        self.global_ids = np.asarray(geometry["reserve_global_ids"], dtype=np.int64)
        self.points = np.asarray(geometry["reserve_means"], dtype=np.float64)
        self.normals = np.asarray(geometry["reserve_normals"], dtype=np.float64)
        self.scores = np.asarray(geometry["reserve_scores"], dtype=np.float64)
        self.covariances = np.asarray(geometry["reserve_covariances"], dtype=np.float64)
        self.supported = np.asarray(
            geometry.get("reserve_supported", np.ones(self.points.shape[0], dtype=bool)), dtype=bool
        )
        self.plan_normal_source = str(np.asarray(geometry.get("reserve_normal_source", "sidecar_reserve_normals")))
        if self.points.shape[0] == 0:
            self.unavailable_reason = "empty_reserve"
            return
        spatial = cfg.get("spatial", {}) or {}
        self.index = MapSpatialIndex(
            self.points,
            backend=str(cfg.get("spatial_backend", spatial.get("backend", "auto"))),
            voxel_size=float(cfg.get("grid_voxel", spatial.get("voxel_size", 0.5))),
        )
        self.available = True
        self.unavailable_reason = None

    @staticmethod
    def empty_metrics(mode: str = "off", reason: str | None = None) -> dict[str, Any]:
        return {
            "reserve_attempted": False,
            "reserve_triggered": False,
            "reserve_trigger_reasons": "",
            "reserve_trigger_unavailable_reason": None,
            "reserve_visible": 0,
            "reserve_selected": 0,
            "reserve_correspondence": None,
            "reserve_candidate_optimizer_valid": None,
            "reserve_candidate_fitness": None,
            "reserve_candidate_rmse": None,
            "reserve_candidate_effective_rank": None,
            "reserve_candidate_hessian_trace": None,
            "reserve_candidate_normalized_lambda_min": None,
            "reserve_selection_score": None,
            "reserve_primary_condition": None,
            "reserve_spectral_coordinates": "world_raw",
            "reserve_spectral_length": None,
            "reserve_spectral_primary_lambda": None,
            "reserve_spectral_candidate_lambda": None,
            "reserve_spectral_primary_condition": None,
            "reserve_spectral_candidate_condition": None,
            "reserve_augmented_condition": None,
            "reserve_primary_lambda_min": None,
            "reserve_augmented_lambda_min": None,
            "reserve_accepted": False,
            "reserve_used": False,
            "reserve_primary_accepted": None,
            "reserve_tracking_accepted": False,
            "reserve_consistency_accepted": False,
            "reserve_relative_improvement_accepted": False,
            "reserve_relative_improvement_protocol": None,
            "reserve_relative_improvement_mode": None,
            "reserve_relative_condition_available": None,
            "reserve_relative_condition_pass": None,
            "reserve_relative_condition_ratio": None,
            "reserve_relative_lambda_available": None,
            "reserve_relative_lambda_pass": None,
            "reserve_relative_lambda_ratio": None,
            "reserve_relative_improvement_reasons": "",
            "reserve_information_v3_branch": None,
            "reserve_information_v3_condition_base_pass": None,
            "reserve_information_v3_lambda_base_pass": None,
            "reserve_information_v3_condition_strong_pass": None,
            "reserve_information_v3_lambda_strong_pass": None,
            "reserve_information_v3_refinement_translation_pass": None,
            "reserve_information_v3_refinement_rotation_pass": None,
            "reserve_information_v3_prospective_streak": None,
            "reserve_information_v3_cooldown_remaining_frames": 0,
            "reserve_information_v3_accepted": None,
            "reserve_information_v3_reason": None,
            "reserve_stability_gate_enabled": False,
            "reserve_stability_gate_mode": "off",
            "reserve_stability_would_accept": None,
            "reserve_stability_reasons": "",
            "reserve_stability_accepted": True,
            "reserve_stability_latency_ms": 0.0,
            "reserve_refinement_twist_rx": None,
            "reserve_refinement_twist_ry": None,
            "reserve_refinement_twist_rz": None,
            "reserve_refinement_twist_tx": None,
            "reserve_refinement_twist_ty": None,
            "reserve_refinement_twist_tz": None,
            "reserve_refinement_twist_norm": None,
            "reserve_refinement_rotation_deg": None,
            "reserve_refinement_translation_m": None,
            "reserve_candidate_init_twist_norm": None,
            "reserve_candidate_init_rotation_deg": None,
            "reserve_candidate_init_translation_m": None,
            "reserve_candidate_temporal_rotation_deg": None,
            "reserve_information_v4_candidate_init_translation_pass": None,
            "reserve_information_v4_candidate_init_rotation_pass": None,
            "reserve_information_v4_joint_rotation_pass": None,
            "reserve_information_v4_prospective_global_streak": None,
            "reserve_information_v4_global_cooldown_remaining_frames": 0,
            "reserve_information_v4_global_streak_pass": None,
            "reserve_information_v4_global_cooldown_pass": None,
            "reserve_information_v4_accepted": None,
            "reserve_information_v4_reason": None,
            "reserve_different_target_fitness_ratio": None,
            "reserve_different_target_ptp_rmse_ratio": None,
            "reserve_different_target_robust_cost_ratio": None,
            "reserve_common_target_cost_ratio": None,
            "reserve_common_target_rmse_ratio": None,
            "reserve_common_target_fitness_ratio": None,
            "reserve_common_target_corr_retention": None,
            "reserve_stability_rolling_translation_l1_m": None,
            "reserve_stability_prospective_streak": None,
            "reserve_stability_cooldown_remaining_frames": 0,
            "reserve_adoption_reason": None,
            "reserve_plan_pose_source": None,
            "reserve_rerun_init_source": None,
            "reserve_plan_normal_source": None,
            "reserve_rerun_normal_source": None,
            "reserve_corr_policy": None,
            "reserve_corr_status": "not_applicable",
            "reserve_corr_reason": None,
            "reserve_corr_source_count": 0,
            "reserve_corr_normal_valid_rate": None,
            "reserve_corr_distance_valid_rate": None,
            "reserve_corr_eligible": 0,
            "reserve_corr_p_mean": None,
            "reserve_corr_p_nonzero_rate": None,
            "reserve_corr_selected_p_mean": None,
            "reserve_corr_selected_p10": None,
            "reserve_corr_selected_p_min": None,
            "reserve_corr_adaptive_k": None,
            "reserve_corr_characteristic_length": None,
            "reserve_corr_latency_ms": 0.0,
            "reserve_corr_normal_source": None,
            "reserve_corr_distance_source": None,
            "reserve_hessian_rotation_trace": None,
            "reserve_hessian_translation_trace": None,
            "reserve_hessian_cross_frobenius": None,
            "reserve_hessian_rotation_lambda_min": None,
            "reserve_hessian_translation_lambda_min": None,
            "reserve_weak_rotation_energy": None,
            "reserve_weak_translation_energy": None,
            "reserve_weak_dominant_block": None,
            "reserve_latency_ms": 0.0,
            "reserve_backend": "disabled" if mode == "off" else reason,
        }

    def _trigger(
        self, result: dict[str, Any], observation: RuntimeReserveObservation | None = None,
    ) -> tuple[bool, list[str], str | None]:
        pose = np.asarray(result.get("transform", []), dtype=np.float64)
        if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
            return False, [], "primary_pose_non_finite"
        hessian = np.asarray(result.get("hessian", []), dtype=np.float64)
        if hessian.shape != (6, 6) or np.any(~np.isfinite(hessian)):
            return False, [], "hessian_unavailable"
        normalized = None
        if getattr(self, "spectral_coordinates", "world_raw") == "primary_camera_depth":
            if observation is None:
                return False, [], "missing_observation"
            try:
                normalized = normalized_hessian_diagnostics(hessian, pose, observation.source_points_camera)
            except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
                return False, [], "spectral_normalization_failed"
            result = dict(result, condition_number=normalized["condition_number"],
                          effective_rank=normalized["effective_rank"])
            eigenvalues = normalized["eigenvalues"]
            trace = float(np.sum(eigenvalues))
        else:
            symmetric = 0.5 * (hessian + hessian.T)
            try:
                eigenvalues = np.linalg.eigvalsh(symmetric)
            except np.linalg.LinAlgError:
                return False, [], "hessian_zero_or_invalid_spectrum"
            trace = float(np.trace(symmetric))
        if not np.all(np.isfinite(eigenvalues)) or trace <= 0.0:
            return False, [], "hessian_zero_or_invalid_spectrum"
        correspondences = result.get("num_correspondences")
        if correspondences is None or int(correspondences) <= 0:
            return False, [], "correspondences_unavailable"
        reasons = []
        limit = self.trigger.get("condition_number")
        if limit is not None and float(result.get("condition_number", float("inf"))) > float(limit):
            reasons.append("condition_number")
        limit = self.trigger.get("normalized_lambda_min")
        normalized_lambda = normalized["normalized_lambda_min"] if normalized is not None else float(eigenvalues[0]) / trace
        if limit is not None and normalized_lambda < float(limit):
            reasons.append("normalized_lambda_min")
        limit = self.trigger.get("iterations")
        if limit is not None and int(result.get("iterations", 0)) >= int(limit):
            reasons.append("iterations")
        limit = self.trigger.get("fitness")
        if limit is not None and float(result.get("fitness", 0.0)) < float(limit):
            reasons.append("fitness")
        rank_limit = self.trigger.get("effective_rank_below", 6)
        effective_rank = result.get("effective_rank")
        if effective_rank is None:
            threshold = max(float(np.max(eigenvalues)) * 1e-9, np.finfo(np.float64).eps)
            effective_rank = int(np.count_nonzero(eigenvalues > threshold))
        if rank_limit is not None and int(effective_rank) < int(rank_limit):
            reasons.append("effective_rank_below")
        if bool(self.trigger.get("degenerate", False)) and bool(result.get("degenerate", False)):
            reasons.append("degenerate")
        return bool(reasons), reasons, None

    @staticmethod
    def _tree_distances(tree: Any, points: np.ndarray) -> np.ndarray:
        try:
            distances, _ = tree.query(points, k=1, workers=-1)
        except TypeError:
            distances, _ = tree.query(points, k=1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        if distances.shape != (points.shape[0],) or np.any(distances < 0.0):
            raise ValueError("primary target tree returned invalid 1NN distances")
        return distances

    def _corr_scores(
        self,
        observation: RuntimeReserveObservation,
        visible: np.ndarray,
        pose: np.ndarray,
        hessian: np.ndarray,
        intrinsics: np.ndarray,
        metrics: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corr_start = time.perf_counter()
        metrics["reserve_corr_source_count"] = int(observation.source_points_camera.shape[0])
        if observation.source_normals_camera is None:
            source_normals, normal_valid, normal_confidence = organized_depth_source_normals(
                observation.depth, intrinsics, observation.source_pixel_ids
            )
            metrics["reserve_corr_normal_source"] = "organized_depth"
        else:
            source_normals = observation.source_normals_camera
            lengths = np.linalg.norm(source_normals, axis=1)
            normal_valid = np.isfinite(lengths) & (lengths > 0.0)
            normal_confidence = (
                observation.source_normal_confidence
                if observation.source_normal_confidence is not None
                else normal_valid.astype(np.float64)
            )
            metrics["reserve_corr_normal_source"] = "observation"
        metrics["reserve_corr_normal_valid_rate"] = float(np.mean(normal_valid)) if normal_valid.size else 0.0

        source_world = observation.source_points_camera @ pose[:3, :3].T + pose[:3, 3]
        if observation.primary_source_target_distances is None:
            primary_distances = self._tree_distances(observation.primary_target_tree, source_world)
            metrics["reserve_corr_distance_source"] = "primary_target_tree_1nn"
        else:
            primary_distances = observation.primary_source_target_distances
            metrics["reserve_corr_distance_source"] = "observation"
        distance_valid = np.isfinite(primary_distances)
        metrics["reserve_corr_distance_valid_rate"] = float(np.mean(distance_valid)) if distance_valid.size else 0.0

        eligible = visible[self.supported[visible]]
        base_ids = set(int(value) for value in observation.primary_target_global_ids)
        eligible = eligible[np.asarray(
            [int(self.global_ids[index]) not in base_ids for index in eligible], dtype=bool
        )]
        if eligible.size:
            _, unique_positions = np.unique(self.global_ids[eligible], return_index=True)
            eligible = eligible[np.sort(unique_positions)]
        metrics["reserve_corr_eligible"] = int(eligible.size)

        probability = correspondence_probability(
            self.points[eligible], observation.source_points_camera, pose,
            observation.depth, intrinsics, observation.source_pixel_ids,
            reserve_normals_world=self.normals[eligible], source_normals_camera=source_normals,
            source_normal_valid=normal_valid, source_normal_confidence=normal_confidence,
            primary_source_target_distances=primary_distances,
            **dict(self.config.get("correspondence", {}) or {}),
        )["probability"]
        scores, details = correspondence_aware_weak_scores(
            self.points, self.normals, pose, hessian, observation.source_points_camera,
            probability, visible_indices=eligible,
            min_dim=int(self.config.get("corr_weak_min_dim", 1)),
            max_dim=int(self.config.get("corr_weak_max_dim", min(3, max(1, self.weak_dim)))),
            alpha=0.0 if getattr(self, "corr_rank_control", "corr_weak") == "weak" else 1.0,
        )
        weak = details["weak"]
        metrics.update({
            "reserve_corr_status": "ok",
            "reserve_corr_reason": None,
            "reserve_corr_p_mean": float(np.mean(probability)) if probability.size else None,
            "reserve_corr_p_nonzero_rate": float(np.mean(probability > 0.0)) if probability.size else 0.0,
            "reserve_corr_adaptive_k": int(weak["k"]),
            "reserve_corr_characteristic_length": float(weak["characteristic_length"]),
            "reserve_corr_latency_ms": float((time.perf_counter() - corr_start) * 1000.0),
        })
        return scores, eligible, np.asarray(probability, dtype=np.float64)

    def plan(
        self,
        frame_id: int,
        result: dict[str, Any],
        intrinsics: np.ndarray,
        image_height: int,
        image_width: int,
        *,
        observation: RuntimeReserveObservation | None = None,
    ) -> ReservePlan:
        metrics = self.empty_metrics(self.mode, self.unavailable_reason)
        metrics["reserve_spectral_coordinates"] = getattr(self, "spectral_coordinates", "world_raw")
        metrics["reserve_stability_gate_enabled"] = bool(getattr(self, "stability_active", False))
        metrics["reserve_stability_gate_mode"] = str(getattr(self, "stability_mode", "off"))
        if getattr(self, "selection", None) == "runtime_corr_weak":
            metrics["reserve_corr_policy"] = "runtime_corr_weak"
            metrics["reserve_corr_status"] = "not_triggered"
        if not self.active or not self.available:
            return ReservePlan(metrics, np.empty(0, dtype=np.int64))
        start = time.perf_counter()
        triggered, reasons, unavailable_reason = self._trigger(result, observation)
        metrics["reserve_triggered"] = triggered
        metrics["reserve_trigger_reasons"] = "|".join(reasons)
        metrics["reserve_trigger_unavailable_reason"] = unavailable_reason
        hessian = np.asarray(result.get("hessian", []), dtype=np.float64)
        if metrics["reserve_spectral_coordinates"] == "primary_camera_depth":
            if unavailable_reason is None:
                spectral = normalized_hessian_diagnostics(
                    hessian, result["transform"], observation.source_points_camera
                )
                metrics.update({
                    "reserve_spectral_length": float(spectral["characteristic_length"]),
                    "reserve_primary_condition": float(spectral["condition_number"]),
                    "reserve_primary_lambda_min": float(spectral["eigenvalues"][0]),
                    "reserve_spectral_primary_condition": float(spectral["condition_number"]),
                    "reserve_spectral_primary_lambda": float(spectral["eigenvalues"][0]),
                })
        else:
            metrics["reserve_primary_condition"] = result.get("condition_number")
            if hessian.shape == (6, 6) and np.all(np.isfinite(hessian)):
                try:
                    eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
                    metrics["reserve_primary_lambda_min"] = float(eigenvalues[0])
                except np.linalg.LinAlgError:
                    pass
        if unavailable_reason is None:
            metrics.update(hessian_block_diagnostics(hessian, result.get("transform", [])))
        metrics["reserve_backend"] = self.index.backend
        metrics["reserve_plan_pose_source"] = "primary_result"
        metrics["reserve_plan_normal_source"] = self.plan_normal_source
        if not triggered:
            metrics["reserve_latency_ms"] = float((time.perf_counter() - start) * 1000.0)
            return ReservePlan(metrics, np.empty(0, dtype=np.int64))
        if (getattr(self, "candidate_target_mode", "append") == "rerun_primary"
                and getattr(self, "attempt_without_selection", False)):
            metrics["reserve_latency_ms"] = float((time.perf_counter() - start) * 1000.0)
            return ReservePlan(metrics, np.empty(0, dtype=np.int64))
        if self.selection == "runtime_corr_weak" and not isinstance(observation, RuntimeReserveObservation):
            metrics.update({
                "reserve_corr_status": "unavailable",
                "reserve_corr_reason": "missing_observation",
                "reserve_latency_ms": float((time.perf_counter() - start) * 1000.0),
            })
            return ReservePlan(metrics, np.empty(0, dtype=np.int64))
        query_config = LocalSubmapConfig(
            near=self.local_config.near,
            far=self.local_config.far,
            margin_px=self.local_config.margin_px,
            radius_margin=self.local_config.radius_margin,
            max_points=self.max_visible,
            sampling_policy="coverage",
            tile_rows=self.local_config.tile_rows,
            tile_cols=self.local_config.tile_cols,
            depth_bins=self.local_config.depth_bins,
            frustum_culling=str(self.config.get("frustum_culling", "radius")),
            block_size_m=float(self.config.get("block_size_m", self.local_config.block_size_m)),
        )
        pose = np.asarray(result["transform"], dtype=np.float64)
        query = self.index.query_local_submap(
            pose, intrinsics, image_height, image_width, query_config, scores=self.scores
        )
        visible = np.asarray(query["indices"], dtype=np.int64)
        metrics["reserve_visible"] = int(visible.size)
        eligible = visible
        corr_probability = None
        if self.selection == "runtime_weak":
            weight_mode = str(self.config.get("score_weight", "unit"))
            weights = None
            if weight_mode == "normalized_score":
                values = np.maximum(self.scores, 0.0)
                weights = values / max(float(np.mean(values)), np.finfo(float).eps)
            elif weight_mode != "unit":
                raise ValueError("observable_reserve.score_weight must be unit or normalized_score")
            selection_scores, _, _ = runtime_weak_scores(
                self.points, self.normals, pose, hessian, self.weak_dim, weights=weights
            )
        elif self.selection == "runtime_corr_weak":
            try:
                selection_scores, eligible, corr_probability = self._corr_scores(
                    observation, visible, pose, hessian, np.asarray(intrinsics, dtype=np.float64), metrics
                )
                if getattr(self, "corr_rank_control", "corr_weak") == "random":
                    selection_scores = np.random.default_rng(self.seed + int(frame_id)).random(self.points.shape[0])
            except Exception as exc:
                metrics.update({
                    "reserve_corr_status": "unavailable",
                    "reserve_corr_reason": f"{type(exc).__name__}: {exc}",
                    "reserve_corr_latency_ms": float((time.perf_counter() - start) * 1000.0),
                    "reserve_latency_ms": float((time.perf_counter() - start) * 1000.0),
                })
                return ReservePlan(metrics, np.empty(0, dtype=np.int64))
        elif self.selection == "saliency":
            selection_scores = self.scores.copy()
        else:
            selection_scores = np.random.default_rng(self.seed + int(frame_id)).random(self.points.shape[0])
        selected = select_diverse_reserve(
            eligible,
            selection_scores,
            self.points,
            pose,
            np.asarray(intrinsics, dtype=np.float64),
            image_height,
            image_width,
            min(self.budget, self.max_points),
            int((self.config.get("diversity", {}) or {}).get("tile_rows", self.config.get("tile_rows", self.local_config.tile_rows))),
            int((self.config.get("diversity", {}) or {}).get("tile_cols", self.config.get("tile_cols", self.local_config.tile_cols))),
            int((self.config.get("diversity", {}) or {}).get("depth_bins", self.config.get("depth_bins", self.local_config.depth_bins))),
            float((self.config.get("diversity", {}) or {}).get("voxel_size", self.config.get("voxel_size", 0.25))),
            (self.config.get("diversity", {}) or {}).get("tile_cap", self.config.get("tile_cap")),
            (self.config.get("diversity", {}) or {}).get("depth_cap", self.config.get("depth_cap")),
            (self.config.get("diversity", {}) or {}).get("voxel_cap", self.config.get("voxel_cap", 1)),
            self.global_ids,
        )
        metrics["reserve_selected"] = int(selected.size)
        metrics["reserve_selection_score"] = float(np.sum(selection_scores[selected])) if selected.size else 0.0
        selected_probability = None
        if corr_probability is not None:
            by_index = {int(index): float(value) for index, value in zip(eligible, corr_probability)}
            selected_probability = np.asarray([by_index[int(index)] for index in selected], dtype=np.float64)
            metrics.update({
                "reserve_corr_selected_p_mean": float(np.mean(selected_probability)) if selected_probability.size else None,
                "reserve_corr_selected_p10": float(np.percentile(selected_probability, 10)) if selected_probability.size else None,
                "reserve_corr_selected_p_min": float(np.min(selected_probability)) if selected_probability.size else None,
            })
        metrics["reserve_latency_ms"] = float((time.perf_counter() - start) * 1000.0)
        return ReservePlan(
            metrics=metrics,
            selected_indices=selected,
            selected_scores=np.asarray(selection_scores[selected], dtype=np.float64),
            selected_probabilities=selected_probability,
            selected_global_ids=np.asarray(self.global_ids[selected], dtype=np.int64),
            selection_policy=self.selection,
        )

    def evaluate_improvement(
        self, primary: dict[str, Any], candidate: dict[str, Any], context_metrics: dict[str, Any] | None = None,
        *, observation: RuntimeReserveObservation | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        diagnostics = {
            "reserve_relative_improvement_protocol": self.accept_only_if.get("protocol"),
            "reserve_relative_improvement_mode": str(self.accept_only_if.get("mode", "all")),
            "reserve_relative_condition_available": None,
            "reserve_relative_condition_pass": None,
            "reserve_relative_condition_ratio": None,
            "reserve_relative_lambda_available": None,
            "reserve_relative_lambda_pass": None,
            "reserve_relative_lambda_ratio": None,
            "reserve_relative_improvement_reasons": "",
            "reserve_spectral_coordinates": getattr(self, "spectral_coordinates", "world_raw"),
            "reserve_spectral_length": None,
            "reserve_spectral_primary_lambda": None,
            "reserve_spectral_candidate_lambda": None,
            "reserve_spectral_primary_condition": None,
            "reserve_spectral_candidate_condition": None,
        }
        if diagnostics["reserve_spectral_coordinates"] == "primary_camera_depth":
            if observation is None:
                diagnostics["reserve_relative_improvement_reasons"] = "missing_observation"
                return False, "spectral_observation_unavailable", diagnostics
            try:
                primary_spectral = normalized_hessian_diagnostics(
                    primary.get("hessian", []), primary.get("transform", []), observation.source_points_camera
                )
                candidate_spectral = normalized_hessian_diagnostics(
                    candidate.get("hessian", []), primary.get("transform", []),
                    observation.source_points_camera, primary_spectral["characteristic_length"],
                )
            except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
                diagnostics["reserve_relative_improvement_reasons"] = "spectral_normalization_failed"
                return False, "spectral_normalization_failed", diagnostics
            primary = dict(primary, eigenvalues=primary_spectral["eigenvalues"],
                           condition_number=primary_spectral["condition_number"])
            candidate = dict(candidate, eigenvalues=candidate_spectral["eigenvalues"],
                             condition_number=candidate_spectral["condition_number"])
            diagnostics["reserve_spectral_length"] = float(primary_spectral["characteristic_length"])
            for name, spectral in (("primary", primary_spectral), ("candidate", candidate_spectral)):
                diagnostics[f"reserve_spectral_{name}_lambda"] = float(spectral["eigenvalues"][0])
                diagnostics[f"reserve_spectral_{name}_condition"] = float(spectral["condition_number"])
        if not self.accept_only_if:
            return True, "relative_rules_not_configured", diagnostics
        try:
            primary_condition = float(primary.get("condition_number", float("inf")))
            candidate_condition = float(candidate.get("condition_number", float("inf")))
        except (TypeError, ValueError):
            primary_condition = candidate_condition = float("inf")
        try:
            primary_eigen = np.asarray(primary.get("eigenvalues", []), dtype=np.float64).reshape(-1)
            candidate_eigen = np.asarray(candidate.get("eigenvalues", []), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            primary_eigen = candidate_eigen = np.empty(0, dtype=np.float64)
        mode = str(self.accept_only_if.get("mode", "all"))
        legacy = "mode" not in self.accept_only_if and "lambda_ratio" not in self.accept_only_if
        configured: list[tuple[str, bool]] = []
        unavailable: list[str] = []
        rejected: list[str] = []
        ratio = self.accept_only_if.get("condition_ratio")
        if ratio is not None:
            available = bool(
                np.isfinite(primary_condition) and np.isfinite(candidate_condition) and primary_condition > 0.0
            )
            actual_ratio = candidate_condition / primary_condition if available else None
            passed = bool(available and actual_ratio <= float(ratio))
            diagnostics.update({
                "reserve_relative_condition_available": available,
                "reserve_relative_condition_pass": passed,
                "reserve_relative_condition_ratio": actual_ratio,
            })
            configured.append(("condition", passed))
            if not available:
                unavailable.append("condition_unavailable")
            elif not passed:
                rejected.append("condition_not_improved")
        increase = self.accept_only_if.get("lambda_increase")
        lambda_ratio = self.accept_only_if.get("lambda_ratio")
        if increase is not None or lambda_ratio is not None:
            available = bool(
                primary_eigen.size and candidate_eigen.size
                and np.isfinite(primary_eigen[0]) and np.isfinite(candidate_eigen[0])
            )
            actual_ratio = None
            if available and primary_eigen[0] > 0.0:
                actual_ratio = float(candidate_eigen[0] / primary_eigen[0])
            ratio_available = lambda_ratio is None or actual_ratio is not None
            passed = bool(
                available and ratio_available
                and (increase is None or candidate_eigen[0] >= primary_eigen[0] + float(increase))
                and (lambda_ratio is None or actual_ratio >= float(lambda_ratio))
            )
            diagnostics.update({
                "reserve_relative_lambda_available": bool(available and ratio_available),
                "reserve_relative_lambda_pass": passed,
                "reserve_relative_lambda_ratio": actual_ratio,
            })
            configured.append(("lambda", passed))
            if not available or not ratio_available:
                unavailable.append("lambda_unavailable")
            elif not passed:
                rejected.append("lambda_not_improved")
        quality = self.accept_only_if.get("quality")
        if quality:
            try:
                primary_fitness = float(primary.get("fitness", float("nan")))
                candidate_fitness = float(candidate.get("fitness", float("nan")))
            except (TypeError, ValueError):
                primary_fitness = candidate_fitness = float("nan")
            available = bool(np.isfinite(primary_fitness) and np.isfinite(candidate_fitness))
            passed = bool(available and candidate_fitness >= primary_fitness)
            configured.append(("quality", passed))
            if not available:
                unavailable.append("quality_unavailable")
            elif not passed:
                rejected.append("quality_not_preserved")
        diagnostics["reserve_relative_improvement_reasons"] = "|".join(unavailable + rejected)
        if self.accept_only_if.get("protocol") in (
            "branch_aware_information_v3", "branch_aware_information_v4",
        ):
            return self.evaluate_branch_adoption(diagnostics, context_metrics or {})
        if legacy and unavailable:
            accepted = self.relative_unavailable_policy == "tracking_only"
            reason = "relative_metrics_unavailable_tracking_only" if accepted else "relative_metrics_unavailable"
            return accepted, reason, diagnostics
        if legacy and rejected:
            legacy_reasons = {
                "condition_not_improved": "condition_ratio_not_improved",
                "lambda_not_improved": "lambda_increase_not_met",
                "quality_not_preserved": "quality_not_preserved",
            }
            return False, legacy_reasons[rejected[0]], diagnostics
        accepted = all(passed for _, passed in configured) if mode == "all" else any(passed for _, passed in configured)
        if accepted:
            if legacy:
                return True, "relative_improvement_accepted", diagnostics
            passing = [name for name, passed in configured if passed]
            if mode == "any" and len(passing) > 1:
                reason = "relative_improvement_any_accepted"
            elif passing:
                reason = f"relative_improvement_{passing[0]}"
            else:
                reason = "relative_improvement_accepted"
            return True, reason, diagnostics
        reason_parts = [f"relative_improvement_{item}" for item in unavailable + rejected]
        return False, "|".join(reason_parts) or "relative_improvement_not_accepted", diagnostics

    def evaluate_branch_adoption(
        self, improvement_diagnostics: dict[str, Any], reserve_metrics: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        """Pure branch-aware v3/v4 decision; commit_branch_adoption owns all state changes."""
        condition_pass = improvement_diagnostics.get("reserve_relative_condition_pass") is True
        lambda_pass = improvement_diagnostics.get("reserve_relative_lambda_pass") is True
        condition_available = improvement_diagnostics.get("reserve_relative_condition_available") is True
        lambda_available = improvement_diagnostics.get("reserve_relative_lambda_available") is True
        if condition_pass and lambda_pass:
            branch = "both"
        elif condition_pass:
            branch = "condition_only"
        elif lambda_pass:
            branch = "lambda_only"
        else:
            branch = "neither"
        diagnostics = dict(improvement_diagnostics)
        diagnostics.update({
            "reserve_information_v3_branch": branch,
            "reserve_information_v3_condition_base_pass": condition_pass,
            "reserve_information_v3_lambda_base_pass": lambda_pass,
            "reserve_information_v3_condition_strong_pass": False,
            "reserve_information_v3_lambda_strong_pass": False,
            "reserve_information_v3_refinement_translation_pass": False,
            "reserve_information_v3_refinement_rotation_pass": False,
            "reserve_information_v3_prospective_streak": 0,
            "reserve_information_v3_cooldown_remaining_frames": 0,
            "reserve_information_v3_accepted": False,
        })
        protocol = self.accept_only_if.get("protocol")
        is_v4 = protocol == "branch_aware_information_v4"
        if is_v4:
            candidate_init_translation = reserve_metrics.get("reserve_candidate_init_translation_m")
            candidate_init_rotation = reserve_metrics.get("reserve_candidate_init_rotation_deg")
            refinement_rotation = reserve_metrics.get("reserve_refinement_rotation_deg")
            temporal_rotation = reserve_metrics.get("reserve_candidate_temporal_rotation_deg")
            candidate_init_translation_pass = bool(
                candidate_init_translation is not None and np.isfinite(float(candidate_init_translation))
                and float(candidate_init_translation) <= float(self.accept_only_if["max_candidate_init_translation_m"])
            )
            candidate_init_rotation_pass = bool(
                candidate_init_rotation is not None and np.isfinite(float(candidate_init_rotation))
                and float(candidate_init_rotation) <= float(self.accept_only_if["max_candidate_init_rotation_deg"])
            )
            refinement_rotation_available = (
                refinement_rotation is not None and np.isfinite(float(refinement_rotation))
            )
            temporal_rotation_available = temporal_rotation is not None and np.isfinite(float(temporal_rotation))
            joint_metrics_available = refinement_rotation_available and temporal_rotation_available
            joint_rotation_pass = bool(
                joint_metrics_available
                and not (
                    float(refinement_rotation) > float(self.accept_only_if["joint_refinement_rotation_deg"])
                    and float(temporal_rotation) > float(self.accept_only_if["joint_temporal_rotation_deg"])
                )
            )
            if not joint_metrics_available and not bool(self.accept_only_if.get("require_joint_metrics", True)):
                joint_rotation_pass = True
            frame = int(reserve_metrics.get("frame_id", -1))
            prospective_global_streak = self.global_consecutive_adoptions + 1
            global_cooldown_remaining = (
                max(0, self.global_cooldown_until_frame - frame + 1)
                if self.global_cooldown_until_frame is not None and frame >= 0 else 0
            )
            global_streak_pass = prospective_global_streak <= int(
                self.accept_only_if["max_global_consecutive_adoptions"]
            )
            global_cooldown_pass = global_cooldown_remaining == 0
            diagnostics.update({
                "reserve_information_v4_candidate_init_translation_pass": candidate_init_translation_pass,
                "reserve_information_v4_candidate_init_rotation_pass": candidate_init_rotation_pass,
                "reserve_information_v4_joint_rotation_pass": joint_rotation_pass,
                "reserve_information_v4_prospective_global_streak": prospective_global_streak,
                "reserve_information_v4_global_cooldown_remaining_frames": global_cooldown_remaining,
                "reserve_information_v4_global_streak_pass": global_streak_pass,
                "reserve_information_v4_global_cooldown_pass": global_cooldown_pass,
                "reserve_information_v4_accepted": False,
            })
        if branch == "neither":
            reason = (
                f"information_{'v4' if is_v4 else 'v3'}_base_metrics_unavailable"
                if not condition_available or not lambda_available
                else f"information_{'v4' if is_v4 else 'v3'}_neither"
            )
            diagnostics["reserve_information_v3_reason"] = reason
            if is_v4:
                diagnostics["reserve_information_v4_reason"] = reason
            diagnostics["reserve_relative_improvement_reasons"] = reason
            return False, reason, diagnostics

        branch_policy = getattr(self, "branch_policy", self.accept_only_if.get("branch_policy", {}) or {})
        policy = branch_policy[branch]
        condition_ratio = improvement_diagnostics.get("reserve_relative_condition_ratio")
        lambda_ratio = improvement_diagnostics.get("reserve_relative_lambda_ratio")
        translation = reserve_metrics.get("reserve_refinement_translation_m")
        rotation = reserve_metrics.get("reserve_refinement_rotation_deg")
        condition_strong = bool(
            condition_ratio is not None and np.isfinite(float(condition_ratio))
            and float(condition_ratio) <= float(policy["condition_ratio"])
        )
        lambda_strong = bool(
            lambda_ratio is not None and np.isfinite(float(lambda_ratio))
            and float(lambda_ratio) >= float(policy["lambda_ratio"])
        )
        translation_pass = bool(
            translation is not None and np.isfinite(float(translation))
            and float(translation) <= float(policy["max_refinement_translation_m"])
        )
        rotation_pass = bool(
            rotation is not None and np.isfinite(float(rotation))
            and float(rotation) <= float(policy["max_refinement_rotation_deg"])
        )
        same_branch = getattr(self, "last_adoption_branch", None) == branch
        prospective_streak = getattr(self, "branch_consecutive_adoptions", 0) + 1 if same_branch else 1
        frame = int(reserve_metrics.get("frame_id", -1))
        cooldown_remaining = (
            max(0, self.branch_cooldown_until_frame - frame + 1)
            if getattr(self, "branch_cooldown_until_frame", None) is not None and frame >= 0 else 0
        )
        diagnostics.update({
            "reserve_information_v3_condition_strong_pass": condition_strong,
            "reserve_information_v3_lambda_strong_pass": lambda_strong,
            "reserve_information_v3_refinement_translation_pass": translation_pass,
            "reserve_information_v3_refinement_rotation_pass": rotation_pass,
            "reserve_information_v3_prospective_streak": prospective_streak,
            "reserve_information_v3_cooldown_remaining_frames": cooldown_remaining,
        })
        if is_v4:
            diagnostics.update({
                "reserve_information_v4_candidate_init_translation_pass": candidate_init_translation_pass,
                "reserve_information_v4_candidate_init_rotation_pass": candidate_init_rotation_pass,
                "reserve_information_v4_joint_rotation_pass": joint_rotation_pass,
                "reserve_information_v4_prospective_global_streak": prospective_global_streak,
                "reserve_information_v4_global_cooldown_remaining_frames": global_cooldown_remaining,
                "reserve_information_v4_global_streak_pass": global_streak_pass,
                "reserve_information_v4_global_cooldown_pass": global_cooldown_pass,
            })
        reasons = []
        if not condition_strong:
            reasons.append(f"information_{'v4' if is_v4 else 'v3'}_condition_threshold")
        if not lambda_strong:
            reasons.append(f"information_{'v4' if is_v4 else 'v3'}_lambda_threshold")
        if not translation_pass:
            reasons.append(f"information_{'v4' if is_v4 else 'v3'}_refinement_translation")
        if not is_v4 and not rotation_pass:
            reasons.append("information_v3_refinement_rotation")
        if prospective_streak > int(policy["max_consecutive_adoptions"]):
            reasons.append(f"information_{'v4' if is_v4 else 'v3'}_max_consecutive")
        if cooldown_remaining > 0:
            reasons.append(f"information_{'v4' if is_v4 else 'v3'}_cooldown")
        if is_v4:
            if not candidate_init_translation_pass:
                reasons.append("information_v4_candidate_init_translation")
            if not candidate_init_rotation_pass:
                reasons.append("information_v4_candidate_init_rotation")
            if not joint_rotation_pass:
                reason = (
                    "information_v4_joint_metrics_unavailable"
                    if not joint_metrics_available else "information_v4_joint_rotation_spike"
                )
                reasons.append(reason)
            if not global_streak_pass:
                reasons.append("information_v4_global_max_consecutive")
            if not global_cooldown_pass:
                reasons.append("information_v4_global_cooldown")
        accepted = not reasons
        reason = f"information_{'v4' if is_v4 else 'v3'}_{branch}" if accepted else "|".join(reasons)
        diagnostics["reserve_information_v3_accepted"] = accepted
        diagnostics["reserve_information_v3_reason"] = reason
        if is_v4:
            diagnostics["reserve_information_v4_accepted"] = accepted
            diagnostics["reserve_information_v4_reason"] = reason
        diagnostics["reserve_relative_improvement_reasons"] = "" if accepted else reason
        return accepted, reason, diagnostics

    def commit_branch_adoption(self, frame: int, used: bool, branch: str | None) -> None:
        """Commit only active-production outcomes; rejected/non-adoption frames reset the streak."""
        protocol = self.accept_only_if.get("protocol")
        if protocol not in ("branch_aware_information_v3", "branch_aware_information_v4"):
            return
        frame = int(frame)
        if not used:
            self.branch_consecutive_adoptions = 0
            if protocol == "branch_aware_information_v4":
                self.global_consecutive_adoptions = 0
            return
        branch_policy = getattr(self, "branch_policy", self.accept_only_if.get("branch_policy", {}) or {})
        if branch not in branch_policy:
            raise ValueError("committed v3 branch must be both, condition_only, or lambda_only")
        self.branch_consecutive_adoptions = (
            self.branch_consecutive_adoptions + 1 if self.last_adoption_branch == branch else 1
        )
        self.last_adoption_branch = branch
        policy = branch_policy[branch]
        if self.branch_consecutive_adoptions >= int(policy["max_consecutive_adoptions"]):
            self.branch_cooldown_until_frame = frame + int(policy["cooldown_frames"])
        if protocol == "branch_aware_information_v4":
            self.global_consecutive_adoptions += 1
            if self.global_consecutive_adoptions >= int(self.accept_only_if["max_global_consecutive_adoptions"]):
                self.global_cooldown_until_frame = frame + int(self.accept_only_if["global_cooldown_frames"])

    def improvement_accepts(self, primary: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
        accepted, reason, _ = self.evaluate_improvement(primary, candidate)
        return accepted, reason

    @staticmethod
    def _finite_ratio(candidate: Any, primary: Any) -> float | None:
        try:
            numerator = float(candidate)
            denominator = float(primary)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0.0:
            return None
        if denominator == 0.0:
            return 1.0 if numerator == 0.0 else float("inf")
        return numerator / denominator

    def prospective_stability(
        self, metrics: dict[str, Any], candidate_pose: np.ndarray, primary_pose: np.ndarray,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Pure candidate decision; state changes only in commit_stability."""
        start = time.perf_counter()
        refinement = relative_se3_log(primary_pose, candidate_pose)
        frame = int(metrics.get("frame_id", -1))
        retained = [(old_frame, twist) for old_frame, twist in self.adopted_twists
                    if frame < 0 or old_frame >= frame - self.rolling_window_frames + 1]
        rolling_translation = float(sum(np.linalg.norm(twist[3:], ord=1) for _, twist in retained)
                                    + np.linalg.norm(refinement["twist"][3:], ord=1))
        prospective_streak = self.consecutive_adoptions + 1
        cooldown_remaining = 0
        if self.consecutive_adoptions == 0 and self.last_adoption_frame is not None and frame >= 0:
            cooldown_remaining = max(0, self.last_adoption_frame + self.cooldown_frames + 1 - frame)
        reasons = []
        limits = self.stability_config
        if limits.get("max_refinement_translation_m") is not None and refinement["translation_norm_m"] > float(limits["max_refinement_translation_m"]):
            reasons.append("stability_gate_refinement_translation")
        if limits.get("max_refinement_rotation_deg") is not None and refinement["rotation_norm_deg"] > float(limits["max_refinement_rotation_deg"]):
            reasons.append("stability_gate_refinement_rotation")
        if limits.get("max_rolling_translation_l1_m") is not None and rolling_translation > float(limits["max_rolling_translation_l1_m"]):
            reasons.append("stability_gate_rolling_translation")
        if self.max_consecutive_adoptions and prospective_streak > self.max_consecutive_adoptions:
            reasons.append("stability_gate_consecutive_adoptions")
        if cooldown_remaining > 0:
            reasons.append("stability_gate_cooldown")
        common = limits.get("common_target", {}) or {}
        for metric_name, config_name, reason, comparison in (
            ("reserve_common_target_cost_ratio", "max_cost_ratio", "stability_gate_common_target_cost", "max"),
            ("reserve_common_target_rmse_ratio", "max_rmse_ratio", "stability_gate_common_target_rmse", "max"),
            ("reserve_common_target_corr_retention", "min_corr_retention", "stability_gate_common_target_corr_retention", "min"),
        ):
            if common.get(config_name) is None:
                continue
            value = metrics.get(metric_name)
            failed = value is None or not np.isfinite(float(value))
            if not failed:
                failed = float(value) > float(common[config_name]) if comparison == "max" else float(value) < float(common[config_name])
            if failed:
                reasons.append(reason)
        diagnostics = {
            "reserve_stability_rolling_translation_l1_m": rolling_translation,
            "reserve_stability_prospective_streak": prospective_streak,
            "reserve_stability_cooldown_remaining_frames": cooldown_remaining,
            "reserve_stability_latency_ms": float((time.perf_counter() - start) * 1000.0),
        }
        return not reasons, "accepted" if not reasons else "|".join(reasons), diagnostics

    def commit_stability(self, frame: int, used: bool, twist: np.ndarray | None) -> None:
        """Commit only the final active-production adoption outcome."""
        frame = int(frame)
        while self.adopted_twists and self.adopted_twists[0][0] < frame - self.rolling_window_frames + 1:
            self.adopted_twists.popleft()
        if not used:
            self.consecutive_adoptions = 0
            return
        value = np.asarray(twist, dtype=np.float64)
        if value.shape != (6,) or np.any(~np.isfinite(value)):
            raise ValueError("committed stability twist must be finite with shape [6]")
        self.adopted_twists.append((frame, value.copy()))
        self.consecutive_adoptions += 1
        self.last_adoption_frame = frame

    def add_stability_attempt(
        self, frame: int, primary_pose: np.ndarray, candidate_pose: np.ndarray,
        init_pose: np.ndarray, metrics: dict[str, Any], actual: bool,
    ) -> None:
        if not self.stability_active:
            return
        self.stability_attempts.append({
            "frame": int(frame), "primary_pose": np.asarray(primary_pose, dtype=np.float64).copy(),
            "candidate_pose": np.asarray(candidate_pose, dtype=np.float64).copy(),
            "init_pose": np.asarray(init_pose, dtype=np.float64).copy(), "metrics": dict(metrics),
            "actual": bool(actual),
        })

    def save_stability_attempts(self, output_dir: str | Path) -> Path | None:
        if not self.stability_active:
            return None
        output = Path(output_dir) / "reserve_stability_attempts.npz"
        rows = self.stability_attempts

        def metric(name: str, default: Any = np.nan) -> np.ndarray:
            return np.asarray([row["metrics"].get(name, default) for row in rows])

        candidate_init = [relative_se3_log(row["init_pose"], row["candidate_pose"])["twist"] for row in rows]
        canonical_config = json.dumps(self.stability_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        np.savez(
            output, schema_version=np.asarray(1, dtype=np.int64),
            metric_definition_version=np.asarray("reserve_stability_v1", dtype="U32"),
            mode=np.asarray(self.stability_mode, dtype="U16"),
            stability_config_sha256=np.asarray(hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(), dtype="U64"),
            coordinate_system=np.asarray("relative right transform log [rotation_rad,translation_m]", dtype="U64"),
            common_target_semantics=np.asarray("same primary target; pose-dependent 1NN", dtype="U64"),
            frame=np.asarray([row["frame"] for row in rows], dtype=np.int64),
            primary_pose=np.stack([row["primary_pose"] for row in rows]) if rows else np.empty((0, 4, 4)),
            candidate_pose=np.stack([row["candidate_pose"] for row in rows]) if rows else np.empty((0, 4, 4)),
            init_pose=np.stack([row["init_pose"] for row in rows]) if rows else np.empty((0, 4, 4)),
            refinement_twist=np.asarray([[row["metrics"].get(f"reserve_refinement_twist_{axis}", np.nan)
                                          for axis in ("rx", "ry", "rz", "tx", "ty", "tz")] for row in rows], dtype=np.float64).reshape(-1, 6),
            candidate_init_twist=np.asarray(candidate_init, dtype=np.float64).reshape(-1, 6),
            different_target_fitness_ratio=metric("reserve_different_target_fitness_ratio").astype(np.float64),
            different_target_ptp_rmse_ratio=metric("reserve_different_target_ptp_rmse_ratio").astype(np.float64),
            different_target_robust_cost_ratio=metric("reserve_different_target_robust_cost_ratio").astype(np.float64),
            common_target_cost_ratio=metric("reserve_common_target_cost_ratio").astype(np.float64),
            common_target_rmse_ratio=metric("reserve_common_target_rmse_ratio").astype(np.float64),
            common_target_fitness_ratio=metric("reserve_common_target_fitness_ratio").astype(np.float64),
            common_target_corr_retention=metric("reserve_common_target_corr_retention").astype(np.float64),
            rolling_translation_l1_m=metric("reserve_stability_rolling_translation_l1_m").astype(np.float64),
            prospective_streak=metric("reserve_stability_prospective_streak", -1).astype(np.int64),
            gate_latency_ms=metric("reserve_stability_latency_ms", 0.0).astype(np.float64),
            reasons=np.asarray([str(row["metrics"].get("reserve_stability_reasons", "")) for row in rows], dtype="U256"),
            would_accept=metric("reserve_stability_would_accept", False).astype(bool),
            actual=np.asarray([row["actual"] for row in rows], dtype=bool),
        )
        return output

    def add_record(self, metrics: dict[str, Any]) -> None:
        if self.active:
            self.records.append(dict(metrics))

    def summary(self) -> dict[str, Any]:
        records = self.records
        triggered = [bool(row["reserve_triggered"]) for row in records]
        attempted = [bool(row.get("reserve_attempted", False)) for row in records]
        used = [bool(row["reserve_used"]) for row in records]
        primary_accepted = [row.get("reserve_primary_accepted") for row in records]
        tracking_frames = len(records)
        attempt_count = sum(attempted)
        adoption_count = sum(used)
        rescue_count = sum(u and a is False for u, a in zip(used, primary_accepted))
        primary_rejected_count = sum(a is False for a in primary_accepted)

        def finite_pairs(mask: list[bool], primary_key: str, final_key: str) -> list[tuple[float, float]]:
            pairs = []
            for include, row in zip(mask, records):
                if not include or row.get(primary_key) is None or row.get(final_key) is None:
                    continue
                primary = float(row[primary_key])
                final = float(row[final_key])
                if np.isfinite(primary) and np.isfinite(final):
                    pairs.append((primary, final))
            return pairs

        def conditional_metrics(scope: str, mask: list[bool]) -> dict[str, Any]:
            condition_pairs = finite_pairs(mask, "reserve_primary_condition", "reserve_augmented_condition")
            lambda_pairs = finite_pairs(mask, "reserve_primary_lambda_min", "reserve_augmented_lambda_min")
            condition_reductions = [primary - final for primary, final in condition_pairs]
            condition_percent = [100.0 * (primary - final) / primary for primary, final in condition_pairs if primary != 0.0]
            lambda_increases = [final - primary for primary, final in lambda_pairs]
            lambda_percent = [100.0 * (final - primary) / abs(primary) for primary, final in lambda_pairs if primary != 0.0]
            return {
                f"reserve_{scope}_primary_condition_mean": float(np.mean([p for p, _ in condition_pairs])) if condition_pairs else None,
                f"reserve_{scope}_final_condition_mean": float(np.mean([f for _, f in condition_pairs])) if condition_pairs else None,
                f"reserve_{scope}_condition_reduction_abs_mean": float(np.mean(condition_reductions)) if condition_reductions else None,
                f"reserve_{scope}_condition_reduction_percent_mean": float(np.mean(condition_percent)) if condition_percent else None,
                f"reserve_{scope}_primary_lambda_min_mean": float(np.mean([p for p, _ in lambda_pairs])) if lambda_pairs else None,
                f"reserve_{scope}_final_lambda_min_mean": float(np.mean([f for _, f in lambda_pairs])) if lambda_pairs else None,
                f"reserve_{scope}_lambda_min_increase_abs_mean": float(np.mean(lambda_increases)) if lambda_increases else None,
                f"reserve_{scope}_lambda_min_increase_percent_mean": float(np.mean(lambda_percent)) if lambda_percent else None,
            }

        attempted_metrics = conditional_metrics("attempted", attempted)
        used_metrics = conditional_metrics("used", used)
        condition_improvements = [
            primary - final for primary, final in finite_pairs(attempted, "reserve_primary_condition", "reserve_augmented_condition")
        ]
        lambda_improvements = [
            final - primary for primary, final in finite_pairs(attempted, "reserve_primary_lambda_min", "reserve_augmented_lambda_min")
        ]
        selected = [float(row["reserve_selected"]) for row, include in zip(records, attempted) if include]
        triggered_selected = [float(row["reserve_selected"]) for row, include in zip(records, triggered) if include]
        correspondences = [
            float(row["reserve_correspondence"])
            for row, include in zip(records, attempted)
            if include and row.get("reserve_correspondence") is not None
        ]
        corr_rows = [row for row in records if row.get("reserve_corr_policy") == "runtime_corr_weak"]
        stability_rows = [row for row in records if row.get("reserve_stability_would_accept") is not None]
        relative_rows = [row for row, include in zip(records, attempted) if include]
        branch_rows = [row for row in relative_rows if row.get("reserve_information_v3_branch") is not None]
        v4_rows = [row for row in relative_rows if row.get("reserve_information_v4_reason") is not None]

        def corr_mean(key: str) -> float | None:
            values = [
                float(row[key]) for row in corr_rows
                if row.get(key) is not None and np.isfinite(float(row[key]))
            ]
            return float(np.mean(values)) if values else None

        result = {
            "observable_reserve_enabled": self.active,
            "observable_reserve_mode": self.mode,
            "observable_reserve_available": self.available,
            "observable_reserve_unavailable_reason": self.unavailable_reason,
            "observable_reserve_geometry_path": self.geometry_path,
            "reserve_relative_improvement_protocol": getattr(self, "accept_only_if", {}).get("protocol"),
            "reserve_relative_improvement_mode": str(getattr(self, "accept_only_if", {}).get("mode", "all")),
            "observable_reserve_trigger_rate": float(np.mean(triggered)) if triggered else 0.0,
            "observable_reserve_use_rate": float(np.mean(used)) if used else 0.0,
            "observable_reserve_recovery_rate": (
                float(
                    sum(u and a is False for u, a in zip(used, primary_accepted))
                    / max(sum(a is False for a in primary_accepted), 1)
                )
                if records else 0.0
            ),
            "observable_reserve_condition_improvement_mean": (
                float(np.mean(condition_improvements)) if condition_improvements else None
            ),
            "observable_reserve_lambda_improvement_mean": (
                float(np.mean(lambda_improvements)) if lambda_improvements else None
            ),
            "observable_reserve_extra_latency_mean_ms": (
                float(np.mean([float(row["reserve_latency_ms"]) for row in records])) if records else 0.0
            ),
            "observable_reserve_triggered_latency_mean_ms": (
                float(np.mean([
                    float(row["reserve_latency_ms"]) for row in records if bool(row["reserve_triggered"])
                ])) if any(triggered) else 0.0
            ),
            "num_reserve_attempt_frames": attempt_count,
            "num_reserve_adopted_frames": adoption_count,
            "num_reserve_rescue_frames": rescue_count,
            "reserve_attempt_rate": attempt_count / tracking_frames if tracking_frames else 0.0,
            "reserve_adoption_rate": adoption_count / tracking_frames if tracking_frames else 0.0,
            "reserve_conditional_adoption_rate": adoption_count / attempt_count if attempt_count else 0.0,
            "reserve_rescue_rate": rescue_count / primary_rejected_count if primary_rejected_count else 0.0,
            "reserve_selected_attempted_mean": float(np.mean(selected)) if selected else None,
            "reserve_selected_triggered_mean": float(np.mean(triggered_selected)) if triggered_selected else None,
            "reserve_correspondence_attempted_mean": float(np.mean(correspondences)) if correspondences else None,
            "reserve_selected_mean": float(np.mean(selected)) if selected else None,
            "reserve_correspondence_mean": float(np.mean(correspondences)) if correspondences else None,
            "reserve_latency_all_tracking_frames_mean_ms": (
                float(np.mean([float(row["reserve_latency_ms"]) for row in records])) if records else 0.0
            ),
            "reserve_latency_triggered_mean_ms": (
                float(np.mean([float(row["reserve_latency_ms"]) for row in records if bool(row["reserve_triggered"])]))
                if any(triggered) else 0.0
            ),
            "reserve_corr_frame_count": len(corr_rows),
            "reserve_corr_ok_frame_count": sum(row.get("reserve_corr_status") == "ok" for row in corr_rows),
            "reserve_corr_unavailable_frame_count": sum(
                row.get("reserve_corr_status") == "unavailable" for row in corr_rows
            ),
            "reserve_corr_source_count_mean": corr_mean("reserve_corr_source_count"),
            "reserve_corr_normal_valid_rate_mean": corr_mean("reserve_corr_normal_valid_rate"),
            "reserve_corr_distance_valid_rate_mean": corr_mean("reserve_corr_distance_valid_rate"),
            "reserve_corr_eligible_mean": corr_mean("reserve_corr_eligible"),
            "reserve_corr_p_mean": corr_mean("reserve_corr_p_mean"),
            "reserve_corr_p_nonzero_rate_mean": corr_mean("reserve_corr_p_nonzero_rate"),
            "reserve_corr_selected_p_mean": corr_mean("reserve_corr_selected_p_mean"),
            "reserve_corr_selected_p10_mean": corr_mean("reserve_corr_selected_p10"),
            "reserve_corr_selected_p_min_mean": corr_mean("reserve_corr_selected_p_min"),
            "reserve_corr_adaptive_k_mean": corr_mean("reserve_corr_adaptive_k"),
            "reserve_corr_characteristic_length_mean": corr_mean("reserve_corr_characteristic_length"),
            "reserve_corr_latency_mean_ms": corr_mean("reserve_corr_latency_ms"),
            "reserve_relative_condition_pass_count": sum(
                row.get("reserve_relative_condition_pass") is True for row in relative_rows
            ),
            "reserve_relative_lambda_pass_count": sum(
                row.get("reserve_relative_lambda_pass") is True for row in relative_rows
            ),
            "reserve_relative_any_only_count": sum(
                (row.get("reserve_relative_condition_pass") is True)
                != (row.get("reserve_relative_lambda_pass") is True)
                for row in relative_rows
            ),
            "reserve_relative_both_count": sum(
                row.get("reserve_relative_condition_pass") is True
                and row.get("reserve_relative_lambda_pass") is True
                for row in relative_rows
            ),
            "reserve_relative_condition_unavailable_count": sum(
                row.get("reserve_relative_condition_available") is False for row in relative_rows
            ),
            "reserve_relative_lambda_unavailable_count": sum(
                row.get("reserve_relative_lambda_available") is False for row in relative_rows
            ),
            **attempted_metrics,
            **used_metrics,
        }
        finite = lambda key: [float(row[key]) for row in stability_rows
                              if row.get(key) is not None and np.isfinite(float(row[key]))]
        refinement = finite("reserve_refinement_translation_m")
        rotation = finite("reserve_refinement_rotation_deg")
        reason_counts = Counter(
            reason for row in stability_rows for reason in str(row.get("reserve_stability_reasons", "")).split("|") if reason
        )
        result.update({
            "reserve_stability_gate_enabled": bool(getattr(self, "stability_active", False)),
            "reserve_stability_gate_mode": str(getattr(self, "stability_mode", "off")),
            "reserve_stability_attempted_count": len(stability_rows),
            "reserve_stability_would_accept_count": sum(bool(row["reserve_stability_would_accept"]) for row in stability_rows),
            "reserve_stability_rejected_reason_counts": dict(sorted(reason_counts.items())),
            "reserve_stability_actual_gate_adoption_count": sum(bool(row.get("reserve_used")) for row in stability_rows),
            "reserve_refinement_translation_mean_m": float(np.mean(refinement)) if refinement else None,
            "reserve_refinement_translation_p95_m": float(np.percentile(refinement, 95)) if refinement else None,
            "reserve_refinement_translation_max_m": float(np.max(refinement)) if refinement else None,
            "reserve_refinement_rotation_mean_deg": float(np.mean(rotation)) if rotation else None,
            "reserve_refinement_rotation_p95_deg": float(np.percentile(rotation, 95)) if rotation else None,
            "reserve_refinement_rotation_max_deg": float(np.max(rotation)) if rotation else None,
            "reserve_stability_rolling_translation_l1_mean_m": float(np.mean(finite("reserve_stability_rolling_translation_l1_m"))) if finite("reserve_stability_rolling_translation_l1_m") else None,
            "reserve_stability_prospective_streak_max": int(max(finite("reserve_stability_prospective_streak"), default=0)),
            "reserve_common_target_cost_ratio_mean": float(np.mean(finite("reserve_common_target_cost_ratio"))) if finite("reserve_common_target_cost_ratio") else None,
            "reserve_common_target_rmse_ratio_mean": float(np.mean(finite("reserve_common_target_rmse_ratio"))) if finite("reserve_common_target_rmse_ratio") else None,
            "reserve_common_target_corr_retention_mean": float(np.mean(finite("reserve_common_target_corr_retention"))) if finite("reserve_common_target_corr_retention") else None,
            "reserve_stability_gate_latency_mean_ms": float(np.mean(finite("reserve_stability_latency_ms"))) if stability_rows else 0.0,
        })
        branch_reason_counts = Counter(
            reason for row in branch_rows
            for reason in str(row.get("reserve_information_v3_reason") or "").split("|") if reason
        )
        for branch in ("both", "condition_only", "lambda_only", "neither"):
            rows = [row for row in branch_rows if row.get("reserve_information_v3_branch") == branch]
            result[f"reserve_information_v3_{branch}_attempt_count"] = len(rows)
            result[f"reserve_information_v3_{branch}_adopt_count"] = sum(bool(row.get("reserve_used")) for row in rows)
            result[f"reserve_information_v3_{branch}_reject_count"] = sum(not bool(row.get("reserve_used")) for row in rows)
        result["reserve_information_v3_reason_counts"] = dict(sorted(branch_reason_counts.items()))
        v4_reason_counts = Counter(
            reason for row in v4_rows
            for reason in str(row.get("reserve_information_v4_reason") or "").split("|") if reason
        )
        result.update({
            "reserve_information_v4_attempt_count": len(v4_rows),
            "reserve_information_v4_adopt_count": sum(bool(row.get("reserve_used")) for row in v4_rows),
            "reserve_information_v4_reject_count": sum(not bool(row.get("reserve_used")) for row in v4_rows),
            "reserve_information_v4_candidate_init_translation_pass_count": sum(
                row.get("reserve_information_v4_candidate_init_translation_pass") is True for row in v4_rows
            ),
            "reserve_information_v4_candidate_init_rotation_pass_count": sum(
                row.get("reserve_information_v4_candidate_init_rotation_pass") is True for row in v4_rows
            ),
            "reserve_information_v4_joint_rotation_pass_count": sum(
                row.get("reserve_information_v4_joint_rotation_pass") is True for row in v4_rows
            ),
            "reserve_information_v4_reason_counts": dict(sorted(v4_reason_counts.items())),
        })
        return result
