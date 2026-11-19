"""State-free paired shadow evaluation for observable reserve policies."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_registration import build_target_index, register_pointcloud
from utils.fsm_robust_registration import evaluate_point_to_plane_diagnostics
from utils.fsm_correspondence_weak import (
    adaptive_weak_subspace,
    correspondence_aware_weak_scores,
    correspondence_probability,
    normalized_hessian_diagnostics,
    organized_depth_source_normals,
)
from utils.fsm_runtime_reserve import (
    rank_diverse_reserve,
    reserve_point_jacobians_camera,
    runtime_weak_scores,
)


POLICIES = ("runtime_weak", "runtime_corr_weak", "saliency", "random")
SCHEMA_VERSION = 2
METRIC_DEFINITION_VERSION = "paired_reserve_metrics_v2"
_EPS = np.finfo(np.float64).eps
_CORRESPONDENCE_DEFAULTS = {
    "k": 4, "distance_sigma": 0.05, "dmax": 0.15,
    "depth_sigma_front": 0.10, "depth_sigma_back": 0.03,
    "missing_depth_prior": 0.35, "normal_missing_prior": 0.5,
    "normal_power": 2.0, "competition_sigma": 0.02,
}


_CSV_BASE_FIELDS = [
    "pair_id", "frame_id", "timestamp", "source", "policy", "seed", "budget",
    "eligible_count", "selected_count", "shortfall", "status", "error_type",
    "error_message", "query_count", "ranking_count", "primary_pose_source",
    "rerun_init_source", "primary_normal_source", "reserve_normal_source",
    "augmented_normal_source", "candidate_feedback", "primary_fixed_fitness",
    "primary_fixed_inlier_rmse", "primary_fixed_num_correspondences",
    "primary_fixed_condition_number", "primary_fixed_effective_rank",
    "augmented_fixed_fitness", "augmented_fixed_inlier_rmse",
    "augmented_fixed_num_correspondences", "augmented_fixed_condition_number",
    "augmented_fixed_effective_rank", "rerun_fitness", "rerun_inlier_rmse",
    "rerun_num_correspondences", "rerun_condition_number", "rerun_effective_rank",
    "rerun_pose_sha256", "latency_standalone_ms", "latency_amortized_ms",
]
_CSV_V2_FIELDS = [
    "schema_version", "metric_definition_version", "row_index",
    "characteristic_length", "weak_subspace_dim", "ridge_rho",
    "selection_score_sum", "selection_score_mean", "predicted_weak_gain",
    "primary_h_trace", "primary_h_lambda_min", "primary_h_lambda_max",
    "primary_h_trace_normalized_lambda_min", "primary_h_trace_normalized_lambda_max",
    "augmented_h_trace", "augmented_h_lambda_min", "augmented_h_lambda_max",
    "augmented_h_trace_normalized_lambda_min", "augmented_h_trace_normalized_lambda_max",
    "actual_reserve_h_trace", "actual_reserve_h_lambda_min", "actual_reserve_h_lambda_max",
    "actual_reserve_h_trace_normalized_lambda_min", "actual_reserve_h_trace_normalized_lambda_max",
    "primary_logdet_gain", "augmented_logdet_gain", "net_logdet_gain",
    "actual_correspondence_count", "actual_correspondence_unique_count",
    "actual_correspondence_precision", "actual_correspondence_fraction",
    "reserve_correspondence_row_share", "selected_reserve_hit_rate",
    "baseline_weak_information", "actual_reserve_weak_gain", "net_weak_gain",
    "p_corr_available", "p_corr_mean", "p_corr_min", "p_corr_max",
    "p_corr_actual", "p_corr_brier", "p_corr_logloss",
]


def _frozen_array(value: Any, dtype: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.ndim != len(shape) or any(expected is not None and array.shape[i] != expected for i, expected in enumerate(shape)):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.number) and np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _hash_array(digest: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


@dataclass(frozen=True)
class BaselineSnapshot:
    """Immutable inputs captured immediately after the primary registration."""

    frame_id: int
    source: str
    source_points: np.ndarray
    primary_pose: np.ndarray
    primary_hessian: np.ndarray
    target_points: np.ndarray
    target_normals: np.ndarray
    target_global_ids: np.ndarray
    target_tree: Any
    intrinsics: np.ndarray
    image: np.ndarray
    depth: np.ndarray
    source_pixel_ids: np.ndarray
    timestamp: float = 0.0
    primary_init_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    primary_accepted: bool = False
    pair_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, (bool, np.bool_)) or int(self.frame_id) != self.frame_id:
            raise ValueError("frame_id must be an integer")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        source_points = _frozen_array(self.source_points, np.float64, (None, 3), "source_points")
        primary_pose = _frozen_array(self.primary_pose, np.float64, (4, 4), "primary_pose")
        primary_init_pose = _frozen_array(self.primary_init_pose, np.float64, (4, 4), "primary_init_pose")
        primary_hessian = _frozen_array(self.primary_hessian, np.float64, (6, 6), "primary_hessian")
        target_points = _frozen_array(self.target_points, np.float64, (None, 3), "target_points")
        target_normals = _frozen_array(self.target_normals, np.float64, target_points.shape, "target_normals")
        target_global_ids = _frozen_array(self.target_global_ids, np.int64, (target_points.shape[0],), "target_global_ids")
        intrinsics = _frozen_array(self.intrinsics, np.float64, (3, 3), "intrinsics")
        image = np.array(self.image, copy=True, order="C")
        depth_value = np.asarray(self.depth)
        if depth_value.ndim == 3 and depth_value.shape[2] == 1:
            depth_value = depth_value[..., 0]
        depth = _frozen_array(depth_value, np.float64, (None, None), "depth")
        source_pixel_ids = _frozen_array(self.source_pixel_ids, np.int64, (source_points.shape[0],), "source_pixel_ids")
        if image.ndim not in (2, 3) or image.shape[:2] != depth.shape:
            raise ValueError("image and depth must share image dimensions")
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            raise ValueError("source_points and target_points must be non-empty")
        if np.unique(target_global_ids).size != target_global_ids.size:
            raise ValueError("target_global_ids must be unique")
        normal_lengths = np.linalg.norm(target_normals, axis=1)
        if np.any(normal_lengths <= 1e-12):
            raise ValueError("target_normals must be non-zero")
        image.setflags(write=False)
        for name, value in (
            ("source_points", source_points), ("primary_pose", primary_pose),
            ("primary_init_pose", primary_init_pose),
            ("primary_hessian", primary_hessian), ("target_points", target_points),
            ("target_normals", target_normals), ("target_global_ids", target_global_ids),
            ("intrinsics", intrinsics), ("image", image), ("depth", depth),
            ("source_pixel_ids", source_pixel_ids),
        ):
            object.__setattr__(self, name, value)
        digest = hashlib.sha256()
        digest.update(f"{int(self.frame_id)}\0{float(self.timestamp):.17g}\0{self.source}\0{int(self.primary_accepted)}\0".encode("utf-8"))
        for name in (
            "source_points", "primary_pose", "primary_init_pose", "primary_hessian", "target_points",
            "target_normals", "target_global_ids", "intrinsics", "image", "depth",
            "source_pixel_ids",
        ):
            _hash_array(digest, name, getattr(self, name))
        object.__setattr__(self, "pair_id", digest.hexdigest())


def runtime_corr_weak_scores(
    means_world: np.ndarray,
    normals_world: np.ndarray,
    candidate_c2w: np.ndarray,
    tracker_hessian: np.ndarray,
    primary_jacobians: np.ndarray,
    weak_subspace_dim: int = 1,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure weak-direction score discounted by correlation with primary rows."""
    weak_scores, weak_vectors, eigenvalues = runtime_weak_scores(
        means_world, normals_world, candidate_c2w, tracker_hessian,
        weak_subspace_dim=weak_subspace_dim, weights=weights,
    )
    candidate_rows = reserve_point_jacobians_camera(means_world, normals_world, candidate_c2w)
    primary_rows = np.asarray(primary_jacobians, dtype=np.float64)
    if primary_rows.ndim != 2 or primary_rows.shape[1] != 6 or np.any(~np.isfinite(primary_rows)):
        raise ValueError("primary_jacobians must be finite with shape [N, 6]")
    if primary_rows.shape[0] == 0:
        return weak_scores, weak_vectors, eigenvalues
    candidate_norm = np.linalg.norm(candidate_rows, axis=1)
    primary_norm = np.linalg.norm(primary_rows, axis=1)
    usable = primary_norm > 1e-12
    if not np.any(usable):
        return weak_scores, weak_vectors, eigenvalues
    normalized_primary = primary_rows[usable] / primary_norm[usable, None]
    correlation = np.max(np.abs(candidate_rows @ normalized_primary.T), axis=1)
    correlation /= np.maximum(candidate_norm, 1e-12)
    return weak_scores * np.maximum(0.0, 1.0 - correlation * correlation), weak_vectors, eigenvalues


@dataclass(frozen=True)
class PairedEvaluation:
    pair_id: str
    triggered: bool
    trigger_reasons: tuple[str, ...]
    eligible_global_ids: np.ndarray
    rows: tuple[dict[str, Any], ...]
    selected_global_ids: tuple[np.ndarray, ...]
    selected_p_corr: tuple[np.ndarray, ...]
    selected_actual_correspondence: tuple[np.ndarray, ...]
    primary_hessian: tuple[np.ndarray, ...]
    augmented_hessian: tuple[np.ndarray, ...]
    actual_reserve_hessian: tuple[np.ndarray, ...]
    primary_hessian_normalized: tuple[np.ndarray, ...]
    augmented_hessian_normalized: tuple[np.ndarray, ...]
    actual_reserve_hessian_normalized: tuple[np.ndarray, ...]
    baseline_weak_basis: tuple[np.ndarray, ...]
    query_count: int
    ranking_counts: dict[str, int]
    query_latency_ms: float
    primary_pose: np.ndarray
    candidate_poses: tuple[np.ndarray, ...]
    shadow_latency_ms: float


class PairedShadowReserve:
    """Evaluate reserve policy arms without mutating tracker or reserve state."""

    def __init__(
        self,
        config: dict[str, Any],
        reserve_points: np.ndarray,
        reserve_normals: np.ndarray,
        reserve_scores: np.ndarray,
        reserve_global_ids: np.ndarray,
        index: MapSpatialIndex,
        local_config: LocalSubmapConfig,
        reserve_supported: np.ndarray | None = None,
        registration_fn: Callable[..., dict[str, Any]] = register_pointcloud,
        diagnostics_fn: Callable[..., dict[str, Any]] = evaluate_point_to_plane_diagnostics,
    ) -> None:
        cfg = config.get("paired_reserve", config)
        if not isinstance(cfg, dict):
            raise ValueError("paired_reserve must be a dictionary")
        if str(cfg.get("mode", "shadow")) != "shadow":
            raise ValueError("paired_reserve.mode must be shadow")
        observable = config.get("observable_reserve", {}) if "paired_reserve" in config else cfg.get("observable_reserve", {})
        if isinstance(observable, dict) and bool(observable.get("enabled", False)) and observable.get("mode") == "active":
            raise ValueError("observable_reserve active mode is forbidden during paired shadow evaluation")
        self.policies = tuple(str(value) for value in cfg.get("policies", POLICIES))
        if not self.policies or len(set(self.policies)) != len(self.policies) or any(value not in POLICIES for value in self.policies):
            raise ValueError(f"policies must be unique members of {POLICIES}")
        self.random_seeds = tuple(int(value) for value in cfg.get("random_seeds", cfg.get("seeds", (0,))))
        if "random" in self.policies and not self.random_seeds:
            raise ValueError("random policy requires random_seeds")
        self.budgets = tuple(int(value) for value in cfg.get("budgets", (64, 128, 256)))
        if not self.budgets or any(value <= 0 for value in self.budgets) or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be unique, positive, and strictly increasing")
        self.eligible_pool = str(cfg.get("eligible_pool", "visible"))
        if self.eligible_pool not in ("visible", "supported"):
            raise ValueError("eligible_pool must be visible or supported")
        self.weak_dim = int(cfg.get("weak_subspace_dim", 1))
        self.ridge_rho = float(cfg.get("ridge_rho", 1e-6))
        if not np.isfinite(self.ridge_rho) or self.ridge_rho <= 0.0:
            raise ValueError("ridge_rho must be positive and finite")
        self.trigger = dict(cfg.get("trigger", {"effective_rank_below": 6}) or {})
        self.registration = dict(cfg.get("registration", {}) or {})
        self.diversity = dict(cfg.get("diversity", {}) or {})
        self.points = _frozen_array(reserve_points, np.float64, (None, 3), "reserve_points")
        self.normals = _frozen_array(reserve_normals, np.float64, self.points.shape, "reserve_normals")
        self.scores = _frozen_array(reserve_scores, np.float64, (self.points.shape[0],), "reserve_scores")
        self.global_ids = _frozen_array(reserve_global_ids, np.int64, (self.points.shape[0],), "reserve_global_ids")
        if np.unique(self.global_ids).size != self.global_ids.size:
            raise ValueError("reserve_global_ids must be unique")
        if index.num_points != self.points.shape[0]:
            raise ValueError("index must cover reserve_points")
        if reserve_supported is None:
            supported = np.ones(self.points.shape[0], dtype=bool)
        else:
            supported = _frozen_array(reserve_supported, bool, (self.points.shape[0],), "reserve_supported")
        supported.setflags(write=False)
        self.supported = supported
        self.index = index
        self.local_config = local_config
        self.registration_fn = registration_fn
        self.diagnostics_fn = diagnostics_fn
        self.config = cfg

    @classmethod
    def from_runtime_reserve(
        cls, config: dict[str, Any], runtime_reserve: Any, local_config: LocalSubmapConfig,
        **kwargs: Any,
    ) -> "PairedShadowReserve":
        if getattr(runtime_reserve, "mode", None) == "active":
            raise ValueError("active RuntimeObservableReserve cannot be reused for paired shadow evaluation")
        return cls(
            config, runtime_reserve.points, runtime_reserve.normals, runtime_reserve.scores,
            runtime_reserve.global_ids, runtime_reserve.index, local_config,
            reserve_supported=getattr(runtime_reserve, "supported", None), **kwargs,
        )

    def _trigger(self, snapshot: BaselineSnapshot, primary: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
        if bool(self.trigger.get("always", False)):
            return True, ("always",)
        eigenvalues = np.linalg.eigvalsh(0.5 * (snapshot.primary_hessian + snapshot.primary_hessian.T))
        trace = float(np.sum(eigenvalues))
        reasons = []
        if "condition_number" in self.trigger and float(primary.get("condition_number", np.inf)) > float(self.trigger["condition_number"]):
            reasons.append("condition_number")
        if "normalized_lambda_min" in self.trigger and (float(eigenvalues[0]) / trace if trace > 0 else 0.0) < float(self.trigger["normalized_lambda_min"]):
            reasons.append("normalized_lambda_min")
        if "effective_rank_below" in self.trigger and int(primary.get("effective_rank", 0)) < int(self.trigger["effective_rank_below"]):
            reasons.append("effective_rank_below")
        return bool(reasons), tuple(reasons)

    def _query_config(self) -> LocalSubmapConfig:
        return LocalSubmapConfig(
            near=self.local_config.near, far=self.local_config.far,
            margin_px=self.local_config.margin_px, radius_margin=self.local_config.radius_margin,
            max_points=None, sampling_policy="coverage", tile_rows=self.local_config.tile_rows,
            tile_cols=self.local_config.tile_cols, depth_bins=self.local_config.depth_bins,
            frustum_culling=str(self.config.get("frustum_culling", self.local_config.frustum_culling)),
            block_size_m=float(self.config.get("block_size_m", self.local_config.block_size_m)),
        )

    def _diagnostics(
        self, snapshot: BaselineSnapshot, points: np.ndarray, normals: np.ndarray,
        target_global_ids: np.ndarray, tree: Any,
    ) -> dict[str, Any]:
        target_ids = np.asarray(target_global_ids, dtype=np.int64)
        if target_ids.shape != (points.shape[0],) or np.unique(target_ids).size != target_ids.size:
            raise ValueError("target_global_ids must match targets and be strictly unique")
        return self.diagnostics_fn(
            snapshot.source_points, points, normals, snapshot.primary_pose,
            max_correspondence_distance=float(self.registration.get("max_correspondence_distance", 0.05)),
            target_tree=tree, loss=str(self.registration.get("robust_loss", "huber")),
            loss_scale=float(self.registration.get("loss_scale", 0.02)),
            min_correspondences=int(self.registration.get("min_correspondences", 6)),
            rank_tolerance=float(self.registration.get("rank_tolerance", 1e-9)),
            source_ids=snapshot.source_pixel_ids, target_ids=target_ids,
            return_correspondences=True,
        )

    def _scores(
        self,
        policy: str,
        snapshot: BaselineSnapshot,
        primary_rows: np.ndarray,
        eligible: np.ndarray,
        source_normals: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        if policy == "runtime_weak":
            scores = runtime_weak_scores(
                self.points, self.normals, snapshot.primary_pose,
                snapshot.primary_hessian, self.weak_dim,
            )[0]
            return scores, None, scores.copy()
        if policy == "runtime_corr_weak":
            normals, normal_valid, normal_confidence = source_normals or organized_depth_source_normals(
                snapshot.depth, snapshot.intrinsics, snapshot.source_pixel_ids
            )
            corr_cfg = dict(self.config.get("correspondence", {}) or {})
            probability = correspondence_probability(
                self.points[eligible], snapshot.source_points, snapshot.primary_pose,
                snapshot.depth, snapshot.intrinsics, snapshot.source_pixel_ids,
                reserve_normals_world=self.normals[eligible], source_normals_camera=normals,
                source_normal_valid=normal_valid, source_normal_confidence=normal_confidence,
                base_target_tree=snapshot.target_tree, **corr_cfg,
            )["probability"]
            scores, details = correspondence_aware_weak_scores(
                self.points, self.normals, snapshot.primary_pose, snapshot.primary_hessian,
                snapshot.source_points, probability, visible_indices=eligible,
                min_dim=int(self.config.get("corr_weak_min_dim", 1)),
                max_dim=int(self.config.get("corr_weak_max_dim", 3)),
            )
            predicted = np.zeros(self.points.shape[0], dtype=np.float64)
            predicted[eligible] = np.asarray(details["weak_score"], dtype=np.float64)
            full_probability = np.full(self.points.shape[0], np.nan, dtype=np.float64)
            full_probability[eligible] = probability
            return scores, full_probability, predicted
        if policy == "saliency":
            return self.scores.copy(), None, np.zeros(self.points.shape[0], dtype=np.float64)
        seed_digest = hashlib.sha256(f"{snapshot.pair_id}:{int(seed)}".encode("ascii")).digest()
        derived_seed = int.from_bytes(seed_digest[:8], "little", signed=False)
        scores = np.random.default_rng(derived_seed).random(self.points.shape[0])
        return scores, None, np.zeros(self.points.shape[0], dtype=np.float64)

    def _register(self, snapshot: BaselineSnapshot, points: np.ndarray, normals: np.ndarray, tree: Any) -> dict[str, Any]:
        kwargs = dict(self.registration)
        kwargs.setdefault("backend", "numpy_robust_point_to_plane")
        kwargs.setdefault("max_correspondence_distance", 0.05)
        kwargs.setdefault("max_iterations", 30)
        kwargs["source_points"] = snapshot.source_points
        kwargs["target_points"] = points
        kwargs["target_normals"] = normals
        kwargs["target_tree"] = tree
        kwargs["init_transform"] = snapshot.primary_pose.copy()
        return self.registration_fn(**kwargs)

    def _weak_subspace(self, snapshot: BaselineSnapshot) -> dict[str, Any]:
        weak_cfg = dict(self.config.get("adaptive_weak", {}) or {})
        return adaptive_weak_subspace(
            snapshot.primary_hessian, snapshot.primary_pose, snapshot.source_points,
            min_dim=int(weak_cfg.get("min_dim", self.config.get("corr_weak_min_dim", 1))),
            max_dim=int(weak_cfg.get(
                "max_dim", self.config.get("corr_weak_max_dim", 3)
            )),
            trace_threshold=float(weak_cfg.get("trace_threshold", 1e-3)),
            relative_threshold=float(weak_cfg.get("relative_threshold", 1e-2)),
            characteristic_depth_min=float(weak_cfg.get("characteristic_length_min", 0.5)),
            characteristic_depth_max=float(weak_cfg.get("characteristic_length_max", 5.0)),
        )

    def evaluate(self, snapshot: BaselineSnapshot) -> PairedEvaluation:
        evaluate_start = time.perf_counter()
        if not isinstance(snapshot, BaselineSnapshot):
            raise TypeError("snapshot must be a BaselineSnapshot")
        primary = self._diagnostics(
            snapshot, snapshot.target_points, snapshot.target_normals,
            snapshot.target_global_ids, snapshot.target_tree,
        )
        triggered, reasons = self._trigger(snapshot, primary)
        empty_ids = np.empty(0, dtype=np.int64)
        empty_ids.setflags(write=False)
        if not triggered:
            primary_pose = snapshot.primary_pose.copy()
            primary_pose.setflags(write=False)
            return PairedEvaluation(
                pair_id=snapshot.pair_id, triggered=False, trigger_reasons=reasons,
                eligible_global_ids=empty_ids, rows=(), selected_global_ids=(),
                selected_p_corr=(), selected_actual_correspondence=(), primary_hessian=(),
                augmented_hessian=(), actual_reserve_hessian=(), primary_hessian_normalized=(),
                augmented_hessian_normalized=(), actual_reserve_hessian_normalized=(),
                baseline_weak_basis=(), query_count=0, ranking_counts={}, query_latency_ms=0.0,
                primary_pose=primary_pose, candidate_poses=(),
                shadow_latency_ms=float((time.perf_counter() - evaluate_start) * 1000.0),
            )

        query_start = time.perf_counter()
        query = self.index.query_local_submap(
            snapshot.primary_pose, snapshot.intrinsics, snapshot.depth.shape[0], snapshot.depth.shape[1],
            self._query_config(), scores=self.scores,
        )
        query_latency = float((time.perf_counter() - query_start) * 1000.0)
        eligible = np.asarray(query["indices"], dtype=np.int64)
        if self.eligible_pool == "supported":
            eligible = eligible[self.supported[eligible]]
        target_ids = set(int(value) for value in snapshot.target_global_ids)
        eligible = eligible[np.asarray([int(self.global_ids[i]) not in target_ids for i in eligible], dtype=bool)]
        eligible_ids = self.global_ids[eligible].copy()
        eligible_ids.setflags(write=False)

        primary_rows = np.asarray(primary.get("jacobians", np.empty((0, 6))), dtype=np.float64)
        source_normals = (
            organized_depth_source_normals(snapshot.depth, snapshot.intrinsics, snapshot.source_pixel_ids)
            if "runtime_corr_weak" in self.policies else None
        )
        rows: list[dict[str, Any]] = []
        selections: list[np.ndarray] = []
        selected_probabilities: list[np.ndarray] = []
        selected_actual: list[np.ndarray] = []
        primary_raw_hessians: list[np.ndarray] = []
        augmented_raw_hessians: list[np.ndarray] = []
        reserve_raw_hessians: list[np.ndarray] = []
        primary_hessians: list[np.ndarray] = []
        augmented_hessians: list[np.ndarray] = []
        reserve_hessians: list[np.ndarray] = []
        weak_bases: list[np.ndarray] = []
        candidate_poses: list[np.ndarray] = []
        ranking_counts: dict[str, int] = {}
        ranking_latencies: dict[tuple[str, int | None], float] = {}
        arms = [(policy, seed) for policy in self.policies for seed in (self.random_seeds if policy == "random" else (None,))]
        orders: dict[tuple[str, int | None], np.ndarray | Exception] = {}
        score_cache: dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
        for policy, seed in arms:
            key = (policy, seed)
            start = time.perf_counter()
            try:
                scores, p_corr, predicted = self._scores(
                    policy, snapshot, primary_rows, eligible, source_normals, seed
                )
                score_cache[key] = (scores, p_corr, predicted)
                orders[key] = rank_diverse_reserve(
                    indices=eligible, scores=scores, means_world=self.points,
                    candidate_c2w=snapshot.primary_pose, intrinsics=snapshot.intrinsics,
                    image_height=snapshot.depth.shape[0], image_width=snapshot.depth.shape[1],
                    tile_rows=int(self.diversity.get("tile_rows", self.local_config.tile_rows)),
                    tile_cols=int(self.diversity.get("tile_cols", self.local_config.tile_cols)),
                    depth_bins=int(self.diversity.get("depth_bins", self.local_config.depth_bins)),
                    voxel_size=float(self.diversity.get("voxel_size", 0.25)),
                    tile_cap=self.diversity.get("tile_cap"), depth_cap=self.diversity.get("depth_cap"),
                    voxel_cap=self.diversity.get("voxel_cap", 1), global_ids=self.global_ids,
                )
                ranking_counts[f"{policy}:{'' if seed is None else seed}"] = 1
            except Exception as exc:
                orders[key] = exc
            ranking_latencies[key] = float((time.perf_counter() - start) * 1000.0)

        total_rows = max(len(arms) * len(self.budgets), 1)
        for policy, seed in arms:
            key = (policy, seed)
            order_or_error = orders[key]
            for budget in self.budgets:
                selected = empty_ids.copy()
                selected_p_corr = np.empty(0, dtype=np.float64)
                actual_flags = np.empty(0, dtype=bool)
                candidate_pose = np.full((4, 4), np.nan, dtype=np.float64)
                primary_h = np.full((6, 6), np.nan, dtype=np.float64)
                augmented_h = np.full((6, 6), np.nan, dtype=np.float64)
                reserve_h = np.full((6, 6), np.nan, dtype=np.float64)
                primary_raw_h = np.full((6, 6), np.nan, dtype=np.float64)
                augmented_raw_h = np.full((6, 6), np.nan, dtype=np.float64)
                reserve_raw_h = np.full((6, 6), np.nan, dtype=np.float64)
                weak_basis = np.full((6, 3), np.nan, dtype=np.float64)
                start = time.perf_counter()
                row: dict[str, Any] = {
                    "pair_id": snapshot.pair_id, "frame_id": int(snapshot.frame_id),
                    "timestamp": float(snapshot.timestamp), "source": snapshot.source,
                    "policy": policy, "seed": "" if seed is None else int(seed), "budget": int(budget),
                    "eligible_count": int(eligible.size), "selected_count": 0, "shortfall": int(budget),
                    "status": "ok", "error_type": "", "error_message": "",
                    "query_count": 1, "ranking_count": 1,
                    "primary_pose_source": "baseline_primary_pose", "rerun_init_source": "baseline_primary_pose",
                    "primary_normal_source": "snapshot_frozen", "reserve_normal_source": "sidecar_frozen",
                    "augmented_normal_source": "frozen_concat", "candidate_feedback": False,
                    "schema_version": SCHEMA_VERSION,
                    "metric_definition_version": METRIC_DEFINITION_VERSION,
                    "row_index": len(rows), "ridge_rho": self.ridge_rho,
                }
                try:
                    if isinstance(order_or_error, Exception):
                        raise order_or_error
                    indices = order_or_error[:budget]
                    scores, p_corr, predicted = score_cache[key]
                    selected = self.global_ids[indices].copy()
                    selected.setflags(write=False)
                    row["selected_count"] = int(indices.size)
                    row["shortfall"] = int(budget - indices.size)
                    augmented_points = np.concatenate((snapshot.target_points, self.points[indices]), axis=0)
                    augmented_normals = np.concatenate((snapshot.target_normals, self.normals[indices]), axis=0)
                    augmented_ids = np.concatenate((snapshot.target_global_ids, selected))
                    if np.unique(augmented_ids).size != augmented_ids.size:
                        raise ValueError("primary and augmented target IDs must be strictly unique")
                    augmented_tree = build_target_index(augmented_points)["target_tree"]
                    fixed = self._diagnostics(
                        snapshot, augmented_points, augmented_normals, augmented_ids, augmented_tree
                    )
                    rerun = self._register(snapshot, augmented_points, augmented_normals, augmented_tree)
                    pose = np.asarray(rerun.get("transform"), dtype=np.float64)
                    if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
                        candidate_pose = pose.copy()
                    for prefix, result in (("primary_fixed", primary), ("augmented_fixed", fixed), ("rerun", rerun)):
                        for metric in ("fitness", "inlier_rmse", "num_correspondences", "condition_number", "effective_rank"):
                            value = result.get(metric)
                            row[f"{prefix}_{metric}"] = value if value is None or np.isscalar(value) else ""
                    row["rerun_pose_sha256"] = _pose_hash(rerun.get("transform"))
                    weak = self._weak_subspace(snapshot)
                    length = float(weak["characteristic_length"])
                    primary_h = np.asarray(weak["normalized_hessian"], dtype=np.float64)
                    augmented_h = _normalized_diagnostic_hessian(fixed, length, snapshot.primary_pose)
                    reserve_h, actual_flags, actual_count, actual_unique = _actual_reserve_hessian(
                        fixed, selected, length, snapshot.primary_pose
                    )
                    primary_raw_h = _psd(snapshot.primary_hessian)
                    augmented_raw_h = _psd(np.asarray(fixed.get("hessian"), dtype=np.float64))
                    reserve_raw_h, _, _, _ = _actual_reserve_hessian(
                        fixed, selected, None, snapshot.primary_pose
                    )
                    _add_hessian_metrics(row, "primary_h", primary_raw_h)
                    _add_hessian_metrics(row, "augmented_h", augmented_raw_h)
                    _add_hessian_metrics(row, "actual_reserve_h", reserve_raw_h)
                    ridge = self.ridge_rho * np.eye(6, dtype=np.float64)
                    baseline_logdet = _logdet(ridge)
                    primary_logdet = _logdet(primary_h + ridge)
                    augmented_logdet = _logdet(augmented_h + ridge)
                    basis = np.asarray(weak["basis"], dtype=np.float64)
                    weak_basis = np.zeros((6, 3), dtype=np.float64)
                    weak_basis[:, :basis.shape[1]] = basis
                    row.update({
                        "characteristic_length": length,
                        "weak_subspace_dim": int(weak["k"]),
                        "selection_score_sum": float(np.sum(scores[indices])),
                        "selection_score_mean": float(np.mean(scores[indices])) if indices.size else 0.0,
                        "predicted_weak_gain": _ideal_weak_gain(
                            self.points[indices], self.normals[indices], snapshot.primary_pose,
                            length, basis,
                        ),
                        "primary_logdet_gain": primary_logdet - baseline_logdet,
                        "augmented_logdet_gain": augmented_logdet - baseline_logdet,
                        "net_logdet_gain": augmented_logdet - primary_logdet,
                        "actual_correspondence_count": actual_count,
                        "actual_correspondence_unique_count": actual_unique,
                        "actual_correspondence_precision": float(
                            actual_count / max(int(fixed.get("num_correspondences", 0)), 1)
                        ),
                        "actual_correspondence_fraction": float(actual_unique / indices.size) if indices.size else 0.0,
                        "reserve_correspondence_row_share": float(
                            actual_count / max(int(fixed.get("num_correspondences", 0)), 1)
                        ),
                        "selected_reserve_hit_rate": float(actual_unique / indices.size) if indices.size else 0.0,
                        "baseline_weak_information": _weak_information(primary_h, basis),
                        "actual_reserve_weak_gain": _weak_information(reserve_h, basis),
                        "net_weak_gain": _weak_information(augmented_h - primary_h, basis),
                    })
                    selected_p_corr = (
                        np.asarray(p_corr[indices], dtype=np.float64)
                        if p_corr is not None else np.full(indices.size, np.nan, dtype=np.float64)
                    )
                    _add_calibration_metrics(row, selected_p_corr, actual_flags)
                except Exception as exc:
                    row["status"] = "error"
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)
                if selected_p_corr.shape != (selected.size,):
                    selected_p_corr = np.full(selected.size, np.nan, dtype=np.float64)
                if actual_flags.shape != (selected.size,):
                    actual_flags = np.zeros(selected.size, dtype=bool)
                standalone = float((time.perf_counter() - start) * 1000.0)
                row["latency_standalone_ms"] = standalone
                row["latency_amortized_ms"] = standalone + query_latency / total_rows + ranking_latencies[key] / len(self.budgets)
                rows.append(row)
                selections.append(selected)
                selected_probabilities.append(selected_p_corr)
                selected_actual.append(actual_flags)
                primary_raw_hessians.append(primary_raw_h)
                augmented_raw_hessians.append(augmented_raw_h)
                reserve_raw_hessians.append(reserve_raw_h)
                primary_hessians.append(primary_h)
                augmented_hessians.append(augmented_h)
                reserve_hessians.append(reserve_h)
                weak_bases.append(weak_basis)
                candidate_pose.setflags(write=False)
                candidate_poses.append(candidate_pose)
        primary_pose = snapshot.primary_pose.copy()
        primary_pose.setflags(write=False)
        return PairedEvaluation(
            pair_id=snapshot.pair_id, triggered=True, trigger_reasons=reasons,
            eligible_global_ids=eligible_ids, rows=tuple(rows),
            selected_global_ids=tuple(selections), selected_p_corr=tuple(selected_probabilities),
            selected_actual_correspondence=tuple(selected_actual),
            primary_hessian=tuple(primary_raw_hessians),
            augmented_hessian=tuple(augmented_raw_hessians),
            actual_reserve_hessian=tuple(reserve_raw_hessians),
            primary_hessian_normalized=tuple(primary_hessians),
            augmented_hessian_normalized=tuple(augmented_hessians),
            actual_reserve_hessian_normalized=tuple(reserve_hessians),
            baseline_weak_basis=tuple(weak_bases), query_count=1,
            ranking_counts=ranking_counts, query_latency_ms=query_latency,
            primary_pose=primary_pose, candidate_poses=tuple(candidate_poses),
            shadow_latency_ms=float((time.perf_counter() - evaluate_start) * 1000.0),
        )

    def save(
        self,
        evaluation: PairedEvaluation | list[PairedEvaluation] | tuple[PairedEvaluation, ...],
        output_dir: str | Path,
        parent_trajectory_sha256: str = "",
    ) -> dict[str, Any]:
        evaluations = (evaluation,) if isinstance(evaluation, PairedEvaluation) else tuple(evaluation)
        if any(not isinstance(item, PairedEvaluation) for item in evaluations):
            raise TypeError("evaluation must contain PairedEvaluation values")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = [row for item in evaluations for row in item.rows]
        selections = [ids for item in evaluations for ids in item.selected_global_ids]
        probabilities = [values for item in evaluations for values in item.selected_p_corr]
        actual = [values for item in evaluations for values in item.selected_actual_correspondence]
        candidates = [pose for item in evaluations for pose in item.candidate_poses]
        lengths = np.asarray([ids.size for ids in selections], dtype=np.int64)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
        flat_ids = np.concatenate(selections) if int(np.sum(lengths)) else np.empty(0, dtype=np.int64)
        eligible_lengths = np.asarray([item.eligible_global_ids.size for item in evaluations], dtype=np.int64)
        eligible_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(eligible_lengths)))
        eligible_ids = (
            np.concatenate([item.eligible_global_ids for item in evaluations])
            if int(np.sum(eligible_lengths)) else np.empty(0, dtype=np.int64)
        )
        np.savez(
            output / "paired_reserve_ids.npz",
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
            metric_definition_version=np.asarray(METRIC_DEFINITION_VERSION, dtype="U32"),
            offsets=offsets, global_ids=flat_ids,
            selected_p_corr=np.concatenate(probabilities) if flat_ids.size else np.empty(0, dtype=np.float64),
            selected_actual_correspondence=np.concatenate(actual) if flat_ids.size else np.empty(0, dtype=bool),
            eligible_offsets=eligible_offsets, eligible_global_ids=eligible_ids,
            frame_id=np.asarray([int(row["frame_id"]) for row in rows], dtype=np.int64),
            timestamp=np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64),
            pair_id=np.asarray([str(row["pair_id"]) for row in rows], dtype="U64"),
            policy=np.asarray([str(row["policy"]) for row in rows], dtype="U32"),
            seed=np.asarray([int(row["seed"]) if row["seed"] != "" else -1 for row in rows], dtype=np.int64),
            budget=np.asarray([int(row["budget"]) for row in rows], dtype=np.int64),
            row_index=np.arange(len(rows), dtype=np.int64),
            actual_correspondence_count=np.asarray([
                int(row.get("actual_correspondence_count", 0)) for row in rows
            ], dtype=np.int64),
            primary_hessian=_stack_hessians(evaluations, "primary_hessian"),
            augmented_hessian=_stack_hessians(evaluations, "augmented_hessian"),
            actual_reserve_hessian=_stack_hessians(evaluations, "actual_reserve_hessian"),
            primary_hessian_normalized=_stack_hessians(evaluations, "primary_hessian_normalized"),
            augmented_hessian_normalized=_stack_hessians(evaluations, "augmented_hessian_normalized"),
            actual_reserve_hessian_normalized=_stack_hessians(evaluations, "actual_reserve_hessian_normalized"),
            baseline_weak_basis=_stack_values(evaluations, "baseline_weak_basis", (6, 3)),
            **_row_metric_arrays(rows),
            primary_pose=np.stack([
                item.primary_pose for item in evaluations for _ in item.rows
            ]) if rows else np.empty((0, 4, 4), dtype=np.float64),
            candidate_pose=np.stack(candidates) if candidates else np.empty((0, 4, 4), dtype=np.float64),
            parent_trajectory_sha256=np.asarray(str(parent_trajectory_sha256), dtype="U64"),
        )
        extras = sorted(set(key for row in rows for key in row) - set(_CSV_BASE_FIELDS) - set(_CSV_V2_FIELDS))
        fieldnames = _CSV_BASE_FIELDS + _CSV_V2_FIELDS + extras
        with open(output / "paired_reserve_rows.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        ok = [row for row in rows if row["status"] == "ok"]
        triggered = [item for item in evaluations if item.triggered]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "trigger_pairs": len(triggered), "pair_count": len(evaluations), "arm_count": len(rows),
            "paired_no_trigger": len(triggered) == 0,
            "row_count": len(rows),
            "successful_rows": len(ok), "failed_rows": len(rows) - len(ok),
            "query_count": int(sum(item.query_count for item in evaluations)),
            "normal_count": len(triggered) if "runtime_corr_weak" in self.policies else 0,
            "registration_count": len(rows),
            "query_latency_ms": float(sum(item.query_latency_ms for item in evaluations)),
            "standalone_latency_total_ms": float(sum(float(row["latency_standalone_ms"]) for row in rows)),
            "amortized_latency_total_ms": float(sum(float(row["latency_amortized_ms"]) for row in rows)),
            "total_shadow_latency_ms": float(sum(item.shadow_latency_ms for item in evaluations)),
            "reserve_correspondence_row_share_mean": _finite_row_mean(ok, "reserve_correspondence_row_share"),
            "selected_reserve_hit_rate_mean": _finite_row_mean(ok, "selected_reserve_hit_rate"),
            "actual_correspondence_precision_mean": _finite_row_mean(ok, "actual_correspondence_precision"),
            "actual_correspondence_fraction_mean": _finite_row_mean(ok, "actual_correspondence_fraction"),
            **_calibration_summary(rows, probabilities, actual),
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "mode": "shadow", "policies": list(self.policies),
            "trigger_protocol": self.config.get("trigger_protocol"),
            "trigger": dict(self.trigger),
            "random_seeds": list(self.random_seeds), "budgets": list(self.budgets),
            "eligible_pool": self.eligible_pool, "normal_protocol": "primary_frozen+sidecar_frozen",
            "registration_init": "primary_pose", "candidate_feedback": False,
            "snapshot_fields": [item.name for item in fields(BaselineSnapshot)],
            "parent_trajectory_sha256": str(parent_trajectory_sha256),
            "ridge_rho": self.ridge_rho,
            "correspondence_model": {
                **_CORRESPONDENCE_DEFAULTS,
                **dict(self.config.get("correspondence", {}) or {}),
            },
            "adaptive_weak_model": {
                "min_dim": int(dict(self.config.get("adaptive_weak", {}) or {}).get(
                    "min_dim", self.config.get("corr_weak_min_dim", 1)
                )),
                "max_dim": int(dict(self.config.get("adaptive_weak", {}) or {}).get(
                    "max_dim", self.config.get("corr_weak_max_dim", 3)
                )),
                "trace_threshold": float(dict(self.config.get("adaptive_weak", {}) or {}).get("trace_threshold", 1e-3)),
                "relative_threshold": float(dict(self.config.get("adaptive_weak", {}) or {}).get("relative_threshold", 1e-2)),
                "characteristic_length_min": float(dict(self.config.get("adaptive_weak", {}) or {}).get("characteristic_length_min", 0.5)),
                "characteristic_length_max": float(dict(self.config.get("adaptive_weak", {}) or {}).get("characteristic_length_max", 5.0)),
            },
            "metric_semantics": {
                "coordinate_system": "camera twist [rotation/characteristic_length, translation]",
                "baseline_hessian": "snapshot primary Hessian transformed to normalized camera coordinates",
                "actual_reserve_hessian": "PSD sum of augmented fixed diagnostic Jacobian outer products for selected target IDs",
                "selection_score": "policy-specific ranking score",
                "predicted_weak_gain": "probability-free normalized geometric weak gain",
                "p_corr": "runtime_corr_weak cached correspondence probability; unavailable for other policies",
                "actual_correspondence": "selected target ID appears in augmented fixed diagnostics",
                "actual_correspondence_precision": "reserve correspondence rows / all augmented fixed correspondence rows",
                "actual_correspondence_fraction": "unique selected reserve IDs matched / selected reserve IDs",
                "reserve_correspondence_row_share": "exact alias of actual_correspondence_precision",
                "selected_reserve_hit_rate": "exact alias of actual_correspondence_fraction",
            },
        }
        for name, value in (("paired_reserve_summary.json", summary), ("paired_reserve_provenance.json", provenance)):
            with open(output / name, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        return summary


def _pose_hash(value: Any) -> str:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
        return ""
    return hashlib.sha256(np.ascontiguousarray(pose).tobytes()).hexdigest()


def _normalized_diagnostic_hessian(
    result: dict[str, Any], length: float, reference_pose: np.ndarray,
) -> np.ndarray:
    """Map world-left diagnostic rows to the primary's normalized camera frame."""
    jacobians = np.asarray(result.get("jacobians", np.empty((0, 6))), dtype=np.float64)
    weights = np.asarray(result.get("weights", np.ones(jacobians.shape[0])), dtype=np.float64)
    if jacobians.ndim != 2 or jacobians.shape[1] != 6 or weights.shape != (jacobians.shape[0],):
        raise ValueError("fixed diagnostics must provide aligned jacobians and weights")
    hessian_world = (jacobians.T * weights) @ jacobians
    return _psd(normalized_hessian_diagnostics(
        hessian_world, reference_pose, characteristic_length=length,
    )["hessian"])


def _actual_reserve_hessian(
    result: dict[str, Any], selected_ids: np.ndarray, length: float | None,
    reference_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sum selected world-left rows, optionally in the shared normalized frame."""
    target_ids = np.asarray(result.get("target_ids", np.empty(0, dtype=np.int64)), dtype=np.int64)
    jacobians = np.asarray(result.get("jacobians", np.empty((0, 6))), dtype=np.float64)
    weights = np.asarray(result.get("weights", np.ones(jacobians.shape[0])), dtype=np.float64)
    if target_ids.shape != (jacobians.shape[0],) or weights.shape != (jacobians.shape[0],):
        raise ValueError("augmented fixed correspondence IDs, jacobians and weights must align")
    mask = np.isin(target_ids, selected_ids)
    rows = jacobians[mask].copy()
    hessian = (
        _psd((rows.T * weights[mask]) @ rows) if length is None
        else _normalized_diagnostic_hessian(
            {"jacobians": rows, "weights": weights[mask]}, length, reference_pose,
        )
    )
    matched = np.unique(target_ids[mask])
    flags = np.isin(selected_ids, matched)
    return hessian, flags, int(np.count_nonzero(mask)), int(matched.size)


def _psd(hessian: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise ValueError("constructed Hessian is not PSD")
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _add_hessian_metrics(row: dict[str, Any], prefix: str, hessian: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(_psd(hessian))
    trace = float(np.sum(eigenvalues))
    row[f"{prefix}_trace"] = trace
    row[f"{prefix}_lambda_min"] = float(eigenvalues[0])
    row[f"{prefix}_lambda_max"] = float(eigenvalues[-1])
    row[f"{prefix}_trace_normalized_lambda_min"] = float(eigenvalues[0] / trace) if trace > 0.0 else 0.0
    row[f"{prefix}_trace_normalized_lambda_max"] = float(eigenvalues[-1] / trace) if trace > 0.0 else 0.0


def _logdet(hessian: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("ridge Hessian must be positive definite")
    return float(value)


def _weak_information(hessian: np.ndarray, basis: np.ndarray) -> float:
    return float(np.trace(basis.T @ (0.5 * (hessian + hessian.T)) @ basis))


def _ideal_weak_gain(
    points: np.ndarray, normals: np.ndarray, pose: np.ndarray, length: float, basis: np.ndarray,
) -> float:
    if points.shape[0] == 0:
        return 0.0
    jacobians = reserve_point_jacobians_camera(points, normals, pose)
    jacobians[:, :3] /= length
    return _weak_information(jacobians.T @ jacobians, basis)


def _add_calibration_metrics(row: dict[str, Any], probability: np.ndarray, actual: np.ndarray) -> None:
    available = probability.size > 0 and np.all(np.isfinite(probability))
    row["p_corr_available"] = bool(available)
    if not available:
        for name in ("p_corr_mean", "p_corr_min", "p_corr_max", "p_corr_actual", "p_corr_brier", "p_corr_logloss"):
            row[name] = ""
        return
    labels = np.asarray(actual, dtype=np.float64)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    row.update({
        "p_corr_mean": float(np.mean(probability)), "p_corr_min": float(np.min(probability)),
        "p_corr_max": float(np.max(probability)), "p_corr_actual": float(np.mean(labels)),
        "p_corr_brier": float(np.mean((probability - labels) ** 2)),
        "p_corr_logloss": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))),
    })


def _stack_hessians(evaluations: tuple[PairedEvaluation, ...], name: str) -> np.ndarray:
    return _stack_values(evaluations, name, (6, 6))


def _stack_values(
    evaluations: tuple[PairedEvaluation, ...], name: str, trailing_shape: tuple[int, ...],
) -> np.ndarray:
    values = [value for item in evaluations for value in getattr(item, name)]
    return np.stack(values) if values else np.empty((0, *trailing_shape), dtype=np.float64)


def _row_metric_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    excluded = {"schema_version", "metric_definition_version", "row_index", "actual_correspondence_count"}
    names = [name for name in _CSV_V2_FIELDS if name not in excluded]
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        if name == "p_corr_available":
            arrays[name] = np.asarray([bool(row.get(name, False)) for row in rows], dtype=bool)
        else:
            arrays[name] = np.asarray([
                float(row[name]) if row.get(name, "") not in ("", None) else np.nan for row in rows
            ], dtype=np.float64)
    return arrays


def _finite_row_mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) not in (None, "") and np.isfinite(float(row[name]))]
    return float(np.mean(values)) if values else None


def _ece(probability: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if probability.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    total = 0.0
    for index in range(bins):
        mask = bucket == index
        if np.any(mask):
            total += float(np.mean(mask)) * abs(float(np.mean(probability[mask]) - np.mean(actual[mask])))
    return total


def _calibration_summary(
    rows: list[dict[str, Any]], probabilities: list[np.ndarray], actual: list[np.ndarray],
) -> dict[str, Any]:
    available_rows = [row for row in rows if row.get("p_corr_available", False)]
    pairs = [(p, np.asarray(a, dtype=np.float64)) for p, a in zip(probabilities, actual) if p.size and np.all(np.isfinite(p))]
    if not pairs:
        return {"p_corr_calibration_row_count": 0, "p_corr_macro_brier": None,
                "p_corr_macro_logloss": None, "p_corr_micro_brier": None,
                "p_corr_micro_logloss": None, "p_corr_ece": None}
    probability = np.concatenate([item[0] for item in pairs])
    labels = np.concatenate([item[1] for item in pairs])
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "p_corr_calibration_row_count": len(available_rows),
        "p_corr_macro_brier": float(np.mean([row["p_corr_brier"] for row in available_rows])),
        "p_corr_macro_logloss": float(np.mean([row["p_corr_logloss"] for row in available_rows])),
        "p_corr_micro_brier": float(np.mean((probability - labels) ** 2)),
        "p_corr_micro_logloss": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))),
        "p_corr_ece": _ece(probability, labels),
    }
