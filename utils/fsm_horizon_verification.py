"""Offline short-horizon verification of paired tracking hypotheses."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from utils.fsm_dataset import rgbd_to_pointcloud
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_map import load_fsm_map
from utils.fsm_registration import build_target_index, register_pointcloud
from utils.fsm_robust_registration import estimate_normals
from utils.fsm_runtime_reserve import evaluate_common_target_pose, relative_se3_log


SCHEMA_VERSION = 1
DEFAULT_POLICY = "runtime_corr_weak"
DEFAULT_BUDGET = 128
DEFAULT_HORIZON = 3


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
class TrackingObservation:
    """Immutable tracker input that physically cannot expose ground truth poses."""

    frame_id: int
    output_idx: int
    timestamp: float
    source_points: np.ndarray
    source_pixel_ids: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray

    def __post_init__(self) -> None:
        points = _frozen_array(self.source_points, np.float64, (None, 3), "source_points")
        pixels = _frozen_array(self.source_pixel_ids, np.int64, (points.shape[0],), "source_pixel_ids")
        depth_value = np.asarray(self.depth)
        if depth_value.ndim == 3 and depth_value.shape[-1] == 1:
            depth_value = depth_value[..., 0]
        depth = _frozen_array(depth_value, np.float64, (None, None), "depth")
        intrinsics = _frozen_array(self.intrinsics, np.float64, (3, 3), "intrinsics")
        if np.any(pixels < 0) or np.any(pixels >= depth.size):
            raise ValueError("source_pixel_ids must index depth")
        object.__setattr__(self, "source_points", points)
        object.__setattr__(self, "source_pixel_ids", pixels)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "intrinsics", intrinsics)

    @classmethod
    def from_frame(
        cls,
        frame: dict[str, Any],
        output_idx: int,
        dataset_config: dict[str, Any],
        seed: int,
    ) -> "TrackingObservation":
        pointcloud = rgbd_to_pointcloud(
            color=frame.get("color"), depth=frame["depth"], intrinsics=frame["intrinsics"],
            c2w=None, depth_min=float(dataset_config.get("depth_min", 0.0)),
            depth_max=dataset_config.get("depth_max"), max_points=dataset_config.get("max_points"),
            seed=int(seed) + int(frame.get("frame_id", output_idx)),
        )
        return cls(
            frame_id=int(frame.get("frame_id", output_idx)), output_idx=int(output_idx),
            timestamp=float(frame.get("timestamp", frame.get("frame_id", output_idx))),
            source_points=pointcloud["points"], source_pixel_ids=pointcloud["indices"],
            depth=frame["depth"], intrinsics=frame["intrinsics"],
        )


@dataclass
class RegistrationCache:
    local_indices: np.ndarray | None = None
    local_tree: Any = None
    local_normals: np.ndarray | None = None


@dataclass
class TrackerBranchState:
    branch: str
    last_valid_estimate: np.ndarray
    valid_pose_history: list[tuple[int, np.ndarray, bool]]
    cache: RegistrationCache = field(default_factory=RegistrationCache)

    def __post_init__(self) -> None:
        self.last_valid_estimate = _pose_copy(self.last_valid_estimate, "last_valid_estimate")
        self.valid_pose_history = [
            (int(index), _pose_copy(pose, "history pose"), bool(recovered))
            for index, pose, recovered in self.valid_pose_history
        ]

    def clone(self, branch: str) -> "TrackerBranchState":
        return TrackerBranchState(branch, self.last_valid_estimate, self.valid_pose_history)


@dataclass(frozen=True)
class TrackStep:
    branch: str
    frame_id: int
    output_idx: int
    pose: np.ndarray
    accepted: bool
    recovered: bool
    tracking_status: str
    init_source: str
    prediction_guard: str
    acceptance_reason: str
    termination_reason: str
    target_global_ids: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _frozen_array(self.pose, np.float64, (4, 4), "pose"))
        object.__setattr__(
            self, "target_global_ids",
            _frozen_array(self.target_global_ids, np.int64, (None,), "target_global_ids"),
        )


@dataclass(frozen=True)
class VerificationConfig:
    horizon_frames: int = DEFAULT_HORIZON
    min_common_valid_frames: int = 2
    max_cost_ratio: float = 0.90
    min_fitness_retention: float = 0.95
    min_corr_retention: float = 0.95
    common_target_max_points: int | None = None

    def __post_init__(self) -> None:
        if self.horizon_frames < 1 or self.min_common_valid_frames < 1:
            raise ValueError("horizon and minimum evidence must be positive")
        if self.min_common_valid_frames > self.horizon_frames:
            raise ValueError("minimum evidence cannot exceed the horizon")
        if self.max_cost_ratio <= 0.0 or self.min_fitness_retention < 0.0 or self.min_corr_retention < 0.0:
            raise ValueError("decision ratios must be non-negative and cost ratio positive")
        if self.common_target_max_points is not None and self.common_target_max_points < 1:
            raise ValueError("common_target_max_points must be positive")


def observation_schema() -> list[str]:
    return [item.name for item in fields(TrackingObservation)]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_separate_output(output_dir: str | Path, parent_dir: str | Path) -> None:
    output = Path(output_dir).resolve()
    parent = Path(parent_dir).resolve()
    if output == parent or parent in output.parents:
        raise ValueError("horizon verification output must be outside the parent localization directory")


def load_paired_hypotheses(
    paired_npz: str | Path,
    *,
    policy: str = DEFAULT_POLICY,
    budget: int = DEFAULT_BUDGET,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    path = Path(paired_npz)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "frame_id", "timestamp", "pair_id", "policy", "seed", "budget",
            "primary_pose", "candidate_pose", "parent_trajectory_sha256",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"paired artifact missing fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required[:-1]}
        parent_hash = str(np.asarray(archive["parent_trajectory_sha256"]).item())
    count = int(arrays["frame_id"].size)
    if arrays["primary_pose"].shape != (count, 4, 4) or arrays["candidate_pose"].shape != (count, 4, 4):
        raise ValueError("paired pose arrays have inconsistent shapes")
    selected = []
    expected_seed = -1 if seed is None else int(seed)
    for index in range(count):
        if str(arrays["policy"][index]) != policy or int(arrays["budget"][index]) != int(budget):
            continue
        if int(arrays["seed"][index]) != expected_seed:
            continue
        selected.append({
            "row_index": index, "frame_id": int(arrays["frame_id"][index]),
            "timestamp": float(arrays["timestamp"][index]), "pair_id": str(arrays["pair_id"][index]),
            "policy": policy, "seed": expected_seed, "budget": int(budget),
            "primary_pose": _pose_copy(arrays["primary_pose"][index], "primary_pose"),
            "candidate_pose": _pose_copy(arrays["candidate_pose"][index], "candidate_pose"),
        })
    return selected, parent_hash


def reconstruct_branch_state(
    trajectory_poses: np.ndarray,
    frame_rows: list[dict[str, str]],
    event_frame_id: int,
    endpoint_pose: np.ndarray,
    branch: str,
) -> TrackerBranchState:
    event_positions = [index for index, row in enumerate(frame_rows) if int(row["frame_id"]) == int(event_frame_id)]
    if len(event_positions) != 1:
        raise ValueError(f"event frame {event_frame_id} has {len(event_positions)} frame-metric matches")
    event_index = event_positions[0]
    if trajectory_poses.shape != (len(frame_rows), 4, 4):
        raise ValueError("trajectory and frame metrics must have identical row counts")
    history = []
    for index in range(event_index):
        if _csv_bool(frame_rows[index].get("tracking_valid", "")):
            history.append((index, trajectory_poses[index], _csv_bool(frame_rows[index].get("recovered", ""))))
    history.append((event_index, endpoint_pose, branch == "candidate"))
    return TrackerBranchState(branch, endpoint_pose, history)


class StrictReplayTracker:
    """Small offline replay implementation using production strict-mode primitives."""

    def __init__(self, config: dict[str, Any], map_path: str | Path) -> None:
        from scripts.fsm_localize import (
            _is_finite_pose, _pose_consistency_acceptance, _query_recovery_target,
            _registration_failure, _strict_motion_initialization, _tracking_acceptance,
        )

        self._is_finite_pose = _is_finite_pose
        self._pose_consistency_acceptance = _pose_consistency_acceptance
        self._query_recovery_target = _query_recovery_target
        self._registration_failure = _registration_failure
        self._strict_motion_initialization = _strict_motion_initialization
        self._tracking_acceptance = _tracking_acceptance
        self.config = config
        self.registration = dict(config.get("registration", {}) or {})
        if self.registration.get("evaluation_protocol") != "mvp_strict":
            raise ValueError("horizon replay requires registration.evaluation_protocol='mvp_strict'")
        self.backend = str(self.registration.get("backend", "numpy_robust_point_to_plane"))
        self.dataset = dict(config.get("dataset", config.get("data", {})) or {})
        local = dict(config.get("local_submap", self.registration.get("local_submap", {})) or {})
        if not bool(local.get("enabled", False)):
            raise ValueError("horizon replay requires local_submap.enabled")
        self.local = LocalSubmapConfig(
            near=float(local.get("near", self.dataset.get("depth_min", 0.0))),
            far=float(local.get("far", self.dataset.get("depth_max", 5.0))),
            margin_px=float(local.get("margin", local.get("margin_px", 0.0))),
            radius_margin=float(local.get("radius_margin", 0.0)), max_points=local.get("max_points"),
            sampling_policy=str(local.get("sampling_policy", "legacy")),
            tile_rows=int(local.get("tile_rows", 12)), tile_cols=int(local.get("tile_cols", 16)),
            depth_bins=int(local.get("depth_bins", 8)), frustum_culling=str(local.get("frustum_culling", "radius")),
            block_size_m=float(local.get("block_size_m", 0.3)), projection_pre_cap=local.get("projection_pre_cap"),
            projection_sampling_policy=str(local.get("projection_sampling_policy", "block_coverage")),
            block_test_chunk=int(local.get("block_test_chunk", 65536)), oracle_mode="off",
        )
        self.local_min_points = int(local.get("min_points", 1))
        fsm_map = load_fsm_map(str(map_path))
        self.points = np.asarray(fsm_map["means"], dtype=np.float64)
        self.covariances = np.asarray(fsm_map["covariances"], dtype=np.float64) if "covariances" in fsm_map else None
        self.normals = np.asarray(fsm_map["normals"], dtype=np.float64) if "normals" in fsm_map else None
        scores = np.asarray(fsm_map.get("scores", []), dtype=np.float64)
        self.scores = scores if scores.shape == (self.points.shape[0],) else None
        self.global_ids = np.asarray(fsm_map.get("indices", np.arange(self.points.shape[0])), dtype=np.int64)
        self.index = MapSpatialIndex(
            self.points, backend=str(local.get("spatial_backend", "auto")),
            voxel_size=float(local.get("grid_voxel", local.get("grid_voxel_size", 1.0))),
        )
        self.recovery = dict(config.get("recovery", self.registration.get("recovery", {})) or {})
        self.motion = config.get("motion_model", {}) or {}

    def make_observation(self, frame: dict[str, Any], output_idx: int) -> TrackingObservation:
        return TrackingObservation.from_frame(frame, output_idx, self.dataset, int(self.config.get("seed", 0)))

    def _attempt(
        self, observation: TrackingObservation, state: TrackerBranchState,
        attempt_pose: np.ndarray, tier: int, backend: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        query = self._query_recovery_target(
            self.index, attempt_pose, observation.intrinsics, observation.depth.shape[0], observation.depth.shape[1],
            self.local, self.scores, None, tier, self.recovery,
        )
        indices = np.asarray(query["indices"], dtype=np.int64)
        if indices.size < self.local_min_points:
            reason = "empty_local_submap" if indices.size == 0 else "undersized_local_submap"
            return self._registration_failure(attempt_pose, observation.source_points.shape[0], indices.size, reason), indices
        target_points = self.points[indices]
        same = state.cache.local_indices is not None and np.array_equal(indices, state.cache.local_indices)
        tree = state.cache.local_tree if same else build_target_index(target_points)["target_tree"]
        selected_backend = str(backend or self.backend)
        normal_source = str(self.registration.get("normal_source", "map"))
        target_normals = self.normals[indices] if self.normals is not None else None
        if normal_source in ("estimate", "local_pca") and selected_backend != "numpy_point_to_point":
            target_normals = None
        if normal_source == "local_pca" and selected_backend != "numpy_point_to_point":
            if same and state.cache.local_normals is not None:
                target_normals = state.cache.local_normals
            else:
                target_normals = estimate_normals(
                    target_points, k=int(self.registration.get("normal_k", self.registration.get("normal_max_nn", 30))),
                    target_tree=tree,
                )
            state.cache.local_indices = indices.copy()
            state.cache.local_tree = tree
            state.cache.local_normals = target_normals
        result = register_pointcloud(
            source_points=observation.source_points, target_points=target_points, init_transform=attempt_pose,
            backend=selected_backend, voxel_size=self.registration.get("voxel_size"),
            max_correspondence_distance=float(self.registration.get("max_correspondence_distance", 0.05)),
            max_iterations=int(self.registration.get("max_iterations", 30)),
            normal_radius=self.registration.get("normal_radius"),
            normal_max_nn=int(self.registration.get("normal_max_nn", 30)),
            target_covariances=self.covariances[indices] if self.covariances is not None else None,
            target_normals=target_normals, target_tree=tree,
            robust_loss=str(self.registration.get("robust_loss", "huber")),
            loss_scale=float(self.registration.get("loss_scale", 0.02)),
            lm_damping=float(self.registration.get("lm_damping", 1e-3)),
            min_correspondences=int(self.registration.get("min_correspondences", 6)),
            normal_k=int(self.registration.get("normal_k", 20)), step_tolerance=self.registration.get("step_tolerance"),
            cost_tolerance=self.registration.get("cost_tolerance"),
            rotation_step_tolerance_deg=float(self.registration.get("rotation_step_tolerance_deg", 0.01)),
            translation_step_tolerance_m=float(self.registration.get("translation_step_tolerance_m", 1e-4)),
            relative_cost_tolerance=float(self.registration.get("relative_cost_tolerance", 1e-5)),
            convergence_patience=int(self.registration.get("convergence_patience", 2)),
            min_iterations=int(self.registration.get("min_iterations", 2)),
            rank_tolerance=float(self.registration.get("rank_tolerance", 1e-9)),
        )
        return result, indices

    def track(self, observation: TrackingObservation, state: TrackerBranchState) -> TrackStep:
        registration = self.registration
        motion_type = str(registration.get("motion_model", self.motion.get("type", "previous_pose")))
        init, init_source, _, _, guard = self._strict_motion_initialization(
            motion_type, state.valid_pose_history, observation.output_idx,
            float(registration.get("prediction_max_translation_m", self.motion.get("max_translation_m", float("inf")))),
            float(registration.get("prediction_max_rotation_deg", self.motion.get("max_rotation_deg", float("inf")))),
            float(registration.get("prediction_max_total_translation_m", float("inf"))),
            float(registration.get("prediction_max_total_rotation_deg", float("inf"))),
            int(registration.get("prediction_max_horizon_frames", DEFAULT_HORIZON)),
        )
        gap = observation.output_idx - state.valid_pose_history[-1][0]
        result, target_ids = self._attempt(observation, state, init, 0)
        accepted, reason = self._tracking_acceptance(result, dict(registration.get("tracking_acceptance", {}) or {}), self.backend)
        consistency = dict(registration.get("pose_consistency", {}) or {})
        if accepted and bool(consistency.get("enabled", False)):
            accepted, reason, _ = self._pose_consistency_acceptance(
                init, result["transform"], state.last_valid_estimate, init_source, gap, consistency,
            )
        recovered = False
        recovery_backend = None
        if not accepted and bool(self.recovery.get("enabled", False)):
            for tier in range(1, int(self.recovery.get("max_tier", 0)) + 1):
                result, target_ids = self._attempt(observation, state, state.last_valid_estimate, tier)
                accepted, reason = self._tracking_acceptance(
                    result, dict(registration.get("tracking_acceptance", {}) or {}), self.backend,
                )
                if accepted and bool(consistency.get("enabled", False)):
                    accepted, reason, _ = self._pose_consistency_acceptance(
                        state.last_valid_estimate, result["transform"], state.last_valid_estimate,
                        "previous_pose", gap, consistency, float(self.recovery.get("correction_limit_multiplier", 1.0)),
                    )
                if accepted:
                    recovered = True
                    break
            verifier = self.recovery.get("verifier_backend")
            if not accepted and verifier:
                result, target_ids = self._attempt(observation, state, state.last_valid_estimate, 3, str(verifier))
                accepted, reason = self._tracking_acceptance(
                    result, dict(self.recovery.get("verifier_acceptance", {}) or {}), str(verifier),
                )
                if accepted and bool(consistency.get("enabled", False)):
                    accepted, reason, _ = self._pose_consistency_acceptance(
                        state.last_valid_estimate, result["transform"], state.last_valid_estimate,
                        "previous_pose", gap, consistency, float(self.recovery.get("correction_limit_multiplier", 1.0)),
                    )
                recovered = accepted
                recovery_backend = str(verifier) if accepted else None
        valid = accepted and self._is_finite_pose(
            result.get("transform"), float(registration.get("max_pose_translation_norm", 1e6)),
        )
        if valid:
            pose = np.asarray(result["transform"], dtype=np.float64)
            state.last_valid_estimate = pose.copy()
            state.valid_pose_history.append((observation.output_idx, pose.copy(), recovered))
            status = "tracked"
        else:
            pose = state.last_valid_estimate.copy()
            status = "lost_hold_last_valid"
        return TrackStep(
            branch=state.branch, frame_id=observation.frame_id, output_idx=observation.output_idx,
            pose=pose, accepted=valid, recovered=recovered, tracking_status=status,
            init_source=init_source, prediction_guard=guard,
            acceptance_reason="" if valid else str(reason or result.get("failure_reason") or "rejected"),
            termination_reason=str(result.get("termination_reason", recovery_backend or "")),
            target_global_ids=self.global_ids[target_ids],
        )

    def common_metrics(
        self, observation: TrackingObservation, primary: TrackStep, candidate: TrackStep,
        max_points: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        ids = np.union1d(primary.target_global_ids, candidate.target_global_ids)
        if max_points is not None and ids.size > max_points:
            if self.scores is None:
                ids = ids[:max_points]
            else:
                score_lookup = {int(value): float(self.scores[index]) for index, value in enumerate(self.global_ids)}
                scores = np.asarray([score_lookup[int(value)] for value in ids], dtype=np.float64)
                order = np.lexsort((ids, -scores))
                ids = np.sort(ids[order[:max_points]])
        lookup = {int(value): index for index, value in enumerate(self.global_ids)}
        positions = np.asarray([lookup[int(value)] for value in ids], dtype=np.int64)
        points = self.points[positions]
        tree = build_target_index(points)["target_tree"]
        normals = self.normals[positions] if self.normals is not None else estimate_normals(
            points, k=int(self.registration.get("normal_k", self.registration.get("normal_max_nn", 30))),
            target_tree=tree,
        )
        kwargs = dict(
            source_points=observation.source_points, target_points=points, target_normals=normals,
            target_tree=tree, max_correspondence_distance=float(self.registration.get("max_correspondence_distance", 0.05)),
            robust_loss=str(self.registration.get("robust_loss", "huber")),
            loss_scale=float(self.registration.get("loss_scale", 0.02)),
            min_correspondences=int(self.registration.get("min_correspondences", 6)),
            rank_tolerance=float(self.registration.get("rank_tolerance", 1e-9)),
        )
        return (
            evaluate_common_target_pose(pose=primary.pose, **kwargs),
            evaluate_common_target_pose(pose=candidate.pose, **kwargs), int(ids.size),
        )


def verify_event(
    event: dict[str, Any], observations: Iterable[TrackingObservation],
    primary_state: TrackerBranchState, candidate_state: TrackerBranchState,
    track_fn: Callable[[TrackingObservation, TrackerBranchState], TrackStep],
    common_fn: Callable[[TrackingObservation, TrackStep, TrackStep, int | None], tuple[dict[str, Any], dict[str, Any], int]],
    decision: VerificationConfig = VerificationConfig(),
    pose_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_rows = []
    primary_lost = candidate_lost = 0
    cost_ratios = []
    fitness_ratios = []
    corr_ratios = []
    for offset, observation in enumerate(observations, start=1):
        try:
            primary = track_fn(observation, primary_state)
        except Exception as exc:
            primary = _exception_step("primary", observation, primary_state, exc)
        try:
            candidate = track_fn(observation, candidate_state)
        except Exception as exc:
            candidate = _exception_step("candidate", observation, candidate_state, exc)
        primary_lost += int(not primary.accepted)
        candidate_lost += int(not candidate.accepted)
        primary_common = candidate_common = None
        common_count = 0
        if primary.accepted and candidate.accepted:
            primary_common, candidate_common, common_count = common_fn(
                observation, primary, candidate, decision.common_target_max_points,
            )
            cost_ratios.append(_ratio(candidate_common["robust_cost_per_corr"], primary_common["robust_cost_per_corr"]))
            fitness_ratios.append(_ratio(candidate_common["fitness"], primary_common["fitness"]))
            corr_ratios.append(_ratio(candidate_common["corr_count"], primary_common["corr_count"]))
        separation = relative_se3_log(primary.pose, candidate.pose)
        for step, common in ((primary, primary_common), (candidate, candidate_common)):
            if pose_rows is not None:
                pose_rows.append({
                    "pair_id": event["pair_id"], "offset": offset,
                    "frame_id": observation.frame_id, "timestamp": observation.timestamp,
                    "branch": step.branch, "accepted": step.accepted,
                    "pose": np.asarray(step.pose, dtype=np.float64).copy(),
                })
            step_rows.append({
                "pair_id": event["pair_id"], "offset": offset, "frame_id": observation.frame_id,
                "branch": step.branch, "accepted": step.accepted, "recovered": step.recovered,
                "tracking_status": step.tracking_status, "init_source": step.init_source,
                "prediction_guard": step.prediction_guard, "acceptance_reason": step.acceptance_reason,
                "termination_reason": step.termination_reason, "common_target_count": common_count,
                "common_robust_cost_per_corr": _finite_or_blank(common, "robust_cost_per_corr"),
                "common_point_to_plane_rmse": _finite_or_blank(common, "point_to_plane_rmse"),
                "common_fitness": _finite_or_blank(common, "fitness"),
                "common_corr_count": "" if common is None else int(common["corr_count"]),
                "pose_separation_translation_m": separation["translation_norm_m"],
                "pose_separation_rotation_deg": separation["rotation_norm_deg"],
            })
    observed = len(step_rows) // 2
    common_valid = len(cost_ratios)
    status = "complete" if observed == decision.horizon_frames else "right_censored"
    selected = "primary"
    reason = "primary_default"
    if observed < decision.horizon_frames:
        status, reason = "inconclusive", "right_censored_horizon"
    elif candidate_lost < primary_lost:
        selected, reason = "candidate", "fewer_lost_frames"
    elif candidate_lost > primary_lost:
        reason = "more_candidate_lost_frames"
    elif common_valid < decision.min_common_valid_frames:
        status, reason = "inconclusive", "insufficient_common_valid_frames"
    else:
        median_cost = float(np.median(cost_ratios))
        median_fitness = float(np.median(fitness_ratios))
        median_corr = float(np.median(corr_ratios))
        if (
            median_cost <= decision.max_cost_ratio
            and median_fitness >= decision.min_fitness_retention
            and median_corr >= decision.min_corr_retention
        ):
            selected, reason = "candidate", "common_target_improvement"
        else:
            reason = "candidate_improvement_not_conservative"
    summary = {
        "pair_id": event["pair_id"], "frame_id": event["frame_id"], "policy": event["policy"],
        "seed": "" if event["seed"] < 0 else event["seed"], "budget": event["budget"],
        "horizon_requested": decision.horizon_frames, "horizon_observed": observed,
        "status": status, "decision": selected, "decision_reason": reason,
        "primary_lost_count": primary_lost, "candidate_lost_count": candidate_lost,
        "common_valid_count": common_valid,
        "median_common_cost_ratio": _median_or_blank(cost_ratios),
        "median_common_fitness_ratio": _median_or_blank(fitness_ratios),
        "median_common_corr_retention": _median_or_blank(corr_ratios),
        "primary_final_pose_sha256": _pose_hash(primary_state.last_valid_estimate),
        "candidate_final_pose_sha256": _pose_hash(candidate_state.last_valid_estimate),
    }
    return summary, step_rows


def save_results(
    output_dir: str | Path, events: list[dict[str, Any]], steps: list[dict[str, Any]],
    provenance: dict[str, Any], pose_rows: list[dict[str, Any]] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "pair_id", "frame_id", "policy", "seed", "budget", "horizon_requested", "horizon_observed",
        "status", "decision", "decision_reason", "primary_lost_count", "candidate_lost_count",
        "common_valid_count", "median_common_cost_ratio", "median_common_fitness_ratio",
        "median_common_corr_retention", "primary_final_pose_sha256", "candidate_final_pose_sha256",
    ]
    step_fields = [
        "pair_id", "offset", "frame_id", "branch", "accepted", "recovered", "tracking_status",
        "init_source", "prediction_guard", "acceptance_reason", "termination_reason", "common_target_count",
        "common_robust_cost_per_corr", "common_point_to_plane_rmse", "common_fitness", "common_corr_count",
        "pose_separation_translation_m", "pose_separation_rotation_deg",
    ]
    _write_csv(output / "horizon_events.csv", event_fields, events)
    _write_csv(output / "horizon_steps.csv", step_fields, steps)
    if pose_rows is not None:
        _save_branch_poses(output, pose_rows)
        provenance = dict(provenance)
        provenance["branch_pose_npz_sha256"] = sha256_file(output / "horizon_branch_poses.npz")
        provenance["branch_pose_csv_sha256"] = sha256_file(output / "horizon_branch_poses.csv")
    with open(output / "provenance.json", "w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, names: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _save_branch_poses(output: Path, rows: list[dict[str, Any]]) -> None:
    count = len(rows)
    poses = np.empty((count, 4, 4), dtype=np.float64)
    csv_rows = []
    for index, row in enumerate(rows):
        poses[index] = _pose_copy(row["pose"], "branch pose")
        csv_row = {
            "row_index": index, "pair_id": str(row["pair_id"]), "offset": int(row["offset"]),
            "frame_id": int(row["frame_id"]), "timestamp": f"{float(row['timestamp']):.9f}",
            "branch": str(row["branch"]), "accepted": bool(row["accepted"]),
        }
        csv_row.update({f"pose_{r}{c}": float(poses[index, r, c]) for r in range(4) for c in range(4)})
        csv_rows.append(csv_row)
    np.savez(
        output / "horizon_branch_poses.npz",
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        row_index=np.arange(count, dtype=np.int64),
        pair_id=np.asarray([str(row["pair_id"]) for row in rows], dtype="U128"),
        offset=np.asarray([int(row["offset"]) for row in rows], dtype=np.int64),
        frame_id=np.asarray([int(row["frame_id"]) for row in rows], dtype=np.int64),
        timestamp=np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64),
        branch=np.asarray([str(row["branch"]) for row in rows], dtype="U16"),
        accepted=np.asarray([bool(row["accepted"]) for row in rows], dtype=np.bool_),
        pose=poses,
    )
    fields = ["row_index", "pair_id", "offset", "frame_id", "timestamp", "branch", "accepted"]
    fields.extend(f"pose_{r}{c}" for r in range(4) for c in range(4))
    _write_csv(output / "horizon_branch_poses.csv", fields, csv_rows)


def _pose_copy(value: Any, name: str) -> np.ndarray:
    pose = np.array(value, dtype=np.float64, copy=True)
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose")
    return pose


def _pose_hash(pose: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pose, dtype=np.float64).tobytes()).hexdigest()


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _ratio(numerator: Any, denominator: Any) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("inf")
    return numerator / denominator


def _finite_or_blank(values: dict[str, Any] | None, name: str) -> Any:
    if values is None:
        return ""
    value = float(values[name])
    return value if np.isfinite(value) else ""


def _median_or_blank(values: list[float]) -> Any:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else ""


def _exception_step(
    branch: str, observation: TrackingObservation, state: TrackerBranchState, exc: Exception,
) -> TrackStep:
    return TrackStep(
        branch=branch, frame_id=observation.frame_id, output_idx=observation.output_idx,
        pose=state.last_valid_estimate, accepted=False, recovered=False,
        tracking_status="exception_hold_last_valid", init_source="", prediction_guard="",
        acceptance_reason=f"{type(exc).__name__}: {exc}", termination_reason="exception",
        target_global_ids=np.empty(0, dtype=np.int64),
    )
