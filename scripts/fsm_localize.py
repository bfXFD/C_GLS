import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from copy import copy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_activation_diagnostics import RuntimeActivationDiagnostics
from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_map import load_fsm_map
from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_metrics import compute_ate_rmse, compute_long_horizon_metrics, compute_rpe_rmse, save_tum_trajectory
from utils.fsm_final_target_oracle import FinalTargetOracleConfig, FinalTargetOracleRuntime
from utils.fsm_paired_reserve import BaselineSnapshot, PairedShadowReserve
from utils.fsm_registration import build_target_index, compose_transform, register_pointcloud
from utils.fsm_robust_registration import estimate_normals
from utils.fsm_robust_registration import se3_exp
from utils.fsm_candidate_verification import (
    FrozenCandidateVerifier, VERIFIER_METRIC_FIELDS, split_source_validation,
    validate_verifier_config,
)
from utils.fsm_runtime_memory import get_process_memory_mb
from utils.fsm_runtime_reserve import (
    RuntimeObservableReserve,
    RuntimeReserveObservation,
    evaluate_common_target_pose,
    relative_se3_log,
    se3_log,
)


AUDIT_FRAME_METRIC_FIELDS = [
    "frame_id",
    "timestamp",
    "fitness",
    "inlier_rmse",
    "latency_ms",
    "num_source_points",
    "num_target_points",
    "converged",
    "fallback_used",
]
STRICT_FRAME_METRIC_FIELDS = AUDIT_FRAME_METRIC_FIELDS + [
    "accepted",
    "optimizer_valid",
    "termination_reason",
    "acceptance_reason",
    "tracking_valid",
    "tracking_lost",
    "tracking_status",
    "init_source",
    "prediction_translation_m",
    "prediction_rotation_deg",
    "prediction_guard",
    "local_query_latency_ms",
    "local_radius_latency_ms",
    "local_projection_latency_ms",
    "local_sampling_latency_ms",
    "total_frame_latency_ms",
    "production_equivalent_total_frame_latency_ms",
    "local_index_build_ms",
    "block_index_build_ms",
    "block_index_memory_bytes",
    "local_candidate_count",
    "local_culling_mode",
    "local_tested_block_count",
    "local_intersecting_block_count",
    "local_block_candidate_count",
    "local_projection_input_count",
    "local_projection_cap",
    "local_projection_cap_applied",
    "local_projection_cap_bypassed",
    "local_projection_cap_utilization",
    "local_block_culling_latency_ms",
    "local_precap_latency_ms",
    "local_precap_strategy",
    "local_representative_points_considered",
    "local_precap_block_count",
    "local_precap_selected_block_count",
    "local_precap_block_coverage",
    "local_precap_bucket_count",
    "local_precap_selected_bucket_count",
    "local_precap_bucket_coverage",
    "local_precap_boundary_block_count",
    "local_precap_selected_boundary_block_count",
    "local_precap_boundary_coverage",
    "local_oracle_latency_ms",
    "local_oracle_count",
    "local_oracle_block_recall",
    "local_oracle_projection_recall",
    "local_oracle_frustum_recall",
    "local_oracle_overflow",
    "local_oracle_status",
    "local_oracle_reason",
    "local_frustum_count",
    "local_positive_depth_count",
    "local_depth_range_count",
    "local_precap_count",
    "local_final_sampling_input_count",
    "local_nonempty_tile_count",
    "local_selected_tile_count",
    "local_nonempty_depth_bin_count",
    "local_selected_depth_bin_count",
    "local_nonempty_tile_depth_count",
    "local_selected_tile_depth_count",
    "local_target_count",
    "local_target_fraction",
    "failure_reason",
    "primary_termination_reason",
    "primary_rejection_reason",
    "primary_optimizer_valid",
    "primary_fitness",
    "primary_inlier_rmse",
    "primary_point_to_plane_rmse",
    "primary_num_correspondences",
    "primary_effective_rank",
    "primary_condition_number",
    "primary_lambda_min",
    "primary_hessian_trace",
    "primary_normalized_lambda_min",
    "primary_consistency_reason",
    "primary_correction_translation_m",
    "primary_correction_rotation_deg",
    "primary_correction_translation_limit_m",
    "primary_correction_rotation_limit_deg",
    "primary_temporal_translation_m",
    "primary_temporal_rotation_deg",
    "primary_temporal_translation_limit_m",
    "primary_temporal_rotation_limit_deg",
    "recovery_attempted",
    "recovery_stage",
    "recovery_count",
    "recovered",
    "recovery_backend",
    "final_consistency_reason",
    "final_correction_translation_m",
    "final_correction_rotation_deg",
    "final_correction_translation_limit_m",
    "final_correction_rotation_limit_deg",
    "final_temporal_translation_m",
    "final_temporal_rotation_deg",
    "final_temporal_translation_limit_m",
    "final_temporal_rotation_limit_deg",
    "temporal_gap",
    "final_failure_reason",
    "iterations",
    "num_correspondences",
    "point_to_plane_rmse",
    "lambda_min",
    "condition_number",
    "effective_rank",
    "pose_covariance_diagonal",
    "final_step_rotation_deg",
    "final_step_translation_m",
    "initial_mean_robust_cost",
    "final_mean_robust_cost",
    "last_relative_reduction",
    "accepted_step_count",
    "rejected_lm_trials",
    "observable_reserve_attempt",
    "paired_triggered",
    "pair_id",
    "arm_count",
    "shadow_latency_ms",
    "final_target_oracle_status",
    "final_target_oracle_latency_ms",
    "reserve_attempted",
    "reserve_triggered",
    "reserve_trigger_reasons",
    "reserve_trigger_unavailable_reason",
    "reserve_visible",
    "reserve_selected",
    "reserve_correspondence",
    "reserve_candidate_optimizer_valid",
    "reserve_candidate_fitness",
    "reserve_candidate_rmse",
    "reserve_candidate_effective_rank",
    "reserve_candidate_hessian_trace",
    "reserve_candidate_normalized_lambda_min",
    "reserve_selection_score",
    "reserve_primary_condition",
    "reserve_augmented_condition",
    "reserve_primary_lambda_min",
    "reserve_augmented_lambda_min",
    "reserve_accepted",
    "reserve_used",
    "reserve_primary_accepted",
    "reserve_tracking_accepted",
    "reserve_consistency_accepted",
    "reserve_relative_improvement_accepted",
    "reserve_relative_improvement_protocol",
    "reserve_relative_improvement_mode",
    "reserve_relative_condition_available",
    "reserve_relative_condition_pass",
    "reserve_relative_condition_ratio",
    "reserve_relative_lambda_available",
    "reserve_relative_lambda_pass",
    "reserve_relative_lambda_ratio",
    "reserve_relative_improvement_reasons",
    "reserve_information_v3_branch",
    "reserve_information_v3_condition_base_pass",
    "reserve_information_v3_lambda_base_pass",
    "reserve_information_v3_condition_strong_pass",
    "reserve_information_v3_lambda_strong_pass",
    "reserve_information_v3_refinement_translation_pass",
    "reserve_information_v3_refinement_rotation_pass",
    "reserve_information_v3_prospective_streak",
    "reserve_information_v3_cooldown_remaining_frames",
    "reserve_information_v3_accepted",
    "reserve_information_v3_reason",
    "reserve_stability_gate_enabled",
    "reserve_stability_gate_mode",
    "reserve_stability_would_accept",
    "reserve_stability_reasons",
    "reserve_stability_accepted",
    "reserve_stability_latency_ms",
    "reserve_refinement_twist_rx",
    "reserve_refinement_twist_ry",
    "reserve_refinement_twist_rz",
    "reserve_refinement_twist_tx",
    "reserve_refinement_twist_ty",
    "reserve_refinement_twist_tz",
    "reserve_refinement_twist_norm",
    "reserve_refinement_rotation_deg",
    "reserve_refinement_translation_m",
    "reserve_candidate_init_twist_norm",
    "reserve_candidate_init_rotation_deg",
    "reserve_candidate_init_translation_m",
    "reserve_candidate_temporal_rotation_deg",
    "reserve_information_v4_candidate_init_translation_pass",
    "reserve_information_v4_candidate_init_rotation_pass",
    "reserve_information_v4_joint_rotation_pass",
    "reserve_information_v4_prospective_global_streak",
    "reserve_information_v4_global_cooldown_remaining_frames",
    "reserve_information_v4_global_streak_pass",
    "reserve_information_v4_global_cooldown_pass",
    "reserve_information_v4_accepted",
    "reserve_information_v4_reason",
    "reserve_different_target_fitness_ratio",
    "reserve_different_target_ptp_rmse_ratio",
    "reserve_different_target_robust_cost_ratio",
    "reserve_common_target_cost_ratio",
    "reserve_common_target_rmse_ratio",
    "reserve_common_target_fitness_ratio",
    "reserve_common_target_corr_retention",
    "reserve_stability_rolling_translation_l1_m",
    "reserve_stability_prospective_streak",
    "reserve_stability_cooldown_remaining_frames",
    "reserve_adoption_reason",
    "reserve_plan_pose_source",
    "reserve_rerun_init_source",
    "reserve_plan_normal_source",
    "reserve_rerun_normal_source",
    "reserve_corr_policy",
    "reserve_corr_status",
    "reserve_corr_reason",
    "reserve_corr_source_count",
    "reserve_corr_normal_valid_rate",
    "reserve_corr_distance_valid_rate",
    "reserve_corr_eligible",
    "reserve_corr_p_mean",
    "reserve_corr_p_nonzero_rate",
    "reserve_corr_selected_p_mean",
    "reserve_corr_selected_p10",
    "reserve_corr_selected_p_min",
    "reserve_corr_adaptive_k",
    "reserve_corr_characteristic_length",
    "reserve_corr_latency_ms",
    "reserve_corr_normal_source",
    "reserve_corr_distance_source",
    "reserve_hessian_rotation_trace",
    "reserve_hessian_translation_trace",
    "reserve_hessian_cross_frobenius",
    "reserve_hessian_rotation_lambda_min",
    "reserve_hessian_translation_lambda_min",
    "reserve_weak_rotation_energy",
    "reserve_weak_translation_energy",
    "reserve_weak_dominant_block",
    "reserve_latency_ms",
    "reserve_backend",
    "rss_mb",
    "peak_rss_mb",
]

STRICT_FRAME_METRIC_FIELDS += list(VERIFIER_METRIC_FIELDS) + [
    "reserve_verifier_latency_ms", "reserve_candidate_target_mode",
    "reserve_candidate_base_removed", "reserve_candidate_target_count",
    "reserve_spectral_coordinates", "reserve_spectral_length",
    "reserve_spectral_primary_lambda", "reserve_spectral_candidate_lambda",
    "reserve_spectral_primary_condition", "reserve_spectral_candidate_condition",
    "source_training_count", "source_validation_count",
]

EVALUATION_PROTOCOLS = ("audit_legacy", "mvp_strict")
FIRST_POSE_PROTOCOLS = ("gt_first_frame", "identity", "explicit")
FROZEN_AUDIT_DIRNAME = "audit_v2_full_rerun"


def parse_args():
    parser = argparse.ArgumentParser(description="Localize RGB-D frames against an FSM map.")
    parser.add_argument("--config", required=True, help="Path to FSM localization config.")
    parser.add_argument("--map", required=True, help="Path to fsm_map.npz.")
    parser.add_argument("--output-dir", required=True, help="Output directory for trajectories and metrics.")
    parser.add_argument("--start", type=int, default=None, help="Start frame index override.")
    parser.add_argument("--end", type=int, default=None, help="End frame index override, exclusive.")
    parser.add_argument("--backend", default=None, help="Registration backend override.")
    parser.add_argument("--protocol", choices=EVALUATION_PROTOCOLS, default=None, help="Evaluation protocol override.")
    parser.add_argument(
        "--motion-model",
        choices=("previous_pose", "constant_velocity"),
        default=None,
        help="Strict-mode motion model override.",
    )
    parser.add_argument(
        "--first-pose-protocol",
        choices=FIRST_POSE_PROTOCOLS,
        default=None,
        help="Strict-mode first-pose protocol override.",
    )
    parser.add_argument(
        "--initial-pose",
        default=None,
        help="Path to a 4x4 .npy, .npz, JSON, or whitespace text pose for first-pose-protocol=explicit.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    if not hasattr(experiment, "config"):
        raise AttributeError(f"Config file {config_path} must define a variable named 'config'")
    return experiment.config


def _config_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def _frame_range(num_frames: int, start: int | None, end: int | None) -> range:
    start = 0 if start is None else int(start)
    end = num_frames if end is None or int(end) < 0 else int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame range start={start}, end={end}")
    return range(start, min(end, num_frames))


def _write_frame_metrics(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ReserveSelectionEventWriter:
    FIELDS = (
        "frame_id", "rank", "selected_global_id", "selection_score", "p_corr",
        "attempted", "used", "selection_policy",
    )

    def __init__(self, output_root: Path, config: dict[str, Any], reserve: RuntimeObservableReserve, map_path: str):
        artifact_cfg = config.get("observable_reserve", {}).get("selection_artifact", {}) or {}
        if not isinstance(artifact_cfg, dict):
            raise ValueError("observable_reserve.selection_artifact must be a dictionary")
        self.enabled = bool(reserve.active and artifact_cfg.get("enabled", False))
        self.flush_each_frame = bool(artifact_cfg.get("flush_each_frame", True))
        self.stream = None
        self.writer = None
        if not self.enabled:
            return
        self.stream = open(output_root / "reserve_selection_events.csv", "w", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        reserve_cfg = config.get("observable_reserve", {}) or {}
        canonical = json.dumps(reserve_cfg, sort_keys=True, separators=(",", ":"), allow_nan=False)
        provenance = {
            "schema_version": 1,
            "metric_definition_version": "reserve_selection_events_v1",
            "row_semantics": "one row per selected reserve Gaussian on each triggered plan",
            "id_space": "source params global Gaussian IDs",
            "selection_policy": reserve.selection,
            "budget": reserve.budget,
            "map_path": str(Path(map_path).resolve()),
            "map_sha256": _sha256_file(map_path),
            "reserve_geometry_path": reserve.geometry_path,
            "reserve_geometry_sha256": (
                _sha256_file(reserve.geometry_path)
                if reserve.geometry_path and Path(reserve.geometry_path).is_file() else None
            ),
            "observable_reserve_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        with open(output_root / "reserve_selection_events_provenance.json", "w") as stream:
            json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    def write(self, frame_id: int, plan: Any, metrics: dict[str, Any]) -> None:
        if not self.enabled or self.writer is None or not plan.selected_indices.size:
            return
        for rank in range(plan.selected_indices.size):
            self.writer.writerow({
                "frame_id": int(frame_id),
                "rank": rank,
                "selected_global_id": int(plan.selected_global_ids[rank]),
                "selection_score": float(plan.selected_scores[rank]),
                "p_corr": "" if plan.selected_probabilities is None else float(plan.selected_probabilities[rank]),
                "attempted": bool(metrics.get("reserve_attempted", False)),
                "used": bool(metrics.get("reserve_used", False)),
                "selection_policy": plan.selection_policy,
            })
        if self.flush_each_frame:
            self.stream.flush()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def _write_summary(path: str, summary: dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_json(summary), f, indent=2, sort_keys=True, allow_nan=False)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _tracking_summary_counts(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tracking_rows = frame_rows[1:]
    denominator = len(tracking_rows)
    primary_accepted = sum(bool(row["reserve_primary_accepted"]) for row in tracking_rows)
    base_only = sum(
        bool(row["tracking_valid"])
        and bool(row["reserve_primary_accepted"])
        and not bool(row["reserve_used"])
        and not bool(row["recovered"])
        for row in tracking_rows
    )
    final_complete = sum(bool(row["tracking_valid"]) for row in tracking_rows)

    def rate(numerator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "num_tracking_attempt_frames": denominator,
        "num_primary_accepted_tracking_frames": primary_accepted,
        "num_base_only_final_frames": base_only,
        "num_final_complete_tracking_frames": final_complete,
        "primary_attempt_acceptance_rate": rate(primary_accepted),
        "primary_tracking_completeness": rate(primary_accepted),
        "base_only_final_fraction": rate(base_only),
        "final_tracking_completeness": rate(final_complete),
    }


def _csv_value(value: Any) -> Any:
    sanitized = _sanitize_json(value)
    if sanitized is None:
        return ""
    if isinstance(sanitized, (list, dict)):
        return json.dumps(sanitized, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return sanitized


def _csv_condition_number(value: Any) -> Any:
    if value is not None and np.isinf(float(value)):
        return "inf"
    return _csv_value(value)


def _select_initial_transform(init_mode: str, frame_idx: int, previous_estimate: np.ndarray | None, gt_pose: np.ndarray) -> np.ndarray:
    if init_mode == "identity":
        return np.eye(4, dtype=np.float64)
    if init_mode == "gt_previous":
        if previous_estimate is None:
            return gt_pose.astype(np.float64)
        return previous_estimate.astype(np.float64)
    if init_mode == "previous_estimate":
        if previous_estimate is None:
            return gt_pose.astype(np.float64)
        return previous_estimate.astype(np.float64)
    raise ValueError(f"Unknown init_mode '{init_mode}'")


def _load_initial_pose(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        pose = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if "pose" in archive:
            pose = archive["pose"]
        elif len(archive.files) == 1:
            pose = archive[archive.files[0]]
        else:
            raise ValueError(f"Initial-pose archive {path} must contain a 'pose' array or exactly one array")
    elif suffix == ".json":
        with open(path) as f:
            pose = json.load(f)
    else:
        pose = np.loadtxt(path)
    pose = np.asarray(pose, dtype=np.float64)
    if pose.size == 16:
        pose = pose.reshape(4, 4)
    if not _is_finite_pose(pose, float("inf")):
        raise ValueError(f"Initial pose in {path} must be a finite 4x4 transform")
    return pose


def _strict_first_pose(
    first_pose_protocol: str,
    gt_pose: np.ndarray,
    explicit_initial_pose: np.ndarray | None,
    first_pose_left_twist: np.ndarray | None = None,
) -> np.ndarray:
    if first_pose_protocol == "gt_first_frame":
        pose = gt_pose.astype(np.float64)
    elif first_pose_protocol == "identity":
        pose = np.eye(4, dtype=np.float64)
    elif first_pose_protocol == "explicit":
        if explicit_initial_pose is None:
            raise ValueError("mvp_strict with first_pose_protocol='explicit' requires an explicit initial pose")
        pose = np.asarray(explicit_initial_pose, dtype=np.float64)
        if not _is_finite_pose(pose, float("inf")):
            raise ValueError("explicit_initial_pose must be a finite 4x4 transform")
        pose = pose.copy()
    else:
        raise ValueError(f"Unknown first_pose_protocol '{first_pose_protocol}'")
    if first_pose_left_twist is not None:
        twist = np.asarray(first_pose_left_twist, dtype=np.float64)
        if twist.shape != (6,) or np.any(~np.isfinite(twist)):
            raise ValueError("registration.first_pose_left_twist must contain six finite values")
        pose = se3_exp(twist) @ pose
    return pose


def _stable_replacement_positions(global_ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    count = min(max(int(count), 0), ids.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    unsigned = ids.astype(np.uint64, copy=False)
    keys = unsigned ^ np.uint64(int(seed) & ((1 << 64) - 1))
    keys ^= keys >> np.uint64(30)
    keys *= np.uint64(0xBF58476D1CE4E5B9)
    keys ^= keys >> np.uint64(27)
    keys *= np.uint64(0x94D049BB133111EB)
    keys ^= keys >> np.uint64(31)
    return np.lexsort((ids, keys))[:count].astype(np.int64)


def _candidate_target_arrays(
    *,
    mode: str,
    base_points: np.ndarray,
    base_covariances: np.ndarray | None,
    base_normals: np.ndarray | None,
    base_global_ids: np.ndarray,
    reserve_points: np.ndarray,
    reserve_covariances: np.ndarray | None,
    reserve_normals: np.ndarray,
    replacement_seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    if mode == "rerun_primary":
        return base_points.copy(), None if base_covariances is None else base_covariances.copy(), (
            None if base_normals is None else base_normals.copy()
        ), 0
    keep = np.ones(base_points.shape[0], dtype=bool)
    removed = 0
    if mode == "replace":
        positions = _stable_replacement_positions(base_global_ids, reserve_points.shape[0], replacement_seed)
        keep[positions] = False
        removed = int(positions.size)
    elif mode != "append":
        raise ValueError(f"Unknown candidate target mode {mode!r}")
    points = np.concatenate([base_points[keep], reserve_points], axis=0)
    covariances = None
    if base_covariances is not None and reserve_covariances is not None:
        covariances = np.concatenate([base_covariances[keep], reserve_covariances], axis=0)
    normals = None
    if base_normals is not None:
        normals = np.concatenate([base_normals[keep], reserve_normals], axis=0)
    return points, covariances, normals, removed


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_pose_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Measure candidate motion in the reference pose coordinate frame."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if not _is_finite_pose(reference, float("inf")) or not _is_finite_pose(candidate, float("inf")):
        raise ValueError("reference and candidate must be finite 4x4 poses")
    relative = np.linalg.inv(reference) @ candidate
    return {
        "translation_m": float(np.linalg.norm(relative[:3, 3])),
        "rotation_deg": _rotation_angle_deg(relative[:3, :3]),
    }


def _strict_motion_initialization(
    motion_model: str,
    valid_pose_history: list[tuple[int, np.ndarray] | tuple[int, np.ndarray, bool]],
    current_output_idx: int,
    max_translation_m: float,
    max_rotation_deg: float,
    max_total_translation_m: float = float("inf"),
    max_total_rotation_deg: float = float("inf"),
    prediction_max_horizon_frames: int = 3,
) -> tuple[np.ndarray, str, float, float, str]:
    if not valid_pose_history:
        raise RuntimeError("mvp_strict has no valid pose for initialization")
    last_entry = valid_pose_history[-1]
    last_valid_output_idx, previous_value = last_entry[:2]
    last_was_recovered = bool(last_entry[2]) if len(last_entry) > 2 else False
    previous = np.asarray(previous_value, dtype=np.float64)
    if motion_model == "previous_pose":
        return previous.copy(), "previous_pose", 0.0, 0.0, "not_applicable"
    if motion_model != "constant_velocity":
        raise ValueError("motion_model must be 'previous_pose' or 'constant_velocity'")
    if len(valid_pose_history) < 2:
        return previous.copy(), "previous_pose", 0.0, 0.0, "insufficient_history"
    if last_was_recovered:
        return previous.copy(), "previous_pose", 0.0, 0.0, "nonconsecutive_valid_history"

    gap = int(current_output_idx) - int(last_valid_output_idx)
    if gap < 1:
        raise ValueError("current_output_idx must follow the last valid output index")
    if gap > prediction_max_horizon_frames:
        return previous.copy(), "previous_pose", 0.0, 0.0, "prediction_horizon_exceeded"

    previous_previous_idx, previous_previous_value = valid_pose_history[-2][:2]
    if int(last_valid_output_idx) - int(previous_previous_idx) != 1:
        return previous.copy(), "previous_pose", 0.0, 0.0, "nonconsecutive_valid_history"
    previous_previous = np.asarray(previous_previous_value, dtype=np.float64)
    relative_step = np.linalg.inv(previous_previous) @ previous
    translation_step = float(np.linalg.norm(relative_step[:3, 3]))
    rotation_step = _rotation_angle_deg(relative_step[:3, :3])
    if translation_step > max_translation_m:
        return previous.copy(), "previous_pose", translation_step, rotation_step, "translation_exceeded"
    if rotation_step > max_rotation_deg:
        return previous.copy(), "previous_pose", translation_step, rotation_step, "rotation_exceeded"
    relative_gap = np.eye(4, dtype=np.float64)
    for _ in range(gap):
        relative_gap = relative_gap @ relative_step
    total_translation = float(np.linalg.norm(relative_gap[:3, 3]))
    total_rotation = _rotation_angle_deg(relative_gap[:3, :3])
    if total_translation > max_total_translation_m:
        return previous.copy(), "previous_pose", translation_step, rotation_step, "total_translation_exceeded"
    if total_rotation > max_total_rotation_deg:
        return previous.copy(), "previous_pose", translation_step, rotation_step, "total_rotation_exceeded"
    return previous @ relative_gap, "constant_velocity", translation_step, rotation_step, "accepted"


def _tracking_acceptance(
    result: dict[str, Any], controls: dict[str, Any], backend: str | None = None
) -> tuple[bool, str | None]:
    optimizer_valid = bool(result.get("optimizer_valid", result.get("converged")))
    if not optimizer_valid:
        return False, result.get("failure_reason") or result.get("termination_reason") or "optimizer_invalid"
    if not _is_finite_pose(result.get("transform"), float("inf")):
        return False, "tracking_quality_non_finite_pose"
    backend = str(backend or result.get("backend", ""))
    correspondences = result.get("num_correspondences")
    if correspondences is None:
        correspondences = float(result.get("fitness", 0.0)) * int(result.get("num_source_points", 0))
    eigenvalues = np.asarray(result.get("eigenvalues", []), dtype=np.float64)
    checks = [
        ("min_fitness", float(result.get("fitness", 0.0)), lambda value, limit: value >= limit),
        ("min_correspondences", float(correspondences), lambda value, limit: value >= limit),
        ("max_inlier_rmse", float(result.get("inlier_rmse", float("inf"))), lambda value, limit: value <= limit),
    ]
    if "point_to_plane" in backend:
        checks.extend([
            ("max_point_to_plane_rmse", float(result.get("point_to_plane_rmse", float("inf"))), lambda value, limit: value <= limit),
            ("max_condition_number", float(result.get("condition_number", float("inf"))), lambda value, limit: value <= limit),
            ("min_lambda_min", float(np.min(eigenvalues)) if eigenvalues.size else 0.0, lambda value, limit: value >= limit),
        ])
    for name, value, predicate in checks:
        if name in controls and not predicate(value, float(controls[name])):
            return False, f"tracking_quality_{name}"
    return True, None


def _pose_consistency_limits(
    config: dict[str, Any], init_source: str, gap: int, recovery_multiplier: float
) -> dict[str, float]:
    correction = config.get("correction", {})
    temporal = config.get("temporal", {})
    correction_key = "constant_velocity" if init_source == "constant_velocity" else "bootstrap"
    correction_limits = correction.get(correction_key, {})
    return {
        "correction_translation_m": float(correction_limits.get("translation_m", float("inf"))) * recovery_multiplier,
        "correction_rotation_deg": float(correction_limits.get("rotation_deg", float("inf"))) * recovery_multiplier,
        "temporal_translation_m": min(
            float(temporal.get("translation_m_per_frame", float("inf"))) * gap,
            float(temporal.get("hard_translation_m", float("inf"))),
        ),
        "temporal_rotation_deg": min(
            float(temporal.get("rotation_deg_per_frame", float("inf"))) * gap,
            float(temporal.get("hard_rotation_deg", float("inf"))),
        ),
    }


def _pose_consistency_acceptance(
    attempt_init: np.ndarray,
    candidate: np.ndarray,
    last_valid: np.ndarray,
    init_source: str,
    gap: int,
    config: dict[str, Any],
    recovery_multiplier: float = 1.0,
) -> tuple[bool, str | None, dict[str, float]]:
    correction = relative_pose_metrics(attempt_init, candidate)
    temporal = relative_pose_metrics(last_valid, candidate)
    limits = _pose_consistency_limits(config, init_source, gap, recovery_multiplier)
    metrics = {
        "correction_translation_m": correction["translation_m"],
        "correction_rotation_deg": correction["rotation_deg"],
        "correction_translation_limit_m": limits["correction_translation_m"],
        "correction_rotation_limit_deg": limits["correction_rotation_deg"],
        "temporal_translation_m": temporal["translation_m"],
        "temporal_rotation_deg": temporal["rotation_deg"],
        "temporal_translation_limit_m": limits["temporal_translation_m"],
        "temporal_rotation_limit_deg": limits["temporal_rotation_deg"],
    }
    checks = (
        ("tracking_consistency_correction_translation", correction["translation_m"], limits["correction_translation_m"]),
        ("tracking_consistency_correction_rotation", correction["rotation_deg"], limits["correction_rotation_deg"]),
        ("tracking_consistency_temporal_translation", temporal["translation_m"], limits["temporal_translation_m"]),
        ("tracking_consistency_temporal_rotation", temporal["rotation_deg"], limits["temporal_rotation_deg"]),
    )
    for reason, value, limit in checks:
        if value > limit:
            return False, reason, metrics
    return True, None, metrics


def _registration_failure(
    init_transform: np.ndarray,
    source_count: int,
    target_count: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "transform": np.asarray(init_transform, dtype=np.float64).copy(),
        "fitness": 0.0,
        "inlier_rmse": float("inf"),
        "latency_ms": 0.0,
        "num_source_points": int(source_count),
        "num_target_points": int(target_count),
        "converged": False,
        "optimizer_valid": False,
        "termination_reason": reason,
        "failure_reason": reason,
        "iterations": 0,
        "num_correspondences": 0,
        "point_to_plane_rmse": float("inf"),
        "eigenvalues": np.zeros(6, dtype=np.float64),
        "condition_number": float("inf"),
        "effective_rank": 0,
        "pose_covariance": np.full((6, 6), np.nan, dtype=np.float64),
    }


def _recovery_local_config(
    base: LocalSubmapConfig,
    recovery_cfg: dict[str, Any],
) -> LocalSubmapConfig:
    max_points = int(recovery_cfg.get("max_points", base.max_points or 10000))
    return LocalSubmapConfig(
        near=base.near,
        far=base.far * float(recovery_cfg.get("far_multiplier", 1.5)),
        margin_px=base.margin_px * float(recovery_cfg.get("margin_multiplier", 2.0)),
        radius_margin=base.radius_margin * float(recovery_cfg.get("radius_multiplier", 2.0)),
        max_points=max_points,
        sampling_policy=base.sampling_policy,
        coverage_representatives=base.coverage_representatives,
        tile_rows=base.tile_rows,
        tile_cols=base.tile_cols,
        depth_bins=base.depth_bins,
        frustum_culling=base.frustum_culling,
        block_size_m=base.block_size_m,
        projection_pre_cap=base.projection_pre_cap,
        projection_sampling_policy=base.projection_sampling_policy,
        block_test_chunk=base.block_test_chunk,
        oracle_mode=base.oracle_mode,
        oracle_overflow_policy=base.oracle_overflow_policy,
        oracle_point_chunk=base.oracle_point_chunk,
    )


def _query_recovery_target(
    spatial_index: MapSpatialIndex,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    base_config: LocalSubmapConfig,
    scores: np.ndarray | None,
    block_score_token: str | None,
    tier: int,
    recovery_cfg: dict[str, Any],
    return_stage_indices: bool = False,
    use_local_map_scores: bool = True,
) -> dict[str, Any]:
    if tier < 3:
        query_config = base_config if tier < 2 else _recovery_local_config(base_config, recovery_cfg)
        query_kwargs = (
            {"block_score_token": block_score_token}
            if block_score_token is not None and query_config.frustum_culling == "voxel_blocks"
            else {"scores": scores if use_local_map_scores else None}
        )
        if return_stage_indices:
            query_kwargs["return_stage_indices"] = True
        return spatial_index.query_local_submap(
            pose, intrinsics, image_height, image_width, query_config, **query_kwargs
        )

    start = time.perf_counter()
    radius = float(recovery_cfg.get("sphere_radius_m", base_config.far))
    max_points = int(recovery_cfg.get("max_points", base_config.max_points or 10000))
    indices = spatial_index.query_radius(np.asarray(pose)[:3, 3], radius, sort_result=False)
    candidate_count = int(indices.size)
    if indices.size > max_points:
        distances = np.linalg.norm(spatial_index.points[indices] - np.asarray(pose)[:3, 3], axis=1)
        if scores is None:
            chosen = np.argpartition(distances, max_points - 1)[:max_points]
            order = np.lexsort((indices[chosen], distances[chosen]))
        else:
            chosen = np.argpartition(-scores[indices], max_points - 1)[:max_points]
            order = np.lexsort((indices[chosen], distances[chosen], -scores[indices[chosen]]))
        indices = indices[chosen[order]]
    return {
        "indices": indices,
        "query_latency_ms": float((time.perf_counter() - start) * 1000.0),
        "candidate_count": candidate_count,
        "projection_input_count": candidate_count,
        "block_candidate_count": candidate_count,
        "tested_block_count": 0,
        "intersecting_block_count": 0,
        "culling_mode": "recovery_radius",
        "projection_cap": None,
        "projection_cap_applied": False,
        "projection_cap_bypassed": False,
        "projection_cap_utilization": None,
        "block_culling_latency_ms": 0.0,
        "precap_latency_ms": 0.0,
        "precap_strategy": "point_reference",
        "representative_points_considered": 0,
        "precap_block_count": 0,
        "precap_selected_block_count": 0,
        "precap_block_coverage": 0.0,
        "precap_bucket_count": 0,
        "precap_selected_bucket_count": 0,
        "precap_bucket_coverage": 0.0,
        "precap_boundary_block_count": 0,
        "precap_selected_boundary_block_count": 0,
        "precap_boundary_coverage": 0.0,
        "oracle_latency_ms": 0.0,
        "oracle_count": 0,
        "oracle_block_recall": 1.0,
        "oracle_projection_recall": 1.0,
        "oracle_frustum_recall": 1.0,
        "oracle_overflow": False,
        "oracle_status": "ok",
        "oracle_reason": "off",
        "positive_depth_count": 0,
        "depth_range_count": 0,
        "precap_count": int(indices.size),
        "frustum_count": 0,
        "count": int(indices.size),
        "backend": spatial_index.backend,
        **(
            {
                "candidate_indices": np.asarray(indices, dtype=np.int64),
                "positive_depth_indices": np.empty(0, dtype=np.int64),
                "depth_range_indices": np.empty(0, dtype=np.int64),
                "frustum_indices": np.empty(0, dtype=np.int64),
                "selected_indices": np.asarray(indices, dtype=np.int64),
            }
            if return_stage_indices else {}
        ),
    }


def _write_long_horizon_csvs(output_root: Path, metrics: dict[str, Any], frame_rows: list[dict[str, Any]]) -> None:
    raw = metrics["raw_absolute"]
    aligned = metrics["aligned_absolute"]
    per_frame_rows = []
    for index, frame_row in enumerate(frame_rows):
        per_frame_rows.append({
            "frame_id": frame_row["frame_id"],
            "timestamp": frame_row["timestamp"],
            "tracking_valid": frame_row["tracking_valid"],
            "raw_translation_error_m": raw["translation_errors_m"][index],
            "raw_rotation_error_deg": raw["rotation_errors_deg"][index],
            "aligned_translation_error_m": aligned["translation_errors_m"][index],
            "aligned_rotation_error_deg": aligned["rotation_errors_deg"][index],
        })
    _write_frame_metrics(
        str(output_root / "per_frame_errors.csv"),
        [{key: _csv_value(value) for key, value in row.items()} for row in per_frame_rows],
        list(per_frame_rows[0].keys()),
    )

    segment_rows = []
    for requested_length, stats in metrics["segment_errors"].items():
        for sample in stats["samples"]:
            segment_rows.append({"requested_length_m": requested_length, **sample})
    segment_fields = [
        "requested_length_m",
        "start_index",
        "end_index",
        "translation_error_m",
        "rotation_error_deg",
        "gt_length_m",
        "translation_drift_percent",
        "rotation_drift_deg_per_m",
    ]
    _write_frame_metrics(
        str(output_root / "segment_errors.csv"),
        [{key: _csv_value(value) for key, value in row.items()} for row in segment_rows],
        segment_fields,
    )


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _compute_valid_rpe(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    valid_mask: np.ndarray,
    delta: int,
) -> dict[str, float] | None:
    weighted_trans_squared = 0.0
    weighted_rot_squared = 0.0
    num_pairs = 0
    run_start = None
    for idx in range(valid_mask.shape[0] + 1):
        is_valid = idx < valid_mask.shape[0] and bool(valid_mask[idx])
        if is_valid and run_start is None:
            run_start = idx
        if not is_valid and run_start is not None:
            run_length = idx - run_start
            if run_length > delta:
                rpe = compute_rpe_rmse(est_poses[run_start:idx], gt_poses[run_start:idx], delta=delta)
                run_pairs = run_length - delta
                weighted_trans_squared += rpe["trans_rmse"] ** 2 * run_pairs
                weighted_rot_squared += rpe["rot_rmse_deg"] ** 2 * run_pairs
                num_pairs += run_pairs
            run_start = None
    if num_pairs == 0:
        return None
    return {
        "trans_rmse": float(np.sqrt(weighted_trans_squared / num_pairs)),
        "rot_rmse_deg": float(np.sqrt(weighted_rot_squared / num_pairs)),
    }


def _assert_strict_output_isolated(output_root: Path) -> None:
    if FROZEN_AUDIT_DIRNAME in output_root.resolve().parts:
        raise ValueError(f"mvp_strict output must not be written inside frozen {FROZEN_AUDIT_DIRNAME}")


def _is_finite_pose(pose: np.ndarray, max_translation_norm: float) -> bool:
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
        return False
    if np.linalg.norm(pose[:3, 3]) > float(max_translation_norm):
        return False
    return True


def run_localization(
    config: dict[str, Any],
    map_path: str,
    output_dir: str,
    frames: list[dict[str, Any]] | None = None,
    dataset: Any | None = None,
    start: int | None = None,
    end: int | None = None,
    backend_override: str | None = None,
    protocol_override: str | None = None,
    first_pose_protocol_override: str | None = None,
    motion_model_override: str | None = None,
    explicit_initial_pose: np.ndarray | None = None,
    stage_profile=None,
    *,
    candidate_probe: Callable[[dict[str, Any]], None] | None = None,
    candidate_force_frames: Iterable[int] | None = None,
    reserve_stop_source_idx: int | None = None,
    primary_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Localize, optionally reporting one reserve arm against the unmodified base trajectory.

    Probe runs must start at source index zero and use identical base settings
    (including the verifier's source holdout) across arms. Forced frame IDs only
    request diagnostic solves; they do not override eligibility or native adoption.
    """
    probe_enabled = candidate_probe is not None
    if reserve_stop_source_idx is not None and (
        type(reserve_stop_source_idx) is not int or reserve_stop_source_idx < 1 or probe_enabled
    ):
        raise ValueError("reserve_stop_source_idx requires a positive integer and no candidate_probe")
    if probe_enabled and not callable(candidate_probe):
        raise TypeError("candidate_probe must be callable")
    force_frames = frozenset(() if candidate_force_frames is None else candidate_force_frames)
    if candidate_force_frames is not None and not probe_enabled:
        raise ValueError("candidate_force_frames requires candidate_probe")
    fsm_map = load_fsm_map(map_path)
    target_points = np.asarray(fsm_map["means"], dtype=np.float64)
    target_covariances = np.asarray(fsm_map["covariances"], dtype=np.float64) if "covariances" in fsm_map else None
    target_normals = np.asarray(fsm_map["normals"], dtype=np.float64) if "normals" in fsm_map else None
    target_scores = np.asarray(fsm_map.get("scores", []), dtype=np.float64)
    target_global_ids = np.asarray(fsm_map.get("indices", np.arange(target_points.shape[0])), dtype=np.int64)
    if target_scores.shape != (target_points.shape[0],):
        target_scores = None

    if frames is None:
        if dataset is None:
            raise ValueError("Either frames or dataset must be provided")
        frame_ids = _frame_range(len(dataset), start, end)
        frame_source = dataset
    else:
        frame_ids = _frame_range(len(frames), start, end)
        frame_source = frames

    if len(frame_ids) == 0:
        raise ValueError("No frames selected for localization")
    if probe_enabled and frame_ids.start != 0:
        raise ValueError("candidate_probe requires replay from source index 0")

    output_root = Path(output_dir)

    registration_cfg = config.get("registration", {})
    dataset_cfg = config.get("dataset", {})
    reserve_cfg = config.get("observable_reserve", {})
    bypass_candidate_gate = reserve_cfg.get("bypass_candidate_gate", False)
    if not isinstance(bypass_candidate_gate, bool):
        raise ValueError("observable_reserve.bypass_candidate_gate must be boolean")
    if bypass_candidate_gate and probe_enabled:
        raise ValueError("bypass_candidate_gate is closed-loop only; candidate_probe already reports both outcomes")
    verifier_cfg = reserve_cfg.get("candidate_verifier", {})
    if not isinstance(verifier_cfg, dict) or not isinstance(verifier_cfg.get("enabled", False), bool):
        raise ValueError("candidate_verifier must be a dictionary with boolean enabled")
    verifier_enabled = verifier_cfg.get("enabled", False)
    verifier_options = {
        key: value for key, value in verifier_cfg.items()
        if key not in ("enabled", "block_size", "folds", "fold", "max_translation_m")
    }
    if "max_translation_m" in verifier_cfg:
        verifier_options["max_translation"] = verifier_cfg["max_translation_m"]
    if verifier_enabled:
        verifier_options = validate_verifier_config(verifier_options)
        split_source_validation(
            np.empty(0, dtype=np.int64), 1,
            block_size=verifier_cfg.get("block_size", 16),
            folds=verifier_cfg.get("folds", 4), fold=verifier_cfg.get("fold", 0),
        )
    eval_cfg = config.get("eval", {})
    memory_cfg = config.get("runtime_memory_metrics", {}) or {}
    if not isinstance(memory_cfg, dict):
        raise ValueError("runtime_memory_metrics must be a dictionary")
    backend = backend_override or registration_cfg.get("backend", "numpy_point_to_point")
    evaluation_protocol = protocol_override or registration_cfg.get("evaluation_protocol", "audit_legacy")
    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(f"Unknown evaluation_protocol '{evaluation_protocol}'")
    if probe_enabled and evaluation_protocol != "mvp_strict":
        raise ValueError("candidate_probe requires mvp_strict")
    if verifier_enabled and (evaluation_protocol != "mvp_strict" or backend != "numpy_robust_point_to_plane"):
        raise ValueError("candidate_verifier requires strict robust point-to-plane registration")
    first_pose_protocol = first_pose_protocol_override or registration_cfg.get("first_pose_protocol")
    first_pose_left_twist = registration_cfg.get("first_pose_left_twist")
    motion_cfg = config.get("motion_model", {})
    if isinstance(motion_cfg, str):
        configured_motion_model = motion_cfg
        motion_cfg = {}
    else:
        configured_motion_model = motion_cfg.get("type", motion_cfg.get("model"))
    motion_model = motion_model_override or registration_cfg.get("motion_model", configured_motion_model or "previous_pose")
    if evaluation_protocol == "mvp_strict":
        if first_pose_protocol not in FIRST_POSE_PROTOCOLS:
            raise ValueError(f"mvp_strict requires explicit first_pose_protocol in {FIRST_POSE_PROTOCOLS}")
        if motion_model not in ("previous_pose", "constant_velocity"):
            raise ValueError("motion_model must be 'previous_pose' or 'constant_velocity'")
        _assert_strict_output_isolated(output_root)
    memory_enabled = evaluation_protocol == "mvp_strict" and bool(memory_cfg.get("enabled", True))
    memory_check_interval = int(memory_cfg.get("check_interval_frames", 25))
    if memory_check_interval < 1:
        raise ValueError("runtime_memory_metrics.check_interval_frames must be positive")
    max_rss_mb = memory_cfg.get("max_rss_mb")
    if max_rss_mb is not None and float(max_rss_mb) <= 0.0:
        raise ValueError("runtime_memory_metrics.max_rss_mb must be positive")
    rss_mb = peak_rss_mb = None
    output_root.mkdir(parents=True, exist_ok=True)

    init_mode = registration_cfg.get("init_mode", "previous_estimate")
    fallback_on_divergence = bool(registration_cfg.get("fallback_on_divergence", True))
    max_pose_translation_norm = float(registration_cfg.get("max_pose_translation_norm", 1e6))
    prediction_max_translation_m = float(
        registration_cfg.get(
            "prediction_max_translation_m",
            motion_cfg.get("max_translation_m", motion_cfg.get("translation_threshold_m", float("inf"))),
        )
    )
    prediction_max_rotation_deg = float(
        registration_cfg.get(
            "prediction_max_rotation_deg",
            motion_cfg.get("max_rotation_deg", motion_cfg.get("rotation_threshold_deg", float("inf"))),
        )
    )
    prediction_max_total_translation_m = float(
        registration_cfg.get("prediction_max_total_translation_m", float("inf"))
    )
    prediction_max_total_rotation_deg = float(
        registration_cfg.get("prediction_max_total_rotation_deg", float("inf"))
    )
    prediction_max_horizon_frames = int(registration_cfg.get("prediction_max_horizon_frames", 3))
    if prediction_max_horizon_frames < 1:
        raise ValueError("prediction_max_horizon_frames must be positive")
    tracking_acceptance_cfg = registration_cfg.get("tracking_acceptance", {})
    if tracking_acceptance_cfg is None:
        tracking_acceptance_cfg = {}
    if not isinstance(tracking_acceptance_cfg, dict):
        raise ValueError("registration.tracking_acceptance must be a dictionary")
    pose_consistency_cfg = registration_cfg.get("pose_consistency", {}) or {}
    if not isinstance(pose_consistency_cfg, dict):
        raise ValueError("registration.pose_consistency must be a dictionary")
    pose_consistency_enabled = evaluation_protocol == "mvp_strict" and bool(pose_consistency_cfg.get("enabled", False))
    if evaluation_protocol == "mvp_strict" and (
        prediction_max_translation_m < 0.0 or prediction_max_rotation_deg < 0.0
    ):
        raise ValueError("prediction motion guard thresholds must be non-negative")

    local_cfg = config.get("local_submap", registration_cfg.get("local_submap", {}))
    use_local_map_scores = local_cfg.get("use_map_scores", True)
    if not isinstance(use_local_map_scores, bool):
        raise ValueError("local_submap.use_map_scores must be boolean")
    local_enabled = bool(local_cfg.get("enabled", False))
    if local_enabled and evaluation_protocol != "mvp_strict":
        raise ValueError("local_submap is only supported with evaluation_protocol='mvp_strict'")
    local_config = None
    local_spatial_index = None
    local_index_build_ms = 0.0
    block_index_build_ms = 0.0
    block_score_hash_ms = 0.0
    block_geometry_sort_ms = 0.0
    block_index_memory_bytes = 0
    block_peak_working_set_estimate_bytes = 0
    block_score_token = None
    local_index_backend = "disabled"
    if local_enabled:
        spatial_cfg = local_cfg.get("spatial", {})
        local_config = LocalSubmapConfig(
            near=float(local_cfg.get("near", dataset_cfg.get("depth_min", 0.0))),
            far=float(local_cfg.get("far", dataset_cfg.get("depth_max", 5.0))),
            margin_px=float(local_cfg.get("margin", local_cfg.get("margin_px", 0.0))),
            radius_margin=float(local_cfg.get("radius_margin", 0.0)),
            max_points=local_cfg.get("max_points"),
            sampling_policy=str(local_cfg.get("sampling_policy", "legacy")),
            coverage_representatives=str(local_cfg.get("coverage_representatives", "ranked")),
            tile_rows=int(local_cfg.get("tile_rows", 12)),
            tile_cols=int(local_cfg.get("tile_cols", 16)),
            depth_bins=int(local_cfg.get("depth_bins", 8)),
            frustum_culling=str(local_cfg.get("frustum_culling", "radius")),
            block_size_m=float(local_cfg.get("block_size_m", 0.3)),
            projection_pre_cap=local_cfg.get("projection_pre_cap"),
            projection_sampling_policy=str(local_cfg.get("projection_sampling_policy", "block_coverage")),
            block_test_chunk=int(local_cfg.get("block_test_chunk", 65536)),
            oracle_mode=str(local_cfg.get("oracle_mode", "off")),
            oracle_overflow_policy=str(local_cfg.get("oracle_overflow_policy", "uncapped_chunked")),
            oracle_point_chunk=int(local_cfg.get("oracle_point_chunk", 262144)),
        )
        local_spatial_index = MapSpatialIndex(
            target_points,
            backend=str(local_cfg.get("spatial_backend", spatial_cfg.get("backend", "auto"))),
            voxel_size=float(
                local_cfg.get(
                    "grid_voxel",
                    local_cfg.get("grid_voxel_size", spatial_cfg.get("grid_voxel", spatial_cfg.get("voxel_size", 1.0))),
                )
            ),
        )
        local_index_build_ms = local_spatial_index.build_latency_ms
        local_index_backend = local_spatial_index.backend
        if local_config.frustum_culling == "voxel_blocks":
            block_metrics = local_spatial_index.prebuild_blocks(
                local_config.block_size_m, target_scores if use_local_map_scores else None,
            )
            block_index_build_ms = float(block_metrics["block_index_build_ms"])
            block_score_hash_ms = float(block_metrics["block_score_hash_ms"])
            block_geometry_sort_ms = float(block_metrics["block_geometry_sort_ms"])
            block_index_memory_bytes = int(block_metrics["block_index_memory_bytes"])
            block_peak_working_set_estimate_bytes = int(block_metrics["peak_working_set_estimate_bytes"])
            block_score_token = str(block_metrics["block_score_token"])
    local_min_points = int(local_cfg.get("min_points", 1))
    if local_min_points <= 0:
        raise ValueError("local_submap.min_points must be positive")
    recovery_cfg = config.get("recovery", registration_cfg.get("recovery", {})) or {}
    recovery_enabled = bool(recovery_cfg.get("enabled", False))
    recovery_max_tier = int(recovery_cfg.get("max_tier", 0))
    verifier_backend = recovery_cfg.get("verifier_backend")
    verifier_acceptance_cfg = recovery_cfg.get("verifier_acceptance", {}) or {}
    if not isinstance(verifier_acceptance_cfg, dict):
        raise ValueError("recovery.verifier_acceptance must be a dictionary")
    correction_recovery_multiplier = float(recovery_cfg.get("correction_limit_multiplier", 1.0))
    if correction_recovery_multiplier < 1.0:
        raise ValueError("recovery.correction_limit_multiplier must be at least 1")
    if recovery_enabled and (evaluation_protocol != "mvp_strict" or not local_enabled):
        raise ValueError("recovery requires strict evaluation with local_submap.enabled")
    if recovery_max_tier < 0 or recovery_max_tier > 3:
        raise ValueError("recovery.max_tier must be between 0 and 3")

    target_index = None
    target_index_build_ms = 0.0
    target_index_backend = "not_applicable"
    if not local_enabled and backend in (
        "numpy_point_to_point",
        "numpy_covariance_weighted",
        "numpy_robust_point_to_plane",
    ):
        index_result = build_target_index(target_points)
        target_index = index_result["target_tree"]
        target_index_build_ms = float(index_result["build_latency_ms"])
        target_index_backend = str(index_result["backend"])

    est_poses = []
    gt_poses = []
    timestamps = []
    frame_rows = []
    previous_estimate = None
    last_valid_estimate = None
    valid_pose_history: list[tuple[int, np.ndarray, bool]] = []
    previous_local_indices: np.ndarray | None = None
    previous_local_tree = None
    previous_local_normals: np.ndarray | None = None
    activation_diagnostics = RuntimeActivationDiagnostics(
        config, fsm_map, map_path, local_config, local_spatial_index, target_points,
        target_normals, target_scores, evaluation_protocol, backend,
    )
    observable_reserve = RuntimeObservableReserve(
        config, fsm_map, map_path, local_config, evaluation_protocol, backend,
    )
    if bypass_candidate_gate or force_frames:
        if not observable_reserve.active or observable_reserve.mode != "active":
            raise ValueError("candidate bypass or force frames require active observable_reserve")
    force_reserve = None
    if force_frames:
        # Reuse the same selection policy and mandatory runtime eligibility checks.
        # Only the diagnostic planner's trigger threshold differs.
        force_reserve = copy(observable_reserve)
        force_reserve.trigger = {"iterations": 0, "effective_rank_below": None}
    selection_event_writer = _ReserveSelectionEventWriter(output_root, config, observable_reserve, map_path)
    paired_cfg = config.get("paired_reserve_evaluation", {}) or {}
    if not isinstance(paired_cfg, dict):
        raise ValueError("paired_reserve_evaluation must be a dictionary")
    paired_enabled = bool(paired_cfg.get("enabled", False))
    paired_shadow = None
    paired_evaluations = []
    if paired_enabled:
        if evaluation_protocol != "mvp_strict" or local_config is None or backend != "numpy_robust_point_to_plane":
            raise ValueError("paired_reserve_evaluation requires strict local robust registration")
        if str(paired_cfg.get("mode", "shadow")) != "shadow":
            raise ValueError("paired_reserve_evaluation.mode must be shadow")
        if observable_reserve.enabled and observable_reserve.mode == "active":
            raise ValueError("paired shadow evaluation forbids observable_reserve active mode")
        loader_config = dict(config)
        loader_config["observable_reserve"] = {
            **dict(config.get("observable_reserve", {}) or {}),
            "enabled": True,
            "mode": "plan_only",
            "geometry_path": paired_cfg.get("geometry_path", "auto"),
        }
        paired_geometry = RuntimeObservableReserve(
            loader_config, fsm_map, map_path, local_config, evaluation_protocol, backend,
        )
        if not paired_geometry.available:
            raise ValueError(f"paired reserve sidecar unavailable: {paired_geometry.unavailable_reason}")
        paired_shadow = PairedShadowReserve.from_runtime_reserve(
            {"paired_reserve": paired_cfg, "observable_reserve": config.get("observable_reserve", {})},
            paired_geometry, local_config,
        )

    oracle_cfg = config.get("final_target_oracle", {}) or {}
    if not isinstance(oracle_cfg, dict):
        raise ValueError("final_target_oracle must be a dictionary")
    final_target_oracle = None
    if bool(oracle_cfg.get("enabled", False)):
        if evaluation_protocol != "mvp_strict" or local_config is None or backend != "numpy_robust_point_to_plane":
            raise ValueError("final_target_oracle requires strict local robust registration")
        runtime_fields = {key: value for key, value in oracle_cfg.items() if key != "enabled"}
        final_target_oracle = FinalTargetOracleRuntime(
            FinalTargetOracleConfig(**runtime_fields), local_spatial_index, local_config,
            scores=None if block_score_token is not None or not use_local_map_scores else target_scores,
            block_score_token=block_score_token,
        )

    for output_idx, source_idx in enumerate(frame_ids):
        frame_start = time.perf_counter()
        frame = get_frame(frame_source, source_idx) if dataset is not None and frames is None else frame_source[source_idx]
        stages = ({"frame_load_ms": (time.perf_counter() - frame_start) * 1000, "attempts": []}
                  if stage_profile is not None else None)
        frame_id = int(frame.get("frame_id", output_idx))
        timestamp = float(frame.get("timestamp", frame_id))
        gt_c2w = np.asarray(frame["c2w"], dtype=np.float64)
        final_attempt_context = None
        paired_triggered = False
        pair_id = ""
        paired_arm_count = 0
        paired_shadow_latency_ms = 0.0
        candidate_pose = None
        candidate_basic_valid = False
        candidate_reserve_count = candidate_target_count = candidate_latency_ms = None
        force_requested = frame_id in force_frames
        force_eligible = forced_attempt = False
        probe_latency_ms = 0.0

        stage_start = time.perf_counter() if stages is not None else 0.0
        pointcloud = rgbd_to_pointcloud(
            color=frame.get("color"),
            depth=frame["depth"],
            intrinsics=frame["intrinsics"],
            c2w=None,
            depth_min=float(dataset_cfg.get("depth_min", 0.0)),
            depth_max=dataset_cfg.get("depth_max"),
            max_points=dataset_cfg.get("max_points"),
            seed=int(config.get("seed", 0)) + frame_id,
        )
        if stages is not None:
            stages["pointcloud_ms"] = (time.perf_counter() - stage_start) * 1000
        validation_points = np.empty((0, 3), dtype=np.float64)
        validation_groups = np.empty(0, dtype=np.int64)
        selection_depth = np.asarray(frame["depth"], dtype=np.float64)
        if verifier_enabled:
            width = np.asarray(frame["depth"]).shape[1]
            validation_mask = split_source_validation(
                pointcloud["indices"], width,
                block_size=verifier_cfg.get("block_size", 16),
                folds=verifier_cfg.get("folds", 4), fold=verifier_cfg.get("fold", 0),
            )
            validation_points = np.asarray(pointcloud["points"][validation_mask], dtype=np.float64).copy()
            pixels = pointcloud["indices"][validation_mask]
            block = verifier_cfg.get("block_size", 16)
            validation_groups = (pixels // width // block) * ((width + block - 1) // block) + pixels % width // block
            selection_depth = selection_depth.copy()
            heldout_pixels = split_source_validation(
                np.arange(selection_depth.size, dtype=np.int64), width,
                block_size=block, folds=verifier_cfg.get("folds", 4), fold=verifier_cfg.get("fold", 0),
            )
            selection_depth.reshape(-1)[heldout_pixels] = 0.0
            for key in ("points", "indices", "colors"):
                if pointcloud.get(key) is not None:
                    pointcloud[key] = pointcloud[key][~validation_mask]

        if evaluation_protocol == "mvp_strict" and output_idx == 0:
            estimated_c2w = _strict_first_pose(
                first_pose_protocol, gt_c2w, explicit_initial_pose, first_pose_left_twist
            )
            result = {
                "transform": estimated_c2w,
                "fitness": 1.0,
                "inlier_rmse": 0.0,
                "latency_ms": 0.0,
                "num_source_points": int(pointcloud["points"].shape[0]),
                "num_target_points": int(target_points.shape[0]),
                "converged": True,
                "optimizer_valid": True,
                "termination_reason": "initialized",
            }
            accepted = True
            fallback_used = False
            tracking_valid = True
            tracking_status = "initialized"
            last_valid_estimate = estimated_c2w
            valid_pose_history.append((output_idx, estimated_c2w.copy(), False))
            init_source = "first_pose_protocol"
            prediction_translation_m = 0.0
            prediction_rotation_deg = 0.0
            prediction_guard = "not_applicable"
            local_query_latency_ms = 0.0
            local_radius_latency_ms = 0.0
            local_projection_latency_ms = 0.0
            local_sampling_latency_ms = 0.0
            local_candidate_count = 0
            local_culling_mode = local_config.frustum_culling if local_config is not None else "disabled"
            local_tested_block_count = 0
            local_intersecting_block_count = 0
            local_block_candidate_count = 0
            local_projection_input_count = 0
            local_projection_cap = local_config.projection_pre_cap if local_config is not None else None
            local_projection_cap_applied = False
            local_projection_cap_bypassed = False
            local_projection_cap_utilization = None
            local_block_culling_latency_ms = 0.0
            local_precap_latency_ms = 0.0
            local_precap_strategy = "point_reference"
            local_representative_points_considered = 0
            local_precap_block_count = 0
            local_precap_selected_block_count = 0
            local_precap_block_coverage = 0.0
            local_precap_bucket_count = 0
            local_precap_selected_bucket_count = 0
            local_precap_bucket_coverage = 0.0
            local_precap_boundary_block_count = 0
            local_precap_selected_boundary_block_count = 0
            local_precap_boundary_coverage = 0.0
            local_oracle_latency_ms = 0.0
            local_oracle_count = 0
            local_oracle_block_recall = 1.0
            local_oracle_projection_recall = 1.0
            local_oracle_frustum_recall = 1.0
            local_oracle_overflow = False
            local_oracle_status = "ok"
            local_oracle_reason = "off"
            local_frustum_count = 0
            local_positive_depth_count = 0
            local_depth_range_count = 0
            local_precap_count = 0
            local_final_sampling_input_count = 0
            local_nonempty_tile_count = 0
            local_selected_tile_count = 0
            local_nonempty_depth_bin_count = 0
            local_selected_depth_bin_count = 0
            local_nonempty_tile_depth_count = 0
            local_selected_tile_depth_count = 0
            local_target_count = 0 if local_enabled else int(target_points.shape[0])
            local_target_fraction = 0.0 if local_enabled else 1.0
            acceptance_reason = None
            primary_termination_reason = "initialized"
            primary_rejection_reason = None
            primary_result = dict(result)
            primary_accepted = True
            recovery_attempted = False
            recovery_stage = 0
            recovery_count = 0
            recovered = False
            recovery_backend = None
            temporal_gap = 0
            primary_consistency_reason = None
            final_consistency_reason = None
            primary_consistency_metrics = {}
            final_consistency_metrics = {}
            reserve_metrics = observable_reserve.empty_metrics(observable_reserve.mode, observable_reserve.unavailable_reason)
            observable_reserve_attempt = False
        else:
            if evaluation_protocol == "mvp_strict":
                (
                    init_transform,
                    init_source,
                    prediction_translation_m,
                    prediction_rotation_deg,
                    prediction_guard,
                ) = _strict_motion_initialization(
                    motion_model,
                    valid_pose_history,
                    output_idx,
                    prediction_max_translation_m,
                    prediction_max_rotation_deg,
                    prediction_max_total_translation_m,
                    prediction_max_total_rotation_deg,
                    prediction_max_horizon_frames,
                )
            else:
                init_transform = _select_initial_transform(init_mode, frame_id, previous_estimate, gt_c2w)
                init_source = "audit_legacy"
                prediction_translation_m = 0.0
                prediction_rotation_deg = 0.0
                prediction_guard = "not_applicable"

            normal_source = str(registration_cfg.get("normal_source", "map"))
            if normal_source not in ("map", "estimate", "local_pca"):
                raise ValueError("normal_source must be 'map', 'estimate', or 'local_pca'")
            if normal_source == "local_pca" and not local_enabled:
                raise ValueError("normal_source='local_pca' requires local_submap.enabled")

            temporal_gap = output_idx - valid_pose_history[-1][0] if evaluation_protocol == "mvp_strict" else 0

            def run_attempt(
                attempt_pose: np.ndarray, tier: int, attempt_backend: str | None = None
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                nonlocal previous_local_indices, previous_local_tree, previous_local_normals, final_attempt_context
                stage = None
                if stages is not None:
                    stage = {"tier": tier, "backend": attempt_backend or backend,
                             "same_query_as_earlier_attempt": any(
                                 old["query_kind"] == min(max(old["tier"], 1), 3)
                                 == min(max(tier, 1), 3) and old["query_pose"] == attempt_pose.tolist()
                                 for old in stages["attempts"]),
                             "query_kind": min(max(tier, 1), 3), "query_pose": attempt_pose.tolist()}
                    stages["attempts"].append(stage)
                registration_target_points = target_points
                registration_target_covariances = target_covariances
                registration_target_normals = target_normals
                registration_target_tree = target_index
                query_metrics = {
                    "query_latency_ms": 0.0,
                    "radius_latency_ms": 0.0,
                    "projection_latency_ms": 0.0,
                    "sampling_latency_ms": 0.0,
                    "candidate_count": 0,
                    "culling_mode": local_config.frustum_culling if local_config is not None else "disabled",
                    "tested_block_count": 0,
                    "intersecting_block_count": 0,
                    "block_candidate_count": 0,
                    "projection_input_count": 0,
                    "projection_cap": local_config.projection_pre_cap if local_config is not None else None,
                    "projection_cap_applied": False,
                    "projection_cap_bypassed": False,
                    "projection_cap_utilization": None,
                    "block_culling_latency_ms": 0.0,
                    "precap_latency_ms": 0.0,
                    "precap_strategy": "point_reference",
                    "representative_points_considered": 0,
                    "precap_block_count": 0,
                    "precap_selected_block_count": 0,
                    "precap_block_coverage": 0.0,
                    "precap_bucket_count": 0,
                    "precap_selected_bucket_count": 0,
                    "precap_bucket_coverage": 0.0,
                    "precap_boundary_block_count": 0,
                    "precap_selected_boundary_block_count": 0,
                    "precap_boundary_coverage": 0.0,
                    "oracle_latency_ms": 0.0,
                    "oracle_count": 0,
                    "oracle_block_recall": 1.0,
                    "oracle_projection_recall": 1.0,
                    "oracle_frustum_recall": 1.0,
                    "oracle_overflow": False,
                    "oracle_status": "ok",
                    "oracle_reason": "off",
                    "frustum_count": 0,
                    "positive_depth_count": 0,
                    "depth_range_count": 0,
                    "precap_count": 0,
                    "final_sampling_input_count": 0,
                    "nonempty_tile_count": 0,
                    "selected_tile_count": 0,
                    "nonempty_depth_bin_count": 0,
                    "selected_depth_bin_count": 0,
                    "nonempty_tile_depth_count": 0,
                    "selected_tile_depth_count": 0,
                    "target_count": int(target_points.shape[0]),
                    "target_fraction": 1.0,
                }
                local_indices = None
                if local_enabled:
                    assert local_spatial_index is not None and local_config is not None
                    image_height, image_width = np.asarray(frame["depth"]).shape[:2]
                    stage_start = time.perf_counter() if stage is not None else 0.0
                    local_query = _query_recovery_target(
                        local_spatial_index,
                        attempt_pose,
                        np.asarray(frame["intrinsics"], dtype=np.float64),
                        image_height,
                        image_width,
                        local_config,
                        target_scores,
                        block_score_token,
                        tier,
                        recovery_cfg,
                        use_local_map_scores=use_local_map_scores,
                    )
                    if stage is not None:
                        stage["query_ms"] = (time.perf_counter() - stage_start) * 1000
                    local_indices = np.asarray(local_query["indices"], dtype=np.int64)
                    query_metrics = {
                        "query_latency_ms": float(local_query["query_latency_ms"]),
                        "radius_latency_ms": float(local_query.get("radius_latency_ms", 0.0)),
                        "projection_latency_ms": float(local_query.get("projection_latency_ms", 0.0)),
                        "sampling_latency_ms": float(local_query.get("sampling_latency_ms", 0.0)),
                        "candidate_count": int(local_query["candidate_count"]),
                        "culling_mode": str(local_query.get("culling_mode", local_config.frustum_culling)),
                        "tested_block_count": int(local_query.get("tested_block_count", 0)),
                        "intersecting_block_count": int(local_query.get("intersecting_block_count", 0)),
                        "block_candidate_count": int(local_query.get("block_candidate_count", local_query["candidate_count"])),
                        "projection_input_count": int(local_query.get("projection_input_count", local_query["candidate_count"])),
                        "projection_cap": local_config.projection_pre_cap,
                        "projection_cap_applied": bool(
                            local_config.projection_pre_cap is not None
                            and int(local_query.get("projection_input_count", local_query["candidate_count"]))
                            < int(local_query["candidate_count"])
                        ),
                        "projection_cap_bypassed": bool(local_query.get("projection_cap_bypassed", False)),
                        "projection_cap_utilization": (
                            float(local_query.get("projection_input_count", local_query["candidate_count"]) / local_config.projection_pre_cap)
                            if local_config.projection_pre_cap is not None else None
                        ),
                        "block_culling_latency_ms": (
                            float(local_query.get("block_culling_latency_ms", 0.0))
                            if local_config.frustum_culling == "voxel_blocks" else 0.0
                        ),
                        "precap_latency_ms": (
                            float(local_query.get("precap_latency_ms", 0.0))
                            if local_config.projection_pre_cap is not None else 0.0
                        ),
                        "precap_strategy": str(local_query.get("precap_strategy", "point_reference")),
                        "representative_points_considered": int(local_query.get("representative_points_considered", 0)),
                        "precap_block_count": int(local_query.get("precap_block_count", 0)),
                        "precap_selected_block_count": int(local_query.get("precap_selected_block_count", 0)),
                        "precap_block_coverage": float(local_query.get("precap_block_coverage", 0.0)),
                        "precap_bucket_count": int(local_query.get("precap_bucket_count", 0)),
                        "precap_selected_bucket_count": int(local_query.get("precap_selected_bucket_count", 0)),
                        "precap_bucket_coverage": float(local_query.get("precap_bucket_coverage", 0.0)),
                        "precap_boundary_block_count": int(local_query.get("precap_boundary_block_count", 0)),
                        "precap_selected_boundary_block_count": int(local_query.get("precap_selected_boundary_block_count", 0)),
                        "precap_boundary_coverage": float(local_query.get("precap_boundary_coverage", 0.0)),
                        "oracle_latency_ms": (
                            float(local_query.get("oracle_latency_ms", 0.0))
                            if local_config.oracle_mode != "off" else 0.0
                        ),
                        "oracle_count": int(local_query.get("oracle_count", 0)),
                        "oracle_block_recall": float(local_query.get("oracle_block_recall", 1.0)),
                        "oracle_projection_recall": float(local_query.get("oracle_projection_recall", 1.0)),
                        "oracle_frustum_recall": float(local_query.get("oracle_frustum_recall", 1.0)),
                        "oracle_overflow": bool(local_query.get("oracle_overflow", False)),
                        "oracle_status": str(local_query.get("status", "ok")),
                        "oracle_reason": str(local_query.get("oracle_reason", "off")),
                        "frustum_count": int(local_query["frustum_count"]),
                        "positive_depth_count": int(local_query.get("positive_depth_count", 0)),
                        "depth_range_count": int(local_query.get("depth_range_count", 0)),
                        "precap_count": int(local_query.get("precap_count", local_query["frustum_count"])),
                        "final_sampling_input_count": int(
                            local_query.get("final_sampling_input_count", local_query["frustum_count"])
                        ),
                        "nonempty_tile_count": int(local_query.get("nonempty_tile_count", 0)),
                        "selected_tile_count": int(local_query.get("selected_tile_count", 0)),
                        "nonempty_depth_bin_count": int(local_query.get("nonempty_depth_bin_count", 0)),
                        "selected_depth_bin_count": int(local_query.get("selected_depth_bin_count", 0)),
                        "nonempty_tile_depth_count": int(local_query.get("nonempty_tile_depth_count", 0)),
                        "selected_tile_depth_count": int(local_query.get("selected_tile_depth_count", 0)),
                        "target_count": int(local_indices.size),
                        "target_fraction": float(local_indices.size / target_points.shape[0]),
                    }
                    if local_query.get("status") == "exact_recall_infeasible":
                        return _registration_failure(
                            attempt_pose, pointcloud["points"].shape[0], 0, "exact_recall_infeasible"
                        ), query_metrics
                    if local_indices.size < local_min_points:
                        reason = "empty_local_submap" if local_indices.size == 0 else "undersized_local_submap"
                        return _registration_failure(
                            attempt_pose, pointcloud["points"].shape[0], local_indices.size, reason
                        ), query_metrics
                    stage_start = time.perf_counter() if stage is not None else 0.0
                    registration_target_points = target_points[local_indices]
                    registration_target_covariances = target_covariances[local_indices] if target_covariances is not None else None
                    registration_target_normals = target_normals[local_indices] if target_normals is not None else None
                    same_local_target = previous_local_indices is not None and np.array_equal(local_indices, previous_local_indices)
                    if stage is not None:
                        stage["gather_ms"] = (time.perf_counter() - stage_start) * 1000
                        stage["tree_reused"] = same_local_target
                        stage_start = time.perf_counter()
                    registration_target_tree = previous_local_tree if same_local_target else build_target_index(registration_target_points)["target_tree"]
                    if stage is not None:
                        stage["tree_ms"] = (time.perf_counter() - stage_start) * 1000

                selected_backend = attempt_backend or backend
                if verifier_enabled and pointcloud["points"].shape[0] < int(registration_cfg.get("min_correspondences", 6)):
                    return _registration_failure(
                        attempt_pose, pointcloud["points"].shape[0], registration_target_points.shape[0],
                        "insufficient_training_source",
                    ), query_metrics
                if normal_source in ("estimate", "local_pca") and selected_backend != "numpy_point_to_point":
                    registration_target_normals = None
                stage_start = time.perf_counter() if stage is not None else 0.0
                if verifier_enabled and normal_source == "estimate" and selected_backend != "numpy_point_to_point":
                    registration_target_normals = estimate_normals(
                        registration_target_points, k=int(registration_cfg.get("normal_k", 20)),
                        target_tree=registration_target_tree,
                    )
                if normal_source == "local_pca" and selected_backend != "numpy_point_to_point":
                    assert local_indices is not None
                    same_local_target = previous_local_indices is not None and np.array_equal(local_indices, previous_local_indices)
                    if stage is not None:
                        stage["normals_reused"] = same_local_target and previous_local_normals is not None
                    if same_local_target and previous_local_normals is not None:
                        registration_target_normals = previous_local_normals
                    else:
                        registration_target_normals = estimate_normals(
                            registration_target_points,
                            k=int(registration_cfg.get("normal_k", registration_cfg.get("normal_max_nn", 30))),
                            target_tree=registration_target_tree,
                        )
                    previous_local_indices = local_indices.copy()
                    previous_local_tree = registration_target_tree
                    previous_local_normals = registration_target_normals
                if stage is not None:
                    stage["normal_prep_ms"] = (time.perf_counter() - stage_start) * 1000
                    stage_start = time.perf_counter()

                attempt_result = register_pointcloud(
                    source_points=pointcloud["points"],
                    target_points=registration_target_points,
                    init_transform=attempt_pose,
                    backend=selected_backend,
                    voxel_size=registration_cfg.get("voxel_size"),
                    max_correspondence_distance=float(registration_cfg.get("max_correspondence_distance", 0.05)),
                    max_iterations=int(registration_cfg.get("max_iterations", 30)),
                    normal_radius=registration_cfg.get("normal_radius"),
                    normal_max_nn=int(registration_cfg.get("normal_max_nn", 30)),
                    target_covariances=registration_target_covariances,
                    target_normals=registration_target_normals,
                    target_tree=registration_target_tree,
                    robust_loss=str(registration_cfg.get("robust_loss", "huber")),
                    loss_scale=float(registration_cfg.get("loss_scale", 0.02)),
                    lm_damping=float(registration_cfg.get("lm_damping", 1e-3)),
                    min_correspondences=int(registration_cfg.get("min_correspondences", 6)),
                    normal_k=int(registration_cfg.get("normal_k", 20)),
                    step_tolerance=registration_cfg.get("step_tolerance"),
                    cost_tolerance=registration_cfg.get("cost_tolerance"),
                    rotation_step_tolerance_deg=float(registration_cfg.get("rotation_step_tolerance_deg", 0.01)),
                    translation_step_tolerance_m=float(registration_cfg.get("translation_step_tolerance_m", 1e-4)),
                    relative_cost_tolerance=float(registration_cfg.get("relative_cost_tolerance", 1e-5)),
                    convergence_patience=int(registration_cfg.get("convergence_patience", 2)),
                    min_iterations=int(registration_cfg.get("min_iterations", 2)),
                    rank_tolerance=float(registration_cfg.get("rank_tolerance", 1e-9)),
                )
                if stage is not None:
                    stage["registration_ms"] = (time.perf_counter() - stage_start) * 1000
                final_attempt_context = {
                    "query_pose": np.asarray(attempt_pose, dtype=np.float64).copy(),
                    "query": local_query if local_enabled else None,
                    "target_points": registration_target_points,
                    "target_covariances": registration_target_covariances,
                    "local_indices": None if local_indices is None else local_indices.copy(),
                    "target_tree": registration_target_tree,
                    "target_normals": registration_target_normals,
                    "target_global_ids": (
                        target_global_ids.copy()
                        if local_indices is None else target_global_ids[local_indices].copy()
                    ),
                }
                return attempt_result, query_metrics

            result, query_metrics = run_attempt(init_transform, 0)
            total_registration_latency_ms = float(result.get("latency_ms", 0.0))
            total_local_query_latency_ms = float(query_metrics["query_latency_ms"])
            total_local_radius_latency_ms = float(query_metrics["radius_latency_ms"])
            total_local_projection_latency_ms = float(query_metrics["projection_latency_ms"])
            total_local_sampling_latency_ms = float(query_metrics["sampling_latency_ms"])
            total_local_block_culling_latency_ms = float(query_metrics["block_culling_latency_ms"])
            total_local_precap_latency_ms = float(query_metrics["precap_latency_ms"])
            total_local_oracle_latency_ms = float(query_metrics["oracle_latency_ms"])
            accepted, acceptance_reason = _tracking_acceptance(result, tracking_acceptance_cfg, backend) if evaluation_protocol == "mvp_strict" else (False, None)
            primary_consistency_reason = None
            primary_consistency_metrics = {}
            if accepted and pose_consistency_enabled:
                accepted, primary_consistency_reason, primary_consistency_metrics = _pose_consistency_acceptance(
                    init_transform, result["transform"], last_valid_estimate, init_source,
                    temporal_gap, pose_consistency_cfg,
                )
                acceptance_reason = primary_consistency_reason
            primary_termination_reason = result.get("termination_reason")
            primary_rejection_reason = acceptance_reason
            primary_result = {
                key: value.copy() if isinstance(value, np.ndarray) else value for key, value in result.items()
            }
            primary_accepted = accepted
            if primary_observer is not None:
                primary_observer(dict(
                    frame_id=frame_id, source_idx=source_idx, initial_pose=init_transform.copy(),
                    primary_pose=primary_result["transform"].copy(), accepted=primary_accepted,
                    target_ids=final_attempt_context["target_global_ids"].copy(),
                ))
            if paired_shadow is not None:
                assert final_attempt_context is not None
                primary_normals = final_attempt_context.get("target_normals")
                if primary_normals is None:
                    primary_normals = estimate_normals(
                        final_attempt_context["target_points"],
                        k=int(registration_cfg.get("normal_k", registration_cfg.get("normal_max_nn", 30))),
                        target_tree=final_attempt_context["target_tree"],
                    )
                local_ids = final_attempt_context.get("local_indices")
                primary_ids = target_global_ids if local_ids is None else target_global_ids[local_ids]
                primary_hessian = np.asarray(primary_result.get("hessian", []), dtype=np.float64)
                if primary_hessian.shape != (6, 6):
                    raise ValueError("paired shadow evaluation requires primary 6x6 Hessian")
                paired_start = time.perf_counter()
                paired_evaluation = paired_shadow.evaluate(BaselineSnapshot(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    source=f"{dataset_cfg.get('sequence', 'sequence')}/frame-{frame_id:06d}",
                    source_points=np.asarray(pointcloud["points"], dtype=np.float64),
                    primary_pose=np.asarray(primary_result["transform"], dtype=np.float64),
                    primary_init_pose=np.asarray(init_transform, dtype=np.float64),
                    primary_accepted=bool(primary_accepted),
                    primary_hessian=primary_hessian,
                    target_points=np.asarray(final_attempt_context["target_points"], dtype=np.float64),
                    target_normals=np.asarray(primary_normals, dtype=np.float64),
                    target_global_ids=np.asarray(primary_ids, dtype=np.int64),
                    target_tree=final_attempt_context["target_tree"],
                    intrinsics=np.asarray(frame["intrinsics"], dtype=np.float64),
                    image=np.asarray(
                        frame["color"] if frame.get("color") is not None
                        else np.zeros((*np.asarray(frame["depth"]).shape, 3), dtype=np.uint8)
                    ),
                    depth=np.asarray(frame["depth"], dtype=np.float64),
                    source_pixel_ids=np.asarray(pointcloud["indices"], dtype=np.int64),
                ))
                paired_shadow_latency_ms = float((time.perf_counter() - paired_start) * 1000.0)
                paired_evaluations.append(paired_evaluation)
                paired_triggered = paired_evaluation.triggered
                pair_id = paired_evaluation.pair_id
                paired_arm_count = len(paired_evaluation.rows)
            recovery_attempted = False
            recovery_stage = 0
            recovery_count = 0
            recovered = False
            recovery_backend = None
            final_consistency_reason = primary_consistency_reason
            final_consistency_metrics = primary_consistency_metrics
            reserve_metrics = observable_reserve.empty_metrics(
                observable_reserve.mode, observable_reserve.unavailable_reason
            )
            reserve_metrics["reserve_primary_accepted"] = primary_accepted
            observable_reserve_attempt = False
            probe_start = time.perf_counter() if probe_enabled else 0.0
            if observable_reserve.active and (reserve_stop_source_idx is None or source_idx < reserve_stop_source_idx):
                image_height, image_width = np.asarray(frame["depth"]).shape[:2]
                reserve_observation = None
                if (
                    (observable_reserve.selection == "runtime_corr_weak"
                     or observable_reserve.spectral_coordinates == "primary_camera_depth")
                    and final_attempt_context is not None
                    and final_attempt_context.get("target_tree") is not None
                ):
                    local_ids = final_attempt_context.get("local_indices")
                    primary_target_ids = target_global_ids if local_ids is None else target_global_ids[local_ids]
                    reserve_observation = RuntimeReserveObservation(
                        source_points_camera=np.asarray(pointcloud["points"], dtype=np.float64),
                        source_pixel_ids=np.asarray(pointcloud["indices"], dtype=np.int64),
                        depth=selection_depth,
                        primary_target_tree=final_attempt_context["target_tree"],
                        primary_target_global_ids=np.asarray(primary_target_ids, dtype=np.int64),
                    )
                reserve_plan_args = (
                    frame_id,
                    primary_result,
                    np.asarray(frame["intrinsics"], dtype=np.float64),
                    image_height,
                    image_width,
                )
                reserve_plan = observable_reserve.plan(*reserve_plan_args, observation=reserve_observation)
                natural_triggered = bool(reserve_plan.metrics["reserve_triggered"])
                force_eligible = (
                    probe_enabled and observable_reserve.mode == "active" and observable_reserve.available
                    and reserve_plan.metrics["reserve_trigger_unavailable_reason"] is None
                    and (observable_reserve.selection != "runtime_corr_weak" or reserve_observation is not None)
                )
                forced_plan_triggered = False
                if force_requested and force_eligible and not natural_triggered:
                    assert force_reserve is not None
                    forced_plan = force_reserve.plan(*reserve_plan_args, observation=reserve_observation)
                    forced_plan_triggered = bool(forced_plan.metrics["reserve_triggered"])
                    forced_plan.metrics.update({
                        key: reserve_plan.metrics[key] for key in (
                            "reserve_triggered", "reserve_trigger_reasons", "reserve_trigger_unavailable_reason",
                        )
                    })
                    forced_plan.metrics["reserve_latency_ms"] += reserve_plan.metrics["reserve_latency_ms"]
                    reserve_plan = forced_plan
                reserve_metrics = reserve_plan.metrics
                reserve_metrics["reserve_primary_accepted"] = primary_accepted
                observable_reserve_attempt = natural_triggered
                if observable_reserve.mode == "active" and (
                    reserve_plan.selected_indices.size or (
                        observable_reserve.attempt_without_selection and (natural_triggered or forced_plan_triggered)
                    )
                ):
                    reserve_metrics["reserve_attempted"] = True
                    observable_reserve_attempt = True
                    forced_attempt = forced_plan_triggered
                    reserve_start = time.perf_counter()
                    assert final_attempt_context is not None
                    base_points = np.asarray(final_attempt_context["target_points"], dtype=np.float64)
                    reserve_indices = reserve_plan.selected_indices
                    if reserve_indices.size:
                        _, unique_positions = np.unique(
                            observable_reserve.global_ids[reserve_indices], return_index=True
                        )
                        reserve_indices = reserve_indices[np.sort(unique_positions)]
                    base_covariances = final_attempt_context.get("target_covariances")
                    base_normals = final_attempt_context.get("target_normals")
                    base_global_ids = np.asarray(final_attempt_context["target_global_ids"], dtype=np.int64)
                    verifier = None
                    verifier_start = time.perf_counter()
                    if verifier_enabled and not bypass_candidate_gate and validation_points.shape[0]:
                        verifier = FrozenCandidateVerifier(
                            validation_points, base_points, base_normals,
                            max_distance=float(registration_cfg.get("max_correspondence_distance", 0.05)),
                            loss_scale=float(registration_cfg.get("loss_scale", 0.02)),
                            robust_loss=str(registration_cfg.get("robust_loss", "huber")),
                            min_correspondences=int(registration_cfg.get("min_correspondences", 6)),
                            spatial_groups=validation_groups,
                        )
                    verifier_setup_ms = (time.perf_counter() - verifier_start) * 1000.0
                    reserve_points = observable_reserve.points[reserve_indices]
                    reserve_covariances = observable_reserve.covariances[reserve_indices]
                    reserve_normals = observable_reserve.normals[reserve_indices]
                    augmented_points, augmented_covariances, frozen_normals, removed_base_count = _candidate_target_arrays(
                        mode=observable_reserve.candidate_target_mode,
                        base_points=base_points,
                        base_covariances=base_covariances,
                        base_normals=base_normals,
                        base_global_ids=base_global_ids,
                        reserve_points=reserve_points,
                        reserve_covariances=reserve_covariances,
                        reserve_normals=reserve_normals,
                        replacement_seed=observable_reserve.replacement_seed,
                    )
                    augmented_tree = build_target_index(augmented_points)["target_tree"]
                    if observable_reserve.candidate_normal_mode == "frozen_concat":
                        if frozen_normals is None:
                            raise ValueError("frozen_concat candidate normals require primary target normals")
                        augmented_normals = frozen_normals
                        normal_source_label = "primary_plus_sidecar_frozen"
                    else:
                        augmented_normals = estimate_normals(
                            augmented_points,
                            k=int(registration_cfg.get("normal_k", registration_cfg.get("normal_max_nn", 30))),
                            target_tree=augmented_tree,
                        )
                        normal_source_label = "augmented_local_pca"
                    reserve_metrics["reserve_rerun_init_source"] = "primary_result"
                    reserve_metrics["reserve_rerun_normal_source"] = normal_source_label
                    reserve_metrics["reserve_candidate_target_mode"] = observable_reserve.candidate_target_mode
                    reserve_metrics["reserve_candidate_base_removed"] = removed_base_count
                    reserve_metrics["reserve_candidate_target_count"] = int(augmented_points.shape[0])
                    candidate_target_count = int(augmented_points.shape[0])
                    candidate_reserve_count = (
                        candidate_target_count - int(base_points.shape[0]) + removed_base_count
                    )
                    candidate_result = register_pointcloud(
                        source_points=pointcloud["points"],
                        target_points=augmented_points,
                        init_transform=np.array(primary_result["transform"], dtype=np.float64, copy=True),
                        backend=backend,
                        voxel_size=registration_cfg.get("voxel_size"),
                        max_correspondence_distance=float(registration_cfg.get("max_correspondence_distance", 0.05)),
                        max_iterations=observable_reserve.rerun_iterations,
                        normal_radius=registration_cfg.get("normal_radius"),
                        normal_max_nn=int(registration_cfg.get("normal_max_nn", 30)),
                        target_covariances=augmented_covariances,
                        target_normals=augmented_normals,
                        target_tree=augmented_tree,
                        robust_loss=str(registration_cfg.get("robust_loss", "huber")),
                        loss_scale=float(registration_cfg.get("loss_scale", 0.02)),
                        lm_damping=float(registration_cfg.get("lm_damping", 1e-3)),
                        min_correspondences=int(registration_cfg.get("min_correspondences", 6)),
                        normal_k=int(registration_cfg.get("normal_k", 20)),
                        step_tolerance=registration_cfg.get("step_tolerance"),
                        cost_tolerance=registration_cfg.get("cost_tolerance"),
                        rotation_step_tolerance_deg=float(registration_cfg.get("rotation_step_tolerance_deg", 0.01)),
                        translation_step_tolerance_m=float(registration_cfg.get("translation_step_tolerance_m", 1e-4)),
                        relative_cost_tolerance=float(registration_cfg.get("relative_cost_tolerance", 1e-5)),
                        convergence_patience=int(registration_cfg.get("convergence_patience", 2)),
                        min_iterations=int(registration_cfg.get("min_iterations", 2)),
                        rank_tolerance=float(registration_cfg.get("rank_tolerance", 1e-9)),
                    )
                    candidate_latency_ms = float(candidate_result.get("latency_ms", 0.0))
                    if not probe_enabled:
                        total_registration_latency_ms += candidate_latency_ms
                    reserve_metrics["reserve_latency_ms"] += float((time.perf_counter() - reserve_start) * 1000.0)
                    tracking_accepted, candidate_reason = _tracking_acceptance(
                        candidate_result, tracking_acceptance_cfg, backend
                    )
                    candidate_consistency_metrics = {}
                    candidate_consistency_reason = None
                    consistency_accepted = tracking_accepted
                    primary_pose = np.asarray(primary_result["transform"], dtype=np.float64)
                    candidate_pose = np.asarray(candidate_result.get("transform"), dtype=np.float64)
                    candidate_pose_valid = _is_finite_pose(candidate_pose, max_pose_translation_norm)
                    if probe_enabled or bypass_candidate_gate:
                        candidate_pose_valid = candidate_pose_valid and bool(
                            np.allclose(candidate_pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-7)
                        )
                        if candidate_pose_valid:
                            try:
                                se3_log(candidate_pose)
                            except ValueError:
                                candidate_pose_valid = False
                    candidate_basic_valid = tracking_accepted and candidate_pose_valid
                    if (
                        tracking_accepted and pose_consistency_enabled and not bypass_candidate_gate
                        and (not probe_enabled or candidate_basic_valid)
                    ):
                        consistency_accepted, candidate_consistency_reason, candidate_consistency_metrics = _pose_consistency_acceptance(
                            np.asarray(primary_result["transform"], dtype=np.float64),
                            candidate_result["transform"], last_valid_estimate, init_source,
                            temporal_gap, pose_consistency_cfg,
                        )
                        candidate_reason = candidate_consistency_reason
                    candidate_eigen = np.asarray(candidate_result.get("eigenvalues", []), dtype=np.float64)
                    candidate_hessian = np.asarray(candidate_result.get("hessian", []), dtype=np.float64)
                    candidate_trace = (
                        float(np.trace(candidate_hessian)) if candidate_hessian.shape == (6, 6) else None
                    )
                    reserve_metrics.update({
                        "reserve_correspondence": candidate_result.get("num_correspondences"),
                        "reserve_candidate_optimizer_valid": bool(
                            candidate_result.get("optimizer_valid", candidate_result.get("converged"))
                        ),
                        "reserve_candidate_fitness": candidate_result.get("fitness"),
                        "reserve_candidate_rmse": candidate_result.get(
                            "point_to_plane_rmse", candidate_result.get("inlier_rmse")
                        ),
                        "reserve_candidate_effective_rank": candidate_result.get("effective_rank"),
                        "reserve_candidate_hessian_trace": candidate_trace,
                        "reserve_candidate_normalized_lambda_min": (
                            float(candidate_eigen[0] / candidate_trace)
                            if candidate_eigen.size and candidate_trace is not None and candidate_trace > 0.0 else None
                        ),
                        "reserve_augmented_condition": candidate_result.get("condition_number"),
                        "reserve_augmented_lambda_min": float(candidate_eigen[0]) if candidate_eigen.size else None,
                        "reserve_tracking_accepted": tracking_accepted,
                        "reserve_consistency_accepted": consistency_accepted,
                    })
                    try:
                        refinement = relative_se3_log(primary_pose, candidate_pose)
                        candidate_init = relative_se3_log(np.asarray(init_transform, dtype=np.float64), candidate_pose)
                    except ValueError as exc:
                        refinement = candidate_init = None
                        tracking_accepted = False
                        consistency_accepted = False
                        candidate_reason = "tracking_invalid_rotation"
                        reserve_metrics["reserve_candidate_pose_diagnostic_error"] = str(exc)
                        reserve_metrics["reserve_tracking_accepted"] = False
                        reserve_metrics["reserve_consistency_accepted"] = False
                    if (probe_enabled or bypass_candidate_gate) and not candidate_pose_valid:
                        tracking_accepted = consistency_accepted = False
                        candidate_reason = "tracking_quality_invalid_pose"
                        reserve_metrics["reserve_tracking_accepted"] = False
                        reserve_metrics["reserve_consistency_accepted"] = False
                    if refinement is not None:
                        for name, value in zip(("rx", "ry", "rz", "tx", "ty", "tz"), refinement["twist"]):
                            reserve_metrics[f"reserve_refinement_twist_{name}"] = float(value)
                    reserve_metrics.update({
                        "frame_id": frame_id,
                        "reserve_refinement_twist_norm": None if refinement is None else refinement["norm"],
                        "reserve_refinement_rotation_deg": None if refinement is None else refinement["rotation_norm_deg"],
                        "reserve_refinement_translation_m": None if refinement is None else refinement["translation_norm_m"],
                        "reserve_candidate_init_twist_norm": None if candidate_init is None else candidate_init["norm"],
                        "reserve_candidate_init_rotation_deg": None if candidate_init is None else candidate_init["rotation_norm_deg"],
                        "reserve_candidate_init_translation_m": None if candidate_init is None else candidate_init["translation_norm_m"],
                        "reserve_candidate_temporal_rotation_deg": candidate_consistency_metrics.get(
                            "temporal_rotation_deg"
                        ),
                        "reserve_different_target_fitness_ratio": observable_reserve._finite_ratio(
                            candidate_result.get("fitness"), primary_result.get("fitness")
                        ),
                        "reserve_different_target_ptp_rmse_ratio": observable_reserve._finite_ratio(
                            candidate_result.get("point_to_plane_rmse", candidate_result.get("inlier_rmse")),
                            primary_result.get("point_to_plane_rmse", primary_result.get("inlier_rmse")),
                        ),
                        "reserve_different_target_robust_cost_ratio": observable_reserve._finite_ratio(
                            candidate_result.get("final_mean_robust_cost"), primary_result.get("final_mean_robust_cost")
                        ),
                    })
                    evaluate_candidate_gate = not bypass_candidate_gate and (not probe_enabled or refinement is not None)
                    if evaluate_candidate_gate and observable_reserve.stability_active:
                        stability_start = time.perf_counter()
                        common_kwargs = dict(
                            source_points=np.asarray(pointcloud["points"], dtype=np.float64),
                            target_points=base_points,
                            target_normals=np.asarray(final_attempt_context["target_normals"], dtype=np.float64),
                            target_tree=final_attempt_context["target_tree"],
                            max_correspondence_distance=float(registration_cfg.get("max_correspondence_distance", 0.05)),
                            robust_loss=str(registration_cfg.get("robust_loss", "huber")),
                            loss_scale=float(registration_cfg.get("loss_scale", 0.02)),
                            min_correspondences=int(registration_cfg.get("min_correspondences", 6)),
                            rank_tolerance=float(registration_cfg.get("rank_tolerance", 1e-9)),
                        )
                        primary_common = evaluate_common_target_pose(pose=primary_pose, **common_kwargs)
                        candidate_common = evaluate_common_target_pose(pose=candidate_pose, **common_kwargs)
                        reserve_metrics.update({
                            "reserve_common_target_cost_ratio": observable_reserve._finite_ratio(
                                candidate_common["robust_cost_per_corr"], primary_common["robust_cost_per_corr"]
                            ),
                            "reserve_common_target_rmse_ratio": observable_reserve._finite_ratio(
                                candidate_common["point_to_plane_rmse"], primary_common["point_to_plane_rmse"]
                            ),
                            "reserve_common_target_fitness_ratio": observable_reserve._finite_ratio(
                                candidate_common["fitness"], primary_common["fitness"]
                            ),
                            "reserve_common_target_corr_retention": observable_reserve._finite_ratio(
                                candidate_common["corr_count"], primary_common["corr_count"]
                            ),
                        })
                    improves, improvement_reason = False, candidate_reason
                    if evaluate_candidate_gate:
                        improves, improvement_reason, improvement_diagnostics = observable_reserve.evaluate_improvement(
                            primary_result, candidate_result, reserve_metrics, observation=reserve_observation
                        )
                        reserve_metrics.update(improvement_diagnostics)
                        reserve_metrics["reserve_relative_improvement_accepted"] = improves
                    base_reserve_accepted = tracking_accepted and consistency_accepted and improves
                    if verifier_enabled and evaluate_candidate_gate:
                        verifier_start = time.perf_counter()
                        verification = (
                            verifier.evaluate_pair(primary_pose, candidate_pose, verifier_options)
                            if verifier is not None else {
                                "accepted": False, "reason": "empty_holdout", "metrics": {
                                    "reserve_verifier_accepted": False,
                                    "reserve_verifier_reason": "empty_holdout",
                                    "reserve_verifier_source_count": 0,
                                },
                            }
                        )
                        reserve_metrics.update(verification["metrics"])
                        verifier_evaluate_ms = (time.perf_counter() - verifier_start) * 1000.0
                        reserve_metrics["reserve_verifier_latency_ms"] = verifier_setup_ms + verifier_evaluate_ms
                        reserve_metrics["reserve_latency_ms"] += verifier_evaluate_ms
                        base_reserve_accepted = base_reserve_accepted and verification["accepted"]
                        if not verification["accepted"]:
                            improvement_reason = "verifier_" + verification["reason"]
                    if not _is_finite_pose(candidate_pose, max_pose_translation_norm):
                        base_reserve_accepted = False
                        candidate_reason = "tracking_quality_non_finite_pose"
                        tracking_accepted = False
                    stability_accepted = True
                    if evaluate_candidate_gate and observable_reserve.stability_active:
                        stability_accepted, stability_reason, stability_diagnostics = observable_reserve.prospective_stability(
                            reserve_metrics, candidate_pose, primary_pose
                        )
                        reserve_metrics.update(stability_diagnostics)
                        reserve_metrics["reserve_stability_latency_ms"] = float(
                            (time.perf_counter() - stability_start) * 1000.0
                        )
                        reserve_metrics.update({
                            "reserve_stability_gate_enabled": True,
                            "reserve_stability_gate_mode": observable_reserve.stability_mode,
                            "reserve_stability_would_accept": stability_accepted,
                            "reserve_stability_reasons": "" if stability_accepted else stability_reason,
                            "reserve_stability_accepted": stability_accepted,
                        })
                        reserve_metrics["reserve_latency_ms"] += float(
                            reserve_metrics["reserve_stability_latency_ms"]
                        )
                    reserve_metrics["reserve_accepted"] = natural_triggered and (
                        candidate_basic_valid if bypass_candidate_gate else base_reserve_accepted and (
                            stability_accepted if observable_reserve.stability_mode == "enforce" else True
                        )
                    )
                    if reserve_metrics["reserve_accepted"]:
                        if bypass_candidate_gate:
                            reserve_metrics["reserve_adoption_reason"] = "candidate_gate_bypassed"
                        elif observable_reserve.accept_only_if.get("protocol") in (
                                "branch_aware_information_v3", "branch_aware_information_v4",
                        ):
                            reserve_metrics["reserve_adoption_reason"] = improvement_reason
                        else:
                            reserve_metrics["reserve_adoption_reason"] = (
                                "rescued_primary_rejection" if not primary_accepted else "improved_primary"
                            )
                        if not probe_enabled:
                            result = candidate_result
                            accepted = True
                            acceptance_reason = None
                            final_consistency_reason = candidate_consistency_reason
                            final_consistency_metrics = candidate_consistency_metrics
                            reserve_metrics["reserve_used"] = True
                    elif not primary_accepted and not probe_enabled:
                        acceptance_reason = primary_rejection_reason or candidate_reason
                    if not reserve_metrics["reserve_accepted"]:
                        if base_reserve_accepted and observable_reserve.stability_mode == "enforce" and not stability_accepted:
                            reserve_metrics["reserve_adoption_reason"] = reserve_metrics["reserve_stability_reasons"]
                        else:
                            reserve_metrics["reserve_adoption_reason"] = (
                                candidate_reason if not tracking_accepted or not consistency_accepted else improvement_reason
                            )
                    if not probe_enabled and not bypass_candidate_gate:
                        observable_reserve.add_stability_attempt(
                            frame_id, primary_pose, candidate_pose, init_transform,
                            reserve_metrics, bool(reserve_metrics["reserve_used"]),
                        )
                if observable_reserve.mode == "active" and not probe_enabled and not bypass_candidate_gate:
                    observable_reserve.commit_branch_adoption(
                        frame_id,
                        bool(reserve_metrics["reserve_used"]),
                        reserve_metrics.get("reserve_information_v3_branch"),
                    )
                    observable_reserve.commit_stability(
                        frame_id,
                        bool(reserve_metrics["reserve_used"]),
                        np.asarray([
                            reserve_metrics.get(f"reserve_refinement_twist_{axis}")
                            for axis in ("rx", "ry", "rz", "tx", "ty", "tz")
                        ], dtype=np.float64) if reserve_metrics["reserve_used"] else None,
                    )
                observable_reserve.add_record(reserve_metrics)
                selection_event_writer.write(frame_id, reserve_plan, reserve_metrics)
            if probe_enabled:
                probe_latency_ms = (time.perf_counter() - probe_start) * 1000.0
            if evaluation_protocol == "mvp_strict" and not accepted and recovery_enabled:
                assert last_valid_estimate is not None
                for tier in range(1, recovery_max_tier + 1):
                    recovery_attempted = True
                    recovery_count += 1
                    recovery_stage = tier
                    result, query_metrics = run_attempt(last_valid_estimate, tier)
                    total_registration_latency_ms += float(result.get("latency_ms", 0.0))
                    total_local_query_latency_ms += float(query_metrics["query_latency_ms"])
                    total_local_radius_latency_ms += float(query_metrics["radius_latency_ms"])
                    total_local_projection_latency_ms += float(query_metrics["projection_latency_ms"])
                    total_local_sampling_latency_ms += float(query_metrics["sampling_latency_ms"])
                    total_local_block_culling_latency_ms += float(query_metrics["block_culling_latency_ms"])
                    total_local_precap_latency_ms += float(query_metrics["precap_latency_ms"])
                    total_local_oracle_latency_ms += float(query_metrics["oracle_latency_ms"])
                    accepted, acceptance_reason = _tracking_acceptance(result, tracking_acceptance_cfg, backend)
                    final_consistency_reason = None
                    final_consistency_metrics = {}
                    if accepted and pose_consistency_enabled:
                        accepted, final_consistency_reason, final_consistency_metrics = _pose_consistency_acceptance(
                            last_valid_estimate, result["transform"], last_valid_estimate, "previous_pose",
                            temporal_gap, pose_consistency_cfg, correction_recovery_multiplier,
                        )
                        acceptance_reason = final_consistency_reason
                    if accepted:
                        recovered = True
                        recovery_backend = backend
                        break
                if not accepted and verifier_backend:
                    recovery_attempted = True
                    recovery_count += 1
                    recovery_stage = recovery_max_tier + 1
                    result, query_metrics = run_attempt(last_valid_estimate, 3, str(verifier_backend))
                    total_registration_latency_ms += float(result.get("latency_ms", 0.0))
                    total_local_query_latency_ms += float(query_metrics["query_latency_ms"])
                    total_local_radius_latency_ms += float(query_metrics["radius_latency_ms"])
                    total_local_projection_latency_ms += float(query_metrics["projection_latency_ms"])
                    total_local_sampling_latency_ms += float(query_metrics["sampling_latency_ms"])
                    total_local_block_culling_latency_ms += float(query_metrics["block_culling_latency_ms"])
                    total_local_precap_latency_ms += float(query_metrics["precap_latency_ms"])
                    total_local_oracle_latency_ms += float(query_metrics["oracle_latency_ms"])
                    accepted, acceptance_reason = _tracking_acceptance(
                        result, verifier_acceptance_cfg, str(verifier_backend)
                    )
                    final_consistency_reason = None
                    final_consistency_metrics = {}
                    if accepted and pose_consistency_enabled:
                        accepted, final_consistency_reason, final_consistency_metrics = _pose_consistency_acceptance(
                            last_valid_estimate, result["transform"], last_valid_estimate, "previous_pose",
                            temporal_gap, pose_consistency_cfg, correction_recovery_multiplier,
                        )
                        acceptance_reason = final_consistency_reason
                    if accepted:
                        recovered = True
                        recovery_backend = str(verifier_backend)

            result["latency_ms"] = total_registration_latency_ms
            local_query_latency_ms = total_local_query_latency_ms
            local_radius_latency_ms = total_local_radius_latency_ms
            local_projection_latency_ms = total_local_projection_latency_ms
            local_sampling_latency_ms = total_local_sampling_latency_ms
            local_candidate_count = query_metrics["candidate_count"]
            local_culling_mode = query_metrics["culling_mode"]
            local_tested_block_count = query_metrics["tested_block_count"]
            local_intersecting_block_count = query_metrics["intersecting_block_count"]
            local_block_candidate_count = query_metrics["block_candidate_count"]
            local_projection_input_count = query_metrics["projection_input_count"]
            local_projection_cap = query_metrics["projection_cap"]
            local_projection_cap_applied = query_metrics["projection_cap_applied"]
            local_projection_cap_bypassed = query_metrics["projection_cap_bypassed"]
            local_projection_cap_utilization = query_metrics["projection_cap_utilization"]
            local_block_culling_latency_ms = total_local_block_culling_latency_ms
            local_precap_latency_ms = total_local_precap_latency_ms
            local_precap_strategy = query_metrics["precap_strategy"]
            local_representative_points_considered = query_metrics["representative_points_considered"]
            local_precap_block_count = query_metrics["precap_block_count"]
            local_precap_selected_block_count = query_metrics["precap_selected_block_count"]
            local_precap_block_coverage = query_metrics["precap_block_coverage"]
            local_precap_bucket_count = query_metrics["precap_bucket_count"]
            local_precap_selected_bucket_count = query_metrics["precap_selected_bucket_count"]
            local_precap_bucket_coverage = query_metrics["precap_bucket_coverage"]
            local_precap_boundary_block_count = query_metrics["precap_boundary_block_count"]
            local_precap_selected_boundary_block_count = query_metrics["precap_selected_boundary_block_count"]
            local_precap_boundary_coverage = query_metrics["precap_boundary_coverage"]
            local_oracle_latency_ms = total_local_oracle_latency_ms
            local_oracle_count = query_metrics["oracle_count"]
            local_oracle_block_recall = query_metrics["oracle_block_recall"]
            local_oracle_projection_recall = query_metrics["oracle_projection_recall"]
            local_oracle_frustum_recall = query_metrics["oracle_frustum_recall"]
            local_oracle_overflow = query_metrics["oracle_overflow"]
            local_oracle_status = query_metrics["oracle_status"]
            local_oracle_reason = query_metrics["oracle_reason"]
            local_frustum_count = query_metrics["frustum_count"]
            local_positive_depth_count = query_metrics["positive_depth_count"]
            local_depth_range_count = query_metrics["depth_range_count"]
            local_precap_count = query_metrics["precap_count"]
            local_final_sampling_input_count = query_metrics["final_sampling_input_count"]
            local_nonempty_tile_count = query_metrics["nonempty_tile_count"]
            local_selected_tile_count = query_metrics["selected_tile_count"]
            local_nonempty_depth_bin_count = query_metrics["nonempty_depth_bin_count"]
            local_selected_depth_bin_count = query_metrics["selected_depth_bin_count"]
            local_nonempty_tile_depth_count = query_metrics["nonempty_tile_depth_count"]
            local_selected_tile_depth_count = query_metrics["selected_tile_depth_count"]
            local_target_count = query_metrics["target_count"]
            local_target_fraction = query_metrics["target_fraction"]

            fallback_used = False
            if evaluation_protocol == "audit_legacy":
                if fallback_on_divergence and not _is_finite_pose(result["transform"], max_pose_translation_norm):
                    estimated_c2w = gt_c2w
                    fallback_used = True
                else:
                    estimated_c2w = result["transform"]
                if fallback_on_divergence and not _is_finite_pose(estimated_c2w, max_pose_translation_norm):
                    estimated_c2w = gt_c2w
                    fallback_used = True
                tracking_valid = _is_finite_pose(estimated_c2w, max_pose_translation_norm)
                tracking_status = "fallback" if fallback_used else ("tracked" if tracking_valid else "lost")
            else:
                tracking_valid = accepted and _is_finite_pose(result["transform"], max_pose_translation_norm)
                if accepted and not tracking_valid:
                    acceptance_reason = "tracking_quality_non_finite_pose"
                if acceptance_reason and (not result.get("failure_reason") or accepted):
                    result["failure_reason"] = acceptance_reason
                if tracking_valid:
                    estimated_c2w = np.asarray(result["transform"], dtype=np.float64)
                    last_valid_estimate = estimated_c2w
                    valid_pose_history.append((output_idx, estimated_c2w.copy(), recovered))
                    tracking_status = "tracked"
                else:
                    estimated_c2w = last_valid_estimate.copy()
                    tracking_status = "lost_hold_last_valid"

        previous_estimate = estimated_c2w if _is_finite_pose(estimated_c2w, max_pose_translation_norm) else None
        est_poses.append(estimated_c2w)
        gt_poses.append(gt_c2w)
        timestamps.append(timestamp)
        if activation_diagnostics.enabled:
            image_height, image_width = np.asarray(frame["depth"]).shape[:2]
            activation_diagnostics.evaluate_frame(
                frame_id=frame_id,
                timestamp=timestamp,
                source_points=np.asarray(pointcloud["points"], dtype=np.float64),
                pose=np.asarray(estimated_c2w, dtype=np.float64),
                intrinsics=np.asarray(frame["intrinsics"], dtype=np.float64),
                image_height=image_height,
                image_width=image_width,
                registration_result=result,
                final_attempt=final_attempt_context,
            )
        final_target_oracle_status = "disabled"
        final_target_oracle_latency_ms = 0.0
        if final_target_oracle is not None:
            image_height, image_width = np.asarray(frame["depth"]).shape[:2]
            final_target_oracle.evaluate_frame(
                frame_id, timestamp, np.asarray(pointcloud["points"], dtype=np.float64),
                np.asarray(estimated_c2w, dtype=np.float64), np.asarray(frame["intrinsics"], dtype=np.float64),
                image_height, image_width,
            )
            final_target_oracle_status = str(final_target_oracle.records[-1]["status"])
            final_target_oracle_latency_ms = float(final_target_oracle.records[-1]["diagnostic_latency_ms"])
        total_frame_latency_ms = float((time.perf_counter() - frame_start) * 1000.0)
        if stages is not None:
            stage_profile(dict(stages, frame_id=frame_id, total_frame_ms=total_frame_latency_ms,
                               tracking_valid=tracking_valid, recovered=recovered,
                               recovery_attempted=recovery_attempted))
        diagnostic_latency_ms = (
            float(activation_diagnostics.records[-1]["diagnostic_latency_ms"])
            if activation_diagnostics.enabled and activation_diagnostics.records else 0.0
        )
        plan_only_latency_ms = (
            float(reserve_metrics["reserve_latency_ms"])
            if evaluation_protocol == "mvp_strict" and observable_reserve.mode == "plan_only" else 0.0
        )
        oracle_diagnostic_latency_ms = (
            float(local_oracle_latency_ms)
            if evaluation_protocol == "mvp_strict" and local_config is not None and local_config.oracle_mode == "measure"
            else 0.0
        )
        production_equivalent_total_frame_latency_ms = max(
            0.0,
            total_frame_latency_ms - diagnostic_latency_ms - plan_only_latency_ms - oracle_diagnostic_latency_ms
            - paired_shadow_latency_ms - final_target_oracle_latency_ms
            - (probe_latency_ms if observable_reserve.mode != "plan_only" else 0.0),
        )
        if candidate_probe is not None:
            natural_triggered = bool(reserve_metrics["reserve_triggered"])
            gated_adopted = bool(reserve_metrics["reserve_accepted"])
            ungated_adopted = natural_triggered and candidate_basic_valid
            baseline_source = (
                "seed" if output_idx == 0 else "hold_last_valid" if not tracking_valid
                else "recovery" if recovered else "primary"
            )
            probe_row = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source_idx": source_idx,
                "init_pose": (estimated_c2w if output_idx == 0 else init_transform).copy(),
                "gt_pose": gt_c2w.copy(),
                "natural_triggered": natural_triggered,
                "force_eligible": force_eligible,
                "force_requested": force_requested,
                "forced_attempt": forced_attempt,
                "candidate_attempted": bool(reserve_metrics["reserve_attempted"]),
                "candidate_submitted": bool(reserve_metrics["reserve_used"]),
                "candidate_basic_valid": candidate_basic_valid,
                "candidate_latency_ms": candidate_latency_ms,
                "probe_latency_ms": probe_latency_ms,
                "candidate_reserve_count": candidate_reserve_count,
                "candidate_target_count": candidate_target_count,
                "primary_target_count": primary_result["num_target_points"],
                "gated_adopted": gated_adopted,
                "ungated_adopted": ungated_adopted,
                "baseline_recovery_attempted": recovery_attempted,
                "baseline_recovery_stage": recovery_stage,
                "reserve_metrics": dict(reserve_metrics),
            }
            for label, pose, valid, source in (
                ("primary", primary_result["transform"], primary_accepted, "seed" if output_idx == 0 else "primary"),
                ("candidate", candidate_pose, candidate_basic_valid, "reserve" if candidate_pose is not None else None),
                ("baseline", estimated_c2w, tracking_valid, baseline_source),
                ("gated", candidate_pose if gated_adopted else estimated_c2w, gated_adopted or tracking_valid,
                 "reserve" if gated_adopted else baseline_source),
                ("ungated", candidate_pose if ungated_adopted else estimated_c2w, ungated_adopted or tracking_valid,
                 "reserve" if ungated_adopted else baseline_source),
            ):
                finite = _is_finite_pose(pose, float("inf"))
                probe_row[f"{label}_pose"] = np.array(pose, dtype=np.float64, copy=True) if finite else None
                probe_row[f"{label}_tracking_valid"] = bool(valid and finite)
                probe_row[f"{label}_source"] = source
                # Ground truth is used only for reporting, never for either adoption decision.
                error_valid = valid and finite and _is_finite_pose(gt_c2w, float("inf"))
                probe_row[f"{label}_raw_translation_error_m"] = (
                    float(np.linalg.norm(pose[:3, 3] - gt_c2w[:3, 3])) if error_valid else None
                )
                probe_row[f"{label}_raw_rotation_error_deg"] = (
                    _rotation_angle_deg(gt_c2w[:3, :3].T @ pose[:3, :3]) if error_valid else None
                )
            candidate_probe(probe_row)
        if memory_enabled and (output_idx % memory_check_interval == 0 or rss_mb is None):
            rss_mb, peak_rss_mb = get_process_memory_mb()
            if max_rss_mb is not None and rss_mb > float(max_rss_mb):
                raise MemoryError(
                    f"runtime RSS {rss_mb:.1f} MiB exceeds configured max_rss_mb={float(max_rss_mb):.1f} MiB"
                )
        row = {
                "frame_id": frame_id,
                "timestamp": f"{timestamp:.9f}",
                "fitness": f"{result['fitness']:.12g}",
                "inlier_rmse": f"{result['inlier_rmse']:.12g}",
                "latency_ms": f"{result['latency_ms']:.12g}",
                "num_source_points": result["num_source_points"],
                "num_target_points": result["num_target_points"],
                "converged": result["converged"],
                "fallback_used": fallback_used,
                "tracking_valid": tracking_valid,
                "tracking_lost": not tracking_valid,
                "tracking_status": tracking_status,
            }
        if evaluation_protocol == "mvp_strict":
            eigenvalues = np.asarray(result.get("eigenvalues", []), dtype=np.float64)
            covariance = np.asarray(result.get("pose_covariance", []), dtype=np.float64)
            row.update({
                "accepted": accepted,
                "optimizer_valid": bool(result.get("optimizer_valid", result.get("converged"))),
                "termination_reason": _csv_value(result.get("termination_reason")),
                "acceptance_reason": _csv_value(acceptance_reason),
                "fitness": _csv_value(result["fitness"]),
                "inlier_rmse": _csv_value(result["inlier_rmse"]),
                "latency_ms": _csv_value(result["latency_ms"]),
                "init_source": init_source,
                "prediction_translation_m": _csv_value(prediction_translation_m),
                "prediction_rotation_deg": _csv_value(prediction_rotation_deg),
                "prediction_guard": prediction_guard,
                "local_query_latency_ms": _csv_value(local_query_latency_ms),
                "local_radius_latency_ms": _csv_value(local_radius_latency_ms),
                "local_projection_latency_ms": _csv_value(local_projection_latency_ms),
                "local_sampling_latency_ms": _csv_value(local_sampling_latency_ms),
                "total_frame_latency_ms": _csv_value(total_frame_latency_ms),
                "production_equivalent_total_frame_latency_ms": _csv_value(
                    production_equivalent_total_frame_latency_ms
                ),
                "local_index_build_ms": _csv_value(local_index_build_ms),
                "block_index_build_ms": _csv_value(block_index_build_ms),
                "block_index_memory_bytes": block_index_memory_bytes,
                "rss_mb": _csv_value(rss_mb),
                "peak_rss_mb": _csv_value(peak_rss_mb),
                "local_candidate_count": local_candidate_count,
                "local_culling_mode": local_culling_mode,
                "local_tested_block_count": local_tested_block_count,
                "local_intersecting_block_count": local_intersecting_block_count,
                "local_block_candidate_count": local_block_candidate_count,
                "local_projection_input_count": local_projection_input_count,
                "local_projection_cap": _csv_value(local_projection_cap),
                "local_projection_cap_applied": local_projection_cap_applied,
                "local_projection_cap_bypassed": local_projection_cap_bypassed,
                "local_projection_cap_utilization": _csv_value(local_projection_cap_utilization),
                "local_block_culling_latency_ms": _csv_value(local_block_culling_latency_ms),
                "local_precap_latency_ms": _csv_value(local_precap_latency_ms),
                "local_precap_strategy": local_precap_strategy,
                "local_representative_points_considered": local_representative_points_considered,
                "local_precap_block_count": local_precap_block_count,
                "local_precap_selected_block_count": local_precap_selected_block_count,
                "local_precap_block_coverage": _csv_value(local_precap_block_coverage),
                "local_precap_bucket_count": local_precap_bucket_count,
                "local_precap_selected_bucket_count": local_precap_selected_bucket_count,
                "local_precap_bucket_coverage": _csv_value(local_precap_bucket_coverage),
                "local_precap_boundary_block_count": local_precap_boundary_block_count,
                "local_precap_selected_boundary_block_count": local_precap_selected_boundary_block_count,
                "local_precap_boundary_coverage": _csv_value(local_precap_boundary_coverage),
                "local_oracle_latency_ms": _csv_value(local_oracle_latency_ms),
                "local_oracle_count": local_oracle_count,
                "local_oracle_block_recall": _csv_value(local_oracle_block_recall),
                "local_oracle_projection_recall": _csv_value(local_oracle_projection_recall),
                "local_oracle_frustum_recall": _csv_value(local_oracle_frustum_recall),
                "local_oracle_overflow": local_oracle_overflow,
                "local_oracle_status": local_oracle_status,
                "local_oracle_reason": local_oracle_reason,
                "local_frustum_count": local_frustum_count,
                "local_positive_depth_count": local_positive_depth_count,
                "local_depth_range_count": local_depth_range_count,
                "local_precap_count": local_precap_count,
                "local_final_sampling_input_count": local_final_sampling_input_count,
                "local_nonempty_tile_count": local_nonempty_tile_count,
                "local_selected_tile_count": local_selected_tile_count,
                "local_nonempty_depth_bin_count": local_nonempty_depth_bin_count,
                "local_selected_depth_bin_count": local_selected_depth_bin_count,
                "local_nonempty_tile_depth_count": local_nonempty_tile_depth_count,
                "local_selected_tile_depth_count": local_selected_tile_depth_count,
                "local_target_count": local_target_count,
                "local_target_fraction": _csv_value(local_target_fraction),
                "failure_reason": _csv_value(result.get("failure_reason")),
                "primary_termination_reason": _csv_value(primary_termination_reason),
                "primary_rejection_reason": _csv_value(primary_rejection_reason),
                "primary_optimizer_valid": bool(primary_result.get("optimizer_valid", primary_result.get("converged"))),
                "primary_fitness": _csv_value(primary_result.get("fitness")),
                "primary_inlier_rmse": _csv_value(primary_result.get("inlier_rmse")),
                "primary_point_to_plane_rmse": _csv_value(primary_result.get("point_to_plane_rmse")),
                "primary_num_correspondences": _csv_value(primary_result.get("num_correspondences")),
                "primary_effective_rank": _csv_value(primary_result.get("effective_rank")),
                "primary_condition_number": _csv_condition_number(primary_result.get("condition_number")),
                "primary_lambda_min": _csv_value(
                    np.min(np.asarray(primary_result.get("eigenvalues", []), dtype=np.float64))
                    if np.asarray(primary_result.get("eigenvalues", [])).size else None
                ),
                "primary_hessian_trace": _csv_value(
                    np.trace(np.asarray(primary_result.get("hessian"), dtype=np.float64))
                    if np.asarray(primary_result.get("hessian", [])).shape == (6, 6) else None
                ),
                "primary_normalized_lambda_min": _csv_value(
                    (np.min(np.asarray(primary_result.get("eigenvalues"), dtype=np.float64)) /
                     np.trace(np.asarray(primary_result.get("hessian"), dtype=np.float64)))
                    if np.asarray(primary_result.get("eigenvalues", [])).size
                    and np.asarray(primary_result.get("hessian", [])).shape == (6, 6)
                    and np.trace(np.asarray(primary_result.get("hessian"), dtype=np.float64)) > 0.0 else None
                ),
                "primary_consistency_reason": _csv_value(primary_consistency_reason),
                "recovery_attempted": recovery_attempted,
                "recovery_stage": recovery_stage,
                "recovery_count": recovery_count,
                "recovered": recovered,
                "recovery_backend": _csv_value(recovery_backend),
                "final_consistency_reason": _csv_value(final_consistency_reason),
                "temporal_gap": temporal_gap,
                "final_failure_reason": _csv_value(None if accepted else result.get("failure_reason") or acceptance_reason),
                "iterations": _csv_value(result.get("iterations")),
                "num_correspondences": _csv_value(result.get("num_correspondences")),
                "point_to_plane_rmse": _csv_value(result.get("point_to_plane_rmse")),
                "lambda_min": _csv_value(np.min(eigenvalues) if eigenvalues.size else None),
                "condition_number": _csv_condition_number(result.get("condition_number")),
                "effective_rank": _csv_value(result.get("effective_rank")),
                "pose_covariance_diagonal": _csv_value(np.diag(covariance) if covariance.shape == (6, 6) else None),
                "final_step_rotation_deg": _csv_value(result.get("final_step_rotation_deg")),
                "final_step_translation_m": _csv_value(result.get("final_step_translation_m")),
                "initial_mean_robust_cost": _csv_value(result.get("initial_mean_robust_cost")),
                "final_mean_robust_cost": _csv_value(result.get("final_mean_robust_cost")),
                "last_relative_reduction": _csv_value(result.get("last_relative_reduction")),
                "accepted_step_count": _csv_value(result.get("accepted_step_count")),
                "rejected_lm_trials": _csv_value(result.get("rejected_lm_trials")),
                "observable_reserve_attempt": observable_reserve_attempt,
                "paired_triggered": paired_triggered,
                "pair_id": pair_id,
                "arm_count": paired_arm_count,
                "shadow_latency_ms": _csv_value(paired_shadow_latency_ms),
                "final_target_oracle_status": final_target_oracle_status,
                "final_target_oracle_latency_ms": _csv_value(final_target_oracle_latency_ms),
                **{key: _csv_condition_number(value) if "condition" in key else _csv_value(value)
                   for key, value in reserve_metrics.items()},
            })
            for prefix, values in (
                ("primary", primary_consistency_metrics),
                ("final", final_consistency_metrics),
            ):
                for name in (
                    "correction_translation_m", "correction_rotation_deg",
                    "correction_translation_limit_m", "correction_rotation_limit_deg",
                    "temporal_translation_m", "temporal_rotation_deg",
                    "temporal_translation_limit_m", "temporal_rotation_limit_deg",
                ):
                    row[f"{prefix}_{name}"] = _csv_value(values.get(name))
        if evaluation_protocol == "mvp_strict":
            row["source_training_count"] = int(pointcloud["points"].shape[0])
            row["source_validation_count"] = int(validation_points.shape[0])
        frame_rows.append(row)

    est_poses = np.stack(est_poses, axis=0)
    selection_event_writer.close()
    gt_poses = np.stack(gt_poses, axis=0)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    save_tum_trajectory(str(output_root / "trajectory_est.txt"), timestamps, est_poses)
    save_tum_trajectory(str(output_root / "trajectory_gt.txt"), timestamps, gt_poses)
    trajectory_hash = hashlib.sha256((output_root / "trajectory_est.txt").read_bytes()).hexdigest()
    frame_metric_fields = STRICT_FRAME_METRIC_FIELDS if evaluation_protocol == "mvp_strict" else AUDIT_FRAME_METRIC_FIELDS
    _write_frame_metrics(str(output_root / "frame_metrics.csv"), frame_rows, frame_metric_fields)

    valid_mask = np.asarray([bool(row["tracking_valid"]) for row in frame_rows], dtype=bool)
    metric_est_poses = est_poses[valid_mask]
    metric_gt_poses = gt_poses[valid_mask]
    ate_rmse = None
    if metric_est_poses.shape[0] >= (3 if bool(eval_cfg.get("align_ate", True)) else 1):
        ate_rmse = compute_ate_rmse(metric_est_poses, metric_gt_poses, align=bool(eval_cfg.get("align_ate", True)))
    valid_values = valid_mask.tolist()
    num_lost_frames = int(np.count_nonzero(~valid_mask))
    num_tracking_failures = int(
        sum(not valid_values[idx] and (idx == 0 or valid_values[idx - 1]) for idx in range(len(valid_values)))
    )
    summary = {
        "ate_rmse": ate_rmse,
        "rpe_trans_rmse": None,
        "rpe_rot_rmse_deg": None,
        "latency_mean_ms": float(np.mean([float(row["latency_ms"]) for row in frame_rows])),
        "latency_p95_ms": float(np.percentile([float(row["latency_ms"]) for row in frame_rows], 95)),
        "target_index_build_ms": target_index_build_ms,
        "target_index_backend": target_index_backend,
        "cold_start_latency_ms": target_index_build_ms + float(frame_rows[0]["latency_ms"]),
        "map_path": map_path,
        "backend": backend,
        "N_star": int(target_points.shape[0]),
        "num_frames": int(est_poses.shape[0]),
        "init_mode": init_mode,
        "num_fallbacks": int(sum(bool(row["fallback_used"]) for row in frame_rows)),
    }
    if evaluation_protocol == "mvp_strict":
        local_query_rows = frame_rows[1:] if local_enabled else []
        local_query_latencies = [float(row["local_query_latency_ms"] or 0.0) for row in local_query_rows]
        local_radius_latencies = [float(row["local_radius_latency_ms"] or 0.0) for row in local_query_rows]
        local_projection_latencies = [float(row["local_projection_latency_ms"] or 0.0) for row in local_query_rows]
        local_sampling_latencies = [float(row["local_sampling_latency_ms"] or 0.0) for row in local_query_rows]
        total_frame_latencies = [float(row["total_frame_latency_ms"] or 0.0) for row in frame_rows]
        production_frame_latencies = [
            float(row["production_equivalent_total_frame_latency_ms"] or 0.0) for row in frame_rows
        ]
        local_target_counts = [int(row["local_target_count"]) for row in local_query_rows]
        coverage_fields = (
            "local_nonempty_tile_count",
            "local_selected_tile_count",
            "local_nonempty_depth_bin_count",
            "local_selected_depth_bin_count",
            "local_nonempty_tile_depth_count",
            "local_selected_tile_depth_count",
        )
        coverage_values = {
            field: [int(row[field]) for row in local_query_rows]
            for field in coverage_fields
        }

        def coverage_ratio(selected_field: str, nonempty_field: str) -> float | None:
            ratios = [
                selected / nonempty
                for selected, nonempty in zip(coverage_values[selected_field], coverage_values[nonempty_field])
                if nonempty > 0
            ]
            return float(np.mean(ratios)) if ratios else None
        lambda_min_values = [float(row["lambda_min"]) for row in frame_rows if row["lambda_min"] != ""]
        condition_values = [
            float(row["condition_number"])
            for row in frame_rows
            if row["condition_number"] not in ("", "inf", "-inf")
        ]
        condition_inf_count = sum(row["condition_number"] in ("inf", "-inf") for row in frame_rows)
        condition_degenerate_count = sum(
            row["effective_rank"] != "" and int(float(row["effective_rank"])) < 6 for row in frame_rows
        )
        accepted_rows = [row for row in frame_rows if bool(row["accepted"])]
        accepted_temporal_translation = [
            float(row["final_temporal_translation_m"])
            for row in accepted_rows if row["final_temporal_translation_m"] != ""
        ]
        accepted_temporal_rotation = [
            float(row["final_temporal_rotation_deg"])
            for row in accepted_rows if row["final_temporal_rotation_deg"] != ""
        ]
        accepted_correction_translation = [
            float(row["final_correction_translation_m"])
            for row in accepted_rows if row["final_correction_translation_m"] != ""
        ]
        accepted_correction_rotation = [
            float(row["final_correction_rotation_deg"])
            for row in accepted_rows if row["final_correction_rotation_deg"] != ""
        ]
        tracking_summary = _tracking_summary_counts(frame_rows)
        summary.update(
            {
                "summary_schema_version": 2,
                "summary_schema_semantics": {
                    "tracking_denominator": "frames after the initialized first frame",
                    "primary_attempt_acceptance_rate": "primary accepted tracking attempts / num_tracking_attempt_frames; reserve adoption does not change this numerator",
                    "primary_tracking_completeness": "deprecated alias of primary_attempt_acceptance_rate",
                    "base_only_final_fraction": "final valid frames using accepted primary output without reserve adoption or recovery / num_tracking_attempt_frames",
                    "reserve_attempt_rate": "active reserve reruns / num_tracking_attempt_frames",
                    "reserve_adoption_rate": "adopted reserve reruns / num_tracking_attempt_frames",
                    "reserve_conditional_adoption_rate": "adopted reserve reruns / active reserve reruns",
                    "reserve_rescue_rate": "reserve adoptions rescuing a rejected primary / primary-rejected tracking attempts",
                    "final_tracking_completeness": "final valid tracking frames / num_tracking_attempt_frames",
                    "reserve_attempted_condition_and_lambda": "paired primary/final means and reductions over active reserve reruns with finite metric pairs",
                    "reserve_used_condition_and_lambda": "paired primary/final means and reductions over adopted reserve reruns with finite metric pairs",
                    "reserve_latency_all_tracking_frames_mean_ms": "reserve planning and rerun latency averaged over all tracking frames",
                    "reserve_latency_triggered_mean_ms": "reserve planning and rerun latency averaged over triggered tracking frames",
                },
                "evaluation_protocol": evaluation_protocol,
                "first_pose_protocol": first_pose_protocol,
                "uses_gt_first_pose": first_pose_protocol == "gt_first_frame",
                "uses_gt_current_frame_initialization": False,
                "uses_gt_fallback": False,
                "num_valid_frames": int(np.count_nonzero(valid_mask)),
                "num_lost_frames": num_lost_frames,
                "lost_frame_ids": [int(frame_rows[idx]["frame_id"]) for idx in np.flatnonzero(~valid_mask)],
                "num_tracking_failures": num_tracking_failures,
                "tracking_completeness": tracking_summary["final_tracking_completeness"],
                "longest_valid_track_frames": _longest_true_run(valid_values),
                "motion_model": motion_model,
                "num_constant_velocity_inits": int(
                    sum(row["init_source"] == "constant_velocity" for row in frame_rows)
                ),
                "num_prediction_guard_fallbacks": int(
                    sum(row["prediction_guard"] not in ("accepted", "not_applicable") for row in frame_rows)
                ),
                "num_consistency_rejections": int(sum(
                    str(row["primary_rejection_reason"]).startswith("tracking_consistency_")
                    for row in frame_rows
                )),
                "max_accepted_temporal_translation_m": max(accepted_temporal_translation, default=None),
                "max_accepted_temporal_rotation_deg": max(accepted_temporal_rotation, default=None),
                "max_accepted_correction_translation_m": max(accepted_correction_translation, default=None),
                "max_accepted_correction_rotation_deg": max(accepted_correction_rotation, default=None),
                "recovery_enabled": recovery_enabled,
                "num_recovery_attempted_frames": int(sum(bool(row["recovery_attempted"]) for row in frame_rows)),
                "num_recovered_frames": int(sum(bool(row["recovered"]) for row in frame_rows)),
                **tracking_summary,
                "local_submap_enabled": local_enabled,
                "local_index_build_ms": local_index_build_ms,
                "block_index_build_ms": block_index_build_ms,
                "block_score_hash_ms": block_score_hash_ms,
                "block_geometry_sort_ms": block_geometry_sort_ms,
                "block_index_memory_bytes": block_index_memory_bytes,
                "block_peak_working_set_estimate_bytes": block_peak_working_set_estimate_bytes,
                "local_index_backend": local_index_backend,
                "local_query_frame_count": len(local_query_rows),
                "local_query_latency_mean_ms": float(np.mean(local_query_latencies)) if local_query_latencies else None,
                "local_query_latency_p95_ms": float(np.percentile(local_query_latencies, 95)) if local_query_latencies else None,
                "local_radius_latency_mean_ms": float(np.mean(local_radius_latencies)) if local_radius_latencies else None,
                "local_projection_latency_mean_ms": float(np.mean(local_projection_latencies)) if local_projection_latencies else None,
                "local_sampling_latency_mean_ms": float(np.mean(local_sampling_latencies)) if local_sampling_latencies else None,
                "total_frame_latency_mean_ms": float(np.mean(total_frame_latencies)),
                "total_frame_latency_p95_ms": float(np.percentile(total_frame_latencies, 95)),
                "production_equivalent_total_frame_latency_mean_ms": float(np.mean(production_frame_latencies)),
                "production_equivalent_total_frame_latency_p95_ms": float(np.percentile(production_frame_latencies, 95)),
                "oracle_diagnostic_latency_mean_ms": (
                    float(np.mean([float(row["local_oracle_latency_ms"] or 0.0) for row in local_query_rows]))
                    if local_query_rows and local_config is not None and local_config.oracle_mode == "measure" else 0.0
                ),
                "local_target_count_mean": float(np.mean(local_target_counts)) if local_target_counts else None,
                "local_nonempty_tile_count_mean": float(np.mean(coverage_values["local_nonempty_tile_count"])) if local_query_rows else None,
                "local_selected_tile_count_mean": float(np.mean(coverage_values["local_selected_tile_count"])) if local_query_rows else None,
                "local_tile_coverage_mean": coverage_ratio("local_selected_tile_count", "local_nonempty_tile_count"),
                "local_nonempty_depth_bin_count_mean": float(np.mean(coverage_values["local_nonempty_depth_bin_count"])) if local_query_rows else None,
                "local_selected_depth_bin_count_mean": float(np.mean(coverage_values["local_selected_depth_bin_count"])) if local_query_rows else None,
                "local_depth_bin_coverage_mean": coverage_ratio("local_selected_depth_bin_count", "local_nonempty_depth_bin_count"),
                "local_nonempty_tile_depth_count_mean": float(np.mean(coverage_values["local_nonempty_tile_depth_count"])) if local_query_rows else None,
                "local_selected_tile_depth_count_mean": float(np.mean(coverage_values["local_selected_tile_depth_count"])) if local_query_rows else None,
                "local_tile_depth_coverage_mean": coverage_ratio("local_selected_tile_depth_count", "local_nonempty_tile_depth_count"),
                "local_target_fraction_mean": float(
                    np.mean([float(row["local_target_fraction"] or 0.0) for row in local_query_rows])
                ) if local_query_rows else None,
                "lambda_min_mean": float(np.mean(lambda_min_values)) if lambda_min_values else None,
                "lambda_min_min": float(np.min(lambda_min_values)) if lambda_min_values else None,
                "condition_number_mean": float(np.mean(condition_values)) if condition_values else None,
                "condition_number_max": float(np.max(condition_values)) if condition_values else None,
                "condition_number_inf_count": int(condition_inf_count),
                "condition_number_degenerate_count": int(condition_degenerate_count),
                "rss_mb": rss_mb,
                "peak_rss_mb": peak_rss_mb,
            }
        )
        summary.update(observable_reserve.summary())
        if probe_enabled:
            summary["candidate_probe"] = True
        if bypass_candidate_gate:
            summary["bypass_candidate_gate"] = True
        if not probe_enabled and not bypass_candidate_gate:
            observable_reserve.save_stability_attempts(output_root)
        if paired_shadow is not None:
            paired_summary = paired_shadow.save(
                paired_evaluations, output_root / "paired_reserve", parent_trajectory_sha256=trajectory_hash
            )
            summary.update({f"paired_reserve_{key}": value for key, value in paired_summary.items()})
        if final_target_oracle is not None:
            oracle_summary = final_target_oracle.save(output_root / "final_target_oracle")
            summary.update({f"final_target_oracle_{key}": value for key, value in oracle_summary.items()})
        block_rows = local_query_rows
        for field in (
            "local_tested_block_count", "local_intersecting_block_count", "local_block_candidate_count",
            "local_projection_input_count", "local_oracle_count", "local_oracle_block_recall",
            "local_oracle_projection_recall", "local_oracle_frustum_recall",
            "local_representative_points_considered", "local_precap_block_count",
            "local_precap_selected_block_count", "local_precap_bucket_count",
            "local_precap_selected_bucket_count", "local_precap_boundary_block_count",
            "local_precap_selected_boundary_block_count",
        ):
            values = [float(row[field]) for row in block_rows if row[field] != ""]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else None
        advanced_query_enabled = bool(
            local_config is not None and (
                local_config.frustum_culling != "radius"
                or local_config.projection_pre_cap is not None
                or local_config.oracle_mode != "off"
            )
        )
        if advanced_query_enabled:
            for field in (
                "local_block_culling_latency_ms", "local_precap_latency_ms", "local_oracle_latency_ms",
            ):
                values = [float(row[field]) for row in block_rows if row[field] != ""]
                summary[f"{field}_mean"] = float(np.mean(values)) if values else None
        strategies = sorted({str(row["local_precap_strategy"]) for row in block_rows if row["local_precap_strategy"]})
        summary["precap_strategy"] = strategies[0] if len(strategies) == 1 else ",".join(strategies)
        summary["representative_considered_mean"] = summary.get("local_representative_points_considered_mean")
        for selected_field, total_field, name in (
            ("local_precap_selected_block_count", "local_precap_block_count", "local_precap_block_coverage_mean"),
            ("local_precap_selected_bucket_count", "local_precap_bucket_count", "local_precap_bucket_coverage_mean"),
            ("local_precap_selected_boundary_block_count", "local_precap_boundary_block_count", "local_precap_boundary_coverage_mean"),
        ):
            ratios = [
                float(row[selected_field]) / float(row[total_field])
                for row in block_rows if row[total_field] != "" and float(row[total_field]) > 0.0
            ]
            summary[name] = float(np.mean(ratios)) if ratios else None
        summary["local_oracle_overflow_count"] = int(sum(bool(row["local_oracle_overflow"]) for row in block_rows))
        summary["local_exact_recall_infeasible_count"] = int(
            sum(row["local_oracle_status"] == "exact_recall_infeasible" for row in block_rows)
        )
    rpe_delta = int(eval_cfg.get("rpe_delta", 1))
    rpe = _compute_valid_rpe(est_poses, gt_poses, valid_mask, rpe_delta)
    if rpe is not None:
        summary["rpe_trans_rmse"] = rpe["trans_rmse"]
        summary["rpe_rot_rmse_deg"] = rpe["rot_rmse_deg"]

    if evaluation_protocol == "mvp_strict":
        long_horizon = compute_long_horizon_metrics(
            est_poses,
            gt_poses,
            valid_mask=valid_mask,
            frame_deltas=tuple(eval_cfg.get("frame_deltas", (1, 10, 30, 100))),
            distance_deltas_m=tuple(eval_cfg.get("distance_deltas_m", (1.0, 5.0, 10.0))),
            segment_lengths_m=tuple(eval_cfg.get("segment_lengths_m", (5.0, 10.0, 20.0))),
            align_with_scale=bool(eval_cfg.get("align_with_scale", False)),
            return_to_start_radius_m=float(
                eval_cfg.get("return_to_start_radius_m", eval_cfg.get("return_radius_m", 0.5))
            ),
        )
        _write_summary(str(output_root / "long_horizon_metrics.json"), long_horizon)
        _write_long_horizon_csvs(output_root, long_horizon, frame_rows)
        raw_absolute = long_horizon["raw_absolute"]
        aligned_absolute = long_horizon["aligned_absolute"]
        drift = long_horizon["drift"]
        endpoint = long_horizon["endpoint"]
        summary.update({
            "raw_ate_rmse_m": raw_absolute["translation_rmse_m"],
            "aligned_ate_rmse_m": aligned_absolute["translation_rmse_m"],
            "raw_rotation_abs_rmse_deg": raw_absolute["rotation_rmse_deg"],
            "aligned_rotation_abs_rmse_deg": aligned_absolute["rotation_rmse_deg"],
            "translation_drift_percent": drift["translation_drift_percent"],
            "rotation_drift_deg_per_m": drift["rotation_drift_deg_per_m"],
            "endpoint_translation_error_m": endpoint["absolute_translation_error_m"],
            "endpoint_rotation_error_deg": endpoint["absolute_rotation_error_deg"],
            "endpoint_relative_translation_error_m": endpoint["relative_translation_error_m"],
            "endpoint_relative_rotation_error_deg": endpoint["relative_rotation_error_deg"],
            "last_valid_endpoint_translation_error_m": endpoint["last_valid_absolute_translation_error_m"],
            "last_valid_endpoint_rotation_error_deg": endpoint["last_valid_absolute_rotation_error_deg"],
            "last_valid_endpoint_relative_translation_error_m": endpoint["last_valid_relative_translation_error_m"],
            "last_valid_endpoint_relative_rotation_error_deg": endpoint["last_valid_relative_rotation_error_deg"],
            "last_valid_frame_id": (
                int(frame_rows[endpoint["last_valid_endpoint_index"]]["frame_id"])
                if endpoint["last_valid_endpoint_index"] is not None else None
            ),
            "final_frame_valid": endpoint["final_frame_valid"],
            "return_to_start_translation_error_m": endpoint["return_to_start_translation_error_m"],
            "return_to_start_rotation_error_deg": endpoint["return_to_start_rotation_error_deg"],
        })
        summary["cold_start_latency_ms"] += local_index_build_ms + block_index_build_ms
        summary["ate_rmse"] = aligned_absolute["translation_rmse_m"]

    _write_summary(str(output_root / "summary.json"), summary)
    activation_diagnostics.save(str(output_root))
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = load_dataset_from_config(args.config)
    explicit_initial_pose = _load_initial_pose(args.initial_pose) if args.initial_pose else None
    summary = run_localization(
        config=config,
        map_path=args.map,
        output_dir=args.output_dir,
        dataset=dataset,
        start=args.start,
        end=args.end,
        backend_override=args.backend,
        protocol_override=args.protocol,
        first_pose_protocol_override=args.first_pose_protocol,
        motion_model_override=args.motion_model,
        explicit_initial_pose=explicit_initial_pose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
