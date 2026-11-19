import argparse
import copy
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.fsm_localize import load_config, run_localization
from utils.fsm_dataset import dataset_pose_provenance, load_dataset_from_config


FROZEN_AUDIT_DIRNAME = "audit_v2_full_rerun"
RUNTIME_SOURCE_PATHS = (
    "scripts/fsm_eval_long_horizon.py",
    "scripts/fsm_localize.py",
    "utils/fsm_local_map.py",
    "utils/fsm_runtime_reserve.py",
    "utils/fsm_paired_reserve.py",
    "utils/fsm_final_target_oracle.py",
    "utils/fsm_correspondence_weak.py",
    "utils/fsm_registration.py",
    "utils/fsm_robust_registration.py",
    "utils/fsm_metrics.py",
    "utils/fsm_activation_diagnostics.py",
    "utils/fsm_runtime_memory.py",
    "utils/fsm_dataset.py",
    "datasets/gradslam_datasets/tum.py",
)
SUMMARY_FIELDS = [
    "dataset",
    "case",
    "map",
    "backend",
    "motion",
    "first_pose_protocol",
    "evaluation_protocol",
    "N",
    "frames",
    "valid_frames",
    "lost_frames",
    "tracking_completeness",
    "num_tracking_attempt_frames",
    "num_primary_accepted_tracking_frames",
    "num_base_only_final_frames",
    "num_final_complete_tracking_frames",
    "primary_attempt_acceptance_rate",
    "primary_tracking_completeness",
    "base_only_final_fraction",
    "final_tracking_completeness",
    "raw_ate_rmse_m",
    "aligned_ate_rmse_m",
    "raw_rotation_abs_rmse_deg",
    "aligned_rotation_abs_rmse_deg",
    "translation_drift_percent",
    "rotation_drift_deg_per_m",
    "endpoint_translation_error_m",
    "endpoint_rotation_error_deg",
    "endpoint_relative_translation_error_m",
    "endpoint_relative_rotation_error_deg",
    "last_valid_endpoint_translation_error_m",
    "last_valid_endpoint_rotation_error_deg",
    "last_valid_endpoint_relative_translation_error_m",
    "last_valid_endpoint_relative_rotation_error_deg",
    "last_valid_frame_id",
    "final_frame_valid",
    "latency_mean_ms",
    "latency_p95_ms",
    "cold_start_latency_ms",
    "target_index_build_ms",
    "local_index_build_ms",
    "block_index_build_ms",
    "block_score_hash_ms",
    "block_geometry_sort_ms",
    "block_index_memory_bytes",
    "block_peak_working_set_estimate_bytes",
    "local_query_latency_mean_ms",
    "local_query_latency_p95_ms",
    "total_frame_latency_mean_ms",
    "total_frame_latency_p95_ms",
    "production_equivalent_total_frame_latency_mean_ms",
    "production_equivalent_total_frame_latency_p95_ms",
    "oracle_diagnostic_latency_mean_ms",
    "local_radius_latency_mean_ms",
    "local_projection_latency_mean_ms",
    "local_sampling_latency_mean_ms",
    "local_nonempty_tile_count_mean",
    "local_selected_tile_count_mean",
    "local_tile_coverage_mean",
    "local_nonempty_depth_bin_count_mean",
    "local_selected_depth_bin_count_mean",
    "local_depth_bin_coverage_mean",
    "local_nonempty_tile_depth_count_mean",
    "local_selected_tile_depth_count_mean",
    "local_tile_depth_coverage_mean",
    "num_consistency_rejections",
    "max_accepted_temporal_translation_m",
    "max_accepted_temporal_rotation_deg",
    "max_accepted_correction_translation_m",
    "max_accepted_correction_rotation_deg",
    "condition_number_mean",
    "condition_number_max",
    "condition_number_inf_count",
    "condition_number_degenerate_count",
    "lambda_min_mean",
    "lambda_min_min",
    "replacement_local_activation_rate",
    "replacement_correspondence_rate",
    "replacement_hessian_trace_fraction_mean",
    "replacement_weak_fraction_mean",
    "removed_local_activation_rate",
    "removed_correspondence_rate",
    "removed_hessian_trace_fraction_mean",
    "removed_weak_fraction_mean",
    "activation_trigger_rate",
    "diagnostic_latency_mean_ms",
    "local_projection_input_count_mean",
    "local_block_candidate_count_mean",
    "local_tested_block_count_mean",
    "local_intersecting_block_count_mean",
    "local_block_culling_latency_ms_mean",
    "local_precap_latency_ms_mean",
    "precap_strategy",
    "representative_considered_mean",
    "local_representative_points_considered_mean",
    "local_precap_block_count_mean",
    "local_precap_selected_block_count_mean",
    "local_precap_block_coverage_mean",
    "local_precap_bucket_count_mean",
    "local_precap_selected_bucket_count_mean",
    "local_precap_bucket_coverage_mean",
    "local_precap_boundary_block_count_mean",
    "local_precap_selected_boundary_block_count_mean",
    "local_precap_boundary_coverage_mean",
    "local_oracle_latency_ms_mean",
    "local_oracle_block_recall_mean",
    "local_oracle_projection_recall_mean",
    "local_oracle_frustum_recall_mean",
    "local_oracle_overflow_count",
    "local_exact_recall_infeasible_count",
    "observable_reserve_trigger_rate",
    "observable_reserve_use_rate",
    "observable_reserve_recovery_rate",
    "observable_reserve_condition_improvement_mean",
    "observable_reserve_lambda_improvement_mean",
    "observable_reserve_extra_latency_mean_ms",
    "observable_reserve_triggered_latency_mean_ms",
    "reserve_relative_improvement_protocol",
    "reserve_relative_improvement_mode",
    "num_reserve_attempt_frames",
    "num_reserve_adopted_frames",
    "num_reserve_rescue_frames",
    "reserve_attempt_rate",
    "reserve_adoption_rate",
    "reserve_conditional_adoption_rate",
    "reserve_rescue_rate",
    "reserve_selected_attempted_mean",
    "reserve_selected_triggered_mean",
    "reserve_correspondence_attempted_mean",
    "reserve_selected_mean",
    "reserve_correspondence_mean",
    "reserve_relative_condition_pass_count",
    "reserve_relative_lambda_pass_count",
    "reserve_relative_any_only_count",
    "reserve_relative_both_count",
    "reserve_relative_condition_unavailable_count",
    "reserve_relative_lambda_unavailable_count",
    "reserve_information_v3_both_attempt_count",
    "reserve_information_v3_both_adopt_count",
    "reserve_information_v3_both_reject_count",
    "reserve_information_v3_condition_only_attempt_count",
    "reserve_information_v3_condition_only_adopt_count",
    "reserve_information_v3_condition_only_reject_count",
    "reserve_information_v3_lambda_only_attempt_count",
    "reserve_information_v3_lambda_only_adopt_count",
    "reserve_information_v3_lambda_only_reject_count",
    "reserve_information_v3_neither_attempt_count",
    "reserve_information_v3_neither_adopt_count",
    "reserve_information_v3_neither_reject_count",
    "reserve_latency_all_tracking_frames_mean_ms",
    "reserve_latency_triggered_mean_ms",
    "reserve_attempted_primary_condition_mean",
    "reserve_attempted_final_condition_mean",
    "reserve_attempted_condition_reduction_abs_mean",
    "reserve_attempted_condition_reduction_percent_mean",
    "reserve_attempted_primary_lambda_min_mean",
    "reserve_attempted_final_lambda_min_mean",
    "reserve_attempted_lambda_min_increase_abs_mean",
    "reserve_attempted_lambda_min_increase_percent_mean",
    "reserve_used_primary_condition_mean",
    "reserve_used_final_condition_mean",
    "reserve_used_condition_reduction_abs_mean",
    "reserve_used_condition_reduction_percent_mean",
    "reserve_used_primary_lambda_min_mean",
    "reserve_used_final_lambda_min_mean",
    "reserve_used_lambda_min_increase_abs_mean",
    "reserve_used_lambda_min_increase_percent_mean",
    "reserve_stability_attempted_count",
    "reserve_stability_would_accept_count",
    "reserve_stability_actual_gate_adoption_count",
    "reserve_refinement_translation_mean_m",
    "reserve_refinement_translation_p95_m",
    "reserve_refinement_translation_max_m",
    "reserve_refinement_rotation_mean_deg",
    "reserve_refinement_rotation_p95_deg",
    "reserve_refinement_rotation_max_deg",
    "reserve_stability_rolling_translation_l1_mean_m",
    "reserve_stability_prospective_streak_max",
    "reserve_common_target_cost_ratio_mean",
    "reserve_common_target_rmse_ratio_mean",
    "reserve_common_target_corr_retention_mean",
    "reserve_stability_gate_latency_mean_ms",
    "paired_reserve_trigger_pairs",
    "paired_reserve_arm_count",
    "paired_reserve_query_count",
    "paired_reserve_normal_count",
    "paired_reserve_registration_count",
    "paired_reserve_total_shadow_latency_ms",
    "final_target_oracle_num_frames",
    "final_target_oracle_num_ok",
    "final_target_oracle_num_errors",
    "final_target_oracle_set_recall_mean",
    "final_target_oracle_set_jaccard_mean",
    "final_target_oracle_joint_histogram_recall_mean",
    "final_target_oracle_tile_histogram_recall_mean",
    "final_target_oracle_depth_histogram_recall_mean",
    "final_target_oracle_score_exact_mean",
    "final_target_oracle_score_production_mean",
    "final_target_oracle_hessian_trace_recall_mean",
    "final_target_oracle_hessian_lambda_min_recall_mean",
    "final_target_oracle_hessian_condition_number_ratio_mean",
    "final_target_oracle_diagnostic_latency_total_ms",
    "final_target_oracle_diagnostic_latency_mean_ms",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing FSM maps with the strict long-horizon protocol.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--case",
        action="append",
        help="Case as name,config,map[,activation_geometry]. Repeat for multiple cases.",
    )
    inputs.add_argument("--plan", help="JSON file containing a list or {'cases': [...]}.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _load_config_write_free(path: str) -> dict[str, Any]:
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return load_config(path)
    finally:
        sys.dont_write_bytecode = old


def _parse_case(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(",", 3)]
    if len(parts) not in (3, 4) or not all(parts):
        raise ValueError("--case must be name,config,map[,activation_geometry]")
    result = {"name": parts[0], "config": parts[1], "map": parts[2]}
    if len(parts) == 4:
        result["activation_geometry"] = parts[3]
    return result


def load_cases(case_args: list[str] | None, plan_path: str | None) -> list[dict[str, Any]]:
    if case_args:
        cases = [_parse_case(value) for value in case_args]
    else:
        assert plan_path is not None
        with open(plan_path) as f:
            document = json.load(f)
        cases = document.get("cases") if isinstance(document, dict) else document
        if not isinstance(cases, list):
            raise ValueError("plan JSON must be a list or contain a 'cases' list")
    normalized = []
    names = set()
    for case in cases:
        if not isinstance(case, dict) or not all(case.get(key) for key in ("name", "config", "map")):
            raise ValueError("each case requires non-empty name, config, and map")
        name = str(case["name"])
        if name in names or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"case name must be unique and path-safe: {name!r}")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "config": str(Path(case["config"]).expanduser().resolve()),
                "map": str(Path(case["map"]).expanduser().resolve()),
                "activation_geometry": (
                    str(Path(case["activation_geometry"]).expanduser().resolve())
                    if case.get("activation_geometry") else None
                ),
                "start": case.get("start"),
                "end": case.get("end"),
            }
        )
    if not normalized:
        raise ValueError("at least one case is required")
    return normalized


def _assert_safe_output(output_root: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if FROZEN_AUDIT_DIRNAME in root.parts:
        raise ValueError(f"output root must not be inside {FROZEN_AUDIT_DIRNAME}")
    return root


def _motion(config: dict[str, Any]) -> str | None:
    registration = config.get("registration", {})
    motion = config.get("motion_model", {})
    if isinstance(motion, str):
        configured = motion
    else:
        configured = motion.get("type", motion.get("model"))
    return registration.get("motion_model", configured)


def validate_case(case: dict[str, Any], default_start: int | None, default_end: int | None) -> dict[str, Any]:
    config_path = Path(case["config"])
    map_path = Path(case["map"])
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    geometry_path = Path(case["activation_geometry"]) if case.get("activation_geometry") else None
    if geometry_path is not None and not geometry_path.is_file():
        raise FileNotFoundError(geometry_path)
    config = _load_config_write_free(str(config_path))
    if geometry_path is not None:
        config = copy.deepcopy(config)
        explicit_path = str(geometry_path)
        diagnostics = config.get("activation_diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            raise ValueError(f"{case['name']}: activation_diagnostics must be a dictionary")
        if bool(diagnostics.get("enabled", False)):
            diagnostics["geometry_path"] = explicit_path
            config["activation_diagnostics"] = diagnostics
        for section in ("observable_reserve", "paired_reserve_evaluation"):
            sidecar_cfg = config.get(section, {}) or {}
            if not isinstance(sidecar_cfg, dict):
                raise ValueError(f"{case['name']}: {section} must be a dictionary")
            if bool(sidecar_cfg.get("enabled", False)) and sidecar_cfg.get("geometry_path") in (None, "auto"):
                sidecar_cfg["geometry_path"] = explicit_path
                config[section] = sidecar_cfg
    registration = config.get("registration", {})
    evaluation_protocol = registration.get("evaluation_protocol")
    first_pose = registration.get("first_pose_protocol")
    backend = registration.get("backend")
    motion = _motion(config)
    local = config.get("local_submap", registration.get("local_submap"))
    metrics = config.get("eval")
    if evaluation_protocol != "mvp_strict":
        raise ValueError(f"{case['name']}: evaluation protocol must be mvp_strict")
    if first_pose not in ("gt_first_frame", "identity"):
        raise ValueError(f"{case['name']}: first pose protocol must be gt_first_frame or identity")
    static_backends = {
        "numpy_point_to_point",
        "numpy_covariance_weighted",
        "numpy_robust_point_to_plane",
    }
    if backend not in static_backends:
        raise ValueError(f"{case['name']}: unsupported long-horizon backend {backend!r}")
    reserve = config.get("observable_reserve", {}) or {}
    paired = config.get("paired_reserve_evaluation", {}) or {}
    reserve_active = bool(reserve.get("enabled", False)) and str(reserve.get("mode", "off")) != "off"
    paired_active = bool(paired.get("enabled", False))
    if (reserve_active or paired_active) and backend != "numpy_robust_point_to_plane":
        raise ValueError(
            f"{case['name']}: observable reserve and paired evaluation require "
            "numpy_robust_point_to_plane"
        )
    if motion != "constant_velocity":
        raise ValueError(f"{case['name']}: motion model must be constant_velocity")
    if not isinstance(local, dict) or not local.get("enabled"):
        raise ValueError(f"{case['name']}: enabled local_submap configuration is required")
    required_metrics = ("frame_deltas", "distance_deltas_m", "segment_lengths_m")
    if not isinstance(metrics, dict) or any(not metrics.get(key) for key in required_metrics):
        raise ValueError(f"{case['name']}: long-horizon metric configuration is incomplete")
    dataset = config.get("dataset", config.get("data", {}))
    pose_provenance = dataset_pose_provenance(config)
    return {
        **case,
        "start": default_start if case.get("start") is None else int(case["start"]),
        "end": default_end if case.get("end") is None else int(case["end"]),
        "dataset": dataset.get("dataset_name", dataset.get("name", "")),
        "sequence": dataset.get("sequence", ""),
        "pose_reference_sequence": dataset.get("pose_reference_sequence"),
        "pose_provenance": pose_provenance,
        "evaluation_protocol": evaluation_protocol,
        "first_pose_protocol": first_pose,
        "uses_gt_first_pose": first_pose == "gt_first_frame",
        "uses_gt_current_frame_initialization": False,
        "uses_gt_fallback": False,
        "backend": backend,
        "motion_model": motion,
        "tracker": registration,
        "local_submap": local,
        "metrics": metrics,
        "config_object": config,
    }


def build_plan(
    cases: list[dict[str, Any]], output_root: str, start: int | None = None, end: int | None = None
) -> dict[str, Any]:
    root = _assert_safe_output(output_root)
    validated = [validate_case(case, start, end) for case in cases]
    printable = [{key: value for key, value in case.items() if key != "config_object"} for case in validated]
    return {
        "dry_run": True,
        "output_root": str(root),
        "creates_map": False,
        "copies_frozen_maps": False,
        "evaluation_protocol": "mvp_strict",
        "cases": printable,
        "_validated_cases": validated,
    }


def _file_record(path: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path, "size_bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def _resolved_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=BASE_DIR, text=True, capture_output=True, check=False
    )
    return result.stdout.splitlines() if result.returncode == 0 else [f"git status failed: {result.stderr.strip()}"]


def _runtime_source_sha256(paths: tuple[str, ...] = RUNTIME_SOURCE_PATHS) -> str:
    digest = hashlib.sha256()
    for relative_path in paths:
        contents = (BASE_DIR / relative_path).read_bytes()
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _git_provenance() -> tuple[str | None, bool | None]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True, capture_output=True, check=False
    )
    if head.returncode != 0:
        return None, None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=BASE_DIR, text=True, capture_output=True, check=False
    )
    return head.stdout.strip(), None if dirty.returncode != 0 else bool(dirty.stdout)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _case_provenance(case: dict[str, Any]) -> dict[str, Any]:
    common_path = Path(case["config"]).parent / "common.py"
    reserve_cfg = case["config_object"].get("observable_reserve", {}) or {}
    selection_artifact_cfg = reserve_cfg.get("selection_artifact", {}) or {}
    stability_cfg = reserve_cfg.get("stability_gate", {}) or {}
    paired_cfg = case["config_object"].get("paired_reserve_evaluation", {}) or {}
    oracle_cfg = case["config_object"].get("final_target_oracle", {}) or {}
    reserve_path = reserve_cfg.get("geometry_path")
    if reserve_path in (None, "auto"):
        automatic = Path(case["map"]).parent / "activation_geometry.npz"
        reserve_path = str(automatic) if automatic.is_file() else None
    elif reserve_path:
        reserve_path = str(Path(reserve_path).expanduser().resolve())
    paired_path = paired_cfg.get("geometry_path")
    if paired_path in (None, "auto"):
        automatic = Path(case["map"]).parent / "activation_geometry.npz"
        paired_path = str(automatic) if automatic.is_file() else None
    elif paired_path:
        paired_path = str(Path(paired_path).expanduser().resolve())
    git_head, dirty = _git_provenance()
    return {
        "case": case["name"],
        "config_sha256": _file_record(case["config"])["sha256"],
        "resolved_config_sha256": _resolved_config_sha256(case["config_object"]),
        "common_config_sha256": _file_record(str(common_path))["sha256"] if common_path.is_file() else None,
        "map_sha256": _file_record(case["map"])["sha256"],
        "start": case["start"],
        "end": case["end"],
        "evaluation_protocol": case["evaluation_protocol"],
        "activation_geometry_sha256": (
            _file_record(case["activation_geometry"])["sha256"] if case.get("activation_geometry") else None
        ),
        "observable_reserve_config_sha256": _resolved_config_sha256(reserve_cfg),
        "observable_reserve_geometry_path": reserve_path,
        "observable_reserve_geometry_sha256": (
            _file_record(reserve_path)["sha256"] if reserve_path and Path(reserve_path).is_file() else None
        ),
        "reserve_selection_artifact_enabled": bool(reserve_cfg.get("enabled", False))
        and str(reserve_cfg.get("mode", "off")) != "off"
        and bool(selection_artifact_cfg.get("enabled", False)),
        "reserve_stability_gate_enabled": bool(stability_cfg.get("enabled", False)) and str(stability_cfg.get("mode", "off")) != "off",
        "reserve_stability_gate_mode": str(stability_cfg.get("mode", "off")),
        "paired_reserve_enabled": bool(paired_cfg.get("enabled", False)),
        "paired_reserve_trigger_protocol": paired_cfg.get("trigger_protocol"),
        "paired_reserve_trigger": dict(paired_cfg.get("trigger", {}) or {}),
        "paired_reserve_policies": list(paired_cfg.get("policies", [])),
        "paired_reserve_budgets": list(paired_cfg.get("budgets", [])),
        "paired_reserve_seeds": list(paired_cfg.get("random_seeds", paired_cfg.get("seeds", []))),
        "paired_reserve_config_sha256": _resolved_config_sha256(paired_cfg),
        "paired_reserve_geometry_path": paired_path,
        "paired_reserve_geometry_sha256": (
            _file_record(paired_path)["sha256"] if paired_path and Path(paired_path).is_file() else None
        ),
        "final_target_oracle_enabled": bool(oracle_cfg.get("enabled", False)),
        "final_target_oracle_config_sha256": _resolved_config_sha256(oracle_cfg),
        "runtime_source_sha256": _runtime_source_sha256(),
        "dataset_pose_provenance": case.get("pose_provenance"),
        "offline_gt_scorer_sha256": _file_record(str(BASE_DIR / "scripts/fsm_score_paired_reserve_gt.py"))["sha256"],
        "git_head": git_head,
        "dirty": dirty,
    }


def _completed_case(summary_path: Path, state_path: Path, provenance: dict[str, Any]) -> dict[str, Any] | None:
    if not summary_path.is_file() or not state_path.is_file():
        return None
    try:
        with open(state_path) as stream:
            state = json.load(stream)
        stored = state.get("provenance")
        if not isinstance(stored, dict):
            return None
        informational = {"git_head", "dirty"}
        if ({key: value for key, value in stored.items() if key not in informational}
                != {key: value for key, value in provenance.items() if key not in informational}):
            return None
        output_dir = summary_path.parent
        if provenance.get("paired_reserve_enabled"):
            required = (
                output_dir / "paired_reserve/paired_reserve_ids.npz",
                output_dir / "paired_reserve/paired_reserve_rows.csv",
                output_dir / "paired_reserve/paired_reserve_summary.json",
                output_dir / "paired_reserve/paired_reserve_provenance.json",
            )
            if any(not path.is_file() for path in required):
                return None
            with np.load(required[0], allow_pickle=False):
                pass
        if provenance.get("final_target_oracle_enabled"):
            required = (
                output_dir / "final_target_oracle/frames.csv",
                output_dir / "final_target_oracle/hessians.npz",
                output_dir / "final_target_oracle/summary.json",
            )
            if any(not path.is_file() for path in required):
                return None
            with np.load(required[1], allow_pickle=False):
                pass
        if provenance.get("reserve_stability_gate_enabled"):
            artifact = output_dir / "reserve_stability_attempts.npz"
            if not artifact.is_file():
                return None
            with np.load(artifact, allow_pickle=False):
                pass
        if provenance.get("reserve_selection_artifact_enabled"):
            required = (
                output_dir / "reserve_selection_events.csv",
                output_dir / "reserve_selection_events_provenance.json",
            )
            if any(not path.is_file() for path in required):
                return None
        with open(summary_path) as stream:
            return json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _condition_stats(path: Path) -> dict[str, Any]:
    values = []
    inf_count = 0
    degenerate_count = 0
    if path.is_file():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                value = row.get("condition_number", "")
                if value:
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if number in (float("inf"), float("-inf")):
                        inf_count += 1
                    elif number == number:
                        values.append(number)
                rank = row.get("effective_rank", "")
                if rank:
                    try:
                        degenerate_count += int(float(rank) < 6)
                    except ValueError:
                        pass
    return {
        "condition_number_mean": sum(values) / len(values) if values else None,
        "condition_number_max": max(values) if values else None,
        "condition_number_inf_count": inf_count,
        "condition_number_degenerate_count": degenerate_count,
    }


def flatten_summary(case: dict[str, Any], summary: dict[str, Any], localize_dir: Path) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "dataset": case["dataset"],
            "case": case["name"],
            "map": case["map"],
            "backend": case["backend"],
            "motion": case["motion_model"],
            "first_pose_protocol": case["first_pose_protocol"],
            "evaluation_protocol": case["evaluation_protocol"],
            "N": summary.get("N_star", ""),
            "frames": summary.get("num_frames", ""),
            "valid_frames": summary.get("num_valid_frames", ""),
            "lost_frames": summary.get("num_lost_frames", ""),
            "tracking_completeness": summary.get("tracking_completeness", ""),
        }
    )
    for field in SUMMARY_FIELDS:
        if field in summary:
            row[field] = summary[field]
    row.update(_condition_stats(localize_dir / "frame_metrics.csv"))
    activation_path = localize_dir / "runtime_activation_summary.json"
    if activation_path.is_file():
        with open(activation_path) as stream:
            activation = json.load(stream)
        for field in (
            "replacement_local_activation_rate",
            "replacement_correspondence_rate",
            "replacement_hessian_trace_fraction_mean",
            "replacement_weak_fraction_mean",
            "removed_local_activation_rate",
            "removed_correspondence_rate",
            "removed_hessian_trace_fraction_mean",
            "removed_weak_fraction_mean",
            "diagnostic_latency_mean_ms",
        ):
            if activation.get(field) is not None:
                row[field] = activation[field]
        if activation.get("trigger_rate") is not None:
            row["activation_trigger_rate"] = activation["trigger_rate"]
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_plan(plan: dict[str, Any], skip_existing: bool = False) -> list[dict[str, Any]]:
    output_root = _assert_safe_output(plan["output_root"])
    cases = plan["_validated_cases"]
    for directory in ("config_snapshot", "localize", "metrics", "logs", ".state"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for case in cases:
        source_snapshot = output_root / "config_snapshot" / f"{case['name']}.py"
        json_snapshot = output_root / "config_snapshot" / f"{case['name']}.json"
        shutil.copy2(case["config"], source_snapshot)
        _write_json(json_snapshot, case["config_object"])
        common_path = Path(case["config"]).parent / "common.py"
        resolved_config_sha256 = _resolved_config_sha256(case["config_object"])
        manifest_cases.append(
            {
                **{key: value for key, value in case.items() if key != "config_object"},
                "config_input": _file_record(case["config"]),
                "map_input": _file_record(case["map"]),
                "activation_geometry_input": (
                    _file_record(case["activation_geometry"]) if case.get("activation_geometry") else None
                ),
                "config_snapshot": str(source_snapshot),
                "resolved_config_snapshot": str(json_snapshot),
                "resolved_config_sha256": resolved_config_sha256,
                "common_config_input": _file_record(str(common_path)) if common_path.is_file() else None,
                "paired_reserve_evaluation": {
                    "enabled": bool(case["config_object"].get("paired_reserve_evaluation", {}).get("enabled", False)),
                    "policies": list(case["config_object"].get("paired_reserve_evaluation", {}).get("policies", [])),
                    "budgets": list(case["config_object"].get("paired_reserve_evaluation", {}).get("budgets", [])),
                    "seeds": list(case["config_object"].get("paired_reserve_evaluation", {}).get("random_seeds", [])),
                    "sidecar_sha256": _case_provenance(case)["paired_reserve_geometry_sha256"],
                },
                "final_target_oracle": {
                    "enabled": bool(case["config_object"].get("final_target_oracle", {}).get("enabled", False)),
                    "config_sha256": _resolved_config_sha256(case["config_object"].get("final_target_oracle", {}) or {}),
                },
                "dataset_pose_provenance": case.get("pose_provenance"),
            }
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "mvp_strict",
        "creates_map": False,
        "copies_frozen_maps": False,
        "git_status_short": _git_status(),
        "runtime_source_sha256": _runtime_source_sha256(),
        "cases": manifest_cases,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as stream:
            existing_manifest = json.load(stream)
        existing_cases = [
            (
                case["name"],
                case["config_input"]["sha256"],
                case.get("resolved_config_sha256"),
                (case.get("common_config_input") or {}).get("sha256"),
                case["map_input"]["sha256"],
                (case.get("activation_geometry_input") or {}).get("sha256"),
                existing_manifest.get("runtime_source_sha256"),
            )
            for case in existing_manifest["cases"]
        ]
        requested_cases = [
            (
                case["name"],
                case["config_input"]["sha256"],
                case["resolved_config_sha256"],
                (case.get("common_config_input") or {}).get("sha256"),
                case["map_input"]["sha256"],
                (case.get("activation_geometry_input") or {}).get("sha256"),
                manifest["runtime_source_sha256"],
            )
            for case in manifest_cases
        ]
        if existing_cases != requested_cases:
            raise ValueError("existing manifest cases differ from requested cases")
    else:
        _write_json(manifest_path, manifest)

    rows = []
    for case in cases:
        localize_dir = output_root / "localize" / case["name"]
        summary_path = localize_dir / "summary.json"
        state_path = output_root / ".state" / f"{case['name']}.done.json"
        log_path = output_root / "logs" / f"{case['name']}.log"
        provenance = _case_provenance(case)
        completed = _completed_case(summary_path, state_path, provenance) if skip_existing else None
        if completed is not None:
            summary = completed
        else:
            dataset = load_dataset_from_config(case["config"])
            try:
                with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    summary = run_localization(
                        config=case["config_object"],
                        map_path=case["map"],
                        output_dir=str(localize_dir),
                        dataset=dataset,
                        start=case["start"],
                        end=case["end"],
                    )
            except Exception:
                with open(log_path, "a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                raise
            _write_json(
                state_path,
                {
                    "case": case["name"],
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "provenance": provenance,
                },
            )
        row = flatten_summary(case, summary, localize_dir)
        rows.append(row)
        _write_json(output_root / "metrics" / f"{case['name']}.json", row)
        _write_summary_csv(output_root / "summary_all.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = load_cases(args.case, args.plan)
    plan = build_plan(cases, args.output_root, args.start, args.end)
    if args.dry_run:
        printable = {key: value for key, value in plan.items() if not key.startswith("_")}
        print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
        return
    run_plan(plan, skip_existing=args.skip_existing)
    print(json.dumps({"completed": len(cases), "output_root": plan["output_root"]}, sort_keys=True))


if __name__ == "__main__":
    main()
