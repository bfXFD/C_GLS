import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.fsm_eval_long_horizon as evaluator


ROOT = Path(__file__).resolve().parent.parent


def make_config(path, backend="numpy_robust_point_to_plane", reserve_enabled=False):
    path.write_text(
        "config = {\n"
        " 'dataset': {'dataset_name': 'fake', 'sequence': 'long'},\n"
        " 'registration': {\n"
        f"  'backend': {backend!r},\n"
        "  'evaluation_protocol': 'mvp_strict',\n"
        "  'first_pose_protocol': 'identity',\n"
        "  'motion_model': 'constant_velocity',\n"
        " },\n"
        f" 'observable_reserve': {{'enabled': {reserve_enabled!r}, 'mode': 'active'}},\n"
        " 'local_submap': {'enabled': True, 'min_points': 1},\n"
        " 'eval': {'frame_deltas': [1], 'distance_deltas_m': [1.0], 'segment_lengths_m': [1.0]},\n"
        "}\n"
    )


def test_covariance_backend_is_allowed_for_static_long_horizon_case(tmp_path):
    config = tmp_path / "covariance.py"
    map_path = tmp_path / "map.npz"
    make_config(config, backend="numpy_covariance_weighted")
    map_path.write_bytes(b"map-placeholder")
    plan = evaluator.build_plan(
        evaluator.load_cases([f"covariance,{config},{map_path}"], None),
        str(tmp_path / "out"),
    )
    assert plan["_validated_cases"][0]["backend"] == "numpy_covariance_weighted"


def test_covariance_backend_is_rejected_when_reserve_is_active(tmp_path):
    config = tmp_path / "covariance_reserve.py"
    map_path = tmp_path / "map.npz"
    make_config(config, backend="numpy_covariance_weighted", reserve_enabled=True)
    map_path.write_bytes(b"map-placeholder")
    with pytest.raises(ValueError, match="observable reserve and paired evaluation require"):
        evaluator.build_plan(
            evaluator.load_cases([f"covariance,{config},{map_path}"], None),
            str(tmp_path / "out"),
        )


def make_inputs(tmp_path):
    config = tmp_path / "config.py"
    map_path = tmp_path / "fsm_map.npz"
    make_config(config)
    map_path.write_bytes(b"map-placeholder")
    return config, map_path


def test_cli_dry_run_is_write_free_and_prints_validated_json(tmp_path):
    config, map_path = make_inputs(tmp_path)
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fsm_eval_long_horizon.py",
            "--case",
            f"case0,{config},{map_path}",
            "--output-root",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(result.stdout)
    assert plan["evaluation_protocol"] == "mvp_strict"
    assert plan["creates_map"] is False
    assert plan["copies_frozen_maps"] is False
    assert plan["cases"][0]["backend"] == "numpy_robust_point_to_plane"
    assert plan["cases"][0]["motion_model"] == "constant_velocity"
    assert plan["cases"][0]["first_pose_protocol"] == "identity"
    assert not output.exists()
    assert not (tmp_path / "__pycache__").exists()


def test_plan_json_and_frozen_output_guard(tmp_path):
    config, map_path = make_inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"cases": [{"name": "case0", "config": str(config), "map": str(map_path)}]}))
    cases = evaluator.load_cases(None, str(plan_path))
    assert cases[0]["name"] == "case0"
    with pytest.raises(ValueError, match="must not be inside"):
        evaluator.build_plan(cases, str(tmp_path / "audit_v2_full_rerun" / "new"))


def test_non_dry_run_writes_manifest_snapshots_state_and_summary(tmp_path, monkeypatch):
    config, map_path = make_inputs(tmp_path)
    output = tmp_path / "results"
    cases = evaluator.load_cases([f"case0,{config},{map_path}"], None)
    plan = evaluator.build_plan(cases, str(output), start=3, end=9)
    calls = []

    monkeypatch.setattr(evaluator, "load_dataset_from_config", lambda path: "dataset")

    def fake_run_localization(**kwargs):
        calls.append(kwargs)
        localize = Path(kwargs["output_dir"])
        localize.mkdir(parents=True)
        (localize / "frame_metrics.csv").write_text("condition_number,effective_rank\n10,6\ninf,5\n20,6\n")
        summary = {
            "N_star": 123,
            "num_frames": 6,
            "num_valid_frames": 5,
            "num_lost_frames": 1,
            "tracking_completeness": 5 / 6,
            "raw_ate_rmse_m": 1.0,
            "aligned_ate_rmse_m": 0.5,
            "raw_rotation_abs_rmse_deg": 2.0,
            "translation_drift_percent": 3.0,
            "endpoint_translation_error_m": 4.0,
            "latency_mean_ms": 5.0,
            "lambda_min_mean": 0.2,
            "lambda_min_min": 0.1,
            "total_frame_latency_mean_ms": 7.0,
            "num_consistency_rejections": 2,
            "block_index_build_ms": 1.25,
            "block_score_hash_ms": 0.25,
            "block_geometry_sort_ms": 1.0,
            "block_index_memory_bytes": 4096,
            "block_peak_working_set_estimate_bytes": 8192,
            "precap_strategy": "block_representative",
        }
        (localize / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(evaluator, "run_localization", fake_run_localization)
    rows = evaluator.run_plan(plan)
    assert calls[0]["dataset"] == "dataset"
    assert calls[0]["start"] == 3
    assert calls[0]["end"] == 9
    assert rows[0]["condition_number_mean"] == 15.0
    assert rows[0]["condition_number_inf_count"] == 1
    assert rows[0]["condition_number_degenerate_count"] == 1
    assert (output / "manifest.json").is_file()
    assert (output / "config_snapshot/case0.py").is_file()
    assert (output / "config_snapshot/case0.json").is_file()
    assert (output / "metrics/case0.json").is_file()
    assert (output / "logs/case0.log").is_file()
    assert (output / ".state/case0.done.json").is_file()
    with open(output / "summary_all.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert set(row) == set(evaluator.SUMMARY_FIELDS)
    assert row["N"] == "123"
    assert row["raw_ate_rmse_m"] == "1.0"
    assert row["total_frame_latency_mean_ms"] == "7.0"
    assert row["num_consistency_rejections"] == "2"
    assert row["block_index_build_ms"] == "1.25"
    assert row["block_score_hash_ms"] == "0.25"
    assert row["block_geometry_sort_ms"] == "1.0"
    assert row["block_index_memory_bytes"] == "4096"
    assert row["block_peak_working_set_estimate_bytes"] == "8192"
    assert row["precap_strategy"] == "block_representative"
    state = json.loads((output / ".state/case0.done.json").read_text())
    assert state["provenance"]["resolved_config_sha256"]
    assert state["provenance"]["common_config_sha256"] is None
    assert state["provenance"]["runtime_source_sha256"]
    assert "git_head" in state["provenance"]
    assert "dirty" in state["provenance"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cases"][0]["resolved_config_sha256"] == state["provenance"]["resolved_config_sha256"]


def test_run_failure_is_written_to_case_log(tmp_path, monkeypatch):
    config, map_path = make_inputs(tmp_path)
    output = tmp_path / "failed"
    plan = evaluator.build_plan(
        evaluator.load_cases([f"case0,{config},{map_path}"], None), str(output)
    )
    monkeypatch.setattr(evaluator, "load_dataset_from_config", lambda path: object())

    def fail(**kwargs):
        raise RuntimeError("paired boom")

    monkeypatch.setattr(evaluator, "run_localization", fail)
    with pytest.raises(RuntimeError, match="paired boom"):
        evaluator.run_plan(plan)
    assert "RuntimeError: paired boom" in (output / "logs/case0.log").read_text()


def test_skip_existing_does_not_load_dataset_or_localize(tmp_path, monkeypatch):
    config, map_path = make_inputs(tmp_path)
    output = tmp_path / "results"
    summary_path = output / "localize/case0/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"N_star": 7, "num_frames": 2}))
    plan = evaluator.build_plan(evaluator.load_cases([f"case0,{config},{map_path}"], None), str(output))
    state_path = output / ".state/case0.done.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"provenance": evaluator._case_provenance(plan["_validated_cases"][0])}))
    monkeypatch.setattr(evaluator, "load_dataset_from_config", lambda path: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(evaluator, "run_localization", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))
    rows = evaluator.run_plan(plan, skip_existing=True)
    assert rows[0]["N"] == 7


def test_runtime_source_change_invalidates_skip(tmp_path, monkeypatch):
    config, map_path = make_inputs(tmp_path)
    output = tmp_path / "results"
    summary_path = output / "localize/case0/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"N_star": 7, "num_frames": 2}))
    plan = evaluator.build_plan(evaluator.load_cases([f"case0,{config},{map_path}"], None), str(output))
    state_path = output / ".state/case0.done.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"provenance": evaluator._case_provenance(plan["_validated_cases"][0])}))
    monkeypatch.setattr(evaluator, "_runtime_source_sha256", lambda paths=evaluator.RUNTIME_SOURCE_PATHS: "0" * 64)
    monkeypatch.setattr(evaluator, "load_dataset_from_config", lambda path: "dataset")
    called = []

    def fake_run(**kwargs):
        called.append(True)
        return {"N_star": 8, "num_frames": 2}

    monkeypatch.setattr(evaluator, "run_localization", fake_run)
    evaluator.run_plan(plan, skip_existing=True)
    assert called


def test_runtime_provenance_includes_summary_and_reserve_sources():
    assert "scripts/fsm_eval_long_horizon.py" in evaluator.RUNTIME_SOURCE_PATHS
    assert "scripts/fsm_localize.py" in evaluator.RUNTIME_SOURCE_PATHS
    assert "utils/fsm_runtime_reserve.py" in evaluator.RUNTIME_SOURCE_PATHS
    assert evaluator._runtime_source_sha256()


def test_provenance_changes_when_imported_common_config_changes(tmp_path):
    common = tmp_path / "common.py"
    config = tmp_path / "config.py"
    map_path = tmp_path / "fsm_map.npz"
    common.write_text("VALUE = 1\n")
    config.write_text(
        "from common import VALUE\n"
        "config = {\n"
        " 'dataset': {'dataset_name': 'fake', 'sequence': str(VALUE)},\n"
        " 'registration': {'backend': 'numpy_robust_point_to_plane', 'evaluation_protocol': 'mvp_strict', 'first_pose_protocol': 'identity', 'motion_model': 'constant_velocity'},\n"
        " 'local_submap': {'enabled': True},\n"
        " 'eval': {'frame_deltas': [1], 'distance_deltas_m': [1.0], 'segment_lengths_m': [1.0]},\n"
        "}\n"
    )
    map_path.write_bytes(b"map-placeholder")
    sys.path.insert(0, str(tmp_path))
    try:
        first = evaluator.build_plan(evaluator.load_cases([f"case0,{config},{map_path}"], None), str(tmp_path / "out"))
        first_provenance = evaluator._case_provenance(first["_validated_cases"][0])
        common.write_text("VALUE = 2\n")
        sys.modules.pop("common", None)
        second = evaluator.build_plan(evaluator.load_cases([f"case0,{config},{map_path}"], None), str(tmp_path / "out2"))
        second_provenance = evaluator._case_provenance(second["_validated_cases"][0])
    finally:
        sys.modules.pop("common", None)
        sys.path.remove(str(tmp_path))
    assert first_provenance["common_config_sha256"] != second_provenance["common_config_sha256"]
    assert first_provenance["resolved_config_sha256"] != second_provenance["resolved_config_sha256"]


def test_optional_activation_geometry_does_not_enable_diagnostics(tmp_path):
    config, map_path = make_inputs(tmp_path)
    geometry = tmp_path / "activation_geometry.npz"
    geometry.write_bytes(b"geometry-v1")
    cases = evaluator.load_cases([f"case0,{config},{map_path},{geometry}"], None)
    plan = evaluator.build_plan(cases, str(tmp_path / "out"))
    case = plan["_validated_cases"][0]
    assert "activation_diagnostics" not in case["config_object"]
    provenance = evaluator._case_provenance(case)
    assert provenance["activation_geometry_sha256"] == evaluator._file_record(str(geometry))["sha256"]
    geometry.write_bytes(b"geometry-v2")
    assert evaluator._case_provenance(case)["activation_geometry_sha256"] != provenance["activation_geometry_sha256"]


def test_optional_activation_geometry_injects_only_enabled_consumers(tmp_path):
    config, map_path = make_inputs(tmp_path)
    geometry = tmp_path / "shared_geometry.npz"
    geometry.write_bytes(b"geometry")
    config.write_text(config.read_text().replace(
        " 'eval':",
        " 'activation_diagnostics': {'enabled': True, 'geometry_path': 'auto'},\n"
        " 'observable_reserve': {'enabled': True, 'mode': 'active', 'geometry_path': None},\n"
        " 'paired_reserve_evaluation': {'enabled': True, 'mode': 'shadow', 'geometry_path': 'auto'},\n"
        " 'eval':",
    ))
    case = evaluator.build_plan(
        evaluator.load_cases([f"case0,{config},{map_path},{geometry}"], None), str(tmp_path / "out")
    )["_validated_cases"][0]
    resolved = case["config_object"]
    assert resolved["activation_diagnostics"] == {"enabled": True, "geometry_path": str(geometry.resolve())}
    assert resolved["observable_reserve"] == {
        "enabled": True, "mode": "active", "geometry_path": str(geometry.resolve()),
    }
    assert resolved["paired_reserve_evaluation"] == {
        "enabled": True, "mode": "shadow", "geometry_path": str(geometry.resolve()),
    }


def test_optional_activation_geometry_preserves_disabled_and_explicit_paths(tmp_path):
    config, map_path = make_inputs(tmp_path)
    geometry = tmp_path / "shared_geometry.npz"
    explicit = tmp_path / "existing_geometry.npz"
    geometry.write_bytes(b"shared")
    explicit.write_bytes(b"existing")
    config.write_text(config.read_text().replace(
        " 'eval':",
        f" 'activation_diagnostics': {{'enabled': False, 'geometry_path': 'auto'}},\n"
        f" 'observable_reserve': {{'enabled': False, 'mode': 'off', 'geometry_path': None}},\n"
        f" 'paired_reserve_evaluation': {{'enabled': True, 'mode': 'shadow', 'geometry_path': {str(explicit)!r}}},\n"
        " 'eval':",
    ))
    resolved = evaluator.build_plan(
        evaluator.load_cases([f"case0,{config},{map_path},{geometry}"], None), str(tmp_path / "out")
    )["_validated_cases"][0]["config_object"]
    assert resolved["activation_diagnostics"] == {"enabled": False, "geometry_path": "auto"}
    assert resolved["observable_reserve"] == {"enabled": False, "mode": "off", "geometry_path": None}
    assert resolved["paired_reserve_evaluation"]["geometry_path"] == str(explicit)


def test_reserve_auto_geometry_and_resolved_config_are_in_provenance(tmp_path):
    config, map_path = make_inputs(tmp_path)
    geometry = tmp_path / "activation_geometry.npz"
    geometry.write_bytes(b"reserve-v2")
    config.write_text(config.read_text().replace(
        " 'eval':",
        " 'observable_reserve': {'enabled': True, 'mode': 'plan_only', 'geometry_path': 'auto'},\n 'eval':",
    ))
    plan = evaluator.build_plan(
        evaluator.load_cases([f"case0,{config},{map_path}"], None), str(tmp_path / "out")
    )
    provenance = evaluator._case_provenance(plan["_validated_cases"][0])
    assert provenance["observable_reserve_config_sha256"]
    assert provenance["observable_reserve_geometry_path"] == str(geometry)
    assert provenance["observable_reserve_geometry_sha256"] == evaluator._file_record(str(geometry))["sha256"]


def test_paired_trigger_protocol_and_thresholds_are_resolved_in_provenance(tmp_path):
    config, map_path = make_inputs(tmp_path)
    config.write_text(config.read_text().replace(
        " 'eval':",
        " 'paired_reserve_evaluation': {'enabled': True, 'trigger_protocol': 'absolute_or_scale_normalized_v1', 'trigger': {'condition_number': 120.0, 'normalized_lambda_min': 0.006, 'effective_rank_below': 6}},\n 'eval':",
    ))
    plan = evaluator.build_plan(
        evaluator.load_cases([f"case0,{config},{map_path}"], None), str(tmp_path / "out")
    )
    provenance = evaluator._case_provenance(plan["_validated_cases"][0])
    assert provenance["paired_reserve_trigger_protocol"] == "absolute_or_scale_normalized_v1"
    assert provenance["paired_reserve_trigger"] == {
        "condition_number": 120.0,
        "normalized_lambda_min": 0.006,
        "effective_rank_below": 6,
    }


def test_skip_existing_requires_stability_artifact_when_gate_enabled(tmp_path):
    config, map_path = make_inputs(tmp_path)
    config.write_text(config.read_text().replace(
        " 'eval':",
        " 'observable_reserve': {'enabled': True, 'mode': 'active', 'stability_gate': {'enabled': True, 'mode': 'report_only'}},\n 'eval':",
    ))
    plan = evaluator.build_plan(evaluator.load_cases([f"case0,{config},{map_path}"], None), str(tmp_path / "out"))
    case = plan["_validated_cases"][0]
    output = tmp_path / "case"
    output.mkdir()
    summary = output / "summary.json"
    state = output / "done.json"
    summary.write_text("{}")
    state.write_text(json.dumps({"provenance": evaluator._case_provenance(case)}))
    assert evaluator._completed_case(summary, state, evaluator._case_provenance(case)) is None
    np.savez(output / "reserve_stability_attempts.npz", frame=np.empty(0, dtype=np.int64))
    assert evaluator._completed_case(summary, state, evaluator._case_provenance(case)) == {}
