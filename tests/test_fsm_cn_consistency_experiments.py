import importlib.util
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cn_experiments", ROOT / "scripts/fsm_cn_consistency_experiments.py"
)
cn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cn)


def test_matrix_and_budgets(tmp_path):
    plan = cn.build_plan(cn.parse_args(["--output-root", str(tmp_path / "new")]))
    assert plan["run_count"] == 22
    assert not plan["partial"] and not plan["arm_subset"]
    assert not (tmp_path / "new").exists()
    assert {run["scene"] for run in plan["runs"]} == {"room2", "office0"}
    for run in plan["runs"]:
        config = run["config"]
        reserve = config["observable_reserve"]
        assert config["seed"] == 0
        assert reserve["budget"] == reserve["max_points"] == 128
        assert reserve["rerun_iterations"] == 15
        assert reserve["accept_only_if"] == {}
        assert not reserve["stability_gate"]["enabled"]
        assert reserve["spectral_coordinates"] == "primary_camera_depth"
        assert reserve["candidate_verifier"] == dict(
            enabled=True, block_size=16, folds=4, fold=0,
            relative_tolerance=0.0, absolute_tolerance=1e-6,
            euclidean_relative_tolerance=0.0, euclidean_absolute_tolerance=1e-6,
            min_coverage_drop=0.0, max_group_cost_increase=1e-5, min_group_size=8,
            max_translation_m=0.012, max_rotation_deg=0.3,
        )
        assert "--skip-existing" not in run["command"]
        assert "--case" in run["command"]
        assert "CGLS_CN_ARM=" + run["arm"] in run["command"]
        assert run["sidecar"] in run["command"][-1]
        assert config["registration"]["tracking_acceptance"]
        assert config["registration"]["pose_consistency"]["enabled"]


def test_normal_and_optimization_separation():
    configs = {arm: cn.resolved_config("replica", "room2", arm) for arm in cn.ARMS}
    for prefix in ("rerun", "corr"):
        frozen = configs[prefix + "_frozen"]
        pca = configs[prefix + "_pca"]
        assert frozen["observable_reserve"]["candidate_normal_mode"] == "frozen_concat"
        assert pca["observable_reserve"]["candidate_normal_mode"] == "augmented_local_pca"
        frozen["observable_reserve"]["candidate_normal_mode"] = "augmented_local_pca"
        frozen.pop("cn_consistency")
        pca.pop("cn_consistency")
        assert frozen == pca
    for arm, config in configs.items():
        reserve = config["observable_reserve"]
        assert reserve["attempt_without_selection"] == arm.startswith("rerun_")
        assert reserve["candidate_target_mode"] == ("rerun_primary" if arm.startswith("rerun_") else "append")
        assert reserve["seed"] == (int(arm[-1]) if arm.startswith("random_") else 0)
        assert reserve["replacement_seed"] == 0
    assert not configs["baseline"]["observable_reserve"]["enabled"]
    assert configs["baseline"]["observable_reserve"]["candidate_verifier"]["enabled"]


def test_regression_and_partial_labels(tmp_path):
    plan = cn.build_plan(cn.parse_args([
        "--stage", "regression", "--start", "2", "--end", "4", "--arms", "baseline",
        "--output-root", str(tmp_path / "regression"),
    ]))
    assert plan["run_count"] == 5
    assert plan["partial"] and plan["arm_subset"]
    assert plan["scene_exposure"] == "historically_exposed_regression_not_untouched"
    assert "no_composite_conclusions_from_subsets" in plan["interpretation"]
    assert "replica_office1_4_untouched_v1" in plan["runs"][0]["map"]
    assert "tum_fps_hybrid_v1" in plan["runs"][-1]["map"]


@pytest.mark.parametrize("argv", [
    ["--start", "-1"], ["--end", "0"], ["--start", "4", "--end", "4"],
    ["--arms", "replace"], ["--arms", "baseline", "baseline"],
    ["--stage", "untouched"], ["--execute", "--collect"], ["--conda-env="],
])
def test_bad_arguments(argv):
    with pytest.raises(SystemExit):
        cn.parse_args(argv)


@pytest.mark.parametrize("dataset,scene,arm", [
    ("bad", "room2", "baseline"), ("replica", "../room2", "baseline"),
    ("replica", "room2", "replace"), ("tum", "office0", "baseline"),
])
def test_bad_config(dataset, scene, arm):
    with pytest.raises(ValueError):
        cn.resolved_config(dataset, scene, arm)


def test_missing_config_environment(monkeypatch):
    for name in ("CGLS_CN_DATASET", "CGLS_CN_SCENE", "CGLS_CN_ARM"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        runpy.run_path(str(cn.CASE))


def test_path_rejections(tmp_path):
    for path in (ROOT, ROOT.parent, ROOT / "experiments/FSM/anything", ROOT / "scripts/output", tmp_path / "audit_v2_full_rerun/new"):
        with pytest.raises(ValueError):
            cn.safe_output(path)
    (tmp_path / "old").mkdir()
    (tmp_path / "old/existing").touch()
    with pytest.raises(ValueError, match="nonempty"):
        cn.safe_output(tmp_path / "old")
    (tmp_path / "link").symlink_to(tmp_path / "old", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        cn.safe_output(tmp_path / "link/new")
    assert cn.safe_output(tmp_path / "old", collect=True) == tmp_path / "old"


def test_default_plan_has_no_side_effects(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("planning must not execute localization or hash input data")
    monkeypatch.setattr(cn.subprocess, "run", forbidden)
    monkeypatch.setattr(cn, "file_record", forbidden)
    output = tmp_path / "new"
    assert cn.main(["--output-root", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["run_count"] == 22
    assert not output.exists()


def test_freeze_before_first_run_and_no_resume(tmp_path, monkeypatch):
    output = tmp_path / "new"
    plan = cn.build_plan(cn.parse_args(["--output-root", str(output), "--arms", "baseline"]))
    map_path, sidecar = tmp_path / "map.npz", tmp_path / "geometry.npz"
    map_path.write_bytes(b"map")
    sidecar.write_bytes(b"geometry")
    for run in plan["runs"]:
        run.update(map=str(map_path), sidecar=str(sidecar))
    calls = []
    def fake_run(command, **kwargs):
        frozen = json.loads((output / "protocol_manifest.json").read_text())
        assert frozen["frozen_before_any_run"]
        assert len(frozen["input_hashes"]) == 2
        assert all(len(record["sha256"]) == 64 for record in frozen["source_hashes"])
        assert frozen["runs"][len(calls)]["command"] == command
        assert kwargs == dict(cwd=ROOT, check=True)
        calls.append(command)
    monkeypatch.setattr(cn.subprocess, "run", fake_run)
    monkeypatch.setattr(cn, "preflight", lambda run, plan: None)
    cn.freeze_and_execute(plan)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="nonempty"):
        cn.freeze_and_execute(plan)


def test_missing_input_fails_before_manifest(tmp_path, monkeypatch):
    output = tmp_path / "new"
    plan = cn.build_plan(cn.parse_args(["--output-root", str(output), "--arms", "baseline"]))
    plan["runs"][0]["map"] = str(tmp_path / "missing.npz")
    monkeypatch.setattr(cn.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(FileNotFoundError):
        cn.freeze_and_execute(plan)
    assert not output.exists()


def test_collect_missing_not_zero_and_available_metrics(tmp_path):
    output = tmp_path / "new"
    plan = cn.build_plan(cn.parse_args(["--output-root", str(output), "--arms", "baseline"]))
    output.mkdir()
    plan["frozen_before_any_run"] = True
    (output / "protocol_manifest.json").write_text(json.dumps(plan))
    run = plan["runs"][0]
    directory = output / "runs" / run["name"]
    local = directory / "localize" / run["name"]
    local.mkdir(parents=True)
    (local / "summary.json").write_text(json.dumps(dict(num_frames=3, num_valid_frames=2, num_lost_frames=1, jts_percent=72)))
    state = directory / ".state"
    state.mkdir()
    (state / f"{run['name']}.done.json").write_text("{}")
    (local / "frame_metrics.csv").write_text(
        "candidate_verifier_decision,production_equivalent_total_frame_latency_ms\n"
        "accepted,2\nrejected,4\n,nan\n"
    )
    result = cn.collect(output)
    row, absent = result["runs"]
    assert row["status"] == "complete" and row["lost"] == 1 and row["valid"] == 2
    assert row["jts"] is None
    assert row["jts_status"] == "missing_frame_or_raw_errors"
    assert row["verifier_decision_counts"]["candidate_verifier_decision"] == dict(counts={"accepted": 1, "rejected": 1}, observed=2, missing=1)
    assert row["frame_latency"]["production_equivalent_total_frame_latency_ms"] == dict(mean_ms=3, observed=2, missing=1)
    assert absent["status"] == "incomplete"
    for field in ("lost", "valid", "frames", "jts", "verifier_decision_counts", "frame_latency", "frame_rows"):
        assert absent[field] is None
    assert (output / "collection/runs.csv").is_file()


def test_collect_requires_frozen_manifest(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        cn.collect(tmp_path)


@pytest.mark.parametrize("stage", ["dev", "regression"])
def test_stage_default_output(stage):
    args = cn.parse_args(["--stage", stage])
    assert args.output_root == cn.MVP / f"cn_consistency_{stage}_v1"
    assert cn.safe_output(args.output_root, collect=True) == args.output_root


def test_only_new_study_roots_in_historical_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "ROOT", tmp_path)
    monkeypatch.setattr(cn, "MVP", tmp_path / "experiments/FSM/mvp_observable_v1")
    allowed = cn.MVP / "cn_consistency_new_v1"
    assert cn.safe_output(allowed) == allowed
    for path in (cn.MVP, cn.MVP / "runtime_reserve_v1", allowed / "child",
                 cn.MVP.parent / "cn_consistency_bad"):
        with pytest.raises(ValueError):
            cn.safe_output(path)
    allowed.mkdir(parents=True)
    (allowed / "history.json").write_text("{}")
    with pytest.raises(ValueError, match="nonempty"):
        cn.safe_output(allowed)
    assert cn.safe_output(allowed, collect=True) == allowed


def test_jts_strict_thresholds_invalid_nan_first_included(tmp_path):
    frames = [dict(frame_id=str(i), timestamp=str(i), tracking_valid=valid)
              for i, valid in enumerate(("True", "True", "False", "True"))]
    (tmp_path / "per_frame_errors.csv").write_text(
        "frame_id,timestamp,tracking_valid,raw_translation_error_m,raw_rotation_error_deg\n"
        "0,0,True,0,0\n1,1,True,0.25,5\n2,2,False,0,0\n3,3,True,nan,nan\n"
    )
    score, status, identities = cn.matched_jts(tmp_path, frames, {"num_frames": 4})
    assert score == 37.5
    assert status == "matched_all_frames_first_included"
    assert identities == [(i, float(i)) for i in range(4)]
    frames[1]["timestamp"] = "1.01"
    with pytest.raises(ValueError, match="timestamp mismatch"):
        cn.matched_jts(tmp_path, frames, {})


@pytest.mark.parametrize("change,match", [
    ("count", "row count"), ("validity", "validity mismatch"),
    ("duplicate", "duplicate or unordered"), ("summary", "summary/frame"),
])
def test_jts_rejects_mismatched_tables(tmp_path, change, match):
    frames = [dict(frame_id=str(i), timestamp=str(i), tracking_valid="True") for i in range(2)]
    (tmp_path / "per_frame_errors.csv").write_text(
        "frame_id,timestamp,tracking_valid,raw_translation_error_m,raw_rotation_error_deg\n"
        "0,0,True,0,0\n1,1,True,0,0\n"
    )
    summary = {}
    if change == "count":
        frames.pop()
    elif change == "validity":
        frames[1]["tracking_valid"] = "False"
    elif change == "duplicate":
        frames[1] = frames[0].copy()
        (tmp_path / "per_frame_errors.csv").write_text(
            "frame_id,timestamp,tracking_valid,raw_translation_error_m,raw_rotation_error_deg\n"
            "0,0,True,0,0\n0,0,True,0,0\n"
        )
    else:
        summary["num_frames"] = 3
    with pytest.raises(ValueError, match=match):
        cn.matched_jts(tmp_path, frames, summary)


@pytest.mark.parametrize("kind", ["input", "source"])
def test_drift_stops_before_next_run(tmp_path, monkeypatch, kind):
    output = tmp_path / "new"
    plan = cn.build_plan(cn.parse_args(["--output-root", str(output), "--arms", "baseline"]))
    dependency = tmp_path / "dependency"
    dependency.write_bytes(b"original")
    for run in plan["runs"]:
        run.update(map=str(dependency), sidecar=str(dependency))
    real_record = cn.file_record
    def record(path):
        if kind == "source" and Path(path) == cn.CASE:
            return real_record(dependency)
        return real_record(path)
    monkeypatch.setattr(cn, "file_record", record)
    monkeypatch.setattr(cn, "preflight", lambda run, plan: None)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        dependency.write_bytes(b"changed")
    monkeypatch.setattr(cn.subprocess, "run", execute)
    with pytest.raises(ValueError, match="frozen source/input changed"):
        cn.freeze_and_execute(plan)
    assert len(calls) == 1


def test_real_preflight_helpers_without_localization(tmp_path):
    plan = cn.build_plan(cn.parse_args(["--output-root", str(tmp_path / "new"), "--arms", "baseline", "rerun_pca"]))
    map_path = tmp_path / "map.npz"
    map_path.write_bytes(b"preflight only checks existence; never loads localization map")
    for run in plan["runs"][:2]:
        run.update(map=str(map_path), sidecar=str(map_path))
        cn.preflight(run, plan)
    run["config"]["observable_reserve"]["candidate_verifier"]["max_translation_m"] = -1
    with pytest.raises(cn.subprocess.CalledProcessError):
        cn.preflight(run, plan)


def test_preflight_failure_before_freeze(tmp_path, monkeypatch):
    plan = cn.build_plan(cn.parse_args(["--output-root", str(tmp_path / "new"), "--arms", "baseline"]))
    monkeypatch.setattr(cn, "file_record", lambda path: {})
    def fail(run, plan):
        raise ValueError("incompatible verifier")
    monkeypatch.setattr(cn, "preflight", fail)
    with pytest.raises(ValueError, match="incompatible"):
        cn.freeze_and_execute(plan)
    assert not (tmp_path / "new").exists()


def test_collect_computes_raw_jts_ate_and_matches_arms(tmp_path):
    output = tmp_path / "new"
    plan = cn.build_plan(cn.parse_args([
        "--output-root", str(output), "--arms", "baseline", "corr_pca",
    ]))
    plan["runs"] = plan["runs"][:2]
    plan["frozen_before_any_run"] = True
    output.mkdir()
    (output / "protocol_manifest.json").write_text(json.dumps(plan))
    locals_ = []
    for run in plan["runs"]:
        directory = output / "runs" / run["name"]
        local = directory / "localize" / run["name"]
        local.mkdir(parents=True)
        locals_.append(local)
        state = directory / ".state"
        state.mkdir()
        (state / f"{run['name']}.done.json").write_text("{}")
        (local / "summary.json").write_text(json.dumps(dict(num_frames=2, raw_ate_rmse_m=.25)))
        (local / "long_horizon_metrics.json").write_text(json.dumps(dict(
            raw_absolute=dict(translation_rmse_m=.75, rotation_rmse_deg=1.5),
        )))
        (local / "frame_metrics.csv").write_text(
            "frame_id,timestamp,tracking_valid\n0,0,True\n1,1,False\n"
        )
        (local / "per_frame_errors.csv").write_text(
            "frame_id,timestamp,tracking_valid,raw_translation_error_m,raw_rotation_error_deg\n"
            "0,0,True,0,0\n1,1,False,0,0\n"
        )
    rows = cn.collect(output)["runs"]
    assert all(row["jts_percent"] == 50 for row in rows)
    assert all(row["raw_ate_rmse_m"] == .25 for row in rows)
    assert all(row["raw_rotation_abs_rmse_deg"] == 1.5 for row in rows)
    # A consistently shifted raw/frame table is still incomparable across arms.
    (locals_[1] / "frame_metrics.csv").write_text(
        "frame_id,timestamp,tracking_valid\n0,0,True\n1,2,False\n"
    )
    (locals_[1] / "per_frame_errors.csv").write_text(
        "frame_id,timestamp,tracking_valid,raw_translation_error_m,raw_rotation_error_deg\n"
        "0,0,True,0,0\n1,2,False,0,0\n"
    )
    with pytest.raises(ValueError, match="arm frame/timestamp mismatch"):
        cn.collect(output)
