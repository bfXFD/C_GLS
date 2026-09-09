import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_all.sh"


def test_runner_uses_registered_mvp_output_root():
    text = RUNNER.read_text()
    assert "experiments/FSM/mvp_observable_v1" in text


def test_runner_defaults_to_plan_only_and_prints_complete_stages(tmp_path):
    output = tmp_path / "must_not_exist"
    env = os.environ.copy()
    env["OUT_ROOT"] = str(output)
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    stdout = result.stdout
    assert stdout.count("fsm_build_observable_map.py") == 21
    assert stdout.count("fsm_eval_long_horizon.py") == 7
    assert "--greedy-core-cap 0" in stdout
    assert "--replacement-count 0" in stdout
    assert "--replacement-count 128" in stdout
    assert "--replacement-count 64" in stdout
    assert "--min-eigen-every 0" in stdout
    assert "--min-eigen-every 16" in stdout
    assert "--max-reference-poses 64" in stdout
    assert "--candidate-cap 4096" in stdout
    assert "--pair-cap 200000" in stdout
    assert stdout.count("--reserve-count 4096") == 14
    cheap_lines = [line for line in stdout.splitlines() if "cheap_baseline/selection.npz" in line]
    assert cheap_lines and all("--reserve-count" not in line for line in cheap_lines)
    logdet_lines = [line for line in stdout.splitlines() if "/logdet/selection.npz" in line]
    assert logdet_lines and all("--min-eigen-every 0" in line and "--min-eigen-bonus 0" in line for line in logdet_lines)
    min_eigen_lines = [line for line in stdout.splitlines() if "/min_eigen/selection.npz" in line]
    assert min_eigen_lines and all("--min-eigen-every 16" in line and "--min-eigen-bonus 0.1" in line for line in min_eigen_lines)
    assert stdout.count("--activation-geometry-output") == 14
    assert "cheap_baseline/activation_geometry.npz" not in stdout
    assert "logdet/activation_geometry.npz" in stdout
    assert "min_eigen/activation_geometry.npz" in stdout
    assert "--skip-existing" in stdout
    assert "--case baseline" in stdout
    assert "--case cheap_scale_opacity_baseline" in stdout
    assert "--case observable_core_cheap_fill" in stdout
    assert "--case observable_core_weak_cheap_fill" in stdout
    assert "--case observable_trace" not in stdout
    assert "audit_v2_full_rerun" in stdout
    assert not output.exists()


def test_runner_filters_datasets_and_stages(tmp_path):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), "DATASETS_FILTER": "replica_room0_0", "STAGES": "maps"})
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    assert result.stdout.count("fsm_build_observable_map.py") == 3
    assert "fsm_eval_long_horizon.py" not in result.stdout


def test_runner_rejects_frozen_audit_output():
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/new")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing" in result.stderr


def test_runner_rejects_invalid_reserve_count(tmp_path):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), "RESERVE_COUNT": "unsafe"})
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "RESERVE_COUNT" in result.stderr
