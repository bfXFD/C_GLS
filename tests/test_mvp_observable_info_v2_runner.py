import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_info_v2.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), **overrides})
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )


def test_info_v2_runner_defaults_to_two_independent_evaluators_and_does_not_execute(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_eval_long_horizon.py") == 2
    assert result.stdout.count("--case") == 18
    assert "/room0" in result.stdout and "/office0" in result.stdout
    assert result.stdout.count("--skip-existing") == 2
    assert result.stdout.count("--start 0") == 2
    assert result.stdout.count("--end 500") == 2
    assert not (tmp_path / "out").exists()


def test_info_v2_runner_scene_filter_and_optional_corr64(tmp_path):
    room = run_plan(tmp_path, SCENES="room0", SKIP_EXISTING="0")
    assert room.stdout.count("fsm_eval_long_horizon.py") == 1
    assert room.stdout.count("--case") == 9
    assert "office0" not in room.stdout
    assert "--skip-existing" not in room.stdout
    office = run_plan(tmp_path, SCENES="office0", INCLUDE_CORR64="1")
    assert office.stdout.count("--case") == 10
    assert "corr_weak64_info_v2.py" in office.stdout


def test_info_v2_runner_rejects_frozen_output(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/info_v2")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing" in result.stderr


def test_info_v2_runner_accepts_explicit_frame_range(tmp_path):
    result = run_plan(tmp_path, SCENES="room0", FRAME_START="10", FRAME_END="20")
    assert "--start 10" in result.stdout
    assert "--end 20" in result.stdout
