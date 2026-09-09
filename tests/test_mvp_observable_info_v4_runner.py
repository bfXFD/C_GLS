import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_info_v4_minimal.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), **overrides})
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )


def test_info_v4_minimal_runner_defaults_to_plan_two_scenes_two_cases_each(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_eval_long_horizon.py") == 2
    assert result.stdout.count("--case") == 4
    assert result.stdout.count("--skip-existing") == 2
    assert result.stdout.count("--start 0") == 2
    assert result.stdout.count("--end 500") == 2
    assert "core_route_v9" not in result.stdout
    assert not (tmp_path / "out").exists()


def test_info_v4_minimal_runner_filter_range_and_safety(tmp_path):
    result = run_plan(tmp_path, SCENES="office0", FRAME_START="240", FRAME_END="254", SKIP_EXISTING="0")
    assert result.stdout.count("--case") == 2
    assert "room0" not in result.stdout
    assert "--start 240" in result.stdout and "--end 254" in result.stdout
    assert "--skip-existing" not in result.stdout
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/info_v4")
    rejected = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert rejected.returncode == 2
    assert "Refusing" in rejected.stderr
