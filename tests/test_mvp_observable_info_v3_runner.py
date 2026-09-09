import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_info_v3.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), **overrides})
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )


def test_info_v3_runner_defaults_to_plan_two_scenes_nine_cases_each(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_eval_long_horizon.py") == 2
    assert result.stdout.count("--case") == 18
    assert result.stdout.count("--skip-existing") == 2
    assert result.stdout.count("--start 0") == 2
    assert result.stdout.count("--end 500") == 2
    assert "core_route_v8" not in result.stdout
    assert not (tmp_path / "out").exists()


def test_info_v3_runner_scene_filter_range_and_safety(tmp_path):
    result = run_plan(tmp_path, SCENES="office0", FRAME_START="10", FRAME_END="20", SKIP_EXISTING="0")
    assert result.stdout.count("--case") == 9
    assert "room0" not in result.stdout
    assert "--start 10" in result.stdout and "--end 20" in result.stdout
    assert "--skip-existing" not in result.stdout
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/info_v3")
    rejected = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert rejected.returncode == 2
    assert "Refusing" in rejected.stderr
