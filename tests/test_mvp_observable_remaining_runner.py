import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_remaining_experiments.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({
        "TIMING_ROOT": str(tmp_path / "timing"),
        "COV_ROOT": str(tmp_path / "covariance"),
        **overrides,
    })
    return subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def test_remaining_runner_plans_all_experiments_serially(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_common_valid.py") == 2
    assert result.stdout.count("fsm_eval_long_horizon.py") == 12
    assert result.stdout.index("freiburg2_xyz/common_valid") < result.stdout.index("office0_repeat1_baseline")
    assert result.stdout.index("office0_repeat3_corrweak") < result.stdout.index("tum_fr2_repeat1_baseline")
    assert not (tmp_path / "timing").exists()
    assert not (tmp_path / "covariance").exists()


def test_remaining_runner_filters_stage(tmp_path):
    result = run_plan(tmp_path, STAGE="office0")
    assert result.stdout.count("fsm_eval_long_horizon.py") == 6
    assert "fsm_common_valid.py" not in result.stdout
    assert "tum_fr2_repeat" not in result.stdout


def test_remaining_runner_rejects_frozen_output(tmp_path):
    env = os.environ.copy()
    env["TIMING_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/timing")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing frozen audit output path" in result.stderr
