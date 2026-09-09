import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_tum_covariance_map_order.sh"


def test_covariance_runner_defaults_to_two_sequence_plan(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(tmp_path / "out")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("fsm_eval_long_horizon.py") == 2
    assert result.stdout.count("--case") == 6
    assert "freiburg2_xyz" in result.stdout
    assert "freiburg3_long_office_household" in result.stdout
    assert not (tmp_path / "out").exists()
