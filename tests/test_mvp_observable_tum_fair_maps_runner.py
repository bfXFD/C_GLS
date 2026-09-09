import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_tum_fair_maps.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), **overrides})
    return subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def test_fair_map_runner_plans_five_strict_cases_per_sequence(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_eval_long_horizon.py") == 3
    assert result.stdout.count("fsm_common_valid.py") == 3
    for case in ("full", "cgls_voxel10", "cgls_fps10", "observable10", "corrweak128"):
        assert result.stdout.count(f"--case {case}") == 3
    assert "audit_v2_full_rerun/main" in result.stdout
    assert not (tmp_path / "out").exists()


def test_fair_map_runner_filters_sequence_and_stage(tmp_path):
    result = run_plan(tmp_path, SEQUENCES="freiburg3_long_office_household", STAGE="localize")
    assert result.stdout.count("fsm_eval_long_horizon.py") == 1
    assert "fsm_common_valid.py" not in result.stdout
    assert "freiburg1_desk" not in result.stdout


def test_fair_map_runner_rejects_frozen_output(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/fair")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing frozen audit output path" in result.stderr
