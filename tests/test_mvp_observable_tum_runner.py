import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_tum_core.sh"


def run_plan(tmp_path, **overrides):
    env = os.environ.copy()
    env.update({
        "OUT_ROOT": str(tmp_path / "out"),
        "MAP_ROOT": str(tmp_path / "maps"),
        **overrides,
    })
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )


def test_tum_runner_defaults_to_complete_serial_plan(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("[sequence]") == 3
    assert result.stdout.count("fsm_build_observable_map.py") == 3
    assert result.stdout.count("fsm_eval_long_horizon.py") == 6
    assert result.stdout.count("fsm_score_paired_reserve_gt.py") == 3
    assert result.stdout.count("fsm_paired_frame_stats.py") == 3
    assert result.stdout.count("fsm_common_valid.py") == 3
    assert result.stdout.count("fsm_critical_adoption.py") == 3
    assert "--ratio 0.1" in result.stdout
    assert "--reserve-count 4096" in result.stdout
    assert "--end 500" in result.stdout
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "maps").exists()


def test_tum_runner_filters_sequence_and_stage(tmp_path):
    result = run_plan(tmp_path, SEQUENCES="freiburg2_xyz", STAGE="closed_loop")
    assert result.stdout.count("fsm_eval_long_horizon.py") == 1
    assert result.stdout.count("--case") == 2
    assert "tum_freiburg2_xyz_corr_weak128_info_v4.py" in result.stdout
    assert "freiburg1_desk" not in result.stdout
    assert "freiburg3_long_office_household" not in result.stdout


def test_tum_runner_rejects_frozen_audit_paths(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/tum_core")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing frozen audit path" in result.stderr
