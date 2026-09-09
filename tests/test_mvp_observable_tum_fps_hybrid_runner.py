import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_tum_fps_hybrid.sh"


def test_fps_hybrid_runner_plans_complete_group(tmp_path):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "out"), "MAP_ROOT": str(tmp_path / "maps")})
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("fsm_build_fixed_map_reserve.py") == 3
    assert result.stdout.count("fsm_eval_long_horizon.py") == 6
    assert result.stdout.count("fsm_score_paired_reserve_gt.py") == 3
    assert result.stdout.count("fsm_paired_frame_stats.py") == 3
    assert result.stdout.count("fsm_common_valid.py") == 3
    assert result.stdout.count("fsm_critical_adoption.py") == 3
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "maps").exists()


def test_fps_hybrid_runner_rejects_frozen_output():
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/hybrid")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing frozen audit output path" in result.stderr
