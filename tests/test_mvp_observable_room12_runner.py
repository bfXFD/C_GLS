import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_room12_generalization.sh"


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


def test_room12_runner_defaults_to_safe_map_plan(tmp_path):
    result = run_plan(tmp_path)
    assert result.stdout.count("fsm_build_observable_map.py") == 2
    assert "--ratio 0.2" in result.stdout
    assert "--max-reference-poses 64" in result.stdout
    assert "--candidate-cap 4096" in result.stdout
    assert "--pair-cap 200000" in result.stdout
    assert "--greedy-core-cap 128" in result.stdout
    assert "--replacement-count 128" in result.stdout
    assert "--reserve-count 4096" in result.stdout
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "maps").exists()


def test_room12_runner_plans_paired_score_stats_and_closed_loop(tmp_path):
    paired = run_plan(tmp_path, STAGE="paired", SCENES="room1")
    assert paired.stdout.count("fsm_eval_long_horizon.py") == 1
    assert "replica_room1_paired_core_route.py" in paired.stdout
    assert "--start 0" in paired.stdout and "--end 500" in paired.stdout
    score = run_plan(tmp_path, STAGE="score", SCENES="room2")
    assert "fsm_score_paired_reserve_gt.py" in score.stdout
    stats = run_plan(tmp_path, STAGE="stats", SCENES="room2")
    assert "fsm_paired_frame_stats.py" in stats.stdout
    assert "--expected-random-seeds 0 1 2 3 4" in stats.stdout
    assert "--require-complete-random-panel" in stats.stdout
    closed = run_plan(tmp_path, STAGE="closed_loop", SCENES="room1", FRAME_END="2000")
    assert closed.stdout.count("--case") == 2
    assert "replica_room1_corr_weak128_info_v4.py" in closed.stdout
    assert "--end 2000" in closed.stdout


def test_room12_runner_rejects_frozen_audit_paths(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(ROOT / "experiments/FSM/audit_v2_full_rerun/room12")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Refusing frozen audit path" in result.stderr
