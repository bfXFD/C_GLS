import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cgls_required_acceptance_experiments.sh"


def plan(tmp_path):
    env = os.environ.copy()
    env.update({
        "PLAN_ONLY": "1", "SKIP_EXISTING": "1", "FRAME_START": "0", "FRAME_END": "-1",
        "OUT_ROOT": str(tmp_path / "required"),
    })
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    ).stdout


def test_master_plan_uses_scene_local_paths_and_unique_evaluations(tmp_path):
    output = plan(tmp_path)
    for scene in ("office0", "room2", "office4", "fr3", "reverse"):
        assert f"/paired/{scene}/localize/paired" in output
        assert f"/horizon_gtfree/{scene}/h1" in output
        assert f"/horizon_labels/{scene}/h10" in output
    for scene in ("room2", "office4", "fr3", "reverse"):
        assert output.count(f"/evaluation/{scene}.json") == 1


def test_master_plan_is_serial_and_freezes_only_office0(tmp_path):
    output = plan(tmp_path)
    lines = output.splitlines()
    reverse = next(i for i, line in enumerate(lines) if "run_fsm_independent_revisit_desk2_to_desk.sh" in line)
    office0_pair = next(i for i, line in enumerate(lines) if "/paired/office0 " in line)
    freeze = next(i for i, line in enumerate(lines) if "fsm_freeze_horizon_thresholds.py" in line)
    evaluate = next(i for i, line in enumerate(lines) if "fsm_evaluate_horizon_test.py" in line)
    assert reverse < office0_pair < freeze < evaluate
    freeze_line = lines[freeze]
    assert "/office0/" in freeze_line
    assert "/room2/" not in freeze_line and "/office4/" not in freeze_line and "/fr3/" not in freeze_line
