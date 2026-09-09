import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_tum_fps_trigger_ablation.sh"


def test_trigger_ablation_runner_defaults_to_plan_only(tmp_path):
    env = os.environ.copy()
    env["OUT_ROOT"] = str(tmp_path / "out")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("--case") == 3
    assert "gate_mid" in result.stdout
    assert "gate_strict" in result.stdout
    assert not (tmp_path / "out").exists()
