import os
import subprocess
from copy import deepcopy
from pathlib import Path

from tests.test_mvp_observable_configs import load_config


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_mvp_observable_office0_closed_loop.sh"


def test_office0_closed_loop_configs_only_change_method_controls():
    names = [
        "replica_office0_corr_weak64",
        "replica_office0_corr_weak128",
        "replica_office0_core_route_weak128",
        "replica_office0_core_route_saliency128",
        *[f"replica_office0_core_route_random128_seed{seed}" for seed in range(5)],
    ]
    configs = [load_config(name)[1] for name in names]
    for config in configs:
        assert config["dataset"]["sequence"] == "office0"
        assert config["observable_reserve"]["mode"] == "active"
    normalized = []
    for config in configs:
        copied = deepcopy(config)
        reserve = copied["observable_reserve"]
        for key in ("selection", "budget", "max_points", "seed"):
            reserve.pop(key)
        normalized.append(copied)
    assert all(config == normalized[0] for config in normalized[1:])


def test_relaxed_gate_is_explicitly_less_strict():
    strict = load_config("replica_room0_reserve_stability_enforce")[1]["observable_reserve"]["stability_gate"]
    relaxed = load_config("replica_room0_reserve_stability_enforce_relaxed")[1]["observable_reserve"]["stability_gate"]
    assert relaxed["max_refinement_translation_m"] > strict["max_refinement_translation_m"]
    assert relaxed["max_rolling_translation_l1_m"] > strict["max_rolling_translation_l1_m"]
    assert relaxed["max_consecutive_adoptions"] > strict["max_consecutive_adoptions"]
    assert relaxed["common_target"]["max_cost_ratio"] > strict["common_target"]["max_cost_ratio"]


def test_office0_runner_defaults_to_plan_only():
    env = os.environ.copy()
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("--case") == 10
    assert "replica_office0_corr_weak64.py" in result.stdout
    assert "replica_office0_corr_weak128.py" in result.stdout
    assert "replica_office0_core_route_weak128.py" in result.stdout
    assert "replica_office0_core_route_saliency128.py" in result.stdout
    assert result.stdout.count("replica_office0_core_route_random128_seed") == 5
