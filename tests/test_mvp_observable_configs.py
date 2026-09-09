import os
import sys
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path

import utils.fsm_dataset as fsm_dataset


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "fsm" / "mvp_observable"
CONFIG_NAMES = [
    "replica_room0",
    "replica_room1",
    "replica_room2",
    "replica_office0",
    "tum_freiburg1_desk",
    "tum_freiburg2_xyz",
    "tum_freiburg3_long_office_household",
]
ABLATION_NAMES = [
    "replica_room0_block_oracle",
    "replica_room0_block_03_cap80k",
    "replica_room0_block_03_cap80k_oracle",
    "replica_room0_reserve_plan",
    "replica_room0_reserve_active",
    "replica_room0_reserve_active_runtime_weak",
    "replica_room0_reserve_active_saliency",
    "replica_room0_reserve_active_random",
]


def load_config(name):
    path = CONFIG_DIR / f"{name}.py"
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = SourceFileLoader(f"test_{name}", str(path)).load_module()
    finally:
        sys.dont_write_bytecode = old
    return path, module.config


def test_all_mvp_observable_configs_load_and_share_strict_contract():
    for name in CONFIG_NAMES:
        path, config = load_config(name)
        assert path.is_file()
        assert config["data"] is config["dataset"]
        registration = config["registration"]
        assert registration["evaluation_protocol"] == "mvp_strict"
        assert registration["first_pose_protocol"] == "gt_first_frame"
        assert registration["backend"] == "numpy_robust_point_to_plane"
        assert registration["motion_model"] == "constant_velocity"
        assert registration["fallback_on_divergence"] is False
        assert registration["normal_source"] == "local_pca"
        assert registration["tracking_acceptance"]["min_correspondences"] > 0
        assert config["local_submap"]["enabled"] is True
        assert config["local_submap"]["sampling_policy"] == "coverage"
        assert registration["pose_consistency"]["enabled"] is True
        assert config["recovery"]["verifier_backend"] == "numpy_point_to_point"
        assert config["eval"]["frame_deltas"]
        assert config["eval"]["distance_deltas_m"]
        assert config["eval"]["segment_lengths_m"]


def test_configs_keep_authoritative_dataset_paths():
    _, replica = load_config("replica_room0")
    _, tum = load_config("tum_freiburg1_desk")
    assert replica["dataset"]["basedir"] == "/home/ras/xbf/SplaTAM/data/Replica"
    assert replica["dataset"]["gradslam_data_cfg"] == str(ROOT / "configs/data/replica.yaml")
    assert tum["dataset"]["basedir"] == str(ROOT / "data/TUM_RGBD")
    assert tum["dataset"]["gradslam_data_cfg"] == str(ROOT / "configs/data/TUM/freiburg1_desk.yaml")


def test_tum_pose_thresholds_are_explicit_and_not_tighter_than_replica():
    _, replica = load_config("replica_room0")
    _, tum = load_config("tum_freiburg1_desk")
    assert tum["registration"]["prediction_max_translation_m"] >= replica["registration"]["prediction_max_translation_m"]
    assert tum["registration"]["prediction_max_rotation_deg"] >= replica["registration"]["prediction_max_rotation_deg"]
    for section in ("constant_velocity", "bootstrap"):
        tum_limits = tum["registration"]["pose_consistency"]["correction"][section]
        replica_limits = replica["registration"]["pose_consistency"]["correction"][section]
        assert tum_limits["translation_m"] >= replica_limits["translation_m"]
        assert tum_limits["rotation_deg"] >= replica_limits["rotation_deg"]


def test_load_dataset_from_config_accepts_thin_config(tmp_path, monkeypatch):
    path, _ = load_config("replica_room1")
    captured = {}

    monkeypatch.setattr("datasets.gradslam_datasets.load_dataset_config", lambda value: {"dataset_name": "replica"})

    def fake_get_dataset(data_cfg, **kwargs):
        captured.update(data_cfg=data_cfg, **kwargs)
        return "dataset"

    monkeypatch.setattr("scripts.splatam.get_dataset", fake_get_dataset)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        result = fsm_dataset.load_dataset_from_config(str(path))
    finally:
        sys.dont_write_bytecode = old
    assert result == "dataset"
    assert captured["sequence"] == "room1"
    assert captured["basedir"] == "/home/ras/xbf/SplaTAM/data/Replica"


def test_room0_ablation_configs_are_thin_and_opt_in():
    _, default = load_config("replica_room0")
    assert default["local_submap"]["frustum_culling"] == "radius"
    assert default["local_submap"]["projection_pre_cap"] is None
    assert default["local_submap"]["oracle_mode"] == "off"
    assert default["observable_reserve"]["enabled"] is False
    assert default["observable_reserve"]["mode"] == "off"
    assert default["paired_reserve_evaluation"]["enabled"] is False
    assert default["final_target_oracle"]["enabled"] is False
    for name in ABLATION_NAMES:
        path, config = load_config(name)
        assert path.is_file()
        assert config["dataset"]["sequence"] == "room0"
    _, oracle = load_config("replica_room0_block_oracle")
    assert oracle["local_submap"]["frustum_culling"] == "voxel_blocks"
    assert oracle["local_submap"]["oracle_mode"] == "measure"
    _, capped = load_config("replica_room0_block_03_cap80k")
    assert capped["local_submap"]["projection_pre_cap"] == 80000
    _, cap_oracle = load_config("replica_room0_block_03_cap80k_oracle")
    assert cap_oracle["local_submap"]["projection_pre_cap"] == 80000
    assert cap_oracle["local_submap"]["oracle_mode"] == "measure"
    _, planned = load_config("replica_room0_reserve_plan")
    _, active = load_config("replica_room0_reserve_active")
    assert planned["observable_reserve"]["mode"] == "plan_only"
    assert active["observable_reserve"]["mode"] == "active"


def test_paired_and_final_oracle_thin_configs():
    _, paired = load_config("replica_room0_paired_core_route")
    assert paired["paired_reserve_evaluation"]["enabled"] is True
    assert paired["paired_reserve_evaluation"]["policies"] == [
        "runtime_weak", "runtime_corr_weak", "saliency", "random"
    ]
    assert paired["paired_reserve_evaluation"]["random_seeds"] == [0, 1, 2, 3, 4]
    assert paired["paired_reserve_evaluation"]["budgets"] == [64, 128, 256]
    assert paired["paired_reserve_evaluation"]["trigger_protocol"] == "absolute_or_scale_normalized_v1"
    assert paired["paired_reserve_evaluation"]["trigger"] == {
        "condition_number": 120.0,
        "normalized_lambda_min": 0.006,
        "effective_rank_below": 6,
    }
    assert paired["observable_reserve"]["trigger_protocol"] == "absolute_or_scale_normalized_v1"
    assert paired["observable_reserve"]["trigger"]["normalized_lambda_min"] == 0.006
    assert paired["observable_reserve"]["mode"] == "off"
    _, oracle = load_config("replica_room0_final_target_oracle")
    assert oracle["local_submap"]["projection_pre_cap"] == 80000
    assert oracle["local_submap"]["oracle_mode"] == "off"
    assert oracle["final_target_oracle"]["enabled"] is True
    _, corr = load_config("replica_room0_corr_weak128")
    assert corr["observable_reserve"]["selection"] == "runtime_corr_weak"
    assert corr["observable_reserve"]["budget"] == 128
    assert corr["observable_reserve"]["correspondence"] == corr["paired_reserve_evaluation"]["correspondence"]
    assert corr["observable_reserve"]["diversity"] == corr["paired_reserve_evaluation"]["diversity"]


def test_stability_and_office0_thin_configs_preserve_profiles_and_matrix():
    _, report = load_config("replica_room0_reserve_stability_report")
    _, enforce = load_config("replica_room0_reserve_stability_enforce")
    _, corr = load_config("replica_room0_corr_weak128")
    for mode, config in (("report_only", report), ("enforce", enforce)):
        assert config["observable_reserve"]["selection"] == "runtime_corr_weak"
        gate = config["observable_reserve"]["stability_gate"]
        assert gate == {
            "enabled": True, "mode": mode,
            "max_refinement_translation_m": 0.012, "max_refinement_rotation_deg": 2.0,
            "rolling_window_frames": 5, "max_rolling_translation_l1_m": 0.05,
            "max_consecutive_adoptions": 8, "cooldown_frames": 3,
            "common_target": {"max_cost_ratio": 1.0, "max_rmse_ratio": 1.0, "min_corr_retention": 0.95},
        }
        normalized = deepcopy(config)
        normalized["observable_reserve"]["stability_gate"] = deepcopy(
            corr["observable_reserve"]["stability_gate"]
        )
        assert normalized == corr
    report_without_mode = deepcopy(report)
    enforce_without_mode = deepcopy(enforce)
    report_without_mode["observable_reserve"]["stability_gate"].pop("mode")
    enforce_without_mode["observable_reserve"]["stability_gate"].pop("mode")
    assert report_without_mode == enforce_without_mode
    _, room = load_config("replica_room0_paired_core_route")
    _, office = load_config("replica_office0_paired_core_route")
    assert office["dataset"]["sequence"] == "office0"
    assert office["dataset"]["dataset_name"] == "replica"
    assert office["paired_reserve_evaluation"] == room["paired_reserve_evaluation"]
    assert office["paired_reserve_evaluation"]["trigger"] == {
        "condition_number": 120.0,
        "normalized_lambda_min": 0.006,
        "effective_rank_below": 6,
    }
    assert office["registration"] == room["registration"]
    assert office["local_submap"] == room["local_submap"]
    _, office_corr = load_config("replica_office0_corr_weak128")
    assert office_corr["dataset"]["sequence"] == "office0"
    assert office_corr["observable_reserve"]["selection"] == "runtime_corr_weak"
    assert office_corr["observable_reserve"]["budget"] == 128


def test_corr_weak128_constructs_real_runtime_reserve_without_selection_error(tmp_path):
    from utils.fsm_local_map import LocalSubmapConfig
    from utils.fsm_runtime_reserve import RuntimeObservableReserve

    _, config = load_config("replica_room0_corr_weak128")
    reserve = RuntimeObservableReserve(
        config,
        {"source_params_sha256": "a" * 64, "selection_controls_sha256": "b" * 64},
        str(tmp_path / "fsm_map.npz"),
        LocalSubmapConfig(near=0.1, far=5.0),
        "mvp_strict",
        "numpy_robust_point_to_plane",
    )
    assert reserve.selection == "runtime_corr_weak"
    assert reserve.unavailable_reason == "activation_geometry_unavailable"


def test_core_route_accepts_runtime_corr_weak_without_helper_bypass():
    common = SourceFileLoader(
        "test_mvp_observable_common_direct", str(CONFIG_DIR / "common.py")
    ).load_module()
    direct = common.make_replica_core_route("room0", "runtime_corr_weak", 128)
    helper = common.make_replica_room0_corr_weak128()
    assert direct == helper
    assert direct["observable_reserve"]["selection"] == "runtime_corr_weak"


def test_active_reserve_selection_ablation_only_changes_selection():
    names = (
        "replica_room0_reserve_active_runtime_weak",
        "replica_room0_reserve_active_saliency",
        "replica_room0_reserve_active_random",
    )
    configs = [load_config(name)[1] for name in names]
    assert [cfg["observable_reserve"]["selection"] for cfg in configs] == ["runtime_weak", "saliency", "random"]
    normalized = []
    for cfg in configs:
        copied = {key: value for key, value in cfg["observable_reserve"].items() if key != "selection"}
        normalized.append(copied)
    assert normalized[0] == normalized[1] == normalized[2]


def test_core_route_matrix_has_exact_budgets_seeds_and_only_intended_diffs():
    deterministic = {
        (selection, budget): load_config(f"replica_room0_core_route_{name}{budget}")[1]
        for selection, name in (("runtime_weak", "weak"), ("saliency", "saliency"))
        for budget in (64, 128, 256)
    }
    for (selection, budget), config in deterministic.items():
        reserve = config["observable_reserve"]
        assert reserve["selection"] == selection
        assert reserve["budget"] == reserve["max_points"] == budget
        assert reserve["seed"] == 0
        assert reserve["mode"] == "active"

    random_configs = {
        (budget, seed): load_config(f"replica_room0_core_route_random{budget}_seed{seed}")[1]
        for budget in (64, 128, 256)
        for seed in range(5)
    }
    assert len(random_configs) == 15
    for (budget, seed), config in random_configs.items():
        reserve = config["observable_reserve"]
        assert reserve["selection"] == "random"
        assert reserve["budget"] == reserve["max_points"] == budget
        assert reserve["seed"] == seed

    baseline = random_configs[(128, 0)]
    for (budget, seed), config in random_configs.items():
        differing = {
            key for key in baseline["observable_reserve"]
            if baseline["observable_reserve"][key] != config["observable_reserve"][key]
        }
        expected = set()
        if budget != 128:
            expected.update(("budget", "max_points"))
        if seed != 0:
            expected.add("seed")
        assert differing == expected
        assert {key: value for key, value in baseline.items() if key != "observable_reserve"} == {
            key: value for key, value in config.items() if key != "observable_reserve"
        }


def test_information_adoption_v2_configs_are_exact_and_fair_while_v1_is_unchanged():
    _, v1 = load_config("replica_room0_corr_weak128")
    assert v1["observable_reserve"]["accept_only_if"] == {"condition_ratio": 0.95}
    v2_names = [
        "replica_room0_corr_weak128_info_v2",
        "replica_room0_weak128_info_v2",
        "replica_room0_saliency128_info_v2",
        *[f"replica_room0_random128_info_v2_seed{seed}" for seed in range(5)],
        "replica_office0_corr_weak128_info_v2",
        "replica_office0_weak128_info_v2",
        "replica_office0_saliency128_info_v2",
        *[f"replica_office0_random128_info_v2_seed{seed}" for seed in range(5)],
    ]
    configs = [load_config(name)[1] for name in v2_names]
    expected_acceptance = {
        "protocol": "condition_or_lambda_v2", "mode": "any",
        "condition_ratio": 0.98, "lambda_ratio": 1.02,
    }
    for config in configs:
        reserve = config["observable_reserve"]
        assert reserve["accept_only_if"] == expected_acceptance
        assert reserve["stability_gate"]["enabled"] is False
        assert reserve["stability_gate"]["mode"] == "off"
        assert reserve["budget"] == reserve["max_points"] == 128
    for scene in ("room0", "office0"):
        scene_configs = [config for config in configs if config["dataset"]["sequence"] == scene]
        normalized = []
        for config in scene_configs:
            copied = deepcopy(config)
            for key in ("selection", "seed"):
                copied["observable_reserve"].pop(key)
            normalized.append(copied)
        assert all(config == normalized[0] for config in normalized[1:])


def test_information_adoption_v3_configs_are_exact_fair_and_stability_off():
    names = [
        f"replica_{scene}_{method}128_info_v3{suffix}"
        for scene in ("room0", "office0")
        for method, suffix in (
            ("corr_weak", ""), ("weak", ""), ("saliency", ""),
            *(("random", f"_seed{seed}") for seed in range(5)),
        )
    ]
    configs = [load_config(name)[1] for name in names]
    policies = [config["observable_reserve"]["accept_only_if"] for config in configs]
    assert all(policy == policies[0] for policy in policies[1:])
    assert policies[0]["protocol"] == "branch_aware_information_v3"
    assert policies[0]["branch_policy"]["both"]["max_consecutive_adoptions"] == 8
    assert policies[0]["branch_policy"]["condition_only"]["cooldown_frames"] == 1
    for config in configs:
        reserve = config["observable_reserve"]
        assert reserve["budget"] == reserve["max_points"] == 128
        assert reserve["stability_gate"]["enabled"] is False
        assert reserve["stability_gate"]["mode"] == "off"
    for scene in ("room0", "office0"):
        scene_configs = [config for config in configs if config["dataset"]["sequence"] == scene]
        normalized = []
        for config in scene_configs:
            copied = deepcopy(config)
            copied["observable_reserve"].pop("selection")
            copied["observable_reserve"].pop("seed")
            normalized.append(copied)
        assert all(config == normalized[0] for config in normalized[1:])


def test_information_adoption_v4_minimal_configs_share_exact_thresholds_and_preserve_v3():
    _, v3 = load_config("replica_room0_corr_weak128_info_v3")
    configs = [
        load_config(f"replica_{scene}_corr_weak128_info_v4")[1]
        for scene in ("room0", "office0")
    ]
    policies = [config["observable_reserve"]["accept_only_if"] for config in configs]
    assert policies[0] == policies[1]
    assert policies[0]["protocol"] == "branch_aware_information_v4"
    assert policies[0]["max_candidate_init_translation_m"] == 0.015
    assert policies[0]["max_candidate_init_rotation_deg"] == 0.50
    assert policies[0]["joint_refinement_rotation_deg"] == 0.20
    assert policies[0]["joint_temporal_rotation_deg"] == 0.70
    assert policies[0]["require_joint_metrics"] is True
    assert policies[0]["max_global_consecutive_adoptions"] == 8
    assert policies[0]["global_cooldown_frames"] == 2
    assert policies[0]["branch_policy"] == v3["observable_reserve"]["accept_only_if"]["branch_policy"]
    assert v3["observable_reserve"]["accept_only_if"]["protocol"] == "branch_aware_information_v3"
    for config in configs:
        reserve = config["observable_reserve"]
        assert reserve["selection"] == "runtime_corr_weak"
        assert reserve["budget"] == reserve["max_points"] == 128
        assert reserve["stability_gate"]["enabled"] is False


def test_room12_generalization_configs_reuse_frozen_paired_and_v4_contracts():
    _, room0_paired = load_config("replica_room0_paired_core_route")
    _, room0_v4 = load_config("replica_room0_corr_weak128_info_v4")
    for scene in ("room1", "room2"):
        _, paired = load_config(f"replica_{scene}_paired_core_route")
        _, v4 = load_config(f"replica_{scene}_corr_weak128_info_v4")
        assert paired["dataset"]["sequence"] == scene
        assert paired["paired_reserve_evaluation"] == room0_paired["paired_reserve_evaluation"]
        assert paired["observable_reserve"]["mode"] == "off"
        assert v4["dataset"]["sequence"] == scene
        assert v4["observable_reserve"]["selection"] == "runtime_corr_weak"
        assert v4["observable_reserve"]["budget"] == 128
        assert v4["observable_reserve"]["accept_only_if"] == room0_v4["observable_reserve"]["accept_only_if"]


def test_tum_core_configs_reuse_frozen_paired_and_v4_contracts():
    _, room0_paired = load_config("replica_room0_paired_core_route")
    _, room0_v4 = load_config("replica_room0_corr_weak128_info_v4")
    for scene in (
        "freiburg1_desk",
        "freiburg2_xyz",
        "freiburg3_long_office_household",
    ):
        _, paired = load_config(f"tum_{scene}_paired_core_route")
        _, v4 = load_config(f"tum_{scene}_corr_weak128_info_v4")
        assert paired["dataset"]["sequence"] == f"rgbd_dataset_{scene}"
        assert paired["paired_reserve_evaluation"] == room0_paired["paired_reserve_evaluation"]
        assert paired["observable_reserve"]["mode"] == "off"
        assert v4["dataset"]["sequence"] == f"rgbd_dataset_{scene}"
        assert v4["observable_reserve"]["selection"] == "runtime_corr_weak"
        assert v4["observable_reserve"]["budget"] == 128
        assert v4["observable_reserve"]["accept_only_if"] == room0_v4["observable_reserve"]["accept_only_if"]


def test_tum_fps_trigger_ablation_only_changes_trigger_thresholds():
    _, frozen = load_config("tum_freiburg3_long_office_household_corr_weak128_info_v4")
    for name, condition, normalized in (
        ("tum_freiburg3_long_office_household_corr_weak128_info_v4_gate_mid", 2000.0, 0.0003),
        ("tum_freiburg3_long_office_household_corr_weak128_info_v4_gate_strict", 3000.0, 0.0003),
    ):
        _, candidate = load_config(name)
        assert candidate["observable_reserve"]["trigger"]["condition_number"] == condition
        assert candidate["observable_reserve"]["trigger"]["normalized_lambda_min"] == normalized
        left = dict(candidate["observable_reserve"])
        right = dict(frozen["observable_reserve"])
        left.pop("trigger")
        right.pop("trigger")
        left.pop("trigger_ablation")
        assert left == right
