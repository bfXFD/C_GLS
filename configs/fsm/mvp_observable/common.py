from copy import deepcopy


REPLICA_BASEDIR = "/home/ras/xbf/SplaTAM/data/Replica"
REPLICA_DATA_CONFIG = "/home/ras/xbf/C_GLS_code/configs/data/replica.yaml"
TUM_BASEDIR = "/home/ras/xbf/C_GLS_code/data/TUM_RGBD"
TUM_DATA_CONFIG_DIR = "/home/ras/xbf/C_GLS_code/configs/data/TUM"


def _corr_weak_defaults():
    return dict(
        corr_weak_min_dim=1,
        corr_weak_max_dim=3,
        correspondence=dict(
            k=4,
            distance_sigma=0.05,
            dmax=0.15,
            depth_sigma_front=0.10,
            depth_sigma_back=0.03,
            missing_depth_prior=0.35,
            normal_missing_prior=0.5,
            normal_power=2.0,
            competition_sigma=0.02,
        ),
        diversity=dict(
            tile_rows=12,
            tile_cols=16,
            depth_bins=8,
            tile_cap=None,
            depth_cap=None,
            voxel_size=0.25,
            voxel_cap=1,
        ),
    )


def _observable_trigger():
    return dict(
        condition_number=120.0,
        normalized_lambda_min=0.006,
        effective_rank_below=6,
    )


def _strict_config(dataset, depth_max, profile):
    dataset = deepcopy(dataset)
    dataset.update(
        start=0,
        end=-1,
        stride=1,
        depth_min=0.0,
        depth_max=float(depth_max),
        max_points=10000,
    )
    return dict(
        seed=0,
        primary_device="cuda:0",
        data=dataset,
        dataset=dataset,
        registration=dict(
            backend="numpy_robust_point_to_plane",
            evaluation_protocol="mvp_strict",
            first_pose_protocol="gt_first_frame",
            motion_model="constant_velocity",
            voxel_size=None,
            max_correspondence_distance=0.07,
            max_iterations=30,
            normal_radius=0.10,
            normal_max_nn=30,
            normal_source="local_pca",
            robust_loss="huber",
            loss_scale=0.02,
            lm_damping=0.001,
            min_correspondences=6,
            rank_tolerance=1e-9,
            rotation_step_tolerance_deg=0.01,
            translation_step_tolerance_m=1e-4,
            relative_cost_tolerance=1e-5,
            convergence_patience=2,
            min_iterations=2,
            fallback_on_divergence=False,
            max_pose_translation_norm=100.0,
            prediction_max_translation_m=profile["prediction_translation_m"],
            prediction_max_rotation_deg=profile["prediction_rotation_deg"],
            prediction_max_horizon_frames=3,
            pose_consistency=dict(
                enabled=True,
                correction=dict(
                    constant_velocity=dict(
                        translation_m=profile["correction_cv_translation_m"],
                        rotation_deg=profile["correction_cv_rotation_deg"],
                    ),
                    bootstrap=dict(
                        translation_m=profile["correction_bootstrap_translation_m"],
                        rotation_deg=profile["correction_bootstrap_rotation_deg"],
                    ),
                ),
                temporal=dict(
                    translation_m_per_frame=profile["temporal_translation_m_per_frame"],
                    rotation_deg_per_frame=profile["temporal_rotation_deg_per_frame"],
                    hard_translation_m=profile["temporal_hard_translation_m"],
                    hard_rotation_deg=profile["temporal_hard_rotation_deg"],
                ),
            ),
            tracking_acceptance=dict(
                min_fitness=0.05,
                min_correspondences=64,
                max_point_to_plane_rmse=0.05,
                max_condition_number=1e8,
                min_lambda_min=1e-8,
            ),
        ),
        motion_model=dict(
            type="constant_velocity",
            max_translation_m=1.0,
            max_rotation_deg=45.0,
        ),
        local_submap=dict(
            enabled=True,
            near=0.05,
            far=float(depth_max),
            margin_px=24.0,
            radius_margin=0.10,
            min_points=64,
            max_points=10000,
            sampling_policy="coverage",
            tile_rows=12,
            tile_cols=16,
            depth_bins=8,
            frustum_culling="radius",
            block_size_m=0.3,
            projection_pre_cap=None,
            projection_sampling_policy="block_coverage",
            oracle_mode="off",
            oracle_overflow_policy="uncapped_chunked",
            spatial_backend="scipy_ckdtree",
            grid_voxel=0.5,
        ),
        activation_diagnostics=dict(
            enabled=False,
            geometry_path=None,
            include_stage_ids=False,
            counterfactual=True,
            save_correspondences=False,
            trigger=dict(
                mode="plan_only",
                condition_number=120.0,
            ),
            reserve_budget=[1000, 2000, 4000],
            weak_subspace_dim=1,
        ),
        runtime_memory_metrics=dict(
            enabled=True,
            check_interval_frames=10,
            max_rss_mb=20000,
        ),
        observable_reserve=dict(
            enabled=False,
            geometry_path="auto",
            mode="off",
            selection_artifact=dict(enabled=True, flush_each_frame=True),
            trigger_protocol="absolute_or_scale_normalized_v1",
            trigger={
                **_observable_trigger(),
                "iterations": None,
                "fitness": None,
                "degenerate": False,
            },
            budget=1000,
            max_points=1000,
            max_visible=10000,
            weak_subspace_dim=1,
            selection="runtime_weak",
            seed=0,
            tile_rows=12,
            tile_cols=16,
            depth_bins=8,
            tile_cap=None,
            depth_cap=None,
            voxel_size=0.25,
            voxel_cap=1,
            rerun_iterations=15,
            accept_only_if=dict(condition_ratio=0.95),
            relative_unavailable_policy="reject",
            stability_gate=dict(
                enabled=False,
                mode="off",
                max_refinement_translation_m=0.012,
                max_refinement_rotation_deg=2.0,
                rolling_window_frames=5,
                max_rolling_translation_l1_m=0.05,
                max_consecutive_adoptions=8,
                cooldown_frames=3,
                common_target=dict(max_cost_ratio=1.0, max_rmse_ratio=1.0, min_corr_retention=0.95),
            ),
            **_corr_weak_defaults(),
        ),
        paired_reserve_evaluation=dict(
            enabled=False,
            mode="shadow",
            geometry_path="auto",
            policies=["runtime_weak", "runtime_corr_weak", "saliency", "random"],
            random_seeds=[0, 1, 2, 3, 4],
            budgets=[64, 128, 256],
            eligible_pool="visible",
            weak_subspace_dim=3,
            trigger_protocol="absolute_or_scale_normalized_v1",
            trigger=_observable_trigger(),
            registration=dict(
                backend="numpy_robust_point_to_plane",
                max_correspondence_distance=0.07,
                max_iterations=15,
                robust_loss="huber",
                loss_scale=0.02,
                min_correspondences=6,
                rank_tolerance=1e-9,
            ),
            **_corr_weak_defaults(),
        ),
        final_target_oracle=dict(
            enabled=False,
            diagnostic_only=True,
            chunk_size=262144,
            normal_k=20,
            max_correspondence_distance=0.07,
            min_correspondences=6,
            robust_loss="huber",
            loss_scale=0.02,
            rank_tolerance=1e-9,
            weak_subspace_dim=1,
        ),
        recovery=dict(
            enabled=True,
            max_tier=3,
            margin_multiplier=2.0,
            radius_multiplier=2.0,
            far_multiplier=1.5,
            sphere_radius_m=float(depth_max),
            max_points=10000,
            verifier_backend="numpy_point_to_point",
            correction_limit_multiplier=1.5,
            verifier_acceptance=dict(
                min_fitness=0.05,
                min_correspondences=64,
                max_inlier_rmse=0.07,
            ),
        ),
        eval=dict(
            align_ate=True,
            align_with_scale=False,
            rpe_delta=1,
            save_tum_format=True,
            frame_deltas=[1, 10, 30, 100],
            distance_deltas_m=[1.0, 5.0, 10.0],
            segment_lengths_m=[5.0, 10.0, 20.0],
            return_to_start_radius_m=0.5,
        ),
    )


def make_replica_config(sequence):
    return _strict_config(
        dict(
            dataset_name="replica",
            basedir=REPLICA_BASEDIR,
            gradslam_data_cfg=REPLICA_DATA_CONFIG,
            sequence=sequence,
            desired_image_height=680,
            desired_image_width=1200,
        ),
        depth_max=6.0,
        profile=dict(
            prediction_translation_m=0.15,
            prediction_rotation_deg=8.0,
            correction_cv_translation_m=0.10,
            correction_cv_rotation_deg=4.0,
            correction_bootstrap_translation_m=0.15,
            correction_bootstrap_rotation_deg=6.0,
            temporal_translation_m_per_frame=0.15,
            temporal_rotation_deg_per_frame=8.0,
            temporal_hard_translation_m=0.45,
            temporal_hard_rotation_deg=20.0,
        ),
    )


def make_tum_config(scene):
    return _strict_config(
        dict(
            dataset_name="tum",
            basedir=TUM_BASEDIR,
            gradslam_data_cfg=f"{TUM_DATA_CONFIG_DIR}/{scene}.yaml",
            sequence=f"rgbd_dataset_{scene}",
            desired_image_height=480,
            desired_image_width=640,
        ),
        depth_max=4.0,
        profile=dict(
            prediction_translation_m=0.25,
            prediction_rotation_deg=15.0,
            correction_cv_translation_m=0.15,
            correction_cv_rotation_deg=8.0,
            correction_bootstrap_translation_m=0.25,
            correction_bootstrap_rotation_deg=12.0,
            temporal_translation_m_per_frame=0.25,
            temporal_rotation_deg_per_frame=15.0,
            temporal_hard_translation_m=0.60,
            temporal_hard_rotation_deg=30.0,
        ),
    )


def make_tum_core_route(scene, selection, budget, seed=0, mode="active"):
    if selection not in ("runtime_weak", "runtime_corr_weak", "saliency", "random"):
        raise ValueError(f"Unknown reserve selection {selection!r}")
    if mode not in ("plan_only", "active"):
        raise ValueError(f"Unknown reserve mode {mode!r}")
    budget = int(budget)
    if budget not in (32, 64, 128, 256):
        raise ValueError("Core-route reserve budget must be 32, 64, 128, or 256")
    config = make_tum_config(scene)
    config["observable_reserve"].update(
        enabled=True,
        mode=mode,
        selection=selection,
        budget=budget,
        max_points=budget,
        seed=int(seed),
    )
    return config


def make_tum_core_route_info_v4(scene, selection, budget, seed=0):
    return apply_information_adoption_v4(
        make_tum_core_route(scene, selection, budget, seed=seed)
    )


def make_tum_paired_core_route(scene, eligible_pool="visible"):
    if eligible_pool not in ("visible", "supported"):
        raise ValueError("eligible_pool must be visible or supported")
    config = make_tum_config(scene)
    config["observable_reserve"].update(enabled=False, mode="off")
    config["paired_reserve_evaluation"].update(enabled=True, eligible_pool=eligible_pool)
    return config


def restrict_paired_to_corrweak128(config):
    paired = config["paired_reserve_evaluation"]
    paired.update(
        policies=["runtime_corr_weak"],
        random_seeds=[0],
        budgets=[128],
    )
    return config


def make_tum_horizon_paired(scene, eligible_pool="visible"):
    return restrict_paired_to_corrweak128(make_tum_paired_core_route(scene, eligible_pool))


def make_replica_horizon_paired(sequence, eligible_pool="visible"):
    return restrict_paired_to_corrweak128(make_replica_paired_core_route(sequence, eligible_pool))


def make_replica_room0_ablation(kind):
    config = make_replica_config("room0")
    if kind == "block_oracle":
        config["local_submap"].update(
            frustum_culling="voxel_blocks", block_size_m=0.3,
            oracle_mode="measure",
        )
    elif kind in ("block_03_cap80k", "block_03_cap80k_oracle"):
        config["local_submap"].update(
            frustum_culling="voxel_blocks", block_size_m=0.3,
            projection_pre_cap=80000,
        )
        if kind == "block_03_cap80k_oracle":
            config["local_submap"]["oracle_mode"] = "measure"
    elif kind in ("reserve_plan", "reserve_active", "reserve_active_runtime_weak", "reserve_active_saliency", "reserve_active_random"):
        config["observable_reserve"].update(
            enabled=True,
            mode="plan_only" if kind == "reserve_plan" else "active",
        )
        if kind.startswith("reserve_active_"):
            config["observable_reserve"]["selection"] = kind.removeprefix("reserve_active_")
    else:
        raise ValueError(f"Unknown room0 ablation {kind!r}")
    return config


def make_replica_core_route(sequence, selection, budget, seed=0, mode="active"):
    if selection not in ("runtime_weak", "runtime_corr_weak", "saliency", "random"):
        raise ValueError(f"Unknown reserve selection {selection!r}")
    if mode not in ("plan_only", "active"):
        raise ValueError(f"Unknown reserve mode {mode!r}")
    budget = int(budget)
    if budget not in (32, 64, 128, 256):
        raise ValueError("Core-route reserve budget must be 32, 64, 128, or 256")
    seed = int(seed)
    config = make_replica_config(sequence)
    config["observable_reserve"].update(
        enabled=True,
        mode=mode,
        selection=selection,
        budget=budget,
        max_points=budget,
        seed=seed,
    )
    return config


def apply_information_adoption_v2(config):
    config["observable_reserve"]["accept_only_if"] = {
        "protocol": "condition_or_lambda_v2",
        "mode": "any",
        "condition_ratio": 0.98,
        "lambda_ratio": 1.02,
    }
    config["observable_reserve"]["stability_gate"].update(enabled=False, mode="off")
    return config


def make_replica_core_route_info_v2(sequence, selection, budget, seed=0):
    return apply_information_adoption_v2(
        make_replica_core_route(sequence, selection, budget, seed=seed)
    )


def apply_information_adoption_v3(config):
    config["observable_reserve"]["accept_only_if"] = {
        "protocol": "branch_aware_information_v3",
        "mode": "any",
        "condition_ratio": 0.98,
        "lambda_ratio": 1.02,
        "branch_policy": {
            "both": {
                "condition_ratio": 0.98,
                "lambda_ratio": 1.02,
                "max_refinement_translation_m": 0.012,
                "max_refinement_rotation_deg": 0.30,
                "max_consecutive_adoptions": 8,
                "cooldown_frames": 2,
            },
            "condition_only": {
                "condition_ratio": 0.97,
                "lambda_ratio": 0.995,
                "max_refinement_translation_m": 0.006,
                "max_refinement_rotation_deg": 0.15,
                "max_consecutive_adoptions": 2,
                "cooldown_frames": 1,
            },
            "lambda_only": {
                "condition_ratio": 0.995,
                "lambda_ratio": 1.03,
                "max_refinement_translation_m": 0.006,
                "max_refinement_rotation_deg": 0.15,
                "max_consecutive_adoptions": 2,
                "cooldown_frames": 1,
            },
        },
    }
    config["observable_reserve"]["stability_gate"].update(enabled=False, mode="off")
    return config


def make_replica_core_route_info_v3(sequence, selection, budget, seed=0):
    return apply_information_adoption_v3(
        make_replica_core_route(sequence, selection, budget, seed=seed)
    )


def apply_information_adoption_v4(config):
    apply_information_adoption_v3(config)
    config["observable_reserve"]["accept_only_if"].update(
        protocol="branch_aware_information_v4",
        max_candidate_init_translation_m=0.015,
        max_candidate_init_rotation_deg=0.50,
        joint_refinement_rotation_deg=0.20,
        joint_temporal_rotation_deg=0.70,
        require_joint_metrics=True,
        max_global_consecutive_adoptions=8,
        global_cooldown_frames=2,
    )
    return config


def make_replica_core_route_info_v4(sequence, selection, budget, seed=0):
    return apply_information_adoption_v4(
        make_replica_core_route(sequence, selection, budget, seed=seed)
    )


def configure_reviewer_ablation(
    config,
    *,
    target_mode="append",
    normal_mode="augmented_local_pca",
    use_v4=True,
    replacement_seed=0,
    first_pose_twist=None,
):
    if target_mode not in ("append", "replace", "rerun_primary"):
        raise ValueError("target_mode must be append, replace, or rerun_primary")
    if normal_mode not in ("augmented_local_pca", "frozen_concat"):
        raise ValueError("normal_mode must be augmented_local_pca or frozen_concat")
    reserve = config["observable_reserve"]
    reserve["candidate_target_mode"] = target_mode
    reserve["candidate_normal_mode"] = normal_mode
    reserve["replacement_seed"] = int(replacement_seed)
    if use_v4:
        apply_information_adoption_v4(config)
    else:
        reserve["accept_only_if"] = {}
        reserve["stability_gate"].update(enabled=False, mode="off")
    if first_pose_twist is not None:
        values = [float(value) for value in first_pose_twist]
        if len(values) != 6:
            raise ValueError("first_pose_twist must contain [rx, ry, rz, tx, ty, tz]")
        config["registration"]["first_pose_left_twist"] = values
    return config


def make_reviewer_ablation(dataset, scene, selection="runtime_corr_weak", budget=128, **kwargs):
    if dataset == "replica":
        config = make_replica_core_route(scene, selection, budget)
    elif dataset == "tum":
        config = make_tum_core_route(scene, selection, budget)
    else:
        raise ValueError("dataset must be replica or tum")
    return configure_reviewer_ablation(config, **kwargs)


def make_replica_room0_core_route(selection, budget, seed=0, mode="active"):
    return make_replica_core_route("room0", selection, budget, seed=seed, mode=mode)


def make_replica_paired_core_route(sequence, eligible_pool="visible"):
    if eligible_pool not in ("visible", "supported"):
        raise ValueError("eligible_pool must be visible or supported")
    config = make_replica_config(sequence)
    config["observable_reserve"].update(enabled=False, mode="off")
    config["paired_reserve_evaluation"].update(enabled=True, eligible_pool=eligible_pool)
    return config


def make_replica_room0_paired_core_route(eligible_pool="visible"):
    return make_replica_paired_core_route("room0", eligible_pool)


def make_replica_stability_gate(mode):
    if mode not in ("report_only", "enforce"):
        raise ValueError("stability gate mode must be report_only or enforce")
    config = make_replica_core_route("room0", "runtime_corr_weak", 128)
    config["observable_reserve"]["stability_gate"].update(enabled=True, mode=mode)
    return config


def make_replica_room0_stability_gate_relaxed(mode="enforce"):
    if mode not in ("report_only", "enforce"):
        raise ValueError("stability gate mode must be report_only or enforce")
    config = make_replica_core_route("room0", "runtime_corr_weak", 128)
    config["observable_reserve"]["stability_gate"].update(
        enabled=True,
        mode=mode,
        max_refinement_translation_m=0.015,
        max_refinement_rotation_deg=2.0,
        rolling_window_frames=5,
        max_rolling_translation_l1_m=0.060,
        max_consecutive_adoptions=12,
        cooldown_frames=2,
        common_target=dict(max_cost_ratio=1.01, max_rmse_ratio=1.01, min_corr_retention=0.95),
    )
    return config


def make_replica_room0_final_target_oracle():
    config = make_replica_room0_ablation("block_03_cap80k")
    config["local_submap"]["oracle_mode"] = "off"
    config["final_target_oracle"]["enabled"] = True
    return config


def make_replica_room0_corr_weak128():
    return make_replica_core_route("room0", "runtime_corr_weak", 128)


def make_replica_office0_corr_weak128():
    return make_replica_core_route("office0", "runtime_corr_weak", 128)
