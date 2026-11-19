"""Controlled CN study; verifier thresholds are global research defaults, not calibrated."""

import os
from pathlib import Path
import runpy


ARMS = (
    "baseline", "rerun_frozen", "rerun_pca", "corr_frozen", "corr_pca",
    "saliency_pca", *(f"random_pca_s{seed}" for seed in range(5)),
)
SCENES = {
    "replica": ("room2", "office0", "office1", "office2", "office3", "office4"),
    "tum": ("freiburg3_long_office_household",),
}
VERIFIER = dict(
    enabled=True, block_size=16, folds=4, fold=0,
    relative_tolerance=0.0, absolute_tolerance=1e-6,
    euclidean_relative_tolerance=0.0, euclidean_absolute_tolerance=1e-6,
    min_coverage_drop=0.0, max_group_cost_increase=1e-5, min_group_size=8,
    max_translation_m=0.012, max_rotation_deg=0.3,
)


def make_config(dataset, scene, arm):
    if dataset not in SCENES or scene not in SCENES[dataset]:
        raise ValueError(f"Unsupported CN dataset/scene: {dataset}:{scene}")
    if arm not in ARMS:
        raise ValueError(f"Unsupported CN arm: {arm}")
    common = runpy.run_path(str(Path(__file__).with_name("common.py")))
    rerun = arm.startswith("rerun_")
    seed = int(arm[-1]) if arm.startswith("random_pca_s") else 0
    selection = (
        "random" if arm.startswith("random_") else
        "saliency" if arm == "saliency_pca" else "runtime_corr_weak"
    )
    normal_mode = "frozen_concat" if arm.endswith("frozen") or arm == "baseline" else "augmented_local_pca"
    config = common["make_reviewer_ablation"](
        dataset, scene, selection=selection, budget=128,
        target_mode="rerun_primary" if rerun else "append",
        normal_mode=normal_mode, use_v4=False, replacement_seed=0,
    )
    config["seed"] = 0
    config["observable_reserve"].update(
        enabled=arm != "baseline", mode="off" if arm == "baseline" else "active",
        seed=seed, budget=128, max_points=128, rerun_iterations=15,
        accept_only_if={}, spectral_coordinates="primary_camera_depth",
        candidate_verifier=dict(VERIFIER), attempt_without_selection=rerun,
    )
    config["cn_consistency"] = dict(
        dataset=dataset, scene=scene, arm=arm, source_seed=0, selection_seed=seed,
        selection=selection, budget=128, rerun_iterations=15,
        target_mode="rerun_primary" if rerun else "append", normal_mode=normal_mode,
        verifier_threshold_status="explicit_global_research_defaults_not_calibrated",
        research_only=True, safety_guarantee=False,
    )
    return config


config = make_config(
    os.environ.get("CGLS_CN_DATASET", ""),
    os.environ.get("CGLS_CN_SCENE", ""),
    os.environ.get("CGLS_CN_ARM", ""),
)
