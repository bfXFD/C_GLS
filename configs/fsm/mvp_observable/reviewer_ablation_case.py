import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path


common = SourceFileLoader(
    "mvp_observable_common", str(Path(__file__).with_name("common.py"))
).load_module()


def _required(name):
    value = os.environ.get(name)
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    return value


dataset = _required("CGLS_REVIEW_DATASET")
scene = _required("CGLS_REVIEW_SCENE")
method = os.environ.get("CGLS_REVIEW_METHOD", "corrweak")
budget = int(os.environ.get("CGLS_REVIEW_BUDGET", "128"))
seed = int(os.environ.get("CGLS_REVIEW_SEED", "0"))
target_mode = os.environ.get("CGLS_REVIEW_TARGET_MODE", "append")
normal_mode = os.environ.get("CGLS_REVIEW_NORMAL_MODE", "augmented_local_pca")
use_v4 = os.environ.get("CGLS_REVIEW_USE_V4", "1") == "1"
twist = json.loads(os.environ.get("CGLS_REVIEW_FIRST_POSE_TWIST", "null"))

if method == "baseline":
    config = common.make_replica_config(scene) if dataset == "replica" else common.make_tum_config(scene)
    if twist is not None:
        config["registration"]["first_pose_left_twist"] = [float(value) for value in twist]
elif method in ("corrweak", "weak", "saliency", "random"):
    selection = {
        "corrweak": "runtime_corr_weak",
        "weak": "runtime_weak",
        "saliency": "saliency",
        "random": "random",
    }[method]
    config = common.make_reviewer_ablation(
        dataset,
        scene,
        selection=selection,
        budget=budget,
        target_mode=target_mode,
        normal_mode=normal_mode,
        use_v4=use_v4,
        replacement_seed=seed,
        first_pose_twist=twist,
    )
    config["observable_reserve"]["seed"] = seed
else:
    raise ValueError(f"Unknown CGLS_REVIEW_METHOD {method!r}")

config["reviewer_ablation"] = {
    "dataset": dataset,
    "scene": scene,
    "method": method,
    "budget": budget,
    "seed": seed,
    "target_mode": target_mode,
    "normal_mode": normal_mode,
    "use_v4": use_v4,
    "first_pose_left_twist": twist,
}
