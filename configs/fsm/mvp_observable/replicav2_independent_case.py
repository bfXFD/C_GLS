import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
scene = os.environ["CGLS_REPLICAV2_SCENE"]
method = os.environ.get("CGLS_REPLICAV2_METHOD", "compact")
config = common.make_replica_config("room0")
config["data"].update(
    dataset_name="replicav2",
    basedir="/home/ras/xbf/C_GLS_code/data/Replica_V2",
    gradslam_data_cfg="/home/ras/xbf/C_GLS_code/configs/data/replica_v2.yaml",
    sequence=scene,
    use_train_split=False,
    query_only_map_frame=True,
    relative_pose=False,
    png_depth_scale=1000.0,
)
config["independent_revisit"] = {"scene": scene, "map_split": "imap/00", "query_split": "imap/01", "map_construction_query_frame_count": 0}
if method == "full" or method == "compact":
    config["observable_reserve"].update(enabled=False, mode="off")
elif method in ("corrweak", "replace"):
    config = common.configure_reviewer_ablation(config, target_mode="replace" if method == "replace" else "append", use_v4=True)
    config["observable_reserve"].update(enabled=True, mode="active", selection="runtime_corr_weak", budget=128)
else:
    raise ValueError(method)
