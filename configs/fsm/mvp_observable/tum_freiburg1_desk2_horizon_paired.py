from importlib.machinery import SourceFileLoader
from pathlib import Path

common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_tum_horizon_paired("freiburg1_desk")
config["data"].update(
    sequence="rgbd_dataset_freiburg1_desk",
    pose_reference_sequence="rgbd_dataset_freiburg1_desk2",
)
config["independent_revisit"] = {
    "map_sequence": "rgbd_dataset_freiburg1_desk2",
    "query_sequence": "rgbd_dataset_freiburg1_desk",
    "map_construction_query_frame_count": 0,
    "pose_frame": "inv(P_map0_abs) @ P_query_abs",
    "arm": "Corr-Weak128 paired shadow",
}
