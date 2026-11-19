from importlib.machinery import SourceFileLoader
from pathlib import Path


common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_tum_core_route_info_v4("freiburg1_desk", "runtime_corr_weak", 128)
config["data"].update(
    sequence="rgbd_dataset_freiburg1_desk",
    pose_reference_sequence="rgbd_dataset_freiburg1_desk2",
)
config["independent_revisit"] = {
    "map_sequence": "rgbd_dataset_freiburg1_desk2",
    "query_sequence": "rgbd_dataset_freiburg1_desk",
    "map_role": "fps_scale_opacity_ratio_0p1_with_map_side_reserve",
    "map_construction_query_frame_count": 0,
    "pose_frame": "inv(P_map0_abs) @ P_query_abs",
    "arm": "Corr-Weak128 V4",
}
