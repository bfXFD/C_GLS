from importlib.machinery import SourceFileLoader
from pathlib import Path

common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_tum_core_route_info_v4("freiburg1_desk2", "runtime_corr_weak", 128)
config["data"]["pose_reference_sequence"] = "rgbd_dataset_freiburg1_desk"
config["independent_revisit"] = {"map_sequence": "rgbd_dataset_freiburg1_desk", "query_sequence": "rgbd_dataset_freiburg1_desk2", "map_construction_query_frame_count": 0, "map_role": "fps10_with_reserve"}
