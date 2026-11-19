from importlib.machinery import SourceFileLoader
from pathlib import Path

common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_replica_core_route_info_v4("office3", "runtime_corr_weak", 128)
