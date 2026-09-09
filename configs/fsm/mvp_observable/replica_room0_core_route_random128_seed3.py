from importlib.machinery import SourceFileLoader
from pathlib import Path

common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_replica_room0_core_route("random", 128, seed=3)
