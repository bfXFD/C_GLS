from importlib.machinery import SourceFileLoader
from pathlib import Path


common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_tum_paired_core_route("freiburg3_long_office_household")
