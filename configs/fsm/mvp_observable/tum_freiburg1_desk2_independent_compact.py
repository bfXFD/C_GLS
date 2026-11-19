from importlib.machinery import SourceFileLoader
from pathlib import Path


base = SourceFileLoader(
    "desk2_independent_full", str(Path(__file__).with_name("tum_freiburg1_desk2_independent_full.py"))
).load_module()
config = base.config
config["independent_revisit"]["map_role"] = "fps_scale_opacity_ratio_0p1"
