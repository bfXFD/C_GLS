from importlib.machinery import SourceFileLoader
from pathlib import Path


base = SourceFileLoader(
    "desk2_independent_corr", str(Path(__file__).with_name("tum_freiburg1_desk2_independent_corr_weak128_v4.py"))
).load_module()
config = base.config
config["observable_reserve"].update(candidate_target_mode="replace", replacement_seed=0)
config["independent_revisit"]["arm"] = "Replace128 V4"
