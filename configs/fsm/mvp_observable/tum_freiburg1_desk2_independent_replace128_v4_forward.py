from importlib.machinery import SourceFileLoader
from pathlib import Path

base = SourceFileLoader("forward_corr", str(Path(__file__).with_name("tum_freiburg1_desk2_independent_corr_weak128_v4_forward.py"))).load_module()
config = base.config
config["observable_reserve"]["candidate_target_mode"] = "replace"
config["observable_reserve"]["replacement_seed"] = 0
config["independent_revisit"]["map_role"] = "fps10_count_matched_replace"
