from importlib.machinery import SourceFileLoader
from pathlib import Path


common = SourceFileLoader("mvp_observable_common", str(Path(__file__).with_name("common.py"))).load_module()
config = common.make_tum_core_route_info_v4("freiburg3_long_office_household", "runtime_corr_weak", 128)
config["observable_reserve"]["trigger"] = dict(
    condition_number=3000.0,
    normalized_lambda_min=0.0003,
    combine="any",
)
config["observable_reserve"]["trigger_ablation"] = "fps_hybrid_gate_strict_v1"
