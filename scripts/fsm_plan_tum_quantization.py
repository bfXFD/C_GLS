import argparse
from pathlib import Path


DEFAULT_TUM_SCENES = ["freiburg1_desk", "freiburg2_xyz"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TUM safe quantization localization ablation commands.")
    parser.add_argument("--output", default="/home/ras/xbf/C_GLS_code/experiments/FSM/scripts/run_tum_quantization_v1_2.sh")
    parser.add_argument("--source-experiment", default="batch_500_tum_pareto_v1_1")
    parser.add_argument("--output-experiment", default="quantization_v1_2")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_TUM_SCENES)
    parser.add_argument("--methods", nargs="+", default=["fps_scale_opacity_ratio_0p1"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--conda-env", default="splatam")
    return parser.parse_args()


def _config_for_scene(scene: str) -> str:
    return f"/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_{scene}_fsm.py"


def build_tum_quantization_commands(
    scenes: list[str],
    methods: list[str],
    source_experiment: str,
    output_experiment: str,
    start: int,
    end: int,
    conda_env: str,
) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    for scene in scenes:
        config = _config_for_scene(scene)
        for method in methods:
            input_map = f"/home/ras/xbf/C_GLS_code/experiments/FSM/tum_{scene}/{source_experiment}/maps/{method}/fsm_map.npz"
            output_root = f"/home/ras/xbf/C_GLS_code/experiments/FSM/tum_{scene}/{output_experiment}/{method}_fp16_drop_scores"
            output_map = f"{output_root}/fsm_map.npz"
            localize_dir = f"{output_root}/localize"
            commands.extend(
                [
                    f"# Safe quantization ablation: TUM {scene} {method}",
                    f"test -f {input_map}",
                    f"test -f {config}",
                    f"mkdir -p {output_root}",
                    f"conda run -n {conda_env} python scripts/fsm_quantize_map.py \\",
                    f"  --map {input_map} \\",
                    f"  --output {output_map} \\",
                    "  --float-dtype float16 --fields means --drop-scores",
                    f"conda run -n {conda_env} python scripts/fsm_localize.py \\",
                    f"  --config {config} \\",
                    f"  --map {output_map} \\",
                    f"  --output-dir {localize_dir} \\",
                    f"  --start {int(start)} --end {int(end)}",
                    "",
                ]
            )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_tum_quantization_commands(
        scenes=args.scenes,
        methods=args.methods,
        source_experiment=args.source_experiment,
        output_experiment=args.output_experiment,
        start=args.start,
        end=args.end,
        conda_env=args.conda_env,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(commands) + "\n")
    print(f"Wrote TUM quantization script to {output_path}")


if __name__ == "__main__":
    main()
