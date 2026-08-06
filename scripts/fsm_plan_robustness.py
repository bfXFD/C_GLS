import argparse
from pathlib import Path


DEFAULT_REPLICA_SCENES = {
    "room0_0": "/home/ras/xbf/C_GLS_code/configs/fsm/replica_fsm.py",
    "room1_0": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_room1_fsm.py",
    "room2_0": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_room2_fsm.py",
    "office0_0": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_office0_fsm.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lightweight FSM initialization robustness scripts.")
    parser.add_argument("--output", default="/home/ras/xbf/C_GLS_code/experiments/FSM/scripts/run_replica_init_robustness_v1_2.sh")
    parser.add_argument("--source-experiment", default="batch_500_pareto_v1_1")
    parser.add_argument("--output-experiment", default="init_robustness_v1_2")
    parser.add_argument("--scenes", nargs="+", default=["room0_0", "room1_0", "room2_0", "office0_0"])
    parser.add_argument("--methods", nargs="+", default=["full_ratio_1", "voxel_scale_opacity_ratio_0p2", "random_ratio_0p2"])
    parser.add_argument("--init-modes", nargs="+", default=["previous_estimate", "identity"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--conda-env", default="splatam")
    return parser.parse_args()


def _patched_config_path(output_experiment: str, scene: str, init_mode: str) -> str:
    return f"/home/ras/xbf/C_GLS_code/experiments/FSM/{output_experiment}/configs/{scene}_{init_mode}.py"


def build_robustness_commands(
    scenes: list[str],
    methods: list[str],
    init_modes: list[str],
    source_experiment: str,
    output_experiment: str,
    start: int,
    end: int,
    conda_env: str,
) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    for scene in scenes:
        if scene not in DEFAULT_REPLICA_SCENES:
            raise KeyError(f"Unsupported Replica scene: {scene}")
        base_config = DEFAULT_REPLICA_SCENES[scene]
        for init_mode in init_modes:
            patched_config = _patched_config_path(output_experiment, scene, init_mode)
            patched_config_dir = str(Path(patched_config).parent)
            commands.extend(
                [
                    f"# Build config for {scene} init_mode={init_mode}",
                    f"mkdir -p {patched_config_dir}",
                    f"python -c \"from pathlib import Path; p=Path('{base_config}'); text=p.read_text(); text=text.replace('init_mode=\\\"previous_estimate\\\"', 'init_mode=\\\"{init_mode}\\\"'); Path('{patched_config}').write_text(text)\"",
                    "",
                ]
            )
            for method in methods:
                map_path = f"/home/ras/xbf/C_GLS_code/experiments/FSM/replica_{scene}/{source_experiment}/maps/{method}/fsm_map.npz"
                output_dir = f"/home/ras/xbf/C_GLS_code/experiments/FSM/replica_{scene}/{output_experiment}/{method}_{init_mode}/localize"
                commands.extend(
                    [
                        f"# Robustness localization: {scene} {method} init={init_mode}",
                        f"test -f {map_path}",
                        f"conda run -n {conda_env} python scripts/fsm_localize.py \\",
                        f"  --config {patched_config} \\",
                        f"  --map {map_path} \\",
                        f"  --output-dir {output_dir} \\",
                        f"  --start {int(start)} --end {int(end)}",
                        "",
                    ]
                )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_robustness_commands(
        scenes=args.scenes,
        methods=args.methods,
        init_modes=args.init_modes,
        source_experiment=args.source_experiment,
        output_experiment=args.output_experiment,
        start=args.start,
        end=args.end,
        conda_env=args.conda_env,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(commands) + "\n")
    print(f"Wrote robustness script to {output_path}")


if __name__ == "__main__":
    main()
