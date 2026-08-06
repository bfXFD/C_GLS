import argparse
from pathlib import Path


DEFAULT_SCENES = {
    "room0_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/replica_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz",
    },
    "room1_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_room1_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/room1_0/params.npz",
    },
    "room2_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_room2_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/room2_0/params.npz",
    },
    "office0_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_office0_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/office0_0/params.npz",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Write shell commands for FSM Pareto localization experiments.")
    parser.add_argument("--output", required=True, help="Output shell script path.")
    parser.add_argument("--experiment-name", default="batch_500_pareto_v1_1", help="Per-scene experiment directory name.")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES.keys()), choices=sorted(DEFAULT_SCENES.keys()))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"],
    )
    parser.add_argument("--ratios", nargs="+", default=["1", "0.25", "0.2", "0.15", "0.1", "0.05"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--conda-env", default="splatam")
    return parser.parse_args()


def build_commands(
    scenes: list[str],
    strategies: list[str],
    ratios: list[str],
    experiment_name: str,
    start: int,
    end: int,
    conda_env: str,
) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    strategy_args = " ".join(strategies)
    ratio_args = " ".join(ratios)
    for scene in scenes:
        scene_cfg = DEFAULT_SCENES[scene]
        output_dir = f"/home/ras/xbf/C_GLS_code/experiments/FSM/replica_{scene}/{experiment_name}"
        commands.extend(
            [
                f"# {scene}",
                f"conda run -n {conda_env} python scripts/fsm_eval.py \\",
                f"  --config {scene_cfg['config']} \\",
                f"  --params {scene_cfg['params']} \\",
                f"  --output-dir {output_dir} \\",
                f"  --strategies {strategy_args} \\",
                f"  --compression-ratios {ratio_args} \\",
                f"  --start {int(start)} --end {int(end)} \\",
                "  --skip-existing",
                "",
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(
        scenes=args.scenes,
        strategies=args.strategies,
        ratios=args.ratios,
        experiment_name=args.experiment_name,
        start=args.start,
        end=args.end,
        conda_env=args.conda_env,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(commands) + "\n")
    print(f"Wrote {len(args.scenes)} FSM Pareto commands to {output_path}")


if __name__ == "__main__":
    main()
