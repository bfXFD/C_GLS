import argparse
from pathlib import Path


REPLICA_SCENES = {
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


TUM_SCENES = {
    "freiburg1_desk": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_freiburg1_desk_fsm.py",
        "params": "/home/ras/xbf/C_GLS_code/experiments/TUM/freiburg1_desk_seed0/params.npz",
    },
    "freiburg2_xyz": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_freiburg2_xyz_fsm.py",
        "params": "/home/ras/xbf/C_GLS_code/experiments/TUM/freiburg2_xyz_seed0/params.npz",
    },
    "freiburg3_long_office_household": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_freiburg3_long_office_household_fsm.py",
        "params": "/home/ras/xbf/C_GLS_code/experiments/TUM/freiburg3_long_office_household_seed0/params.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E1 LightGaussian proxy baseline run script.")
    parser.add_argument(
        "--output-script",
        default="/home/ras/xbf/C_GLS_code/experiments/FSM/scripts/run_lightgaussian_proxy_e1.sh",
        help="Shell script path to write.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ras/xbf/C_GLS_code/experiments/FSM/rendering_oriented_baselines_v1_0",
        help="Root directory for E1 outputs.",
    )
    parser.add_argument("--replica-scenes", nargs="+", default=["room0_0", "office0_0"], choices=sorted(REPLICA_SCENES))
    parser.add_argument(
        "--tum-scenes",
        nargs="+",
        default=["freiburg1_desk", "freiburg2_xyz", "freiburg3_long_office_household"],
        choices=sorted(TUM_SCENES),
    )
    parser.add_argument("--ratios", nargs="+", default=["0.2", "0.1", "0.05"], help="Retention ratios for external_scores.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--conda-env", default="splatam")
    parser.add_argument("--v-pow", type=float, default=0.1)
    parser.add_argument("--opacity-power", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="Forward --overwrite to fsm_eval.py map export.")
    return parser.parse_args()


def _append_scene_commands(
    commands: list[str],
    dataset_prefix: str,
    scene: str,
    scene_cfg: dict[str, str],
    output_root: str,
    ratios: list[str],
    start: int,
    end: int,
    conda_env: str,
    v_pow: float,
    opacity_power: float,
    overwrite: bool,
) -> None:
    output_dir = f"{output_root}/{dataset_prefix}_{scene}/lightgaussian_proxy_vpow_{str(v_pow).replace('.', 'p')}"
    score_path = f"{output_dir}/scores/lightgaussian_proxy_scores.npz"
    metadata_path = f"{output_dir}/scores/lightgaussian_proxy_scores.json"
    ratio_args = " ".join(ratios)
    final_eval_args = ["  --skip-existing"]
    if overwrite:
        final_eval_args.append("  --overwrite")

    commands.extend(
        [
            f"# {dataset_prefix} {scene}",
            f"test -f {scene_cfg['params']}",
            f"test -f {scene_cfg['config']}",
            f"conda run -n {conda_env} python scripts/fsm_lightgaussian_scores.py \\",
            f"  --params {scene_cfg['params']} \\",
            f"  --output {score_path} \\",
            f"  --metadata {metadata_path} \\",
            f"  --v-pow {float(v_pow):.12g} \\",
            f"  --opacity-power {float(opacity_power):.12g}",
            f"conda run -n {conda_env} python scripts/fsm_eval.py \\",
            f"  --config {scene_cfg['config']} \\",
            f"  --params {scene_cfg['params']} \\",
            f"  --output-dir {output_dir} \\",
            "  --strategies external_scores \\",
            f"  --compression-ratios {ratio_args} \\",
            f"  --external-scores {score_path} \\",
            "  --external-scores-key scores \\",
            f"  --start {int(start)} --end {int(end)} \\",
            " \\\n+".join(final_eval_args),
            "",
        ]
    )


def build_commands(args: argparse.Namespace) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    for scene in args.replica_scenes:
        _append_scene_commands(
            commands,
            dataset_prefix="replica",
            scene=scene,
            scene_cfg=REPLICA_SCENES[scene],
            output_root=args.output_root,
            ratios=args.ratios,
            start=args.start,
            end=args.end,
            conda_env=args.conda_env,
            v_pow=args.v_pow,
            opacity_power=args.opacity_power,
            overwrite=args.overwrite,
        )
    for scene in args.tum_scenes:
        _append_scene_commands(
            commands,
            dataset_prefix="tum",
            scene=scene,
            scene_cfg=TUM_SCENES[scene],
            output_root=args.output_root,
            ratios=args.ratios,
            start=args.start,
            end=args.end,
            conda_env=args.conda_env,
            v_pow=args.v_pow,
            opacity_power=args.opacity_power,
            overwrite=args.overwrite,
        )
    return commands


def main() -> None:
    args = parse_args()
    output_script = Path(args.output_script)
    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text("\n".join(build_commands(args)) + "\n")
    print(f"Wrote {output_script}")


if __name__ == "__main__":
    main()
