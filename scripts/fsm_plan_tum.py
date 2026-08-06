import argparse
from pathlib import Path


TUM_SCENES = {
    "freiburg1_desk": "rgbd_dataset_freiburg1_desk",
    "freiburg1_desk2": "rgbd_dataset_freiburg1_desk2",
    "freiburg1_room": "rgbd_dataset_freiburg1_room",
    "freiburg2_xyz": "rgbd_dataset_freiburg2_xyz",
    "freiburg3_long_office_household": "rgbd_dataset_freiburg3_long_office_household",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TUM RGB-D FSM configs and run scripts.")
    parser.add_argument("--output-dir", default="/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum")
    parser.add_argument("--script-dir", default="/home/ras/xbf/C_GLS_code/experiments/FSM/scripts")
    parser.add_argument("--scenes", nargs="+", default=["freiburg1_desk", "freiburg2_xyz"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end", type=int, default=500, help="FSM localization end frame. Use -1 for full sequence.")
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--experiment-name", default="batch_500_tum_pareto_v1_1")
    parser.add_argument("--ratios", nargs="+", default=["1", "0.25", "0.2", "0.15", "0.1"])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"],
    )
    parser.add_argument("--conda-env", default="splatam")
    return parser.parse_args()


def _scene_num(scene: str) -> int:
    return list(TUM_SCENES.keys()).index(scene)


def _config_text(scene: str, seed: int, max_points: int, depth_max: float) -> str:
    sequence = TUM_SCENES[scene]
    return f'''config = dict(
    seed={int(seed)},
    primary_device="cuda:0",
    dataset=dict(
        dataset_name="tum",
        basedir="/home/ras/xbf/C_GLS_code/data/TUM_RGBD",
        gradslam_data_cfg="/home/ras/xbf/C_GLS_code/configs/data/TUM/{scene}.yaml",
        sequence="{sequence}",
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        depth_min=0.0,
        depth_max={float(depth_max)},
        max_points={int(max_points)},
    ),
    registration=dict(
        backend="numpy_point_to_point",
        voxel_size=None,
        max_correspondence_distance=0.07,
        max_iterations=30,
        normal_radius=0.10,
        normal_max_nn=30,
        init_mode="previous_estimate",
        fallback_on_divergence=True,
        max_pose_translation_norm=100.0,
    ),
    eval=dict(
        align_ate=True,
        rpe_delta=1,
        save_tum_format=True,
    ),
)
'''


def write_configs(output_dir: str | Path, scenes: list[str], seed: int, max_points: int, depth_max: float) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for scene in scenes:
        if scene not in TUM_SCENES:
            raise KeyError(f"Unsupported TUM scene: {scene}")
        path = output_root / f"tum_{scene}_fsm.py"
        path.write_text(_config_text(scene, seed=seed, max_points=max_points, depth_max=depth_max))
        paths[scene] = path
    return paths


def build_splatam_commands(scenes: list[str], seed: int, conda_env: str) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    for scene in scenes:
        commands.extend(
            [
                f"# Build TUM params for {scene}",
                f"SEED={int(seed)} SCENE_NUM={_scene_num(scene)} WANDB_MODE=disabled conda run -n {conda_env} python scripts/splatam.py configs/tum/tum_eval.py",
                "",
            ]
        )
    return commands


def build_fsm_commands(
    scenes: list[str],
    config_paths: dict[str, Path],
    experiment_name: str,
    strategies: list[str],
    ratios: list[str],
    end: int,
    conda_env: str,
    seed: int = 0,
) -> list[str]:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", "cd /home/ras/xbf/C_GLS_code", ""]
    strategy_args = " ".join(strategies)
    ratio_args = " ".join(ratios)
    for scene in scenes:
        params = f"/home/ras/xbf/C_GLS_code/experiments/TUM/{scene}_seed{int(seed)}/params.npz"
        output_dir = f"/home/ras/xbf/C_GLS_code/experiments/FSM/tum_{scene}/{experiment_name}"
        commands.extend(
            [
                f"# FSM Pareto for TUM {scene}",
                f"test -f {params}",
                f"conda run -n {conda_env} python scripts/fsm_eval.py \\",
                f"  --config {config_paths[scene]} \\",
                f"  --params {params} \\",
                f"  --output-dir {output_dir} \\",
                f"  --strategies {strategy_args} \\",
                f"  --compression-ratios {ratio_args} \\",
                f"  --start 0 --end {int(end)} \\",
                "  --skip-existing",
                "",
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    config_paths = write_configs(args.output_dir, args.scenes, seed=args.seed, max_points=args.max_points, depth_max=args.depth_max)
    script_root = Path(args.script_dir)
    script_root.mkdir(parents=True, exist_ok=True)
    splatam_script = script_root / "run_tum_splatam_params_v1_1.sh"
    fsm_script = script_root / "run_tum_fsm_pareto_v1_1.sh"
    splatam_script.write_text("\n".join(build_splatam_commands(args.scenes, seed=args.seed, conda_env=args.conda_env)) + "\n")
    fsm_script.write_text(
        "\n".join(
            build_fsm_commands(
                args.scenes,
                config_paths=config_paths,
                experiment_name=args.experiment_name,
                strategies=args.strategies,
                ratios=args.ratios,
                end=args.end,
                conda_env=args.conda_env,
                seed=args.seed,
            )
        )
        + "\n"
    )
    print(f"Wrote {len(config_paths)} TUM FSM configs to {args.output_dir}")
    print(f"Wrote {splatam_script}")
    print(f"Wrote {fsm_script}")


if __name__ == "__main__":
    main()
