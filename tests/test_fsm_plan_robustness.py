from scripts.fsm_plan_robustness import build_robustness_commands


def test_build_robustness_commands_generates_init_mode_configs_and_localization():
    commands = "\n".join(
        build_robustness_commands(
            scenes=["room0_0"],
            methods=["voxel_scale_opacity_ratio_0p2"],
            init_modes=["identity"],
            source_experiment="src",
            output_experiment="robust",
            start=5,
            end=25,
            conda_env="splatam",
        )
    )

    assert "init_mode=\\\"identity\\\"" in commands
    assert "replica_room0_0/src/maps/voxel_scale_opacity_ratio_0p2/fsm_map.npz" in commands
    assert "scripts/fsm_localize.py" in commands
    assert "--start 5 --end 25" in commands
