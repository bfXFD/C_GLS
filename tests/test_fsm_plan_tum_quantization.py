from scripts.fsm_plan_tum_quantization import build_tum_quantization_commands


def test_build_tum_quantization_commands_include_safe_quantization_and_localize():
    commands = "\n".join(
        build_tum_quantization_commands(
            scenes=["freiburg2_xyz"],
            methods=["fps_scale_opacity_ratio_0p1"],
            source_experiment="src",
            output_experiment="quant",
            start=3,
            end=17,
            conda_env="splatam",
        )
    )

    assert "tum_freiburg2_xyz/src/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz" in commands
    assert "--float-dtype float16 --fields means --drop-scores" in commands
    assert "configs/fsm/generated_tum/tum_freiburg2_xyz_fsm.py" in commands
    assert "--start 3 --end 17" in commands
    assert "scripts/fsm_localize.py" in commands
