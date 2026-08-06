from scripts.fsm_plan_tum import build_fsm_commands, build_splatam_commands, write_configs


def test_write_configs_creates_tum_fsm_config(tmp_path):
    paths = write_configs(tmp_path, ["freiburg1_desk"], seed=3, max_points=123, depth_max=4.5)

    text = paths["freiburg1_desk"].read_text()
    assert "rgbd_dataset_freiburg1_desk" in text
    assert "freiburg1_desk.yaml" in text
    assert "max_points=123" in text
    assert "depth_max=4.5" in text


def test_build_tum_scripts_include_params_and_eval_commands(tmp_path):
    paths = write_configs(tmp_path, ["freiburg2_xyz"], seed=0, max_points=10000, depth_max=4.0)

    splatam = "\n".join(build_splatam_commands(["freiburg2_xyz"], seed=0, conda_env="splatam"))
    fsm = "\n".join(
        build_fsm_commands(
            ["freiburg2_xyz"],
            config_paths=paths,
            experiment_name="batch_test",
            strategies=["full", "voxel_scale_opacity"],
            ratios=["1", "0.2"],
            end=25,
            conda_env="splatam",
            seed=2,
        )
    )

    assert "SCENE_NUM=3" in splatam
    assert "scripts/splatam.py configs/tum/tum_eval.py" in splatam
    assert "experiments/TUM/freiburg2_xyz_seed2/params.npz" in fsm
    assert "--compression-ratios 1 0.2" in fsm
    assert "--start 0 --end 25" in fsm
