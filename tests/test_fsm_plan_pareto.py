from scripts.fsm_plan_pareto import build_commands


def test_build_commands_writes_scene_eval_commands():
    commands = build_commands(
        scenes=["room1_0"],
        strategies=["full", "voxel_scale_opacity"],
        ratios=["1", "0.2"],
        experiment_name="batch_test",
        start=0,
        end=10,
        conda_env="splatam",
    )

    text = "\n".join(commands)
    assert "scripts/fsm_eval.py" in text
    assert "replica_room1_0/batch_test" in text
    assert "--strategies full voxel_scale_opacity" in text
    assert "--compression-ratios 1 0.2" in text
    assert "--skip-existing" in text
