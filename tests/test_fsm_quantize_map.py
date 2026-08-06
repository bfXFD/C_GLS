import os

import numpy as np

from scripts.fsm_plan_quantization import build_quantization_commands
from scripts.fsm_quantize_map import quantize_file, quantize_fsm_map
from tests.test_fsm_map import make_params
from utils.fsm_map import build_fsm_map, load_fsm_map, save_fsm_map


def test_quantize_fsm_map_casts_float_fields_and_drops_scores():
    params = make_params(num_gaussians=4)
    fsm_map = build_fsm_map(params, scores=np.arange(4, dtype=np.float32))

    quantized = quantize_fsm_map(fsm_map, float_dtype="float16", fields=["means"], drop_scores=True)

    assert quantized["means"].dtype == np.float16
    assert quantized["rotations"].dtype == fsm_map["rotations"].dtype
    assert quantized["covariances"].dtype == fsm_map["covariances"].dtype
    assert quantized["scores"].shape == (0,)
    assert quantized["indices"].dtype == fsm_map["indices"].dtype


def test_quantize_file_reduces_map_size(tmp_path):
    params = make_params(num_gaussians=16)
    fsm_map = build_fsm_map(params, scores=np.arange(16, dtype=np.float32))
    input_path = tmp_path / "fsm_map.npz"
    output_path = tmp_path / "quantized" / "fsm_map.npz"
    save_fsm_map(fsm_map, str(input_path))

    summary = quantize_file(str(input_path), str(output_path), float_dtype="float16", fields=["means"], drop_scores=True)
    loaded = load_fsm_map(str(output_path))

    assert os.path.exists(output_path)
    assert summary["output_size_mb"] < summary["input_size_mb"]
    assert loaded["means"].dtype == np.float16
    assert loaded["scores"].shape == (0,)


def test_build_quantization_commands_include_localize_steps():
    commands = "\n".join(
        build_quantization_commands(
            scenes=["room0_0"],
            methods=["voxel_scale_opacity_ratio_0p2"],
            source_experiment="src",
            output_experiment="quant",
            conda_env="splatam",
        )
    )

    assert "scripts/fsm_quantize_map.py" in commands
    assert "--float-dtype float16 --fields means --drop-scores" in commands
    assert "scripts/fsm_localize.py" in commands
    assert "replica_room0_0/src/maps/voxel_scale_opacity_ratio_0p2/fsm_map.npz" in commands
