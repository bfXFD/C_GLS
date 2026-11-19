import json
import os
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

from datasets.gradslam_datasets.tum import TUMDataset
from utils.fsm_dataset import dataset_pose_provenance


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/run_fsm_independent_revisit_desk2.sh"


def _write_tum_sequence(root: Path, name: str, translations):
    sequence = root / name
    sequence.mkdir(parents=True)
    timestamps = [float(index) for index in range(len(translations))]
    (sequence / "rgb.txt").write_text("# rgb\n" + "".join(f"{stamp:.6f} rgb/{index}.png\n" for index, stamp in enumerate(timestamps)))
    (sequence / "depth.txt").write_text("# depth\n" + "".join(f"{stamp:.6f} depth/{index}.png\n" for index, stamp in enumerate(timestamps)))
    (sequence / "groundtruth.txt").write_text(
        "# groundtruth\n"
        + "".join(
            f"{stamp:.6f} {xyz[0]} {xyz[1]} {xyz[2]} 0 0 0 1\n"
            for stamp, xyz in zip(timestamps, translations)
        )
    )


def _dataset(root: Path, sequence: str, reference=None):
    config = {
        "dataset_name": "tum",
        "camera_params": {
            "image_height": 2, "image_width": 2, "fx": 1.0, "fy": 1.0,
            "cx": 0.0, "cy": 0.0, "png_depth_scale": 1.0,
        },
    }
    return TUMDataset(
        config, str(root), sequence, stride=1, start=0, end=-1,
        desired_height=2, desired_width=2, device="cpu", relative_pose=True,
        pose_reference_sequence=reference,
    )


def test_default_tum_pose_behavior_remains_first_query_relative(tmp_path):
    _write_tum_sequence(tmp_path, "query", [(4, 0, 0), (5, 0, 0)])
    dataset = _dataset(tmp_path, "query")
    np.testing.assert_allclose(dataset.transformed_poses[0], np.eye(4), atol=1e-7)
    np.testing.assert_allclose(dataset.transformed_poses[1, :3, 3], [1, 0, 0], atol=1e-7)
    assert dataset.pose_reference_transform is None


def test_reference_sequence_poses_equal_inverse_map0_times_query_absolute(tmp_path):
    _write_tum_sequence(tmp_path, "map", [(1, 2, 3), (2, 2, 3)])
    _write_tum_sequence(tmp_path, "query", [(4, 6, 8), (5, 7, 9)])
    dataset = _dataset(tmp_path, "query", "map")
    expected = []
    map0 = np.eye(4)
    map0[:3, 3] = [1, 2, 3]
    for xyz in ((4, 6, 8), (5, 7, 9)):
        query = np.eye(4)
        query[:3, 3] = xyz
        expected.append(np.linalg.inv(map0) @ query)
    np.testing.assert_allclose(dataset.transformed_poses.numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(dataset.pose_reference_transform, expected[0], atol=1e-7)


def test_reference_transform_is_frozen_after_initialization(tmp_path):
    _write_tum_sequence(tmp_path, "map", [(1, 0, 0), (2, 0, 0)])
    _write_tum_sequence(tmp_path, "query", [(3, 0, 0), (4, 0, 0)])
    dataset = _dataset(tmp_path, "query", "map")
    frozen = dataset.transformed_poses.clone()
    dataset.poses[:] = 999.0
    (tmp_path / "query/groundtruth.txt").write_text("invalid after initialization\n")
    np.testing.assert_allclose(dataset.transformed_poses.numpy(), frozen.numpy())


def test_pose_provenance_records_hashes_fixed_transform_and_zero_query_map_frames(tmp_path):
    _write_tum_sequence(tmp_path, "map", [(1, 0, 0)])
    _write_tum_sequence(tmp_path, "query", [(3, 0, 0)])
    config = {"data": {"dataset_name": "tum", "basedir": str(tmp_path), "sequence": "query", "pose_reference_sequence": "map"}}
    record = dataset_pose_provenance(config)
    np.testing.assert_allclose(record["fixed_frame_transform_A"][:3][0][3], 2.0)
    assert record["map_construction_query_frame_count"] == 0
    assert record["query_sources"]["groundtruth.txt"]["sha256"]
    assert record["reference_sources"]["rgb.txt"]["sha256"]


def _load_config(name):
    path = ROOT / "configs/fsm/mvp_observable" / name
    return SourceFileLoader(name, str(path)).load_module().config


def test_independent_revisit_config_semantics():
    full = _load_config("tum_freiburg1_desk2_independent_full.py")
    compact = _load_config("tum_freiburg1_desk2_independent_compact.py")
    corr = _load_config("tum_freiburg1_desk2_independent_corr_weak128_v4.py")
    replace = _load_config("tum_freiburg1_desk2_independent_replace128_v4.py")
    for config in (full, compact, corr, replace):
        assert config["data"]["sequence"] == "rgbd_dataset_freiburg1_desk"
        assert config["data"]["pose_reference_sequence"] == "rgbd_dataset_freiburg1_desk2"
        assert config["registration"]["first_pose_protocol"] == "gt_first_frame"
        assert config["independent_revisit"]["map_construction_query_frame_count"] == 0
    assert full["observable_reserve"]["enabled"] is False
    assert compact["independent_revisit"]["map_role"] == "fps_scale_opacity_ratio_0p1"
    assert corr["observable_reserve"]["selection"] == "runtime_corr_weak"
    assert corr["observable_reserve"]["budget"] == 128
    assert corr["observable_reserve"]["accept_only_if"]["protocol"] == "branch_aware_information_v4"
    assert replace["observable_reserve"]["candidate_target_mode"] == "replace"


def test_runner_plan_is_serial_and_contains_deterministic_builder(tmp_path):
    env = os.environ.copy()
    env.update(OUT_ROOT=str(tmp_path / "out"), PARAMS=str(tmp_path / "missing_params.npz"), PLAN_ONLY="1")
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    output = result.stdout
    assert "SEED=0" in output and "SCENE_NUM=1" in output
    assert "scripts/splatam.py" in output and "configs/tum/tum_eval.py" in output
    assert "--strategies full" in output
    assert "--strategies fps_scale_opacity" in output
    assert "--compression-ratios 0.1" in output
    arm_plan = output.split("fsm_eval_long_horizon.py", 1)[1]
    assert arm_plan.index("full_map") < arm_plan.index("compact_fps10")
    assert arm_plan.index("compact_fps10") < arm_plan.index("corr_weak128_v4")
    assert arm_plan.index("corr_weak128_v4") < arm_plan.index("replace128_v4")
    assert "fsm_build_fixed_map_reserve.py" in output
    assert "--reserve-count 4096" in output
    map_build_plan = output.split("fsm_independent_revisit_provenance.py", 1)[0]
    assert "rgbd_dataset_freiburg1_desk " not in map_build_plan
    assert "fsm_independent_revisit_acceptance.py" in output
    assert "fsm_independent_revisit_table.py" in output


def test_runner_has_two_frame_plan_and_refuses_frozen_output(tmp_path):
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "smoke"), "FRAME_END": "2"})
    result = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "--end 2" in result.stdout

    env["OUT_ROOT"] = str(tmp_path / "audit_v2_full_rerun" / "forbidden")
    refused = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert refused.returncode == 2
    assert "Refusing frozen audit output path" in refused.stderr
