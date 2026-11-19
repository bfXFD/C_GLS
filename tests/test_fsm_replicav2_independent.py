import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.gradslam_datasets.replica import ReplicaV2Dataset


ROOT = Path(__file__).resolve().parents[1]


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return " ".join(str(item) for item in value.reshape(-1)) + "\n"


def make_split(root, split, xs):
    base = root / "room_0" / "imap" / split
    (base / "rgb").mkdir(parents=True)
    (base / "depth").mkdir(parents=True)
    for index in range(len(xs)):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(base / "rgb" / f"rgb_{index}.png")
        Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(base / "depth" / f"depth_{index}.png")
    (base / "traj_w_c.txt").write_text("".join(pose(x) for x in xs))


def test_query_only_replicav2_uses_only_imap01_and_map_frame(tmp_path):
    make_split(tmp_path, "00", [10.0, 11.0])
    make_split(tmp_path, "01", [12.0, 13.0, 14.0])
    dataset = ReplicaV2Dataset(
        {"dataset_name": "replicav2", "camera_params": {"image_height": 2, "image_width": 2, "fx": 1, "fy": 1, "cx": 0, "cy": 0, "png_depth_scale": 1}},
        basedir=str(tmp_path), sequence="room_0", use_train_split=False, query_only_map_frame=True,
        desired_height=2, desired_width=2, relative_pose=False, device="cpu",
    )
    assert len(dataset) == 3
    assert np.isclose(dataset.poses[0][0, 3].item(), 2.0)
    assert np.isclose(dataset.poses[-1][0, 3].item(), 4.0)


def test_replicav2_runner_plan_contains_two_scenes_and_four_arms(tmp_path):
    env = os.environ.copy()
    env.update({"PLAN_ONLY": "1", "SCENES": "room_0,office_0", "OUT_ROOT": str(tmp_path / "out")})
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_fsm_replicav2_independent_workspaces.sh")],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=True,
    )
    output = result.stdout
    for scene in ("room_0", "office_0"):
        for arm in ("full", "compact", "corrweak", "replace"):
            assert f"{scene}_{arm}" in output
    assert "fsm_replicav2_preflight.py" in output
    assert "fsm_replicav2_acceptance.py" in output


def test_replicav2_runner_missing_data_prints_download_command(tmp_path):
    env = os.environ.copy()
    env.update({
        "PLAN_ONLY": "0", "AUTO_DOWNLOAD": "0", "SCENES": "room_0",
        "DATA_ROOT": str(tmp_path / "missing"), "OUT_ROOT": str(tmp_path / "out"),
    })
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_fsm_replicav2_independent_workspaces.sh")],
        cwd=tmp_path, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "download_replicav2.sh" in result.stderr
    assert "AUTO_DOWNLOAD=1" in result.stderr
