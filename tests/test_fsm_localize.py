import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import scripts.fsm_localize as fsm_localize
from scripts.fsm_localize import load_config, run_localization
from utils.fsm_map import build_fsm_map, save_fsm_map


def make_depth_from_points(points, intrinsics, height=4, width=4):
    depth = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x, y, z = point
        u = int(round(intrinsics[0, 0] * x / z + intrinsics[0, 2]))
        v = int(round(intrinsics[1, 1] * y / z + intrinsics[1, 2]))
        if 0 <= u < width and 0 <= v < height:
            depth[v, u] = z
    return depth


def make_fake_map(tmp_path):
    means = np.array(
        [
            [-1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    params = {
        "means3D": means,
        "log_scales": np.log(np.full((means.shape[0], 1), 0.01, dtype=np.float32)),
        "unnorm_rotations": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (means.shape[0], 1)),
    }
    fsm_map = build_fsm_map(params, strategy="full")
    map_path = tmp_path / "fsm_map.npz"
    save_fsm_map(fsm_map, str(map_path))
    return map_path, means


def make_frames(points, num_frames=3):
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    depth = make_depth_from_points(points, K, height=3, width=3)
    frames = []
    for idx in range(num_frames):
        frames.append(
            {
                "frame_id": idx,
                "timestamp": float(idx),
                "color": np.zeros((3, 3, 3), dtype=np.uint8),
                "depth": depth,
                "intrinsics": K,
                "c2w": np.eye(4, dtype=np.float32),
            }
        )
    return frames


def make_config():
    return {
        "seed": 0,
        "dataset": {
            "depth_min": 0.0,
            "depth_max": 2.0,
            "max_points": None,
        },
        "registration": {
            "backend": "numpy_point_to_point",
            "voxel_size": None,
            "max_correspondence_distance": 0.5,
            "max_iterations": 20,
            "init_mode": "identity",
        },
        "eval": {
            "align_ate": True,
            "rpe_delta": 1,
        },
    }


def test_run_localization_dry_run_outputs_files(tmp_path):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=3)
    output_dir = tmp_path / "localize"
    summary = run_localization(make_config(), str(map_path), str(output_dir), frames=frames)

    assert summary["backend"] == "numpy_point_to_point"
    assert summary["N_star"] == points.shape[0]
    assert summary["num_frames"] == 3
    assert summary["ate_rmse"] < 1e-6
    assert summary["rpe_trans_rmse"] is not None
    assert (output_dir / "trajectory_est.txt").exists()
    assert (output_dir / "trajectory_gt.txt").exists()
    assert (output_dir / "frame_metrics.csv").exists()
    assert (output_dir / "summary.json").exists()

    with open(output_dir / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["frame_id"] == "0"
    assert int(rows[0]["num_target_points"]) == points.shape[0]

    loaded_summary = json.loads((output_dir / "summary.json").read_text())
    assert loaded_summary["num_frames"] == 3


def test_run_localization_respects_frame_range(tmp_path):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=5)
    output_dir = tmp_path / "localize"
    summary = run_localization(make_config(), str(map_path), str(output_dir), frames=frames, start=1, end=4)
    assert summary["num_frames"] == 3
    lines = (output_dir / "trajectory_est.txt").read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("1.000000000")


def test_run_localization_falls_back_on_non_finite_pose(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=3)
    bad_transform = np.eye(4, dtype=np.float64)
    bad_transform[0, 3] = np.inf

    calls = {"count": 0}

    def fake_register_pointcloud(**kwargs):
        calls["count"] += 1
        return {
            "transform": bad_transform if calls["count"] == 2 else np.eye(4, dtype=np.float64),
            "fitness": 1.0,
            "inlier_rmse": 0.0,
            "latency_ms": 0.0,
            "num_source_points": 1,
            "num_target_points": points.shape[0],
            "converged": True,
        }

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update(
        {
            "init_mode": "previous_estimate",
            "fallback_on_divergence": True,
            "max_pose_translation_norm": 100.0,
        }
    )
    output_dir = tmp_path / "localize"
    summary = run_localization(config, str(map_path), str(output_dir), frames=frames)
    assert summary["num_fallbacks"] == 1
    with open(output_dir / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["fallback_used"] == "True"


def test_run_localization_requires_frames_or_dataset(tmp_path):
    map_path, _ = make_fake_map(tmp_path)
    with pytest.raises(ValueError):
        run_localization(make_config(), str(map_path), str(tmp_path / "out"))


def test_load_config_reads_config_variable(tmp_path):
    config_path = tmp_path / "config.py"
    config_path.write_text("config = {'registration': {'backend': 'numpy_point_to_point'}}\n")
    config = load_config(str(config_path))
    assert config["registration"]["backend"] == "numpy_point_to_point"


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/fsm_localize.py", "--help"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Localize RGB-D frames against an FSM map" in result.stdout
