import csv
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import scripts.fsm_localize as fsm_localize
from scripts.fsm_localize import load_config, run_localization
from utils.fsm_map import (
    ACTIVATION_GEOMETRY_ID_SPACE,
    ACTIVATION_GEOMETRY_SCHEMA_VERSION,
    build_fsm_map,
    expand_log_scales,
    gaussians_to_covariances,
    normalize_quaternions,
    save_fsm_map,
    selected_ids_sha256,
)


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


def test_strict_first_pose_applies_left_twist_to_gt_pose():
    gt = np.eye(4, dtype=np.float64)
    pose = fsm_localize._strict_first_pose(
        "gt_first_frame", gt, None, np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3])
    )
    np.testing.assert_allclose(pose[:3, 3], [0.1, -0.2, 0.3])


def test_candidate_target_replace_preserves_budget_and_is_deterministic():
    base = np.arange(30, dtype=np.float64).reshape(10, 3)
    base_cov = np.tile(np.eye(3), (10, 1, 1))
    base_normals = np.tile([0.0, 0.0, 1.0], (10, 1))
    reserve = np.arange(9, dtype=np.float64).reshape(3, 3) + 100.0
    reserve_cov = np.tile(2.0 * np.eye(3), (3, 1, 1))
    reserve_normals = np.tile([1.0, 0.0, 0.0], (3, 1))
    kwargs = dict(
        mode="replace", base_points=base, base_covariances=base_cov,
        base_normals=base_normals, base_global_ids=np.arange(10),
        reserve_points=reserve, reserve_covariances=reserve_cov,
        reserve_normals=reserve_normals, replacement_seed=7,
    )
    first = fsm_localize._candidate_target_arrays(**kwargs)
    second = fsm_localize._candidate_target_arrays(**kwargs)
    assert first[0].shape == base.shape
    assert first[3] == reserve.shape[0]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2][-3:], reserve_normals)


def test_candidate_target_rerun_primary_ignores_reserve():
    base = np.arange(12, dtype=np.float64).reshape(4, 3)
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    points, covariances, result_normals, removed = fsm_localize._candidate_target_arrays(
        mode="rerun_primary", base_points=base, base_covariances=None,
        base_normals=normals, base_global_ids=np.arange(4),
        reserve_points=np.ones((2, 3)), reserve_covariances=None,
        reserve_normals=np.tile([1.0, 0.0, 0.0], (2, 1)), replacement_seed=0,
    )
    np.testing.assert_array_equal(points, base)
    np.testing.assert_array_equal(result_normals, normals)
    assert covariances is None
    assert removed == 0


def test_reserve_selection_event_writer_streams_bounded_rows(tmp_path):
    map_path, _ = make_fake_map(tmp_path)
    geometry_path = tmp_path / "geometry.npz"
    np.savez(geometry_path, schema_version=np.asarray(2))
    reserve = type("Reserve", (), {
        "active": True,
        "selection": "runtime_corr_weak",
        "budget": 2,
        "geometry_path": str(geometry_path),
    })()
    config = {"observable_reserve": {
        "enabled": True,
        "mode": "active",
        "selection_artifact": {"enabled": True, "flush_each_frame": True},
    }}
    writer = fsm_localize._ReserveSelectionEventWriter(tmp_path, config, reserve, str(map_path))
    plan = type("Plan", (), {
        "selected_indices": np.asarray([4, 2]),
        "selected_global_ids": np.asarray([104, 102]),
        "selected_scores": np.asarray([3.0, 2.0]),
        "selected_probabilities": np.asarray([0.8, 0.6]),
        "selection_policy": "runtime_corr_weak",
    })()
    writer.write(7, plan, {"reserve_attempted": True, "reserve_used": False})
    writer.close()
    with open(tmp_path / "reserve_selection_events.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["selected_global_id"]) for row in rows] == [104, 102]
    assert all(row["frame_id"] == "7" and row["attempted"] == "True" for row in rows)
    with open(tmp_path / "reserve_selection_events_provenance.json") as stream:
        provenance = json.load(stream)
    assert provenance["budget"] == 2
    assert provenance["reserve_geometry_sha256"]


def make_activation_map(tmp_path):
    coordinates = np.linspace(-0.3, 0.3, 3)
    means = np.array([[x, y, 1.0] for x in coordinates for y in coordinates], dtype=np.float32)
    params = {
        "means3D": means,
        "log_scales": np.log(np.full((means.shape[0], 1), 0.01, dtype=np.float32)),
        "unnorm_rotations": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (means.shape[0], 1)),
    }
    selected = np.array([5, 1, 7, 3, 8, 0, 6], dtype=np.int64)
    preserved = np.array([1, 3, 0, 6], dtype=np.int64)
    replacement = np.array([5, 7, 8], dtype=np.int64)
    removed = np.array([2, 4], dtype=np.int64)
    baseline = np.concatenate([preserved, removed])
    fsm_map = build_fsm_map(
        params,
        indices=selected,
        strategy="logdet",
        source_params_sha256="a" * 64,
        selection_controls_sha256="b" * 64,
    )
    fsm_map["normals"] = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (selected.size, 1))
    map_path = tmp_path / "fsm_map.npz"
    np.savez(map_path, **fsm_map)
    removed_log_scales = params["log_scales"][removed]
    removed_rotations = normalize_quaternions(params["unnorm_rotations"][removed])
    local_by_global = {int(value): index for index, value in enumerate(selected)}
    geometry = {
        "schema_version": np.asarray(ACTIVATION_GEOMETRY_SCHEMA_VERSION, dtype=np.int64),
        "id_space": np.asarray(ACTIVATION_GEOMETRY_ID_SPACE),
        "selected_global_ids": selected,
        "baseline_global_ids": baseline,
        "preserved_global_ids": preserved,
        "replacement_global_ids": replacement,
        "removed_global_ids": removed,
        "removed_means": means[removed],
        "removed_log_scales": removed_log_scales,
        "removed_rotations": removed_rotations,
        "removed_scales": np.exp(expand_log_scales(removed_log_scales)),
        "removed_covariances": gaussians_to_covariances(removed_log_scales, removed_rotations),
        "removed_scores": np.ones(removed.size, dtype=np.float32),
        "removed_normals": np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (removed.size, 1)),
        "normal_source": np.asarray("test"),
        "source_params_sha256": np.asarray("a" * 64),
        "selection_controls_sha256": np.asarray("b" * 64),
        "map_selected_ids_sha256": np.asarray(selected_ids_sha256(selected)),
        "replacement_selected_local_ids": np.asarray([local_by_global[int(value)] for value in replacement]),
        "reserve_global_ids": np.empty(0, dtype=np.int64),
        "reserve_means": np.empty((0, 3), dtype=np.float32),
        "reserve_normals": np.empty((0, 3), dtype=np.float32),
        "reserve_scores": np.empty(0, dtype=np.float32),
        "reserve_covariances": np.empty((0, 3, 3), dtype=np.float32),
        "reserve_scales": np.empty((0, 3), dtype=np.float32),
        "reserve_rotations": np.empty((0, 4), dtype=np.float32),
        "reserve_count": np.asarray(0, dtype=np.int64),
        "reserve_policy": np.asarray("test"),
    }
    geometry_path = tmp_path / "activation_geometry.npz"
    np.savez(geometry_path, **geometry)
    return map_path, geometry_path, means[selected]


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


def translated_pose(x=0.0, yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    pose[0, 3] = x
    return pose


def successful_result(transform, target_count):
    return {
        "transform": np.asarray(transform, dtype=np.float64),
        "fitness": 1.0,
        "inlier_rmse": 0.0,
        "latency_ms": 0.0,
        "num_source_points": 1,
        "num_target_points": target_count,
        "converged": True,
        "optimizer_valid": True,
        "termination_reason": "converged",
    }


def consistency_config(**overrides):
    config = {
        "enabled": True,
        "correction": {
            "constant_velocity": {"translation_m": 0.10, "rotation_deg": 4.0},
            "bootstrap": {"translation_m": 0.15, "rotation_deg": 6.0},
        },
        "temporal": {
            "translation_m_per_frame": 0.15,
            "rotation_deg_per_frame": 8.0,
            "hard_translation_m": 0.45,
            "hard_rotation_deg": 20.0,
        },
    }
    config.update(overrides)
    return config


def test_relative_pose_metrics_uses_reference_inverse():
    reference = translated_pose(1.0, yaw_deg=90.0)
    candidate = reference @ translated_pose(0.25, yaw_deg=10.0)
    metrics = fsm_localize.relative_pose_metrics(reference, candidate)
    assert metrics["translation_m"] == pytest.approx(0.25)
    assert metrics["rotation_deg"] == pytest.approx(10.0)


def test_pose_consistency_small_cv_and_gap_hard_cap():
    accepted, reason, metrics = fsm_localize._pose_consistency_acceptance(
        translated_pose(0.2), translated_pose(0.25), translated_pose(0.1),
        "constant_velocity", 1, consistency_config(),
    )
    assert accepted and reason is None
    assert metrics["correction_translation_m"] == pytest.approx(0.05)

    accepted, reason, metrics = fsm_localize._pose_consistency_acceptance(
        translated_pose(0.4), translated_pose(0.42), translated_pose(0.0),
        "constant_velocity", 10, consistency_config(),
    )
    assert accepted
    assert metrics["temporal_translation_limit_m"] == pytest.approx(0.45)

    accepted, reason, _ = fsm_localize._pose_consistency_acceptance(
        translated_pose(0.5), translated_pose(0.5), translated_pose(0.0),
        "constant_velocity", 10, consistency_config(),
    )
    assert not accepted
    assert reason == "tracking_consistency_temporal_translation"


def test_quality_pass_consistency_reject_does_not_enter_history(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    inits = []

    def fake_register_pointcloud(**kwargs):
        inits.append(np.asarray(kwargs["init_transform"]).copy())
        return successful_result(translated_pose(0.345), points.shape[0])

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "motion_model": "constant_velocity",
        "pose_consistency": consistency_config(),
    })
    output = tmp_path / "consistency_reject"
    summary = run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=3))
    assert summary["num_consistency_rejections"] == 2
    assert inits[1][0, 3] == pytest.approx(0.0)
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["primary_rejection_reason"] == "tracking_consistency_correction_translation"
    assert rows[1]["tracking_valid"] == "False"
    assert rows[2]["init_source"] == "previous_pose"


def test_large_cv_temporal_rejected_even_with_small_correction():
    accepted, reason, metrics = fsm_localize._pose_consistency_acceptance(
        translated_pose(0.40), translated_pose(0.42), translated_pose(0.0),
        "constant_velocity", 1, consistency_config(),
    )
    assert metrics["correction_translation_m"] < 0.10
    assert not accepted
    assert reason == "tracking_consistency_temporal_translation"


def test_backend_aware_acceptance_does_not_require_point_to_plane_diagnostics():
    result = successful_result(np.eye(4), 100)
    result.update(backend="numpy_point_to_point", fitness=0.8, inlier_rmse=0.02, num_source_points=100)
    accepted, reason = fsm_localize._tracking_acceptance(
        result,
        {"min_fitness": 0.5, "min_correspondences": 64, "max_inlier_rmse": 0.05,
         "max_point_to_plane_rmse": 0.001, "min_lambda_min": 100.0},
        "numpy_point_to_point",
    )
    assert accepted and reason is None


def test_recovery_point_to_point_verifier_records_backend_without_gt(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=2)
    frames[1]["c2w"] = translated_pose(999.0).astype(np.float32)
    backends = []

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.arange(points.shape[0]),
            "query_latency_ms": 0.0,
            "candidate_count": points.shape[0],
            "frustum_count": points.shape[0],
            "count": points.shape[0],
            "backend": self.backend,
        }

    def fake_register_pointcloud(**kwargs):
        backends.append(kwargs["backend"])
        result = successful_result(translated_pose(0.05), points.shape[0])
        result.update(backend=kwargs["backend"], num_source_points=100, fitness=0.8, inlier_rmse=0.01)
        if kwargs["backend"] != "numpy_point_to_point":
            result.update(optimizer_valid=False, converged=False, failure_reason="degenerate_geometry")
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({
        "backend": "numpy_robust_point_to_plane",
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "tracking_acceptance": {"min_fitness": 0.5},
        "pose_consistency": consistency_config(),
    })
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["recovery"] = {
        "enabled": True,
        "max_tier": 1,
        "max_points": points.shape[0],
        "verifier_backend": "numpy_point_to_point",
        "verifier_acceptance": {"min_fitness": 0.5, "min_correspondences": 64, "max_inlier_rmse": 0.05},
    }
    output = tmp_path / "verifier"
    summary = run_localization(config, str(map_path), str(output), frames=frames)
    assert backends == ["numpy_robust_point_to_plane", "numpy_robust_point_to_plane", "numpy_point_to_point"]
    assert summary["num_recovered_frames"] == 1
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["recovery_backend"] == "numpy_point_to_point"
    assert rows[1]["tracking_valid"] == "True"


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
    assert "evaluation_protocol" not in summary
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


def test_run_localization_streams_dataset_frames(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=4)

    class TrackingDataset:
        def __init__(self, values):
            self.values = values
            self.read_indices = []

        def __len__(self):
            return len(self.values)

        def __getitem__(self, idx):
            self.read_indices.append(idx)
            frame = self.values[idx]
            return frame["color"], frame["depth"], frame["intrinsics"], frame["c2w"]

    dataset = TrackingDataset(frames)
    registration_calls = {"count": 0}

    def fake_register_pointcloud(**kwargs):
        registration_calls["count"] += 1
        assert dataset.read_indices == list(range(registration_calls["count"]))
        return {
            "transform": np.eye(4, dtype=np.float64),
            "fitness": 1.0,
            "inlier_rmse": 0.0,
            "latency_ms": 0.0,
            "num_source_points": 1,
            "num_target_points": points.shape[0],
            "converged": True,
        }

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    summary = run_localization(
        make_config(),
        str(map_path),
        str(tmp_path / "streamed"),
        dataset=dataset,
    )

    assert summary["num_frames"] == 4
    assert dataset.read_indices == [0, 1, 2, 3]


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
    assert "tracking_status" not in rows[1]


def test_strict_protocol_never_uses_current_frame_gt(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames_a = make_frames(points, num_frames=4)
    frames_b = make_frames(points, num_frames=4)
    for idx in range(1, len(frames_b)):
        frames_b[idx]["c2w"] = np.eye(4, dtype=np.float32)
        frames_b[idx]["c2w"][0, 3] = 1000.0 + idx

    init_transforms = []

    def fake_register_pointcloud(**kwargs):
        init_transform = np.asarray(kwargs["init_transform"], dtype=np.float64)
        init_transforms.append(init_transform.copy())
        transform = init_transform.copy()
        transform[0, 3] += 0.1
        return {
            "transform": transform,
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
            "evaluation_protocol": "mvp_strict",
            "first_pose_protocol": "gt_first_frame",
            "fallback_on_divergence": True,
        }
    )
    run_localization(config, str(map_path), str(tmp_path / "strict_a"), frames=frames_a)
    trajectory_a = (tmp_path / "strict_a" / "trajectory_est.txt").read_text()
    inits_a = [pose.copy() for pose in init_transforms]
    init_transforms.clear()

    run_localization(config, str(map_path), str(tmp_path / "strict_b"), frames=frames_b)
    trajectory_b = (tmp_path / "strict_b" / "trajectory_est.txt").read_text()

    assert trajectory_a == trajectory_b
    assert len(inits_a) == len(init_transforms) == 3
    for pose_a, pose_b in zip(inits_a, init_transforms):
        np.testing.assert_allclose(pose_a, pose_b)


def test_strict_protocol_marks_lost_and_holds_last_valid(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=5)
    calls = {"count": 0}

    def fake_register_pointcloud(**kwargs):
        calls["count"] += 1
        transform = np.asarray(kwargs["init_transform"], dtype=np.float64).copy()
        transform[0, 3] += 0.25
        return {
            "transform": transform,
            "fitness": 0.0 if calls["count"] in (2, 3) else 1.0,
            "inlier_rmse": float("inf") if calls["count"] in (2, 3) else 0.0,
            "latency_ms": 0.0,
            "num_source_points": 1,
            "num_target_points": points.shape[0],
            "converged": calls["count"] not in (2, 3),
        }

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update(
        {
            "evaluation_protocol": "mvp_strict",
            "first_pose_protocol": "identity",
            "fallback_on_divergence": True,
        }
    )
    output_dir = tmp_path / "strict"
    summary = run_localization(config, str(map_path), str(output_dir), frames=frames)

    assert summary["num_fallbacks"] == 0
    assert summary["num_lost_frames"] == 2
    assert summary["lost_frame_ids"] == [2, 3]
    assert summary["num_tracking_failures"] == 1
    assert summary["num_valid_frames"] == 3
    assert summary["tracking_completeness"] == pytest.approx(0.5)
    assert summary["num_tracking_attempt_frames"] == 4
    assert summary["longest_valid_track_frames"] == 2
    assert summary["uses_gt_current_frame_initialization"] is False
    assert summary["uses_gt_fallback"] is False
    with open(output_dir / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["tracking_status"] for row in rows] == [
        "initialized",
        "tracked",
        "lost_hold_last_valid",
        "lost_hold_last_valid",
        "tracked",
    ]
    trajectory_lines = (output_dir / "trajectory_est.txt").read_text().strip().splitlines()
    held_positions = [line.split()[1:4] for line in trajectory_lines[1:4]]
    assert held_positions[0] == held_positions[1] == held_positions[2]


def test_strict_protocol_requires_explicit_first_pose_protocol(tmp_path):
    map_path, points = make_fake_map(tmp_path)
    config = make_config()
    config["registration"]["evaluation_protocol"] = "mvp_strict"
    with pytest.raises(ValueError, match="requires explicit first_pose_protocol"):
        run_localization(config, str(map_path), str(tmp_path / "strict"), frames=make_frames(points))


def test_strict_protocol_rejects_frozen_audit_output(tmp_path):
    map_path, points = make_fake_map(tmp_path)
    config = make_config()
    config["registration"].update(
        {"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"}
    )
    output_dir = tmp_path / "audit_v2_full_rerun" / "localize"
    with pytest.raises(ValueError, match="must not be written"):
        run_localization(config, str(map_path), str(output_dir), frames=make_frames(points))


def test_strict_protocol_accepts_explicit_initial_pose(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=1)
    initial_pose = np.eye(4, dtype=np.float64)
    initial_pose[1, 3] = 2.5

    def fail_if_called(**kwargs):
        raise AssertionError("strict first frame must be initialized without registration")

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fail_if_called)
    config = make_config()
    config["registration"].update(
        {"evaluation_protocol": "mvp_strict", "first_pose_protocol": "explicit"}
    )
    output_dir = tmp_path / "strict"
    summary = run_localization(
        config,
        str(map_path),
        str(output_dir),
        frames=frames,
        explicit_initial_pose=initial_pose,
    )
    assert summary["first_pose_protocol"] == "explicit"
    trajectory_line = (output_dir / "trajectory_est.txt").read_text().strip().split()
    assert float(trajectory_line[2]) == pytest.approx(2.5)


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


def test_constant_velocity_formula_and_history_fallback():
    previous_previous = translated_pose(1.0, yaw_deg=10.0)
    previous = translated_pose(2.0, yaw_deg=20.0)
    predicted, source, trans_step, rot_step, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(0, previous_previous), (1, previous)], 2, 10.0, 180.0
    )
    np.testing.assert_allclose(predicted, previous @ np.linalg.inv(previous_previous) @ previous)
    assert source == "constant_velocity"
    assert trans_step > 0.0
    assert rot_step == pytest.approx(10.0)
    assert guard == "accepted"

    fallback, source, trans_step, rot_step, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(1, previous)], 2, 10.0, 180.0
    )
    np.testing.assert_allclose(fallback, previous)
    assert (source, trans_step, rot_step, guard) == ("previous_pose", 0.0, 0.0, "insufficient_history")


def test_constant_velocity_translation_and_rotation_guards():
    identity = np.eye(4)
    translation = translated_pose(2.0)
    predicted, source, _, _, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(0, identity), (1, translation)], 2, 1.0, 180.0
    )
    np.testing.assert_allclose(predicted, translation)
    assert source == "previous_pose"
    assert guard == "translation_exceeded"

    rotation = translated_pose(0.0, yaw_deg=30.0)
    predicted, source, _, _, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(0, identity), (1, rotation)], 2, 1.0, 10.0
    )
    np.testing.assert_allclose(predicted, rotation)
    assert source == "previous_pose"
    assert guard == "rotation_exceeded"


def test_constant_velocity_prediction_horizon_guard():
    first = translated_pose(0.0)
    last = translated_pose(1.0)
    predicted, source, _, _, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(0, first), (1, last)], 5, 10.0, 180.0, prediction_max_horizon_frames=3
    )
    np.testing.assert_allclose(predicted, last)
    assert source == "previous_pose"
    assert guard == "prediction_horizon_exceeded"


def test_lost_hold_does_not_pollute_constant_velocity_history(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=4)
    inits = []

    def fake_register_pointcloud(**kwargs):
        init = np.asarray(kwargs["init_transform"], dtype=np.float64)
        inits.append(init.copy())
        if len(inits) == 1:
            return successful_result(translated_pose(1.0), points.shape[0])
        if len(inits) == 2:
            result = successful_result(translated_pose(99.0), points.shape[0])
            result["converged"] = False
            result["optimizer_valid"] = False
            return result
        return successful_result(init, points.shape[0])

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "motion_model": "constant_velocity",
    })
    run_localization(config, str(map_path), str(tmp_path / "strict_history"), frames=frames)

    assert len(inits) == 3
    assert inits[1][0, 3] == pytest.approx(2.0)
    assert inits[2][0, 3] == pytest.approx(3.0)


def test_recovered_pose_enters_history_but_blocks_immediate_constant_velocity(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    inits = []
    calls = {"count": 0}

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.arange(points.shape[0]), "query_latency_ms": 0.0,
            "candidate_count": points.shape[0], "frustum_count": points.shape[0],
            "count": points.shape[0], "backend": self.backend,
        }

    def fake_register_pointcloud(**kwargs):
        calls["count"] += 1
        inits.append(np.asarray(kwargs["init_transform"]).copy())
        result = successful_result(translated_pose(0.1), points.shape[0])
        if calls["count"] == 1:
            result.update(optimizer_valid=False, converged=False, failure_reason="primary_failed")
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({
        "evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity",
        "motion_model": "constant_velocity",
    })
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["recovery"] = {"enabled": True, "max_tier": 1, "max_points": points.shape[0]}
    output = tmp_path / "recovered_history"
    run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=3))
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["recovered"] == "True"
    assert rows[2]["init_source"] == "previous_pose"
    assert rows[2]["prediction_guard"] == "nonconsecutive_valid_history"
    assert inits[-1][0, 3] == pytest.approx(0.1)


def test_constant_velocity_rejects_nonconsecutive_history_across_output_gap():
    first = translated_pose(0.0)
    last = translated_pose(1.0)
    predicted, source, step, _, guard = fsm_localize._strict_motion_initialization(
        "constant_velocity", [(0, first), (2, last)], 3, 10.0, 180.0
    )
    assert predicted[0, 3] == pytest.approx(1.0)
    assert source == "previous_pose"
    assert step == pytest.approx(0.0)
    assert guard == "nonconsecutive_valid_history"


def test_strict_tracking_acceptance_rejects_converged_low_quality_pose(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)

    def fake_register_pointcloud(**kwargs):
        result = successful_result(translated_pose(5.0), points.shape[0])
        result.update(
            fitness=0.01,
            num_correspondences=100,
            point_to_plane_rmse=0.01,
            condition_number=10.0,
            eigenvalues=np.ones(6),
        )
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "tracking_acceptance": {"min_fitness": 0.5},
    })
    output = tmp_path / "quality"
    summary = run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=2))
    assert summary["num_lost_frames"] == 1
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["converged"] == "True"
    assert rows[1]["accepted"] == "False"
    assert rows[1]["failure_reason"] == "tracking_quality_min_fitness"
    assert rows[1]["acceptance_reason"] == "tracking_quality_min_fitness"


def test_strict_accepts_valid_max_iterations_solution(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)

    def fake_register_pointcloud(**kwargs):
        result = successful_result(translated_pose(0.25), points.shape[0])
        result.update(
            converged=False,
            optimizer_valid=True,
            termination_reason="max_iterations",
            failure_reason=None,
            fitness=1.0,
        )
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    output = tmp_path / "max_iterations"
    summary = run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=2))
    assert summary["num_valid_frames"] == 2
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["converged"] == "False"
    assert rows[1]["optimizer_valid"] == "True"
    assert rows[1]["termination_reason"] == "max_iterations"
    assert rows[1]["accepted"] == "True"


def test_strict_rejects_optimizer_invalid_even_with_high_fitness(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)

    def fake_register_pointcloud(**kwargs):
        result = successful_result(translated_pose(5.0), points.shape[0])
        result.update(optimizer_valid=False, termination_reason="degenerate_geometry", failure_reason="degenerate_geometry")
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    summary = run_localization(config, str(map_path), str(tmp_path / "invalid"), frames=make_frames(points, num_frames=2))
    assert summary["num_lost_frames"] == 1


def test_local_recovery_uses_last_valid_without_current_gt(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=2)
    frames[1]["c2w"] = translated_pose(1000.0).astype(np.float32)
    query_poses = []
    registration_calls = {"count": 0}

    def fake_query(self, c2w, *args, **kwargs):
        query_poses.append(np.asarray(c2w).copy())
        return {
            "indices": np.arange(points.shape[0]),
            "query_latency_ms": 0.0,
            "candidate_count": points.shape[0],
            "frustum_count": points.shape[0],
            "count": points.shape[0],
            "backend": self.backend,
        }

    def fake_register_pointcloud(**kwargs):
        registration_calls["count"] += 1
        result = successful_result(kwargs["init_transform"], points.shape[0])
        if registration_calls["count"] == 1:
            result.update(optimizer_valid=False, converged=False, failure_reason="lm_step_rejected", termination_reason="lm_step_rejected")
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["recovery"] = {"enabled": True, "max_tier": 1, "max_points": points.shape[0]}
    output = tmp_path / "recovery"
    summary = run_localization(config, str(map_path), str(output), frames=frames)
    assert registration_calls["count"] == 2
    assert all(pose[0, 3] != pytest.approx(1000.0) for pose in query_poses)
    assert summary["num_recovered_frames"] == 1
    assert summary["num_tracking_attempt_frames"] == 1
    assert summary["primary_attempt_acceptance_rate"] == pytest.approx(0.0)
    assert summary["primary_tracking_completeness"] == summary["primary_attempt_acceptance_rate"]
    assert summary["final_tracking_completeness"] == pytest.approx(1.0)
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["primary_termination_reason"] == "lm_step_rejected"
    assert rows[1]["recovery_stage"] == "1"
    assert rows[1]["recovered"] == "True"


def test_recovery_does_not_run_for_quality_accepted_max_iterations(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    calls = {"count": 0}

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.arange(points.shape[0]),
            "query_latency_ms": 0.0,
            "candidate_count": points.shape[0],
            "frustum_count": points.shape[0],
            "count": points.shape[0],
            "backend": self.backend,
        }

    def fake_register_pointcloud(**kwargs):
        calls["count"] += 1
        result = successful_result(kwargs["init_transform"], points.shape[0])
        result.update(converged=False, optimizer_valid=True, termination_reason="max_iterations")
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["recovery"] = {"enabled": True, "max_tier": 3, "max_points": points.shape[0]}
    output = tmp_path / "accepted_no_recovery"
    run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=2))
    assert calls["count"] == 1
    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["accepted"] == "True"
    assert rows[1]["recovery_attempted"] == "False"


def test_primary_audit_is_not_overwritten_by_recovery(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    calls = {"count": 0}

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.arange(points.shape[0]), "query_latency_ms": 0.0,
            "candidate_count": points.shape[0], "frustum_count": points.shape[0],
            "count": points.shape[0], "backend": self.backend,
        }

    def fake_register(**kwargs):
        calls["count"] += 1
        result = successful_result(kwargs["init_transform"], points.shape[0])
        result.update(
            num_correspondences=10 + calls["count"], effective_rank=5 + calls["count"],
            condition_number=100.0 / calls["count"], eigenvalues=np.arange(1.0, 7.0),
            hessian=np.diag(np.arange(1.0, 7.0)),
        )
        if calls["count"] == 1:
            result.update(optimizer_valid=False, termination_reason="degenerate_geometry")
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["recovery"] = {"enabled": True, "max_tier": 1, "max_points": points.shape[0]}
    output = tmp_path / "primary_audit"
    run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=2))
    with open(output / "frame_metrics.csv", newline="") as stream:
        row = list(csv.DictReader(stream))[1]
    assert row["primary_optimizer_valid"] == "False"
    assert row["primary_num_correspondences"] == "11"
    assert row["primary_effective_rank"] == "6"
    assert row["num_correspondences"] == "12"


def test_local_submap_uses_init_pose_and_slices_aligned_fields(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=2)
    frames[1]["c2w"] = translated_pose(1000.0).astype(np.float32)
    selected = np.array([1, 4, 7], dtype=np.int64)
    query_poses = []

    def fake_query(self, c2w, intrinsics, image_height, image_width, config, scores=None):
        query_poses.append(np.asarray(c2w).copy())
        return {
            "indices": selected,
            "query_latency_ms": 0.25,
            "candidate_count": 5,
            "frustum_count": 4,
            "count": selected.size,
            "backend": self.backend,
        }

    def fake_register_pointcloud(**kwargs):
        np.testing.assert_allclose(kwargs["target_points"], points[selected])
        assert kwargs["target_covariances"].shape == (selected.size, 3, 3)
        return successful_result(kwargs["init_transform"], selected.size)

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {
        "enabled": True,
        "near": 0.1,
        "far": 3.0,
        "min_points": 2,
        "spatial_backend": "numpy_grid",
        "grid_voxel": 0.5,
    }
    run_localization(config, str(map_path), str(tmp_path / "strict_local"), frames=frames)

    np.testing.assert_allclose(query_poses[0], np.eye(4))
    assert query_poses[0][0, 3] != frames[1]["c2w"][0, 3]


def test_empty_local_submap_marks_lost_without_registration(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.empty(0, dtype=np.int64),
            "query_latency_ms": 0.1,
            "candidate_count": 0,
            "frustum_count": 0,
            "count": 0,
            "backend": self.backend,
        }

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(
        fsm_localize,
        "register_pointcloud",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("full-map fallback is forbidden")),
    )
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    output_dir = tmp_path / "strict_empty_local"
    summary = run_localization(config, str(map_path), str(output_dir), frames=make_frames(points, num_frames=2))

    assert summary["num_lost_frames"] == 1
    with open(output_dir / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["failure_reason"] == "empty_local_submap"
    assert rows[1]["local_target_count"] == "0"


def test_strict_writes_long_horizon_files_and_standard_json(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    monkeypatch.setattr(
        fsm_localize,
        "register_pointcloud",
        lambda **kwargs: successful_result(kwargs["init_transform"], points.shape[0]),
    )
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["eval"].update({"frame_deltas": [1], "distance_deltas_m": [1.0], "segment_lengths_m": [1.0]})
    output_dir = tmp_path / "strict_metrics"
    summary = run_localization(config, str(map_path), str(output_dir), frames=make_frames(points, num_frames=3))

    assert (output_dir / "long_horizon_metrics.json").exists()
    assert (output_dir / "per_frame_errors.csv").exists()
    assert (output_dir / "segment_errors.csv").exists()
    json.loads((output_dir / "long_horizon_metrics.json").read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    json.loads((output_dir / "summary.json").read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert summary["raw_ate_rmse_m"] == pytest.approx(0.0)
    assert summary["aligned_ate_rmse_m"] == pytest.approx(0.0)
    assert summary["ate_rmse"] == summary["aligned_ate_rmse_m"]
    assert summary["total_frame_latency_mean_ms"] >= 0.0
    assert summary["total_frame_latency_p95_ms"] >= 0.0
    with open(output_dir / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(float(row["total_frame_latency_ms"]) >= 0.0 for row in rows)
    assert float(rows[0]["total_frame_latency_ms"]) >= 0.0


def test_strict_propagates_local_coverage_diagnostics(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)

    def fake_query(self, *args, **kwargs):
        return {
            "indices": np.array([0, 1], dtype=np.int64),
            "query_latency_ms": 1.0,
            "radius_latency_ms": 0.2,
            "projection_latency_ms": 0.3,
            "sampling_latency_ms": 0.4,
            "candidate_count": 10,
            "positive_depth_count": 9,
            "depth_range_count": 8,
            "precap_count": 7,
            "frustum_count": 7,
            "count": 2,
            "nonempty_tile_count": 4,
            "selected_tile_count": 4,
            "nonempty_depth_bin_count": 3,
            "selected_depth_bin_count": 3,
            "nonempty_tile_depth_count": 6,
            "selected_tile_depth_count": 5,
            "backend": self.backend,
        }

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", fake_query)
    monkeypatch.setattr(
        fsm_localize,
        "register_pointcloud",
        lambda **kwargs: successful_result(kwargs["init_transform"], kwargs["target_points"].shape[0]),
    )
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    output = tmp_path / "coverage"
    summary = run_localization(config, str(map_path), str(output), frames=make_frames(points, num_frames=2))

    with open(output / "frame_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["local_nonempty_tile_count"] == "4"
    assert rows[1]["local_selected_tile_depth_count"] == "5"
    assert summary["local_tile_coverage_mean"] == pytest.approx(1.0)
    assert summary["local_depth_bin_coverage_mean"] == pytest.approx(1.0)
    assert summary["local_tile_depth_coverage_mean"] == pytest.approx(5 / 6)


def test_audit_header_and_summary_schema_remain_frozen(tmp_path):
    map_path, points = make_fake_map(tmp_path)
    output_dir = tmp_path / "audit"
    summary = run_localization(make_config(), str(map_path), str(output_dir), frames=make_frames(points))
    with open(output_dir / "frame_metrics.csv", newline="") as f:
        header = next(csv.reader(f))
    assert header == fsm_localize.AUDIT_FRAME_METRIC_FIELDS
    assert set(summary) == {
        "ate_rmse",
        "rpe_trans_rmse",
        "rpe_rot_rmse_deg",
        "latency_mean_ms",
        "latency_p95_ms",
        "target_index_build_ms",
        "target_index_backend",
        "cold_start_latency_ms",
        "map_path",
        "backend",
        "N_star",
        "num_frames",
        "init_mode",
        "num_fallbacks",
    }


def test_tracking_summary_excludes_initialization_and_keeps_primary_acceptance_when_reserve_used():
    rows = [
        {"tracking_valid": True, "reserve_primary_accepted": None, "reserve_used": False, "recovered": False},
        {"tracking_valid": True, "reserve_primary_accepted": True, "reserve_used": True, "recovered": False},
        {"tracking_valid": True, "reserve_primary_accepted": True, "reserve_used": False, "recovered": False},
        {"tracking_valid": False, "reserve_primary_accepted": False, "reserve_used": False, "recovered": False},
    ]
    summary = fsm_localize._tracking_summary_counts(rows)
    assert summary == {
        "num_tracking_attempt_frames": 3,
        "num_primary_accepted_tracking_frames": 2,
        "num_base_only_final_frames": 1,
        "num_final_complete_tracking_frames": 2,
        "primary_attempt_acceptance_rate": pytest.approx(2 / 3),
        "primary_tracking_completeness": pytest.approx(2 / 3),
        "base_only_final_fraction": pytest.approx(1 / 3),
        "final_tracking_completeness": pytest.approx(2 / 3),
    }


def test_information_v3_localizer_evaluates_after_refinement_and_commits_only_final_use():
    source = inspect.getsource(fsm_localize.run_localization)
    refinement_index = source.index("refinement = relative_se3_log(primary_pose, candidate_pose)")
    evaluation_index = source.index("observable_reserve.evaluate_improvement(")
    branch_commit_index = source.index("observable_reserve.commit_branch_adoption(")
    record_index = source.index("observable_reserve.add_record(reserve_metrics)")
    assert refinement_index < evaluation_index < branch_commit_index < record_index
    assert "primary_result, candidate_result, reserve_metrics" in source
    assert "bool(reserve_metrics[\"reserve_used\"])" in source[branch_commit_index:record_index]


def test_information_v4_localizer_injects_temporal_rotation_before_evaluation_and_exports_fields():
    source = inspect.getsource(fsm_localize.run_localization)
    temporal_index = source.index('"reserve_candidate_temporal_rotation_deg"')
    evaluation_index = source.index("observable_reserve.evaluate_improvement(")
    assert temporal_index < evaluation_index
    assert 'candidate_consistency_metrics.get(\n                            "temporal_rotation_deg"' in source
    for field in (
        "reserve_candidate_temporal_rotation_deg",
        "reserve_information_v4_candidate_init_translation_pass",
        "reserve_information_v4_candidate_init_rotation_pass",
        "reserve_information_v4_joint_rotation_pass",
        "reserve_information_v4_prospective_global_streak",
        "reserve_information_v4_global_cooldown_remaining_frames",
        "reserve_information_v4_accepted",
        "reserve_information_v4_reason",
    ):
        assert field in fsm_localize.STRICT_FRAME_METRIC_FIELDS


def test_voxel_block_build_is_counted_in_cold_start_and_not_first_query(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=2)
    config = make_config()
    config["registration"].update({
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "normal_source": "map",
    })
    config["local_submap"] = {
        "enabled": True, "near": 0.1, "far": 3.0, "min_points": 1,
        "frustum_culling": "voxel_blocks", "block_size_m": 0.3,
        "spatial_backend": "numpy_grid",
    }
    config["recovery"] = {"enabled": False}
    config["eval"].update({"frame_deltas": [1], "distance_deltas_m": [1.0], "segment_lengths_m": [1.0]})
    real_prebuild = fsm_localize.MapSpatialIndex.prebuild_blocks
    builds = []

    def counted_prebuild(self, *args, **kwargs):
        result = real_prebuild(self, *args, **kwargs)
        builds.append(result["cached"])
        self._score_array_and_fingerprint = lambda *unused_args, **unused_kwargs: (
            (_ for _ in ()).throw(AssertionError("localizer query unexpectedly rehashed scores"))
        )
        return result

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "prebuild_blocks", counted_prebuild)
    monkeypatch.setattr(
        fsm_localize,
        "register_pointcloud",
        lambda **kwargs: successful_result(kwargs["init_transform"], kwargs["target_points"].shape[0]),
    )
    summary = run_localization(config, str(map_path), str(tmp_path / "block_start"), frames=frames)
    assert builds == [False]
    assert summary["block_index_build_ms"] >= 0.0
    assert summary["block_score_hash_ms"] >= 0.0
    assert summary["block_geometry_sort_ms"] >= 0.0
    assert summary["block_index_memory_bytes"] > 0
    assert summary["block_peak_working_set_estimate_bytes"] >= summary["block_index_memory_bytes"]
    assert summary["cold_start_latency_ms"] >= summary["local_index_build_ms"] + summary["block_index_build_ms"]


def test_activation_diagnostics_is_independent_sidecar_and_preserves_outputs(tmp_path, monkeypatch):
    map_path, geometry_path, points = make_activation_map(tmp_path)
    frames = make_frames(points, num_frames=2)
    config = make_config()
    config["registration"].update({
        "backend": "numpy_robust_point_to_plane",
        "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
        "normal_source": "map",
        "min_correspondences": 1,
    })
    config["local_submap"] = {
        "enabled": True, "near": 0.1, "far": 2.0, "min_points": 1,
        "spatial_backend": "numpy_grid", "grid_voxel": 0.5,
    }
    config["eval"].update({"frame_deltas": [1], "distance_deltas_m": [1.0], "segment_lengths_m": [1.0]})

    def fake_register_pointcloud(**kwargs):
        result = successful_result(kwargs["init_transform"], kwargs["target_points"].shape[0])
        result.update(
            backend="numpy_robust_point_to_plane", num_correspondences=1,
            point_to_plane_rmse=0.0, condition_number=10.0,
            eigenvalues=np.ones(6), pose_covariance=np.eye(6), iterations=1,
        )
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    disabled = tmp_path / "disabled"
    enabled = tmp_path / "enabled"
    disabled_summary = run_localization(config, str(map_path), str(disabled), frames=frames)
    config["activation_diagnostics"] = {
        "enabled": True,
        "geometry_path": str(geometry_path),
        "counterfactual": True,
        "trigger": {"mode": "plan_only", "condition_number": 1.0},
        "reserve_budget": [1, 2],
        "weak_subspace_dim": 1,
    }
    enabled_summary = run_localization(config, str(map_path), str(enabled), frames=frames)
    timing_summary_fields = {
        "local_index_build_ms", "local_query_latency_mean_ms", "local_query_latency_p95_ms",
        "local_radius_latency_mean_ms", "local_projection_latency_mean_ms",
        "local_sampling_latency_mean_ms", "total_frame_latency_mean_ms",
        "total_frame_latency_p95_ms", "production_equivalent_total_frame_latency_mean_ms",
        "production_equivalent_total_frame_latency_p95_ms", "cold_start_latency_ms",
        "rss_mb", "peak_rss_mb",
    }
    assert {
        key: value for key, value in disabled_summary.items() if key not in timing_summary_fields
    } == {
        key: value for key, value in enabled_summary.items() if key not in timing_summary_fields
    }
    assert (disabled / "trajectory_est.txt").read_bytes() == (enabled / "trajectory_est.txt").read_bytes()
    with open(disabled / "frame_metrics.csv", newline="") as stream:
        disabled_rows = list(csv.DictReader(stream))
    with open(enabled / "frame_metrics.csv", newline="") as stream:
        enabled_rows = list(csv.DictReader(stream))
    timing_frame_fields = {
        "local_query_latency_ms", "local_radius_latency_ms", "local_projection_latency_ms",
        "local_sampling_latency_ms", "total_frame_latency_ms", "local_index_build_ms",
        "production_equivalent_total_frame_latency_ms",
        "rss_mb", "peak_rss_mb",
    }
    assert [
        {key: value for key, value in row.items() if key not in timing_frame_fields}
        for row in disabled_rows
    ] == [
        {key: value for key, value in row.items() if key not in timing_frame_fields}
        for row in enabled_rows
    ]
    archive = np.load(enabled / "runtime_activation_diagnostics.npz")
    assert archive["actual_replacement_hessian"].shape == (2, 6, 6)
    assert archive["counterfactual_removed_hessian"].shape == (2, 6, 6)
    assert archive["reserve_predicted_weak_gain"].shape == (2, 2)
    assert np.all(archive["trigger"])
    activation_summary = json.loads(
        (enabled / "runtime_activation_summary.json").read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert activation_summary["counterfactual_scope"] == "preserved_selected_rows_plus_removed_geometry"
    assert activation_summary["reserve_source"] == "replacement_set"
    assert activation_summary["replacement_minus_removed_local_activation_rate"] is not None
    assert activation_summary["replacement_minus_removed_correspondence_rate"] is not None
    assert activation_summary["observed_projection_input_count_p95"] >= 0
    assert activation_summary["observed_frustum_count_p95"] >= 0
    assert activation_summary["projection_pre_cap_sweep"] == []


def test_runtime_memory_metrics_and_guard(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    config = make_config()
    config["registration"].update({"evaluation_protocol": "mvp_strict", "first_pose_protocol": "identity"})
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 3.0, "min_points": 1}
    config["runtime_memory_metrics"] = {"enabled": True, "check_interval_frames": 1, "max_rss_mb": 100.0}
    monkeypatch.setattr(fsm_localize, "get_process_memory_mb", lambda: (80.0, 90.0))
    summary = run_localization(config, str(map_path), str(tmp_path / "memory"), frames=make_frames(points, 2))
    assert summary["rss_mb"] == 80.0
    assert summary["peak_rss_mb"] == 90.0
    with open(tmp_path / "memory/frame_metrics.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["rss_mb"] == "80.0" and row["peak_rss_mb"] == "90.0" for row in rows)

    monkeypatch.setattr(fsm_localize, "get_process_memory_mb", lambda: (101.0, 101.0))
    with pytest.raises(MemoryError, match="exceeds configured max_rss_mb"):
        run_localization(config, str(map_path), str(tmp_path / "guard"), frames=make_frames(points, 1))


def test_activation_diagnostics_no_sidecar_and_no_gt_dependence(tmp_path, monkeypatch):
    map_path, _, points = make_activation_map(tmp_path)
    (tmp_path / "activation_geometry.npz").unlink()
    frames_a = make_frames(points, num_frames=2)
    frames_b = make_frames(points, num_frames=2)
    frames_b[1]["c2w"] = translated_pose(999.0).astype(np.float32)
    config = make_config()
    config["registration"].update({
        "backend": "numpy_robust_point_to_plane", "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity", "normal_source": "map", "min_correspondences": 1,
    })
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 2.0, "min_points": 1}
    config["eval"].update({"frame_deltas": [1], "distance_deltas_m": [1.0], "segment_lengths_m": [1.0]})
    config["activation_diagnostics"] = {"enabled": True, "geometry_path": None, "counterfactual": False}

    def fake_register_pointcloud(**kwargs):
        result = successful_result(kwargs["init_transform"], kwargs["target_points"].shape[0])
        result.update(backend="numpy_robust_point_to_plane", num_correspondences=1, point_to_plane_rmse=0.0,
                      condition_number=10.0, eigenvalues=np.ones(6), pose_covariance=np.eye(6), iterations=1)
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", fake_register_pointcloud)
    out_a, out_b = tmp_path / "nogt_a", tmp_path / "nogt_b"
    run_localization(config, str(map_path), str(out_a), frames=frames_a)
    run_localization(config, str(map_path), str(out_b), frames=frames_b)
    summary = json.loads((out_a / "runtime_activation_summary.json").read_text())
    assert summary["role_available"] is False
    assert summary["role_unavailable_reason"] == "activation_geometry_unavailable"
    for key in np.load(out_a / "runtime_activation_diagnostics.npz").files:
        if "latency" not in key:
            np.testing.assert_array_equal(
                np.load(out_a / "runtime_activation_diagnostics.npz")[key],
                np.load(out_b / "runtime_activation_diagnostics.npz")[key],
            )


def test_activation_geometry_mismatch_fails_before_tracking(tmp_path):
    map_path, geometry_path, points = make_activation_map(tmp_path)
    geometry = dict(np.load(geometry_path))
    geometry["selected_global_ids"] = geometry["selected_global_ids"][::-1]
    np.savez(geometry_path, **geometry)
    config = make_config()
    config["registration"].update({
        "backend": "numpy_robust_point_to_plane", "evaluation_protocol": "mvp_strict",
        "first_pose_protocol": "identity",
    })
    config["local_submap"] = {"enabled": True, "near": 0.1, "far": 2.0, "min_points": 1}
    config["activation_diagnostics"] = {"enabled": True, "geometry_path": str(geometry_path)}
    with pytest.raises(ValueError, match="exactly match"):
        run_localization(config, str(map_path), str(tmp_path / "mismatch"), frames=make_frames(points, 1))
