import os
import importlib.util

import numpy as np
import pytest

from utils.fsm_registration import (
    compose_transform,
    estimate_normals,
    invert_transform,
    register_pointcloud,
    transform_points,
)


OPEN3D_AVAILABLE = importlib.util.find_spec("open3d") is not None
RUN_OPEN3D_REGISTRATION = os.environ.get("FSM_RUN_OPEN3D_TESTS") == "1"


def make_asymmetric_points():
    xs, ys, zs = np.meshgrid(
        np.linspace(-0.2, 0.3, 4),
        np.linspace(-0.1, 0.25, 4),
        np.linspace(0.0, 0.2, 3),
        indexing="xy",
    )
    points = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=1)
    points[:, 2] += 0.1 * points[:, 0] + 0.05 * points[:, 1]
    return points.astype(np.float64)


def make_transform(translation=(0.0, 0.0, 0.0), yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def assert_transform_close(actual, expected, trans_tol=1e-2, rot_tol=1e-2):
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=trans_tol)
    np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=rot_tol)


def test_invert_transform_roundtrip():
    T = make_transform(translation=(0.1, -0.2, 0.3), yaw_deg=15.0)
    identity = compose_transform(T, invert_transform(T))
    np.testing.assert_allclose(identity, np.eye(4), atol=1e-8)


def test_transform_points_applies_transform():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    T = make_transform(translation=(1.0, 2.0, 3.0), yaw_deg=90.0)
    transformed = transform_points(points, T)
    expected = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 3.0]], dtype=np.float64)
    np.testing.assert_allclose(transformed, expected, atol=1e-8)


def test_transform_points_rejects_bad_shape():
    with pytest.raises(ValueError):
        transform_points(np.zeros((3, 2)), np.eye(4))


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D is not installed")
def test_estimate_normals_shape():
    points = make_asymmetric_points()
    normals = estimate_normals(points, radius=1.0, max_nn=20)
    assert normals.shape == points.shape
    assert np.all(np.isfinite(normals))


def test_register_identity_numpy_point_to_point():
    points = make_asymmetric_points()
    result = register_pointcloud(
        points,
        points,
        backend="numpy_point_to_point",
        max_correspondence_distance=0.2,
        max_iterations=50,
    )
    assert result["converged"]
    assert result["fitness"] > 0.99
    assert_transform_close(result["transform"], np.eye(4), trans_tol=1e-6, rot_tol=1e-6)


def test_register_known_translation_numpy_point_to_point():
    target = make_asymmetric_points()
    true_source_to_target = make_transform(translation=(0.03, -0.02, 0.01), yaw_deg=0.0)
    source = transform_points(target, invert_transform(true_source_to_target))
    result = register_pointcloud(
        source,
        target,
        init_transform=np.eye(4),
        backend="numpy_point_to_point",
        max_correspondence_distance=0.2,
        max_iterations=80,
    )
    assert result["converged"]
    assert_transform_close(result["transform"], true_source_to_target, trans_tol=1e-3, rot_tol=1e-3)


def test_register_known_rotation_numpy_point_to_point_with_good_init():
    target = make_asymmetric_points()
    true_source_to_target = make_transform(translation=(0.01, 0.0, 0.0), yaw_deg=8.0)
    source = transform_points(target, invert_transform(true_source_to_target))
    init = make_transform(translation=(0.0, 0.0, 0.0), yaw_deg=5.0)
    result = register_pointcloud(
        source,
        target,
        init_transform=init,
        backend="numpy_point_to_point",
        max_correspondence_distance=0.2,
        max_iterations=100,
    )
    assert result["converged"]
    assert_transform_close(result["transform"], true_source_to_target, trans_tol=5e-3, rot_tol=5e-2)


def test_register_numpy_point_to_point_handles_large_target_without_broadcast():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(50000, 3)).astype(np.float64)
    source = target[:200].copy()
    result = register_pointcloud(
        source,
        target,
        init_transform=np.eye(4),
        backend="numpy_point_to_point",
        max_correspondence_distance=1e-6,
        max_iterations=1,
    )
    assert result["converged"]
    assert result["fitness"] == 1.0
    assert result["num_target_points"] == target.shape[0]


@pytest.mark.skipif(
    not (OPEN3D_AVAILABLE and RUN_OPEN3D_REGISTRATION),
    reason="Open3D registration smoke test requires FSM_RUN_OPEN3D_TESTS=1",
)
def test_register_point_to_plane_smoke():
    target = make_asymmetric_points()
    true_source_to_target = make_transform(translation=(0.01, -0.01, 0.0), yaw_deg=3.0)
    source = transform_points(target, invert_transform(true_source_to_target))
    result = register_pointcloud(
        source,
        target,
        init_transform=np.eye(4),
        backend="open3d_point_to_plane",
        max_correspondence_distance=0.2,
        max_iterations=50,
        normal_radius=1.0,
    )
    assert result["backend"] == "open3d_point_to_plane"
    assert result["num_source_points"] > 0
    assert result["num_target_points"] > 0
    assert np.isfinite(result["inlier_rmse"])
    assert result["latency_ms"] >= 0.0


def test_register_rejects_unknown_backend():
    points = make_asymmetric_points()
    with pytest.raises(ValueError):
        register_pointcloud(points, points, backend="missing")


def test_register_rejects_bad_init_shape():
    points = make_asymmetric_points()
    with pytest.raises(ValueError):
        register_pointcloud(points, points, init_transform=np.eye(3), backend="open3d_point_to_point")
