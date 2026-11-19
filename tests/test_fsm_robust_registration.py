import numpy as np
import pytest

import utils.fsm_robust_registration as registration

from utils.fsm_robust_registration import (
    aggregate_correspondence_hessian,
    evaluate_point_to_plane_diagnostics,
    point_to_plane_residual_jacobian,
    register_robust_point_to_plane,
    se3_exp,
    so3_exp,
    transform_points,
)


def make_three_plane_scene(grid_size=9):
    values = np.linspace(-0.6, 0.6, grid_size)
    a, b = np.meshgrid(values, values, indexing="ij")
    zeros = np.zeros_like(a)
    xy = np.stack([a, b, zeros], axis=-1).reshape(-1, 3)
    yz = np.stack([zeros, a, b], axis=-1).reshape(-1, 3)
    xz = np.stack([a, zeros, b], axis=-1).reshape(-1, 3)
    points = np.concatenate([xy, yz, xz], axis=0)
    normals = np.concatenate(
        [
            np.tile([0.0, 0.0, 1.0], (xy.shape[0], 1)),
            np.tile([1.0, 0.0, 0.0], (yz.shape[0], 1)),
            np.tile([0.0, 1.0, 0.0], (xz.shape[0], 1)),
        ],
        axis=0,
    )
    return points.astype(np.float64), normals.astype(np.float64)


def transform_error(estimate, truth):
    rotation_error = estimate[:3, :3] @ truth[:3, :3].T
    cosine = np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cosine)
    translation = np.linalg.norm(estimate[:3, 3] - truth[:3, 3])
    return float(angle + translation)


def test_so3_and_se3_exp_known_cases():
    rotation = so3_exp(np.array([0.0, 0.0, np.pi / 2.0]))
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) > 0.999999

    transform = se3_exp(np.array([0.0, 0.0, 0.0, 1.0, -2.0, 3.0]))
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], [1.0, -2.0, 3.0], atol=1e-12)


def test_left_point_to_plane_jacobian_matches_finite_difference():
    source = np.array([[0.4, -0.2, 0.7], [-0.1, 0.8, 0.3]], dtype=np.float64)
    target = np.array([[0.3, -0.1, 0.6], [0.0, 0.7, 0.4]], dtype=np.float64)
    normals = np.array([[0.2, -0.3, 0.9], [-0.7, 0.4, 0.1]], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    residuals, jacobian = point_to_plane_residual_jacobian(source, target, normals)
    epsilon = 1e-7
    numerical = np.empty_like(jacobian)
    for column in range(6):
        step = np.zeros(6)
        step[column] = epsilon
        perturbed = transform_points(source, se3_exp(step))
        next_residuals, _ = point_to_plane_residual_jacobian(perturbed, target, normals)
        numerical[:, column] = (next_residuals - residuals) / epsilon
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6, rtol=1e-6)


def test_recovers_known_small_transform():
    target, normals = make_three_plane_scene()
    truth = se3_exp(np.array([0.018, -0.014, 0.012, 0.025, -0.018, 0.02]))
    source = transform_points(target, np.linalg.inv(truth))
    result = register_robust_point_to_plane(
        source,
        target,
        target_normals=normals,
        max_correspondence_distance=0.12,
        max_iterations=60,
        loss="none",
    )
    assert result["converged"], result["failure_reason"]
    assert result["effective_rank"] == 6
    assert result["pose_covariance_valid"]
    assert result["hessian"].shape == (6, 6)
    assert result["gradient"].shape == (6,)
    assert result["eigenvalues"].shape == (6,)
    assert np.isfinite(result["condition_number"])
    assert transform_error(result["transform"], truth) < 2e-3


@pytest.mark.parametrize("loss", ["none", "huber", "cauchy"])
def test_zero_residual_full_rank_warm_start_converges(loss):
    target, normals = make_three_plane_scene()
    warm_start = se3_exp(np.array([0.0, 0.0, 0.0, 0.025, -0.018, 0.02]))
    source = target - warm_start[:3, 3]
    result = register_robust_point_to_plane(
        source,
        target,
        init_transform=warm_start,
        target_normals=normals,
        loss=loss,
        min_iterations=3,
        convergence_patience=2,
    )
    assert result["converged"]
    assert result["optimizer_valid"]
    assert result["failure_reason"] is None
    assert result["termination_reason"] == "converged"
    assert result["effective_rank"] == 6
    assert result["iterations"] == 3
    assert result["accepted_step_count"] == 0
    assert result["rejected_lm_trials"] == 0
    assert result["final_mean_robust_cost"] == 0.0
    assert result["final_step_rotation_deg"] == 0.0
    assert result["final_step_translation_m"] == 0.0
    assert result["last_relative_reduction"] == 0.0
    np.testing.assert_array_equal(result["transform"], warm_start)


def test_near_stationary_nonzero_residual_converges_without_moving():
    rng = np.random.default_rng(42)
    target = rng.normal(size=(100, 3))
    normals = rng.normal(size=target.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    _, jacobian = point_to_plane_residual_jacobian(target, target, normals)
    residuals = rng.normal(scale=1e-3, size=target.shape[0])
    residuals -= jacobian @ np.linalg.lstsq(jacobian, residuals, rcond=None)[0]
    source = target + residuals[:, None] * normals + np.array([1e-12, 0.0, 0.0])
    result = register_robust_point_to_plane(
        source, target, target_normals=normals, loss="none"
    )
    assert result["converged"]
    assert result["optimizer_valid"]
    assert result["failure_reason"] is None
    assert result["effective_rank"] == 6
    assert result["final_mean_robust_cost"] > 0.0
    assert 0.0 < np.linalg.norm(result["gradient"]) < 1e-9
    assert result["accepted_step_count"] == 0
    assert result["rejected_lm_trials"] == 0
    np.testing.assert_array_equal(result["transform"], np.eye(4))


def test_nonstationary_rejected_trials_remain_invalid(monkeypatch):
    target, normals = make_three_plane_scene()
    source = target + np.array([0.01, -0.01, 0.01])
    monkeypatch.setattr(registration, "_robust_cost", lambda *args: 1.0)
    result = register_robust_point_to_plane(
        source, target, target_normals=normals, lm_damping=1e12, loss="none"
    )
    assert not result["converged"]
    assert not result["optimizer_valid"]
    assert result["effective_rank"] == 6
    assert result["failure_reason"] == "lm_step_rejected"
    assert result["termination_reason"] == "lm_step_rejected"
    assert result["accepted_step_count"] == 0
    assert result["rejected_lm_trials"] == 10
    assert np.linalg.norm(result["gradient"]) > 0.1
    np.testing.assert_array_equal(result["transform"], np.eye(4))


def test_large_damping_small_accepted_steps_do_not_imply_convergence():
    target, normals = make_three_plane_scene()
    source = target + np.array([0.01, -0.01, 0.01])
    result = register_robust_point_to_plane(
        source,
        target,
        target_normals=normals,
        lm_damping=1e12,
        max_iterations=2,
        loss="none",
    )
    assert result["accepted_step_count"] == 2
    assert result["final_step_rotation_deg"] < 0.01
    assert result["final_step_translation_m"] < 1e-4
    assert result["last_relative_reduction"] < 1e-5
    assert np.linalg.norm(result["gradient"]) > 0.1
    assert not result["converged"]
    assert result["optimizer_valid"]
    assert result["termination_reason"] == "max_iterations"


def test_cauchy_is_more_robust_to_outliers_than_none():
    rng = np.random.default_rng(23)
    target, normals = make_three_plane_scene(grid_size=11)
    truth = se3_exp(np.array([0.012, -0.01, 0.008, 0.02, -0.015, 0.018]))
    clean_source = transform_points(target, np.linalg.inv(truth))
    outlier_indices = rng.choice(clean_source.shape[0], size=90, replace=False)
    source = clean_source.copy()
    source[outlier_indices] += rng.normal(scale=0.12, size=(outlier_indices.size, 3))
    common = dict(
        source_points=source,
        target_points=target,
        target_normals=normals,
        max_correspondence_distance=0.35,
        max_iterations=80,
    )
    plain = register_robust_point_to_plane(loss="none", **common)
    robust = register_robust_point_to_plane(loss="cauchy", loss_scale=0.025, **common)
    robust_error = transform_error(robust["transform"], truth)
    plain_error = transform_error(plain["transform"], truth)
    assert robust["converged"], robust["failure_reason"]
    assert robust_error < 0.025
    assert robust_error < plain_error * 0.65


def test_single_plane_reports_degeneracy():
    values = np.linspace(-1.0, 1.0, 10)
    x, y = np.meshgrid(values, values, indexing="ij")
    target = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (target.shape[0], 1))
    result = register_robust_point_to_plane(
        target.copy(),
        target,
        target_normals=normals,
        max_correspondence_distance=0.3,
    )
    assert not result["converged"]
    assert result["failure_reason"] == "degenerate_geometry"
    assert result["termination_reason"] == "degenerate_geometry"
    assert not result["optimizer_valid"]
    assert result["effective_rank"] == 3
    assert np.isinf(result["condition_number"])
    assert not result["pose_covariance_valid"]
    assert np.all(np.isfinite(result["pose_covariance"]))


def test_insufficient_correspondences_has_failure_reason():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    target = source.copy()
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    result = register_robust_point_to_plane(
        source,
        target,
        target_normals=normals,
        max_correspondence_distance=0.1,
        min_correspondences=3,
    )
    assert not result["converged"]
    assert result["failure_reason"] == "insufficient_correspondences"
    assert result["num_correspondences"] == 2
    assert result["effective_rank"] == 0


def test_max_iterations_is_valid_without_convergence():
    target, normals = make_three_plane_scene()
    truth = se3_exp(np.array([0.018, -0.014, 0.012, 0.025, -0.018, 0.02]))
    source = transform_points(target, np.linalg.inv(truth))
    result = register_robust_point_to_plane(
        source,
        target,
        target_normals=normals,
        max_correspondence_distance=0.12,
        max_iterations=1,
        loss="none",
    )
    assert not result["converged"]
    assert result["optimizer_valid"]
    assert result["termination_reason"] == "max_iterations"
    assert result["failure_reason"] is None
    assert result["accepted_step_count"] == 1
    assert result["rejected_lm_trials"] == 0
    assert np.isfinite(result["initial_mean_robust_cost"])
    assert np.isfinite(result["final_mean_robust_cost"])
    assert np.isfinite(result["final_step_rotation_deg"])
    assert np.isfinite(result["final_step_translation_m"])


def test_true_optimizer_failure_is_invalid():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    result = register_robust_point_to_plane(
        source,
        source,
        target_normals=normals,
        min_correspondences=3,
    )
    assert not result["optimizer_valid"]
    assert result["termination_reason"] == "insufficient_correspondences"
    assert result["failure_reason"] == "insufficient_correspondences"


def test_fixed_pose_diagnostics_correspondences_and_hessian_identity():
    target, normals = make_three_plane_scene(grid_size=5)
    source = target.copy()
    source_ids = np.arange(source.shape[0], dtype=np.int32) + 1000
    target_ids = np.array([f"map-{index}" for index in range(target.shape[0])])
    common = dict(
        source_points=source,
        target_points=target,
        target_normals=normals,
        transform=np.eye(4),
        max_correspondence_distance=1e-6,
        loss="cauchy",
        loss_scale=0.03,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    aggregate_only = evaluate_point_to_plane_diagnostics(**common)
    detailed = evaluate_point_to_plane_diagnostics(
        **common, return_correspondences=True
    )

    detail_keys = {
        "source_indices",
        "target_indices",
        "source_ids",
        "target_ids",
        "distances",
        "residuals",
        "weights",
        "jacobians",
    }
    assert detail_keys.isdisjoint(aggregate_only)
    assert detail_keys.issubset(detailed)
    for key in aggregate_only:
        if isinstance(aggregate_only[key], np.ndarray):
            np.testing.assert_allclose(aggregate_only[key], detailed[key], equal_nan=True)
        else:
            assert aggregate_only[key] == detailed[key]
    np.testing.assert_array_equal(
        detailed["source_ids"], source_ids[detailed["source_indices"]]
    )
    np.testing.assert_array_equal(
        detailed["target_ids"], target_ids[detailed["target_indices"]]
    )
    expected_hessian = np.einsum(
        "n,ni,nj->ij",
        detailed["weights"],
        detailed["jacobians"],
        detailed["jacobians"],
    )
    np.testing.assert_allclose(expected_hessian, detailed["hessian"], atol=1e-12)


def test_aggregate_correspondence_hessian_mask_and_metrics():
    jacobians = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        ]
    )
    weights = np.array([2.0, 3.0, 4.0])
    result = aggregate_correspondence_hessian(
        jacobians, weights, mask=np.array([True, False, True])
    )
    expected = np.diag([2.0, 0.0, 36.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["hessian"], expected)
    np.testing.assert_allclose(result["eigenvalues"], np.linalg.eigvalsh(expected))
    assert result["trace"] == 38.0
    assert result["rank"] == 2
    assert result["effective_rank"] == 2
    assert result["lambda_min"] == 0.0
    assert result["lambda_max"] == 36.0
    assert result["eigenvectors"].shape == (6, 6)


def test_fixed_pose_insufficient_correspondences_returns_typed_empty_arrays():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = source.copy()
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    result = evaluate_point_to_plane_diagnostics(
        source,
        target,
        normals,
        np.eye(4),
        max_correspondence_distance=0.1,
        min_correspondences=3,
        source_ids=np.array([10, 11], dtype=np.uint32),
        target_ids=np.array(["a", "b"]),
        return_correspondences=True,
    )
    assert result["num_correspondences"] == 2
    assert result["source_indices"].shape == (0,)
    assert result["source_indices"].dtype == np.int64
    assert result["target_indices"].dtype == np.int64
    assert result["source_ids"].dtype == np.uint32
    assert result["target_ids"].dtype.kind == "U"
    for key in ("distances", "residuals", "weights"):
        assert result[key].shape == (0,)
        assert result[key].dtype == np.float64
    assert result["jacobians"].shape == (0, 6)
    assert result["jacobians"].dtype == np.float64
