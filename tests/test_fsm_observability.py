import numpy as np
import pytest
from unittest import mock

from utils.fsm_observability import (
    ReferenceViews,
    VisibilityFactors,
    build_frustum_center_factors,
    compare_replacement_information,
    load_reference_views,
    select_observable_gaussians,
    trajectory_trace_scores,
    weak_direction_score,
)


def make_views(w2c=None):
    if w2c is None:
        w2c = np.eye(4, dtype=np.float64)[None]
    return ReferenceViews(
        w2c=np.asarray(w2c, dtype=np.float64),
        intrinsics=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        width=100,
        height=100,
        frame_ids=np.arange(len(w2c)),
    )


def make_factors(jacobians, gaussian_ids=None, weights=None, num_gaussians=None, pose_ptr=None):
    jacobians = np.asarray(jacobians, dtype=np.float64)
    if gaussian_ids is None:
        gaussian_ids = np.arange(jacobians.shape[0])
    gaussian_ids = np.asarray(gaussian_ids, dtype=np.int64)
    if weights is None:
        weights = np.ones(jacobians.shape[0])
    if num_gaussians is None:
        num_gaussians = int(np.max(gaussian_ids)) + 1
    if pose_ptr is None:
        pose_ptr = [0, jacobians.shape[0]]
    return VisibilityFactors(
        pose_ptr=np.asarray(pose_ptr, dtype=np.int64),
        gaussian_ids=gaussian_ids,
        jacobians=jacobians,
        weights=np.asarray(weights, dtype=np.float64),
        num_gaussians=num_gaussians,
    )


def test_reference_loader_uses_estimated_keyframes_not_ground_truth():
    params = {
        "cam_unnorm_rots": np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:, None], (1, 3))[None],
        "cam_trans": np.array([[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "intrinsics": np.eye(3),
        "org_width": 10,
        "org_height": 10,
        "keyframe_time_indices": np.array([0, 2]),
        "gt_w2c_all_frames": np.repeat((99.0 * np.eye(4))[None], 3, axis=0),
    }
    views = load_reference_views(params)
    np.testing.assert_array_equal(views.frame_ids, [0, 2])
    np.testing.assert_allclose(views.w2c[:, 0, 3], [0.0, 2.0])
    assert "params_keyframes" in views.source


def test_visibility_and_pair_caps_are_deterministic():
    means = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [2.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    normals = np.tile([0.0, 0.0, 1.0], (means.shape[0], 1))
    factors = build_frustum_center_factors(means, normals, make_views(), pair_cap=1, chunk_size=2)
    np.testing.assert_array_equal(factors.pose_ptr, [0, 1])
    np.testing.assert_array_equal(factors.gaussian_ids, [0])
    assert factors.jacobians.shape == (1, 6)


def test_pose_changes_visibility_and_conditioning():
    means = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    normals = np.tile([0.0, 0.0, 1.0], (means.shape[0], 1))
    pose2 = np.eye(4)
    pose2[0, 3] = -0.8
    factors = build_frustum_center_factors(means, normals, make_views(np.stack([np.eye(4), pose2])), pair_cap=None)
    assert factors.pose_ptr.shape == (3,)
    assert not np.allclose(factors.jacobians[0], factors.jacobians[-1])
    information = np.eye(6) * 1e-6
    for jacobian, weight in zip(factors.jacobians, factors.weights):
        information += weight * np.outer(jacobian, jacobian)
    assert np.linalg.eigvalsh(information)[-2] > 1e-6


def test_trace_score_overvalues_repeated_direction():
    repeated = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    factors = make_factors([repeated, repeated, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], gaussian_ids=[0, 0, 1])
    scores = trajectory_trace_scores(factors)
    assert scores[0] == 2.0
    assert scores[1] == 1.0


def test_logdet_greedy_prefers_complementary_direction_over_repeated_trace():
    factors = make_factors(
        [[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0.0, 0.9, 0, 0, 0, 0]],
        gaussian_ids=[0, 1, 2],
    )
    selection = select_observable_gaussians(factors, budget=2, greedy_core_cap=2, candidate_cap=3)
    np.testing.assert_array_equal(selection.indices, [0, 2])


def test_per_pose_objective_differs_from_aggregated_hessian_and_matches_bruteforce():
    vectors = np.array(
        [
            [[-0.9531402414, -0.2554960765], [4.0431090686, 1.6018319497]],
            [[-4.8539615815, -0.0153871610], [-0.8094178120, 0.1929623079]],
            [[-4.6896665464, 0.7050355027], [0.6610370043, 4.4249428426]],
        ]
    )
    jacobians = np.zeros((6, 6), dtype=np.float64)
    jacobians[:3, :2] = vectors[:, 0]
    jacobians[3:, :2] = vectors[:, 1]
    factors = make_factors(
        jacobians,
        gaussian_ids=[0, 1, 2, 0, 1, 2],
        num_gaussians=3,
        pose_ptr=[0, 3, 6],
    )
    regularization = 0.2
    selection = select_observable_gaussians(
        factors, budget=2, candidate_cap=3, greedy_core_cap=2, regularization=regularization
    )

    def per_pose_objective(indices):
        total = 0.0
        for pose_id in range(2):
            information = regularization * np.eye(6)
            for gaussian_id in indices:
                jacobian = jacobians[pose_id * 3 + gaussian_id]
                information += np.outer(jacobian, jacobian)
            total += np.linalg.slogdet(information)[1]
        return total

    def aggregate_objective(indices):
        information = 2.0 * regularization * np.eye(6)
        for pose_id in range(2):
            for gaussian_id in indices:
                jacobian = jacobians[pose_id * 3 + gaussian_id]
                information += np.outer(jacobian, jacobian)
        return np.linalg.slogdet(information)[1]

    pairs = [(0, 1), (0, 2), (1, 2)]
    assert max(pairs, key=per_pose_objective) == (0, 2)
    assert max(pairs, key=aggregate_objective) == (1, 2)
    np.testing.assert_array_equal(selection.indices, [2, 0])
    assert selection.objective == pytest.approx(per_pose_objective(selection.indices))
    assert selection.objective == pytest.approx(np.sum(selection.per_pose_logdet))
    assert selection.per_pose_information.shape == (2, 6, 6)


def test_each_greedy_gain_matches_bruteforce_per_pose_slogdet():
    factors = make_factors(
        [[1.0, 0, 0, 0, 0, 0], [0.5, 1.0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0], [0, 1.5, 0, 0, 0, 0]],
        gaussian_ids=[0, 1, 0, 1],
        num_gaussians=2,
        pose_ptr=[0, 2, 4],
    )
    regularization = 0.5
    first = select_observable_gaussians(factors, budget=1, candidate_cap=2, greedy_core_cap=1, regularization=regularization)
    full = select_observable_gaussians(factors, budget=2, candidate_cap=2, greedy_core_cap=2, regularization=regularization)

    def objective(indices):
        total = 0.0
        for start, stop in zip(factors.pose_ptr[:-1], factors.pose_ptr[1:]):
            information = regularization * np.eye(6)
            for pair_id in range(start, stop):
                if factors.gaussian_ids[pair_id] in indices:
                    j = factors.jacobians[pair_id]
                    information += np.outer(j, j)
            total += np.linalg.slogdet(information)[1]
        return total

    baseline = 2 * np.linalg.slogdet(regularization * np.eye(6))[1]
    gains = [objective([idx]) - baseline for idx in range(2)]
    assert first.indices[0] == int(np.argmax(gains))
    assert first.objective - baseline == pytest.approx(max(gains))
    assert full.objective - first.objective == pytest.approx(objective([0, 1]) - objective(first.indices))


def test_weak_direction_score_rewards_minimum_eigen_direction():
    information = np.diag([1e-3, 2.0, 3.0, 4.0, 5.0, 6.0])
    weak = weak_direction_score(information, np.array([[1.0, 0, 0, 0, 0, 0]]), np.ones(1))
    strong = weak_direction_score(information, np.array([[0.0, 1.0, 0, 0, 0, 0]]), np.ones(1))
    assert weak == 1.0
    assert strong == 0.0


def test_periodic_min_eigen_bonus_can_prefer_weak_direction():
    factors = make_factors(
        [[2.0, 0, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [0.0, 1.0, 0, 0, 0, 0]],
        gaussian_ids=[0, 1, 2],
    )
    without_bonus = select_observable_gaussians(
        factors,
        budget=2,
        candidate_cap=3,
        greedy_core_cap=2,
        regularization=100.0,
    )
    with_bonus = select_observable_gaussians(
        factors,
        budget=2,
        candidate_cap=3,
        greedy_core_cap=2,
        regularization=100.0,
        min_eigen_every=1,
        min_eigen_bonus=1.0,
    )
    np.testing.assert_array_equal(without_bonus.indices, [0, 1])
    np.testing.assert_array_equal(with_bonus.indices, [0, 2])


def test_weak_bonus_eigh_is_once_per_pose_batch_not_per_candidate():
    factors = make_factors(np.eye(6)[:4], num_gaussians=4)
    with mock.patch("utils.fsm_observability.np.linalg.eigh", wraps=np.linalg.eigh) as patched:
        select_observable_gaussians(
            factors, budget=3, candidate_cap=4, greedy_core_cap=3,
            min_eigen_every=1, min_eigen_bonus=0.1,
        )
    assert patched.call_count == 3


def test_candidate_cap_is_hard_and_selection_does_not_fill_beyond_candidates():
    factors = make_factors(np.eye(6)[:4], num_gaussians=4)
    with pytest.raises(ValueError, match="bounded candidate count"):
        select_observable_gaussians(factors, budget=3, candidate_cap=1, greedy_core_cap=1)


def test_eligible_indices_are_a_hard_constraint_with_stable_ties():
    factors = make_factors(np.ones((4, 6)), num_gaussians=4)
    selection = select_observable_gaussians(
        factors, budget=2, candidate_cap=None, greedy_core_cap=None, eligible_indices=np.array([3, 2])
    )
    np.testing.assert_array_equal(selection.indices, [2, 3])
    assert set(selection.candidate_indices) == {2, 3}


def test_replacement_information_diagnostics_match_direct_math():
    factors = make_factors(
        [[1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [0, 0.5, 0, 0, 0, 0]],
        gaussian_ids=[0, 1, 0, 1],
        num_gaussians=2,
        pose_ptr=[0, 2, 4],
    )
    diagnostics = compare_replacement_information(factors, np.array([0]), np.array([1]), regularization=1.0)
    np.testing.assert_allclose(diagnostics["trace_ratio"], [10.0 / 7.0, 6.25 / 7.0])
    expected_logdet = np.log(np.array([5.0, 1.25])) - np.log(np.array([2.0, 2.0]))
    np.testing.assert_allclose(diagnostics["logdet_delta"], expected_logdet)
    np.testing.assert_allclose(diagnostics["lambda_min_delta"], [0.0, 0.0])
    np.testing.assert_array_equal(diagnostics["psd_preserved"], [True, False])
    assert diagnostics["summary"]["psd_preserved_fraction"] == 0.5
    assert "replacement_set_vs_removed_set" in diagnostics["scope"]


def test_factor_jacobian_matches_robust_point_to_plane_tracker():
    from utils.fsm_robust_registration import point_to_plane_residual_jacobian

    means = np.array([[0.4, -0.2, 2.0]])
    normals = np.array([[0.0, 1.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    factors = build_frustum_center_factors(means, normals, make_views(pose[None]), pair_cap=None)
    point_cam = means @ pose[:3, :3].T + pose[:3, 3]
    normal_cam = normals @ pose[:3, :3].T
    _, expected = point_to_plane_residual_jacobian(point_cam, point_cam, normal_cam)
    np.testing.assert_allclose(factors.jacobians, expected)


def test_pair_cap_is_stratified_per_pose_and_reports_coverage():
    means = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    factors = build_frustum_center_factors(means, normals, make_views(np.repeat(np.eye(4)[None], 2, axis=0)), pair_cap=2)
    np.testing.assert_array_equal(factors.pose_ptr, [0, 1, 2])
    assert factors.nonempty_pose_count == 2
    assert factors.pose_coverage == 1.0

    capped_out = build_frustum_center_factors(means, normals, make_views(np.repeat(np.eye(4)[None], 3, axis=0)), pair_cap=2)
    assert capped_out.num_pairs == 0
    assert capped_out.visible_pose_count == 3
    assert capped_out.nonempty_pose_count == 0
