from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from utils.fsm_candidate_verification import (
    FrozenCandidateVerifier, VERIFIER_METRIC_FIELDS, split_source_validation,
    validate_verifier_config,
)
from utils.fsm_robust_registration import _robust_cost, se3_exp, transform_points


def scene():
    points = np.column_stack((np.arange(20) * 2.0, np.zeros(20), np.zeros(20)))
    normals = np.tile([0., 0., 1.], (20, 1))
    return points, normals


def verifier(**kwargs):
    points, normals = scene()
    return FrozenCandidateVerifier(points, points, normals, 0.5, 0.1, **kwargs)


def pose(z=0., x=0.):
    result = np.eye(4)
    result[:3, 3] = [x, 0., z]
    return result


def assert_finite(result):
    for value in result.values():
        if isinstance(value, dict):
            assert_finite(value)
        elif not isinstance(value, str):
            assert np.all(np.isfinite(value))


def test_same_pose_and_fixed_scalar_metric_schema():
    result = verifier().evaluate_pair(np.eye(4), np.eye(4))
    assert result["accepted"] and result["reason"] == "accepted"
    assert tuple(result["metrics"]) == VERIFIER_METRIC_FIELDS
    assert all(np.isscalar(v) for v in result["metrics"].values())
    assert result["metrics"]["reserve_verifier_source_count"] == 20
    assert_finite(result)


@pytest.mark.parametrize("loss", ["none", "huber", "cauchy"])
def test_robust_semantics_and_worse_pose_rejected(loss):
    check = verifier(robust_loss=loss)
    score = check.evaluate(pose(z=0.04))
    expected = _robust_cost(np.full(20, 0.04), loss, 0.1) / 20
    assert score["cost"] == pytest.approx(expected)
    assert score["euclidean_cost"] == pytest.approx(expected)
    assert check.evaluate_pair(np.eye(4), pose(z=0.04))["reason"] == "cost_increase"


def test_tangential_motion_cannot_hide_behind_plane_cost():
    check = verifier()
    assert check.evaluate(pose(x=0.04))["cost"] == 0
    assert check.evaluate_pair(np.eye(4), pose(x=0.04))["reason"] == "euclidean_cost_increase"


def test_independent_tolerances_and_primary_must_be_valid():
    check = verifier()
    config = {"absolute_tolerance": 1., "max_group_cost_increase": 1.}
    assert check.evaluate_pair(np.eye(4), pose(z=0.04), config)["reason"] == "euclidean_cost_increase"
    config["euclidean_absolute_tolerance"] = 1.
    assert check.evaluate_pair(np.eye(4), pose(z=0.04), config)["accepted"]
    assert check.evaluate_pair(pose(z=2), np.eye(4), config)["reason"] == "invalid_primary"
    config = {"relative_tolerance": 4., "euclidean_relative_tolerance": 4.,
              "max_group_cost_increase": 1.}
    assert check.evaluate_pair(pose(z=0.01), pose(z=0.02), config)["accepted"]


def test_relative_log_at_pi_is_finite():
    # Rotation around the x-axis fixes this collinear synthetic geometry.
    candidate = se3_exp(np.array([np.pi, 0., 0., 0., 0., 0.]))
    result = verifier().evaluate_pair(np.eye(4), candidate, {"max_rotation_deg": 180.})
    assert result["accepted"]
    assert result["metrics"]["reserve_verifier_rotation_deg"] == pytest.approx(180.)
    assert_finite(result)


@pytest.mark.parametrize("loss", ["none", "huber", "cauchy"])
def test_unmatched_points_keep_fixed_denominator_and_max_penalty(loss):
    points, normals = scene()
    source = points.copy()
    source[:10, 2] = 0.45
    check = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1, robust_loss=loss)
    before, after = check.evaluate(np.eye(4)), check.evaluate(pose(z=0.1))
    assert after["count"] == 10 and after["coverage"] == 0.5
    assert after["cost"] > before["cost"]
    expected = (_robust_cost(np.full(10, 0.5), loss, 0.1)
                + _robust_cost(np.full(10, 0.1), loss, 0.1)) / 20
    assert after["cost"] == pytest.approx(expected)
    assert after["euclidean_cost"] == pytest.approx(expected)
    empty = check.evaluate(pose(z=2))
    assert not empty["valid"] and empty["coverage"] == 0
    assert empty["cost"] >= after["cost"]
    assert_finite(empty)


def test_spatial_local_degradation_despite_global_improvement():
    points, normals = scene()
    source = points.copy()
    source[:14, 2] = 0.08
    groups = np.r_[np.zeros(14, dtype=int), np.ones(6, dtype=int)]
    check = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1,
                                    spatial_groups=groups)
    candidate = pose(z=-0.04)
    assert check.evaluate(candidate)["cost"] < check.evaluate(np.eye(4))["cost"]
    result = check.evaluate_pair(np.eye(4), candidate)
    assert result["reason"] == "spatial_group_degradation"
    assert result["metrics"]["reserve_verifier_worst_group_delta"] > 0
    assert check.evaluate_pair(np.eye(4), candidate, {"min_group_size": 7})["accepted"]


def test_tangential_group_degradation_despite_global_euclidean_improvement():
    points, normals = scene()
    source = points.copy()
    source[:14, 0] += 0.08
    groups = np.r_[np.zeros(14, dtype=int), np.ones(6, dtype=int)]
    check = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1,
                                    spatial_groups=groups)
    candidate = pose(x=-0.04)
    primary_score, candidate_score = check.evaluate(np.eye(4)), check.evaluate(candidate)
    assert candidate_score["cost"] == primary_score["cost"] == 0
    assert candidate_score["euclidean_cost"] < primary_score["euclidean_cost"]
    result = check.evaluate_pair(np.eye(4), candidate)
    assert result["reason"] == "spatial_group_euclidean_degradation"
    assert not result["accepted"]
    assert result["metrics"]["reserve_verifier_worst_group_delta"] == 0
    delta = result["metrics"]["reserve_verifier_worst_group_euclidean_delta"]
    assert delta == pytest.approx(0.5 * 0.04 ** 2)
    assert_finite(result)
    assert check.evaluate_pair(np.eye(4), candidate, {"min_group_size": 7})["accepted"]
    assert check.evaluate_pair(np.eye(4), candidate, {"max_group_cost_increase": delta})["accepted"]
    assert not check.evaluate_pair(np.eye(4), candidate, {"max_group_cost_increase": delta / 2})["accepted"]
    no_groups = check.evaluate_pair(np.eye(4), candidate, {"min_group_size": 21})
    assert no_groups["metrics"]["reserve_verifier_worst_group_euclidean_delta"] == 0


def test_validate_config_defaults_normalization_and_no_mutation():
    defaults = validate_verifier_config()
    assert defaults == validate_verifier_config({})
    assert len(defaults) == 9
    config = {"min_group_size": np.int64(8), "max_translation": np.float64(0.2)}
    normalized = validate_verifier_config(config)
    assert type(normalized["min_group_size"]) is int
    assert type(normalized["max_translation"]) is float
    assert set(config) == {"min_group_size", "max_translation"}
    normalized["absolute_tolerance"] = 1.
    assert validate_verifier_config() == defaults
    check = verifier()
    assert check.evaluate_pair(np.eye(4), np.eye(4), validate_verifier_config(config))["accepted"]


def test_coverage_guard_is_independent_of_cost_tolerances():
    check = verifier()
    result = check.evaluate_pair(np.eye(4), pose(z=0.6), {
        "max_translation": 1., "absolute_tolerance": 1.,
        "euclidean_absolute_tolerance": 1., "max_group_cost_increase": 1.,
    })
    assert result["reason"] == "invalid_candidate"
    points, normals = scene()
    source = points.copy()
    source[:3, 2] = 0.49
    check = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1)
    config = {"absolute_tolerance": 1., "euclidean_absolute_tolerance": 1.,
              "max_group_cost_increase": 1.}
    assert check.evaluate_pair(np.eye(4), pose(z=0.02), config)["reason"] == "coverage_drop"
    assert check.evaluate_pair(np.eye(4), pose(z=0.02), dict(config, min_coverage_drop=0.2))["accepted"]


def test_immutability_copies_geometry_groups_and_kdtree():
    points, normals = scene()
    source, groups = points.copy(), np.arange(20)
    check = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1, spatial_groups=groups)
    before = check.evaluate(np.eye(4))
    source[:] = 99
    points[:] = -99
    normals[:] = 0
    groups[:] = 999
    assert check.evaluate(np.eye(4)) == before
    for array in (check.source_points, check.target_points, check.target_normals,
                  check.spatial_groups, check._tree.data):
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        check.max_distance = 20


def test_source_split_deterministic_disjoint_and_blockwise():
    ids = np.arange(128 * 96, dtype=np.int64)
    masks = np.stack([split_source_validation(ids, 128, fold=f) for f in range(4)])
    assert masks.dtype == np.bool_
    np.testing.assert_array_equal(masks.sum(axis=0), np.ones(ids.size))
    for f in range(4):
        assert masks[f].any()
        np.testing.assert_array_equal(split_source_validation(ids[::-1], 128, fold=f), masks[f, ::-1])
        image = masks[f].reshape(96, 128)
        for row in range(0, 96, 16):
            for column in range(0, 128, 16):
                assert np.unique(image[row:row + 16, column:column + 16]).size == 1
    assert not split_source_validation(np.array([], dtype=int), 128).size


def test_se3_target_coordinate_invariance_and_relative_log_bounds():
    points, normals = scene()
    source = points + [0.01, 0.0, 0.02]
    primary = se3_exp(np.array([0.001, 0.0, 0.0, 0.01, -0.01, 0.0]))
    delta = np.array([0.002, -0.001, 0.001, 0.01, 0.005, -0.003])
    candidate = primary @ se3_exp(delta)
    change = se3_exp(np.array([0.4, -0.2, 0.3, 5., -8., 2.]))
    original = FrozenCandidateVerifier(source, points, normals, 0.5, 0.1)
    changed = FrozenCandidateVerifier(source, transform_points(points, change),
                                       normals @ change[:3, :3].T, 0.5, 0.1)
    first = original.evaluate_pair(primary, candidate)
    second = changed.evaluate_pair(change @ primary, change @ candidate)
    assert first["reason"] == second["reason"]
    for key, value in first["metrics"].items():
        if not isinstance(value, str):
            assert second["metrics"][key] == pytest.approx(value, abs=1e-12)
    assert first["metrics"]["reserve_verifier_translation"] == pytest.approx(np.linalg.norm(delta[3:]))
    assert first["metrics"]["reserve_verifier_rotation_deg"] == pytest.approx(np.degrees(np.linalg.norm(delta[:3])))
    assert original.evaluate_pair(primary, candidate, {"max_translation": 0.001})["reason"] == "translation_bound"
    assert original.evaluate_pair(primary, candidate, {"max_rotation_deg": 0.001})["reason"] == "rotation_bound"


@pytest.mark.parametrize("bad", [None, np.zeros((4, 4)), np.full((4, 4), np.nan),
                                     np.diag([-1., 1., 1., 1.]), np.diag([2., 1., 1., 1.]),
                                     np.eye(3), "bad"])
def test_invalid_pose_fails_closed_with_finite_metrics(bad):
    check = verifier()
    assert not check.evaluate(bad)["valid"]
    assert_finite(check.evaluate(bad))
    for primary, candidate in ((bad, np.eye(4)), (np.eye(4), bad)):
        result = check.evaluate_pair(primary, candidate)
        assert not result["accepted"]
        assert_finite(result)


@pytest.mark.parametrize("scale", [np.nextafter(np.float32(1), np.float32(0)),
                                   np.nextafter(np.float32(1), np.float32(2))])
def test_float32_pose_defect_is_projected_before_matching(scale):
    points, normals = scene()
    rigid = se3_exp(np.array([0.1, 0.2, -0.3, 0.01, 0, 0]))
    near_rigid = rigid @ np.diag([float(scale), 1., 1., 1.])
    reference = FrozenCandidateVerifier(points, transform_points(points, rigid),
                                        normals @ rigid[:3, :3].T, 0.07, 0.02)
    original = near_rigid.copy()
    result = reference.evaluate_pair(near_rigid, rigid)
    assert result["accepted"]
    assert result["metrics"]["reserve_verifier_primary_count"] == len(points)
    assert result["metrics"]["reserve_verifier_primary_reason"] == "valid"
    np.testing.assert_array_equal(near_rigid, original)


def test_verifier_distinguishes_bad_pose_from_missing_correspondences():
    check = verifier()
    shear = np.eye(4)
    shear[0, 1] = 1e-4
    result = check.evaluate_pair(shear, pose(z=2))
    assert not result["accepted"]
    assert result["metrics"]["reserve_verifier_primary_reason"] == "invalid_pose"
    assert result["metrics"]["reserve_verifier_candidate_reason"] == "insufficient_correspondences"


@pytest.mark.parametrize("config", [
    {"relative_tolerance": -1}, {"absolute_tolerance": np.nan},
    {"euclidean_relative_tolerance": np.inf}, {"euclidean_absolute_tolerance": -1},
    {"min_coverage_drop": 1.1}, {"min_coverage_drop": True},
    {"max_group_cost_increase": -0.1}, {"min_group_size": 1.2},
    {"min_group_size": 0}, {"max_translation": np.inf},
    {"max_rotation_deg": 181}, {"unknown": 1}, [],
])
def test_invalid_config_never_accepts(config):
    with pytest.raises(ValueError):
        validate_verifier_config(config)
    with pytest.raises(ValueError):
        verifier().evaluate_pair(np.eye(4), np.eye(4), config)


@pytest.mark.parametrize("kwargs", [
    {"max_distance": 0}, {"loss_scale": np.inf}, {"robust_loss": "bad"},
    {"min_correspondences": True}, {"min_correspondences": 1.5},
    {"spatial_groups": np.zeros(19)}, {"spatial_groups": np.full(20, -1)},
    {"source_points": np.empty((0, 3))}, {"target_points": np.full((20, 3), np.nan)},
    {"target_normals": np.zeros((20, 3))},
])
def test_invalid_constructor_rejected(kwargs):
    points, normals = scene()
    options = dict(source_points=points, target_points=points, target_normals=normals,
                   max_distance=0.5, loss_scale=0.1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        FrozenCandidateVerifier(**options)


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"block_size": 1.5}, {"folds": 1}, {"fold": 4},
    {"pixel_ids": [-1]}, {"pixel_ids": [1.]}, {"fold": True},
])
def test_invalid_split_rejected(kwargs):
    options = dict(pixel_ids=np.arange(10), width=128)
    options.update(kwargs)
    with pytest.raises(ValueError):
        split_source_validation(**options)
