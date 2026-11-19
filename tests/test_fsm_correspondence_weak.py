import numpy as np
import pytest

from utils.fsm_correspondence_weak import (
    adaptive_weak_subspace,
    normalized_hessian_diagnostics,
)


def _pose(angle=0.4, translation=(0.3, -0.2, 0.5)):
    pose = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = translation
    return pose


def _adjoint(pose):
    rotation, t = pose[:3, :3], pose[:3, 3]
    skew = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, :3] = skew @ rotation
    result[3:, 3:] = rotation
    return result


def test_diagnostics_matches_camera_quadratic_form_and_spectrum():
    pose = _pose()
    hessian = np.diag(np.arange(1.0, 7.0))
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 3.0]])
    result = normalized_hessian_diagnostics(hessian, pose, points)
    coordinate_map = np.diag([0.5] * 3 + [1.0] * 3)
    transform = _adjoint(pose) @ coordinate_map
    expected = transform.T @ hessian @ transform
    np.testing.assert_allclose(result["hessian"], expected, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(expected)
    np.testing.assert_allclose(result["eigenvalues"], eigenvalues)
    assert result["condition_number"] == pytest.approx(eigenvalues[-1] / eigenvalues[0])
    assert result["effective_rank"] == 6
    assert result["characteristic_length"] == 2.0
    assert result["normalized_lambda_min"] == pytest.approx(eigenvalues[0] / np.trace(expected))


def test_common_world_reexpression_preserves_diagnostics():
    rng = np.random.default_rng(14)
    rows = rng.normal(size=(20, 6))
    hessian = rows.T @ rows
    pose = _pose()
    world_change = _pose(-0.7, (2.0, -3.0, 1.2))
    inverse_adjoint = np.linalg.inv(_adjoint(world_change))
    transformed = inverse_adjoint.T @ hessian @ inverse_adjoint
    points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 4.0]])
    first = normalized_hessian_diagnostics(hessian, pose, points)
    second = normalized_hessian_diagnostics(transformed, world_change @ pose, points)
    for key in ("hessian", "eigenvalues", "condition_number", "effective_rank",
                "characteristic_length", "normalized_lambda_min"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-12)


def test_primary_candidate_share_reference_and_explicit_length():
    pose = _pose()
    primary = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    addition = np.outer(np.arange(1.0, 7.0), np.arange(1.0, 7.0))
    first = normalized_hessian_diagnostics(primary, pose, [[0.0, 0.0, 2.0]])
    length = first["characteristic_length"]
    candidate = normalized_hessian_diagnostics(
        primary + addition, pose, [[0.0, 0.0, 5.0]], characteristic_length=length
    )
    delta = normalized_hessian_diagnostics(addition, pose, characteristic_length=length)
    assert candidate["characteristic_length"] == length
    np.testing.assert_allclose(candidate["hessian"] - first["hessian"], delta["hessian"], atol=1e-13)


def test_hessian_symmetrization_and_scale_invariance():
    hessian = np.diag([1e-4, 1.0, 2.0, 3.0, 4.0, 5.0])
    asymmetric = hessian.copy()
    asymmetric[0, 1], asymmetric[1, 0] = 0.2, -0.2
    first = normalized_hessian_diagnostics(hessian, _pose())
    symmetric = normalized_hessian_diagnostics(asymmetric, _pose())
    np.testing.assert_array_equal(first["hessian"], symmetric["hessian"])
    for scale in (1e-20, 17.0, 1e20):
        scaled = normalized_hessian_diagnostics(scale * hessian, _pose())
        for key in ("condition_number", "effective_rank", "normalized_lambda_min"):
            np.testing.assert_allclose(first[key], scaled[key], rtol=1e-10)


@pytest.mark.parametrize("points, expected", [
    (None, 1.0), (np.empty((0, 3)), 1.0),
    ([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]], 1.0),
    ([[0.0, 0.0, 0.1]], 0.5), ([[0.0, 0.0, 8.0]], 5.0),
    ([[0.0, 0.0, -8.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], 3.0),
])
def test_characteristic_length_depth_convention(points, expected):
    result = normalized_hessian_diagnostics(np.eye(6), np.eye(4), points)
    assert result["characteristic_length"] == expected


@pytest.mark.parametrize("length", [0.1, 8.0])
def test_explicit_length_is_not_clipped(length):
    result = normalized_hessian_diagnostics(np.eye(6), np.eye(4), characteristic_length=length)
    assert result["characteristic_length"] == length


@pytest.mark.parametrize("spectrum, rank", [
    ([0.0] * 6, 0), ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 5),
    ([-1e-15, 1.0, 2.0, 3.0, 4.0, 5.0], 5),
])
def test_singular_and_roundoff_psd(spectrum, rank):
    result = normalized_hessian_diagnostics(np.diag(spectrum), np.eye(4))
    assert result["effective_rank"] == rank
    assert result["condition_number"] == np.inf
    assert result["normalized_lambda_min"] == 0.0
    np.testing.assert_allclose(result["eigenvalues"], np.maximum(spectrum, 0.0))


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_negative_spectrum_rejected_at_any_scale(scale):
    with pytest.raises(ValueError, match="positive semidefinite"):
        normalized_hessian_diagnostics(scale * np.diag([-1e-4, 1, 2, 3, 4, 5]), np.eye(4))


@pytest.mark.parametrize("kwargs, message", [
    ({"hessian_world": np.eye(5)}, "hessian_world"),
    ({"hessian_world": np.full((6, 6), np.nan)}, "hessian_world"),
    ({"hessian_world": np.full((6, 6), np.inf)}, "hessian_world"),
    ({"reference_pose": np.eye(3)}, "reference_pose"),
    ({"reference_pose": np.full((4, 4), np.nan)}, "reference_pose"),
    ({"reference_pose": np.diag([1.0, 1.0, 1.0, 2.0])}, "rigid transform"),
    ({"reference_pose": np.diag([2.0, 1.0, 1.0, 1.0])}, "rigid transform"),
    ({"reference_pose": np.diag([-1.0, 1.0, 1.0, 1.0])}, "rigid transform"),
    ({"source_points_camera": np.zeros((2, 2))}, "source_points_camera"),
    ({"source_points_camera": [[np.nan, 0.0, 2.0]]}, "source_points_camera"),
    ({"source_points_camera": [[0.0, 0.0, np.inf]]}, "source_points_camera"),
    ({"characteristic_length": 0.0}, "characteristic_length"),
    ({"characteristic_length": -1.0}, "characteristic_length"),
    ({"characteristic_length": np.nan}, "characteristic_length"),
    ({"characteristic_length": np.inf}, "characteristic_length"),
    ({"characteristic_length": [1.0]}, "characteristic_length"),
])
def test_invalid_inputs_raise_value_error(kwargs, message):
    arguments = {"hessian_world": np.eye(6), "reference_pose": np.eye(4)}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        normalized_hessian_diagnostics(**arguments)


@pytest.mark.parametrize("depth_bounds", [(0.5, 5.0), (0.1, 0.4), (6.0, 10.0)])
@pytest.mark.parametrize("spectrum", [
    [1e-8, 1, 2, 3, 4, 5], [1e-8, 2e-8, 1, 2, 3, 4],
    [1e-8, 2e-8, 3e-8, 4e-8, 1, 2], [1, 1, 1, 1, 1, 1],
])
def test_adaptive_legacy_api_and_numerics(spectrum, depth_bounds):
    pose = _pose()
    hessian = np.diag(spectrum)
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]])
    length = float(np.clip(1.5, *depth_bounds))
    # The legacy path projects the input rotation onto SO(3) using SVD.
    legacy_pose = pose.copy()
    u, _, vt = np.linalg.svd(pose[:3, :3])
    legacy_pose[:3, :3] = u @ vt
    adjoint = _adjoint(legacy_pose)
    camera = adjoint.T @ hessian @ adjoint
    coordinate_map = np.diag([1.0 / length] * 3 + [1.0] * 3)
    normalized = coordinate_map.T @ camera @ coordinate_map
    normalized = 0.5 * (normalized + normalized.T)
    values, vectors = np.linalg.eigh(normalized)
    values = np.maximum(values, 0.0)
    threshold = max(1e-3 * max(float(np.sum(values)), np.finfo(float).eps), 1e-2 * values[-1])
    requested = np.count_nonzero(values < threshold)
    k = int(np.clip(requested, 1, 3))
    deficits = np.maximum(threshold - values[:k], 0.0) / threshold
    if not np.any(deficits > 0.0):
        deficits = threshold / np.maximum(values[:k] + threshold, threshold)
    result = adaptive_weak_subspace(
        hessian, pose, points,
        characteristic_depth_min=depth_bounds[0], characteristic_depth_max=depth_bounds[1],
    )
    expected = {
        "basis": vectors[:, :k], "eigs": values, "eigenvalues": values,
        "deficit_weights": deficits, "k": k, "length": length,
        "characteristic_length": length, "threshold": threshold,
        "truncated": bool(requested > 3), "camera_hessian": camera,
        "normalized_hessian": normalized, "adjoint_c2w": adjoint,
        "coordinate_map": coordinate_map,
    }
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        if key == "basis":
            np.testing.assert_allclose(result[key] @ result[key].T, value @ value.T, atol=1e-12)
        else:
            np.testing.assert_allclose(result[key], value, rtol=1e-12, atol=1e-14)
    diagnostics = normalized_hessian_diagnostics(hessian, pose, points, length)
    np.testing.assert_array_equal(result["normalized_hessian"], diagnostics["hessian"])
    np.testing.assert_array_equal(result["eigenvalues"], diagnostics["eigenvalues"])
