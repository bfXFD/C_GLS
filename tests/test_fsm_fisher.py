import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.fsm_fisher import compute_and_save_fisher, load_retained_indices
from utils.fsm_fisher import (
    compute_geometric_fisher,
    compute_pruning_residuals,
    estimate_map_normals,
    point_to_plane_pose_jacobian,
    predict_error_increment,
    skew,
)
from utils.fsm_map import build_fsm_map, save_fsm_map


def make_non_degenerate_points():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.7, 1.2],
            [0.3, -0.8, 0.6],
            [1.3, 0.2, -0.4],
        ],
        dtype=np.float64,
    )


def make_normals(num_points):
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (num_points, 1))
    normals[1::3] = np.array([0.0, 1.0, 0.0])
    normals[2::3] = np.array([1.0, 0.0, 0.0])
    return normals


def make_params_file(tmp_path):
    points = make_non_degenerate_points().astype(np.float32)
    params_path = tmp_path / "params.npz"
    np.savez(
        params_path,
        means3D=points,
        log_scales=np.log(np.full((points.shape[0], 1), 0.01, dtype=np.float32)),
        unnorm_rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (points.shape[0], 1)),
    )
    return params_path, points


def test_skew_cross_product_equivalence():
    v = np.array([1.0, 2.0, 3.0])
    x = np.array([-1.0, 0.5, 4.0])
    np.testing.assert_allclose(skew(v) @ x, np.cross(v, x))


def test_point_to_plane_pose_jacobian_shape_and_translation_part():
    point = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 2.0])
    jacobian = point_to_plane_pose_jacobian(point, normal)
    assert jacobian.shape == (6,)
    np.testing.assert_allclose(jacobian[3:], np.array([0.0, 0.0, 1.0]))


def test_estimate_map_normals_shape_and_unit_norm():
    points = make_non_degenerate_points()
    normals = estimate_map_normals(points, k=4)
    assert normals.shape == points.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_compute_geometric_fisher_outputs_psd_matrices():
    points = make_non_degenerate_points()
    normals = make_normals(points.shape[0])
    fisher = compute_geometric_fisher(points, normals=normals)
    assert fisher["per_gaussian_fim"].shape == (points.shape[0], 6, 6)
    assert fisher["scores"].shape == (points.shape[0],)
    assert fisher["full_fim"].shape == (6, 6)
    assert np.all(fisher["scores"] >= 0)
    assert np.all(np.linalg.eigvalsh(fisher["per_gaussian_fim"]) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(fisher["full_fim"]) >= -1e-8)


def test_compute_geometric_fisher_weights_scale_scores():
    points = make_non_degenerate_points()
    normals = make_normals(points.shape[0])
    base = compute_geometric_fisher(points, normals=normals)
    weighted = compute_geometric_fisher(points, normals=normals, weights=np.full(points.shape[0], 2.0))
    np.testing.assert_allclose(weighted["scores"], base["scores"] * 2.0)


def test_compute_pruning_residuals_monotonic_with_more_retained():
    points = make_non_degenerate_points()
    normals = make_normals(points.shape[0])
    fisher = compute_geometric_fisher(points, normals=normals)
    few = compute_pruning_residuals(fisher["per_gaussian_fim"], np.array([0, 1]))
    many = compute_pruning_residuals(fisher["per_gaussian_fim"], np.array([0, 1, 2, 3, 4]))
    assert many["delta_trace"] <= few["delta_trace"]
    assert many["delta_spectral"] <= few["delta_spectral"] + 1e-8


def test_predict_error_increment_marks_invalid_when_delta_exceeds_lambda():
    full_fim = np.eye(6)
    pruned_fim = np.eye(6) * 2.0
    prediction = predict_error_increment(full_fim, pruned_fim, regularization=0.0)
    assert not prediction["valid_spectral_bound"]
    assert np.isnan(prediction["pred_delta_spectral"])


def test_predict_error_increment_valid_small_delta():
    full_fim = np.eye(6) * 10.0
    pruned_fim = np.eye(6) * 0.1
    prediction = predict_error_increment(full_fim, pruned_fim, regularization=0.0)
    assert prediction["valid_spectral_bound"]
    assert np.isfinite(prediction["pred_delta_spectral"])


def test_compute_and_save_fisher_from_params(tmp_path):
    params_path, points = make_params_file(tmp_path)
    output_path = tmp_path / "fisher_scores.npz"
    output = compute_and_save_fisher(str(output_path), params_path=str(params_path), normal_k=4)
    assert output_path.exists()
    loaded = np.load(output_path, allow_pickle=True)
    assert loaded["scores"].shape == (points.shape[0],)
    assert loaded["full_fim"].shape == (6, 6)
    assert output["scores"].shape == (points.shape[0],)


def test_compute_and_save_fisher_with_retained_indices(tmp_path):
    params_path, _ = make_params_file(tmp_path)
    output_path = tmp_path / "fisher_scores.npz"
    retained = np.array([0, 1, 2], dtype=np.int64)
    compute_and_save_fisher(str(output_path), params_path=str(params_path), retained_indices=retained, normal_k=4)
    loaded = np.load(output_path, allow_pickle=True)
    np.testing.assert_array_equal(loaded["retained_indices"], retained)
    assert "delta_trace" in loaded
    assert "pred_delta_spectral" in loaded


def test_compute_and_save_fisher_from_fsm_map(tmp_path):
    _, points = make_params_file(tmp_path)
    params = {
        "means3D": points,
        "log_scales": np.log(np.full((points.shape[0], 1), 0.01, dtype=np.float32)),
        "unnorm_rotations": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (points.shape[0], 1)),
    }
    map_path = tmp_path / "fsm_map.npz"
    save_fsm_map(build_fsm_map(params), str(map_path))
    output_path = tmp_path / "fisher_scores.npz"
    compute_and_save_fisher(str(output_path), map_path=str(map_path), normal_k=4)
    loaded = np.load(output_path, allow_pickle=True)
    assert loaded["points"].shape == points.shape


def test_load_retained_indices_from_npz_and_npy(tmp_path):
    indices = np.array([0, 2, 4], dtype=np.int64)
    npz_path = tmp_path / "indices.npz"
    npy_path = tmp_path / "indices.npy"
    np.savez(npz_path, indices=indices)
    np.save(npy_path, indices)
    np.testing.assert_array_equal(load_retained_indices(str(npz_path)), indices)
    np.testing.assert_array_equal(load_retained_indices(str(npy_path)), indices)


def test_cli_smoke_from_params(tmp_path):
    params_path, points = make_params_file(tmp_path)
    output_path = tmp_path / "cli_fisher.npz"
    cmd = [
        sys.executable,
        "scripts/fsm_fisher.py",
        "--params",
        str(params_path),
        "--output",
        str(output_path),
        "--normal-k",
        "4",
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), check=True, text=True, capture_output=True)
    assert f"Saved Fisher scores for {points.shape[0]} Gaussians" in result.stdout
    assert output_path.exists()


def test_compute_pruning_residuals_rejects_duplicate_indices():
    fim = np.zeros((3, 6, 6), dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pruning_residuals(fim, np.array([0, 0]))
