import csv
import json

import numpy as np
import pytest

from utils.fsm_final_target_oracle import (
    FinalTargetOracleConfig,
    FinalTargetOracleRuntime,
    _hessian_comparison,
)
from utils.fsm_local_map import (
    LocalSubmapConfig,
    MapSpatialIndex,
    points_in_camera_frustum,
    stream_exact_final_target,
)


K = np.array([[20.0, 0.0, 10.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])


def point_at_pixel(u, v, depth):
    return np.array([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth])


def brute_force(index, pose, config, scores=None):
    visible = np.flatnonzero(points_in_camera_frustum(index.points, pose, K, 16, 20, config))
    camera = (index.points[visible] - pose[:3, 3]) @ pose[:3, :3]
    depth = camera[:, 2]
    u = K[0, 0] * camera[:, 0] / depth + K[0, 2]
    v = K[1, 1] * camera[:, 1] / depth + K[1, 2]
    return index._coverage_sample(visible, u, v, depth, 16, 20, config, scores)[0]


@pytest.mark.parametrize("chunk", [1, 3, 17, 1000])
@pytest.mark.parametrize("with_scores", [False, True])
def test_exact_streaming_matches_brute_force_across_chunk_pose_and_ties(chunk, with_scores):
    rng = np.random.default_rng(84)
    camera_points = rng.uniform([-1.0, -0.8, 0.3], [1.0, 0.8, 5.0], size=(240, 3))
    camera_points[:8] = np.stack([point_at_pixel(10.0, 8.0, 1.0) for _ in range(8)])
    angle = np.deg2rad(27.0)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    pose[:3, 3] = [0.7, -0.2, 1.1]
    points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.2, far=5.5, max_points=43, sampling_policy="coverage",
        tile_rows=3, tile_cols=4, depth_bins=4,
    )
    scores = rng.integers(-2, 3, size=points.shape[0]).astype(np.float64) if with_scores else None
    exact = stream_exact_final_target(index, pose, K, 16, 20, config, scores=scores, chunk_size=chunk)
    reference = brute_force(index, pose, config, scores)
    np.testing.assert_array_equal(exact["indices"], reference)
    assert exact["retained_count"] == exact["count"] <= config.max_points
    assert exact["peak_retained_bound"] <= config.max_points
    assert exact["visible_count"] >= exact["count"]


def test_exact_streaming_accepts_block_score_token_and_uses_snapshot():
    points = np.stack([point_at_pixel(10.0, 8.0, 1.0 + 0.01 * i) for i in range(20)])
    scores = np.arange(20, dtype=np.float64)
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.5, far=2.0, max_points=5, sampling_policy="coverage",
        frustum_culling="voxel_blocks", block_size_m=0.3,
    )
    token = index.prebuild_blocks(config.block_size_m, scores)["block_score_token"]
    expected = stream_exact_final_target(index, np.eye(4), K, 16, 20, config, block_score_token=token)
    scores[:] = scores[::-1]
    actual = stream_exact_final_target(index, np.eye(4), K, 16, 20, config, block_score_token=token)
    np.testing.assert_array_equal(actual["indices"], expected["indices"])
    runtime = FinalTargetOracleRuntime(FinalTargetOracleConfig(), index, config, block_score_token=token)
    runtime.evaluate_frame(1, 0.0, points.copy(), np.eye(4), K, 16, 20)
    assert runtime.records[0]["scores"]["available"] is True


def test_pipeline_equal_without_projection_precap():
    rng = np.random.default_rng(7)
    points = rng.uniform([-1.5, -1.2, 0.2], [1.5, 1.2, 5.0], size=(300, 3))
    scores = rng.integers(-3, 4, size=points.shape[0]).astype(np.float64)
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.1, far=5.0, max_points=50, sampling_policy="coverage",
        tile_rows=3, tile_cols=4, depth_bins=3,
    )
    production = index.query_local_submap(np.eye(4), K, 16, 20, config, scores=scores)
    exact = index.stream_exact_final_target(np.eye(4), K, 16, 20, config, scores=scores, chunk_size=11)
    np.testing.assert_array_equal(production["indices"], exact["indices"])


def test_projection_precap_loss_is_reported():
    points = np.stack([
        point_at_pixel(2.0 + (i % 8) * 2.0, 2.0 + (i % 6) * 2.0, 0.6 + 0.02 * i)
        for i in range(90)
    ])
    index = MapSpatialIndex(points, backend="numpy_grid")
    local = LocalSubmapConfig(
        near=0.5, far=3.0, max_points=30, sampling_policy="coverage",
        tile_rows=3, tile_cols=4, depth_bins=3, frustum_culling="voxel_blocks",
        block_size_m=0.2, projection_pre_cap=8,
    )
    runtime = FinalTargetOracleRuntime(
        FinalTargetOracleConfig(min_correspondences=3, max_correspondence_distance=0.2), index, local
    )
    runtime.evaluate_frame(1, 0.0, points.copy(), np.eye(4), K, 16, 20)
    record = runtime.records[0]
    assert record["status"] == "ok"
    assert record["sets"]["recall"] < 1.0
    assert record["sets"]["symmetric_difference_count"] > 0
    assert record["histograms"]["joint_recall"] < 1.0


def test_hessian_comparison_detects_weak_direction_loss():
    exact_h = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    production_h = np.diag([0.1, 2.0, 3.0, 4.0, 5.0, 6.0])

    def values(matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        trace = float(np.trace(matrix))
        return {
            "hessian": matrix,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "trace": trace,
            "lambda_min": float(eigenvalues[0]),
            "normalized_lambda_min": float(eigenvalues[0] / trace),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "effective_rank": 6,
            "num_correspondences": 20,
        }

    comparison = _hessian_comparison(values(exact_h), values(production_h), weak_dim=1)
    assert comparison["weak_energy_recall"] == pytest.approx(0.1)
    assert comparison["lambda_min_recall"] == pytest.approx(0.1)
    assert comparison["weak_energy_loss"] > 0.0


def test_diagnostic_errors_are_recorded_and_evaluate_returns_none():
    points = np.stack([point_at_pixel(10.0, 8.0, 1.0 + 0.1 * i) for i in range(8)])
    runtime = FinalTargetOracleRuntime(
        FinalTargetOracleConfig(),
        MapSpatialIndex(points, backend="numpy_grid"),
        LocalSubmapConfig(near=0.5, far=3.0, max_points=5, sampling_policy="coverage"),
    )
    result = runtime.evaluate_frame(4, 1.5, points, np.eye(3), K, 16, 20)
    assert result is None
    assert runtime.records[0]["status"] == "error"
    assert runtime.records[0]["error_type"] == "ValueError"
    assert "pose" not in runtime.records[0]
    with pytest.raises(TypeError):
        runtime.evaluate_frame(5, 2.0, points, np.eye(4), K, 16, 20, gt_pose=np.eye(4))


def test_strict_diagnostic_config_and_bounded_retention():
    with pytest.raises(ValueError, match="diagnostic_only"):
        FinalTargetOracleConfig(diagnostic_only=False)
    points = np.stack([point_at_pixel(10.0, 8.0, 0.6 + 0.001 * i) for i in range(1000)])
    config = LocalSubmapConfig(near=0.5, far=2.0, max_points=17, sampling_policy="coverage")
    result = MapSpatialIndex(points, backend="numpy_grid").stream_exact_final_target(
        np.eye(4), K, 16, 20, config, chunk_size=9
    )
    assert result["visible_count"] == 1000
    assert result["retained_count"] == 17
    assert result["peak_retained_bound"] == 17


def test_csv_npz_and_standard_json_outputs_do_not_include_target_ids(tmp_path):
    points = np.stack([
        point_at_pixel(4.0 + i % 4, 5.0 + i % 3, 1.0 + 0.01 * i) for i in range(30)
    ])
    index = MapSpatialIndex(points, backend="numpy_grid")
    runtime = FinalTargetOracleRuntime(
        FinalTargetOracleConfig(min_correspondences=3, max_correspondence_distance=0.2),
        index,
        LocalSubmapConfig(near=0.5, far=2.0, max_points=20, sampling_policy="coverage"),
    )
    runtime.evaluate_frame(2, 0.2, points.copy(), np.eye(4), K, 16, 20)
    summary = runtime.save(tmp_path)
    assert summary["num_frames"] == 1
    with open(tmp_path / "frames.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert not any("target_ids" in name for name in rows[0])
    arrays = np.load(tmp_path / "hessians.npz")
    assert arrays["exact_hessian"].shape == (1, 6, 6)
    assert arrays["production_hessian"].shape == (1, 6, 6)
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["diagnostic_only"] is True
