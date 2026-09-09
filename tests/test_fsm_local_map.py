import builtins

import numpy as np
import pytest

from utils.fsm_local_map import (
    LocalSubmapConfig,
    MapSpatialIndex,
    camera_frustum_planes_world,
    conservative_aabb_halfspace_test,
    points_in_camera_frustum,
    query_local_submap,
)


K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
CONFIG = LocalSubmapConfig(near=0.5, far=5.0)


def point_at_pixel(u, v, depth, intrinsics=K):
    return np.array(
        [
            (u - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ]
    )


def test_identity_translation_and_rotation_use_c2w_convention():
    identity_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    np.testing.assert_array_equal(
        points_in_camera_frustum(identity_points, np.eye(4), K, 8, 10, CONFIG),
        [True, False],
    )

    translated = np.eye(4)
    translated[:3, 3] = [2.0, 3.0, 4.0]
    translated_points = np.array([[2.0, 3.0, 5.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(
        points_in_camera_frustum(translated_points, translated, K, 8, 10, CONFIG),
        [True, False],
    )

    rotated = np.eye(4)
    rotated[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rotated_points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        points_in_camera_frustum(rotated_points, rotated, K, 8, 10, CONFIG),
        [True, False],
    )


def test_depth_image_boundaries_and_margin_are_exact():
    points = np.stack(
        [
            point_at_pixel(0.0, 0.0, 0.5),
            point_at_pixel(10.0, 4.0, 1.0),
            point_at_pixel(9.999, 7.999, 5.0),
            point_at_pixel(-0.25, 4.0, 1.0),
            point_at_pixel(5.0, 4.0, 5.001),
        ]
    )
    np.testing.assert_array_equal(
        points_in_camera_frustum(points, np.eye(4), K, 8, 10, CONFIG),
        [True, False, True, False, False],
    )
    margin_config = LocalSubmapConfig(near=0.5, far=5.0, margin_px=0.5)
    assert points_in_camera_frustum(points, np.eye(4), K, 8, 10, margin_config)[3]


@pytest.mark.parametrize("backend", ["numpy_grid", "auto"])
def test_far_corner_candidate_query_is_complete(backend):
    far_corner = point_at_pixel(9.999, 7.999, 5.0)
    points = np.array([far_corner, [0.0, 0.0, 2.0], [20.0, 20.0, 20.0]])
    index = MapSpatialIndex(points, backend=backend, voxel_size=0.5)
    result = query_local_submap(index, np.eye(4), K, 8, 10, CONFIG)
    reference = np.flatnonzero(points_in_camera_frustum(points, np.eye(4), K, 8, 10, CONFIG))
    np.testing.assert_array_equal(np.sort(result["indices"]), reference)
    assert result["candidate_count"] >= result["count"] == 2
    assert result["backend"] in ("scipy_ckdtree", "numpy_grid")
    assert result["query_latency_ms"] >= 0.0


def test_auto_backend_has_deterministic_numpy_grid_fallback(monkeypatch):
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    points = np.array([[0.0, 0.0, 1.0], [100.0, 100.0, 100.0]])
    index = MapSpatialIndex(points, backend="auto", voxel_size=0.25)
    assert index.backend == "numpy_grid"
    np.testing.assert_array_equal(index.query_radius(np.zeros(3), 2.0), [0])


def test_cap_is_stable_by_score_then_index_or_depth_then_index():
    points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=0.5)
    capped = LocalSubmapConfig(near=0.1, far=4.0, max_points=2)

    depth_result = index.query_local_submap(np.eye(4), K, 8, 10, capped)
    np.testing.assert_array_equal(depth_result["indices"], [1, 3])
    assert depth_result["frustum_count"] == 4
    assert depth_result["count"] == 2

    scores = np.array([0.9, 1.0, 1.0, 0.1])
    score_result = index.query_local_submap(np.eye(4), K, 8, 10, capped, scores=scores)
    np.testing.assert_array_equal(score_result["indices"], [1, 2])


def test_coverage_cap_preserves_tiles_depth_and_is_order_deterministic(monkeypatch):
    points = np.stack([
        point_at_pixel(1.0, 1.0, 0.6),
        point_at_pixel(1.0, 1.0, 1.0),
        point_at_pixel(1.0, 1.0, 4.0),
        point_at_pixel(8.0, 1.0, 0.6),
        point_at_pixel(8.0, 1.0, 4.0),
        point_at_pixel(1.0, 6.0, 0.6),
        point_at_pixel(1.0, 6.0, 4.0),
        point_at_pixel(8.0, 6.0, 0.6),
        point_at_pixel(8.0, 6.0, 4.0),
    ])
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=0.5)
    config = LocalSubmapConfig(
        near=0.5, far=5.0, max_points=8, sampling_policy="coverage",
        tile_rows=2, tile_cols=2, depth_bins=2,
    )
    scores = np.array([100.0, 99.0, 98.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    forward = index.query_local_submap(np.eye(4), K, 8, 10, config, scores=scores)
    original = index.query_radius
    monkeypatch.setattr(index, "query_radius", lambda *args, **kwargs: original(*args, **kwargs)[::-1])
    reverse = index.query_local_submap(np.eye(4), K, 8, 10, config, scores=scores)
    assert forward["count"] == 8
    assert forward["selected_tile_count"] == 4
    assert forward["selected_depth_bin_count"] == 2
    assert forward["selected_tile_depth_count"] == 8
    np.testing.assert_array_equal(np.sort(forward["indices"]), np.sort(reverse["indices"]))


def test_stage_indices_are_map_rows_nested_and_frustum_is_precap():
    points = np.stack(
        [
            point_at_pixel(5.0, 4.0, -1.0),
            point_at_pixel(5.0, 4.0, 0.25),
            point_at_pixel(5.0, 4.0, 1.0),
            point_at_pixel(20.0, 4.0, 1.0),
            point_at_pixel(2.0, 2.0, 2.0),
            point_at_pixel(8.0, 6.0, 4.0),
            np.array([100.0, 100.0, 100.0]),
        ]
    )
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=0.5)
    config = LocalSubmapConfig(
        near=0.5,
        far=5.0,
        max_points=2,
        sampling_policy="coverage",
        tile_rows=2,
        tile_cols=2,
        depth_bins=2,
    )

    default_result = index.query_local_submap(np.eye(4), K, 8, 10, config)
    for key in (
        "candidate_indices",
        "projection_input_indices",
        "positive_depth_indices",
        "depth_range_indices",
        "frustum_indices",
        "selected_indices",
    ):
        assert key not in default_result

    result = query_local_submap(
        index, np.eye(4), K, 8, 10, config, return_stage_indices=True
    )
    stage_names = (
        "candidate_indices",
        "projection_input_indices",
        "positive_depth_indices",
        "depth_range_indices",
        "frustum_indices",
        "selected_indices",
    )
    for name in stage_names:
        assert result[name].dtype == np.int64
    for parent, child in zip(stage_names, stage_names[1:]):
        assert set(result[child]).issubset(set(result[parent]))
    np.testing.assert_array_equal(result["indices"], result["selected_indices"])
    np.testing.assert_array_equal(np.sort(result["frustum_indices"]), [2, 4, 5])
    assert result["frustum_indices"].size == result["precap_count"] == 3
    assert result["selected_indices"].size == result["count"] == 2
    assert result["candidate_indices"].size == result["candidate_count"]
    assert result["positive_depth_indices"].size == result["positive_depth_count"]
    assert result["depth_range_indices"].size == result["depth_range_count"]


def test_coverage_insufficient_keeps_all_and_reports_nonnegative_breakdown():
    points = np.stack([point_at_pixel(1.0, 1.0, 1.0), point_at_pixel(8.0, 6.0, 4.0)])
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=0.5)
    config = LocalSubmapConfig(
        near=0.5, far=5.0, max_points=10, sampling_policy="coverage",
        tile_rows=2, tile_cols=2, depth_bins=2,
    )
    result = index.query_local_submap(np.eye(4), K, 8, 10, config)
    np.testing.assert_array_equal(np.sort(result["indices"]), [0, 1])
    assert result["positive_depth_count"] >= result["depth_range_count"] >= result["precap_count"] == 2
    for name in ("query_latency_ms", "radius_latency_ms", "projection_latency_ms", "sampling_latency_ms", "final_latency_ms"):
        assert result[name] >= 0.0


def test_query_radius_sort_result_default_and_opt_out():
    points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=4.0)
    np.testing.assert_array_equal(index.query_radius(np.zeros(3), 3.0), [0, 1, 2])
    assert set(index.query_radius(np.zeros(3), 3.0, sort_result=False)) == {0, 1, 2}


@pytest.mark.parametrize(
    "config",
    [
        LocalSubmapConfig(near=0.0, far=1.0),
    ],
)
def test_invalid_camera_and_map_inputs(config):
    with pytest.raises(ValueError):
        MapSpatialIndex(np.empty((0, 3)))
    with pytest.raises(ValueError):
        MapSpatialIndex(np.array([[np.nan, 0.0, 0.0]]))

    points = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        points_in_camera_frustum(points, np.eye(3), K, 8, 10, config)
    bad_k = K.copy()
    bad_k[0, 0] = 0.0
    with pytest.raises(ValueError):
        points_in_camera_frustum(points, np.eye(4), bad_k, 8, 10, config)
    with pytest.raises(ValueError):
        points_in_camera_frustum(points, np.eye(4), K, 0, 10, config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"near": -1.0, "far": 2.0},
        {"near": 1.0, "far": 1.0},
        {"near": 0.0, "far": 2.0, "margin_px": -1.0},
        {"near": 0.0, "far": 2.0, "radius_margin": -1.0},
        {"near": 0.0, "far": 2.0, "max_points": 0},
        {"near": 0.0, "far": 2.0, "sampling_policy": "bad"},
        {"near": 0.0, "far": 2.0, "tile_rows": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LocalSubmapConfig(**kwargs)


def test_frustum_planes_match_exact_centers_for_pose_asymmetry_and_margin():
    asymmetric_k = np.array([[8.0, 0.0, 2.0], [0.0, 12.0, 6.0], [0.0, 0.0, 1.0]])
    angle = np.deg2rad(31.0)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    c2w[:3, 3] = [3.0, -2.0, 5.0]
    config = LocalSubmapConfig(near=0.3, far=7.0, margin_px=0.4)
    camera_points = np.stack(
        [
            point_at_pixel(-0.4, 4.0, 1.0, asymmetric_k),
            point_at_pixel(9.5, 7.0, 6.0, asymmetric_k),
            point_at_pixel(10.5, 4.0, 2.0, asymmetric_k),
            point_at_pixel(2.0, 6.0, 7.1, asymmetric_k),
        ]
    )
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    exact = points_in_camera_frustum(world_points, c2w, asymmetric_k, 8, 10, config)
    planes = camera_frustum_planes_world(c2w, asymmetric_k, 8, 10, config)
    halfspace = np.all(world_points @ planes[:, :3].T + planes[:, 3] >= -1e-12, axis=1)
    np.testing.assert_array_equal(halfspace, exact)


def test_conservative_aabb_halfspace_boundary_and_negative_coordinates():
    planes = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.0]])
    lower = np.array([[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]])
    upper = np.array([[-0.1, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    np.testing.assert_array_equal(
        conservative_aabb_halfspace_test(lower, upper, planes),
        [False, True, True, False],
    )


@pytest.mark.parametrize("size", [0.07, 0.3, 1.1])
def test_voxel_block_random_property_has_exact_center_recall_one(size):
    rng = np.random.default_rng(1234)
    points = rng.uniform([-8.0, -8.0, -3.0], [8.0, 8.0, 9.0], size=(4000, 3))
    index = MapSpatialIndex(points, backend="numpy_grid", voxel_size=0.41)
    config = LocalSubmapConfig(
        near=0.2, far=8.0, margin_px=0.3, frustum_culling="voxel_blocks", block_size_m=size,
    )
    result = index.query_local_submap(np.eye(4), K, 8, 10, config, return_stage_indices=True)
    oracle = np.flatnonzero(points_in_camera_frustum(points, np.eye(4), K, 8, 10, config))
    np.testing.assert_array_equal(np.sort(result["indices"]), oracle)
    assert set(oracle).issubset(set(result["block_candidate_indices"]))


def test_voxel_blocks_are_backend_invariant():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(500, 3))
    points[:, 2] += 2.0
    config = LocalSubmapConfig(near=0.1, far=5.0, frustum_culling="voxel_blocks", block_size_m=0.23)
    numpy_result = MapSpatialIndex(points, backend="numpy_grid").query_local_submap(
        np.eye(4), K, 8, 10, config, return_stage_indices=True
    )
    auto_result = MapSpatialIndex(points, backend="auto").query_local_submap(
        np.eye(4), K, 8, 10, config, return_stage_indices=True
    )
    np.testing.assert_array_equal(numpy_result["block_candidate_indices"], auto_result["block_candidate_indices"])
    np.testing.assert_array_equal(numpy_result["indices"], auto_result["indices"])


def test_projection_pre_cap_is_strict_deterministic_and_stage_nested():
    rng = np.random.default_rng(44)
    points = rng.uniform([-2.0, -2.0, 0.2], [2.0, 2.0, 5.0], size=(300, 3))
    scores = rng.normal(size=points.shape[0])
    config = LocalSubmapConfig(
        near=0.1, far=6.0, frustum_culling="voxel_blocks", block_size_m=0.4,
        projection_pre_cap=37, tile_rows=3, tile_cols=4, depth_bins=3,
    )
    first = MapSpatialIndex(points, backend="numpy_grid").query_local_submap(
        np.eye(4), K, 8, 10, config, scores=scores, return_stage_indices=True
    )
    second = MapSpatialIndex(points, backend="auto").query_local_submap(
        np.eye(4), K, 8, 10, config, scores=scores, return_stage_indices=True
    )
    assert first["projection_input_count"] == 37
    assert first["candidate_count"] >= first["projection_input_count"] >= first["count"]
    assert set(first["projection_input_indices"]).issubset(set(first["block_candidate_indices"]))
    assert set(first["indices"]).issubset(set(first["projection_input_indices"]))
    assert first["candidate_indices"].size == first["candidate_count"]
    assert first["projection_input_indices"].size == first["projection_input_count"]
    np.testing.assert_array_equal(first["projection_input_indices"], second["projection_input_indices"])


def test_oracle_measure_does_not_change_result_and_reports_recall():
    rng = np.random.default_rng(7)
    points = rng.uniform([-2.0, -2.0, 0.1], [2.0, 2.0, 5.0], size=(200, 3))
    index = MapSpatialIndex(points, backend="numpy_grid")
    base = LocalSubmapConfig(near=0.1, far=5.0, frustum_culling="voxel_blocks", projection_pre_cap=25)
    measured = LocalSubmapConfig(
        near=0.1, far=5.0, frustum_culling="voxel_blocks", projection_pre_cap=25,
        oracle_mode="measure", oracle_point_chunk=17,
    )
    base_result = index.query_local_submap(np.eye(4), K, 8, 10, base)
    measured_result = index.query_local_submap(np.eye(4), K, 8, 10, measured)
    np.testing.assert_array_equal(base_result["indices"], measured_result["indices"])
    assert measured_result["oracle_count"] == np.count_nonzero(
        points_in_camera_frustum(points, np.eye(4), K, 8, 10, measured)
    )
    assert measured_result["oracle_block_recall"] == 1.0
    assert 0.0 <= measured_result["oracle_projection_recall"] <= 1.0
    assert measured_result["oracle_projection_recall"] == measured_result["oracle_frustum_recall"]


def test_oracle_enforce_injects_feasible_and_handles_both_overflow_policies():
    points = np.stack([point_at_pixel(5.0, 4.0, 1.0 + i * 0.01) for i in range(60)])
    index = MapSpatialIndex(points, backend="numpy_grid")
    feasible = LocalSubmapConfig(
        near=0.5, far=3.0, frustum_culling="voxel_blocks", projection_pre_cap=80,
        oracle_mode="enforce", oracle_point_chunk=7,
    )
    feasible_result = index.query_local_submap(np.eye(4), K, 8, 10, feasible, return_stage_indices=True)
    assert feasible_result["oracle_projection_recall"] == 1.0
    assert feasible_result["projection_input_count"] <= 80

    fail_closed = LocalSubmapConfig(
        near=0.5, far=3.0, frustum_culling="voxel_blocks", projection_pre_cap=10,
        oracle_mode="enforce", oracle_overflow_policy="fail_closed", oracle_point_chunk=7,
    )
    failed = index.query_local_submap(np.eye(4), K, 8, 10, fail_closed)
    assert failed["status"] == "exact_recall_infeasible"
    assert failed["oracle_overflow"]
    assert failed["indices"].size == 0

    uncapped = LocalSubmapConfig(
        near=0.5, far=3.0, frustum_culling="voxel_blocks", projection_pre_cap=10,
        oracle_mode="enforce", oracle_overflow_policy="uncapped_chunked", oracle_point_chunk=7,
    )
    bypassed = index.query_local_submap(np.eye(4), K, 8, 10, uncapped)
    assert bypassed["projection_cap_bypassed"]
    assert bypassed["projection_input_count"] > 10
    assert bypassed["oracle_frustum_recall"] == 1.0


@pytest.mark.parametrize("with_scores", [False, True])
def test_block_representative_matches_point_reference_randomized(with_scores):
    rng = np.random.default_rng(20260905)
    for size in (0.17, 0.4):
        points = rng.uniform([-2.5, -2.5, -0.3], [2.5, 2.5, 5.5], size=(600, 3))
        points[:20] -= np.array([1.0, 1.0, 0.0])
        scores = None
        if with_scores:
            scores = rng.integers(-3, 4, size=points.shape[0]).astype(np.float64)
        index = MapSpatialIndex(points, backend="numpy_grid")
        for cap in (1, 7, 31, 97):
            config = LocalSubmapConfig(
                near=0.1, far=6.0, frustum_culling="voxel_blocks", block_size_m=size,
                projection_pre_cap=cap, tile_rows=3, tile_cols=4, depth_bins=3,
            )
            score_array, _, _ = index._score_array_and_fingerprint(scores)
            planes = camera_frustum_planes_world(np.eye(4), K, 8, 10, config)
            block_ids, csr = index._query_frustum_blocks(planes, size, config.block_test_chunk, score_array)
            candidates = index.expand_blocks(block_ids, csr)
            if candidates.size <= cap:
                continue
            reference = index._block_coverage_pre_cap(
                candidates, np.eye(4), K, 8, 10, config, score_array
            )
            fast, metrics = index._block_representative_pre_cap(
                block_ids, csr, np.eye(4), K, 8, 10, config
            )
            np.testing.assert_array_equal(fast, reference)
            assert metrics["precap_strategy"] == "block_representative"
            assert metrics["representative_points_considered"] >= cap


def test_block_precap_does_not_expand_candidates_without_diagnostics(monkeypatch):
    rng = np.random.default_rng(11)
    points = rng.uniform([-2.0, -2.0, 0.1], [2.0, 2.0, 5.0], size=(500, 3))
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.1, far=6.0, frustum_culling="voxel_blocks", block_size_m=0.3,
        projection_pre_cap=20, oracle_mode="measure",
    )
    index.prebuild_blocks(config.block_size_m)
    monkeypatch.setattr(index, "expand_blocks", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    result = index.query_local_submap(np.eye(4), K, 8, 10, config)
    assert result["precap_strategy"] == "block_representative"
    assert result["projection_input_count"] == 20
    assert result["oracle_block_recall"] == 1.0


def test_block_snapshot_cache_score_rebuild_and_memory_estimate():
    points = np.array([
        [-0.2, 0.0, 1.0], [-0.1, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0],
    ])
    index = MapSpatialIndex(points, backend="numpy_grid")
    first = index.prebuild_blocks(0.5, np.array([1.0, 2.0, 3.0, 4.0]))
    cached = index.prebuild_blocks(0.5, np.array([1.0, 2.0, 3.0, 4.0]))
    changed = index.prebuild_blocks(0.5, np.array([4.0, 3.0, 2.0, 1.0]))
    assert first["cached"] is False
    assert cached["cached"] is True
    assert changed["cached"] is False
    assert first["score_fingerprint"] != changed["score_fingerprint"]
    csr = index._active_block_csr
    assert csr is not None
    assert csr.scores is not None and not csr.scores.flags.writeable
    assert csr.memory_bytes == (
        csr.coords.nbytes + csr.offsets.nbytes + csr.point_ids.nbytes
        + csr.block_best_score.nbytes + csr.scores.nbytes
    )
    assert csr.point_ids.size == points.shape[0]
    assert index.block_index_build_ms == changed["block_index_build_ms"]
    assert index.block_index_memory_bytes == changed["block_index_memory_bytes"]
    assert changed["block_index_build_ms"] >= changed["block_score_hash_ms"]
    assert changed["block_index_build_ms"] >= changed["block_geometry_sort_ms"]
    assert changed["peak_working_set_estimate_bytes"] >= changed["block_index_memory_bytes"]


def test_public_scores_in_place_mutation_rebuilds_and_changes_selection():
    points = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.02, 0.0, 1.0]])
    scores = np.array([3.0, 2.0, 1.0])
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.1, far=2.0, frustum_culling="voxel_blocks",
        block_size_m=0.5, projection_pre_cap=1,
    )
    first = index.query_local_submap(np.eye(4), K, 8, 10, config, scores=scores)
    first_fingerprint = index._active_block_csr.score_fingerprint
    scores[:] = [1.0, 2.0, 3.0]
    second = index.query_local_submap(np.eye(4), K, 8, 10, config, scores=scores)
    np.testing.assert_array_equal(first["indices"], [0])
    np.testing.assert_array_equal(second["indices"], [2])
    assert index._active_block_csr.score_fingerprint != first_fingerprint


def test_block_cache_keeps_one_snapshot_and_memory_does_not_accumulate():
    rng = np.random.default_rng(6)
    points = rng.normal(size=(100, 3))
    index = MapSpatialIndex(points, backend="numpy_grid")
    memory = []
    for iteration in range(10):
        index.prebuild_blocks(0.4, rng.normal(size=points.shape[0]) + iteration)
        assert index.block_snapshot_count == 1
        memory.append(index.memory_bytes)
    assert index.memory_bytes == memory[-1]
    assert max(memory) == min(memory)


def test_block_score_token_query_does_not_rehash_and_uses_snapshot(monkeypatch):
    points = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]])
    scores = np.array([2.0, 1.0])
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.1, far=2.0, frustum_culling="voxel_blocks",
        block_size_m=0.5, projection_pre_cap=1,
    )
    token = index.prebuild_blocks(config.block_size_m, scores)["block_score_token"]
    scores[:] = [1.0, 2.0]
    monkeypatch.setattr(
        index, "_score_array_and_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected score hash")),
    )
    result = index.query_local_submap(
        np.eye(4), K, 8, 10, config, block_score_token=token,
    )
    np.testing.assert_array_equal(result["indices"], [0])


def test_skewed_bucket_round_robin_removes_exhausted_buckets_promptly():
    points = np.stack([
        point_at_pixel(1.0, 1.0, 1.0 + 0.001 * i) for i in range(80)
    ] + [
        point_at_pixel(8.0, 6.0, 1.0),
    ])
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(
        near=0.5, far=2.0, frustum_culling="voxel_blocks", block_size_m=0.05,
        projection_pre_cap=40, tile_rows=2, tile_cols=2, depth_bins=2,
    )
    result = index.query_local_submap(np.eye(4), K, 8, 10, config)
    assert result["projection_input_count"] == 40
    assert result["representative_bucket_turns"] <= 42


@pytest.mark.parametrize(
    "config",
    [
        LocalSubmapConfig(
            near=0.2, far=5.0, margin_px=0.0, frustum_culling="voxel_blocks",
            block_size_m=0.23, projection_pre_cap=13, tile_rows=2, tile_cols=3, depth_bins=2,
        ),
        LocalSubmapConfig(
            near=0.2, far=5.0, margin_px=0.75, frustum_culling="voxel_blocks",
            block_size_m=0.41, projection_pre_cap=47, tile_rows=3, tile_cols=4, depth_bins=3,
        ),
        LocalSubmapConfig(
            near=0.05, far=6.0, margin_px=1.25, frustum_culling="voxel_blocks",
            block_size_m=0.67, projection_pre_cap=91, tile_rows=4, tile_cols=5, depth_bins=4,
        ),
    ],
)
def test_fast_representative_matches_slow_reference_across_full_active_configs(config):
    rng = np.random.default_rng(90210 + int(config.projection_pre_cap))
    points = rng.uniform([-3.0, -3.0, -0.5], [3.0, 3.0, 6.5], size=(900, 3))
    boundary_camera = np.stack([
        point_at_pixel(-config.margin_px, 4.0, config.near),
        point_at_pixel(10.0 + config.margin_px - 1e-9, 4.0, config.far),
        point_at_pixel(5.0, -config.margin_px, 2.0),
        point_at_pixel(5.0, 8.0 + config.margin_px - 1e-9, 2.0),
    ])
    angle = np.deg2rad(19.0)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    pose[:3, 3] = [0.4, -0.3, 0.2]
    points[: boundary_camera.shape[0]] = boundary_camera @ pose[:3, :3].T + pose[:3, 3]
    scores = rng.integers(-5, 6, size=points.shape[0]).astype(np.float64)
    index = MapSpatialIndex(points, backend="numpy_grid")
    score_array, _, _ = index._score_array_and_fingerprint(scores)
    planes = camera_frustum_planes_world(pose, K, 8, 10, config)
    block_ids, csr = index._query_frustum_blocks(
        planes, config.block_size_m, config.block_test_chunk, score_array,
    )
    candidates = index.expand_blocks(block_ids, csr)
    assert candidates.size > int(config.projection_pre_cap)
    reference = index._block_coverage_pre_cap(
        candidates, pose, K, 8, 10, config, score_array,
    )
    fast, metrics = index._block_representative_pre_cap(
        block_ids, csr, pose, K, 8, 10, config,
    )
    np.testing.assert_array_equal(fast, reference)
    assert metrics["precap_selected_block_count"] <= metrics["precap_block_count"]
    assert metrics["precap_selected_bucket_count"] <= metrics["precap_bucket_count"]
    assert metrics["precap_selected_boundary_block_count"] <= metrics["precap_boundary_block_count"]


def test_first_prebuilt_block_query_has_no_block_build(monkeypatch):
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 2.0]])
    index = MapSpatialIndex(points, backend="numpy_grid")
    config = LocalSubmapConfig(near=0.1, far=3.0, frustum_culling="voxel_blocks", block_size_m=0.3)
    index.prebuild_blocks(config.block_size_m)
    monkeypatch.setattr(index, "prebuild_blocks", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    result = index.query_local_submap(np.eye(4), K, 8, 10, config)
    assert result["count"] == 3


def test_radius_cap_stays_on_point_reference_strategy():
    points = np.stack([point_at_pixel(5.0, 4.0, 1.0 + i * 0.01) for i in range(30)])
    result = MapSpatialIndex(points, backend="numpy_grid").query_local_submap(
        np.eye(4), K, 8, 10,
        LocalSubmapConfig(near=0.5, far=3.0, frustum_culling="radius", projection_pre_cap=7),
    )
    assert result["precap_strategy"] == "point_reference"
    assert result["projection_input_count"] == 7
