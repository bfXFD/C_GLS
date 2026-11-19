import csv
import dataclasses
import json

import numpy as np
import pytest

from utils.fsm_local_map import LocalSubmapConfig
from utils.fsm_correspondence_weak import _camera_geometry, normalized_hessian_diagnostics
from utils.fsm_paired_reserve import (
    BaselineSnapshot,
    PairedShadowReserve,
    _actual_reserve_hessian,
    _normalized_diagnostic_hessian,
    runtime_corr_weak_scores,
)
from utils.fsm_robust_registration import evaluate_point_to_plane_diagnostics, se3_exp
from utils.fsm_runtime_reserve import rank_diverse_reserve


class FakeIndex:
    def __init__(self, count):
        self.num_points = count
        self.calls = 0

    def query_local_submap(self, *args, **kwargs):
        self.calls += 1
        return {"indices": np.arange(self.num_points, dtype=np.int64)}


def make_geometry(count=10):
    x = np.linspace(-0.4, 0.4, count)
    points = np.column_stack((x, np.zeros(count), np.linspace(1.0, 3.0, count)))
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (count, 1))
    scores = np.linspace(1.0, 2.0, count)
    ids = np.arange(100, 100 + count, dtype=np.int64)
    return points, normals, scores, ids


def make_snapshot(target_ids=(10, 11), image_delta=0):
    source = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.2], [-0.1, 0.0, 1.4]])
    target = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.2]])
    return BaselineSnapshot(
        frame_id=7,
        source="sequence/frame-000007",
        source_points=source,
        primary_pose=np.eye(4),
        primary_hessian=np.diag([0.01, 1.0, 2.0, 3.0, 4.0, 5.0]),
        target_points=target,
        target_normals=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        target_global_ids=np.asarray(target_ids, dtype=np.int64),
        target_tree=None,
        intrinsics=np.array([[20.0, 0.0, 4.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]]),
        image=np.full((6, 8, 3), image_delta, dtype=np.uint8),
        depth=np.ones((6, 8), dtype=np.float64),
        source_pixel_ids=np.array([1, 2, 3], dtype=np.int64),
    )


def fake_diagnostics(source_points, target_points, target_normals, transform, **kwargs):
    count = min(len(source_points), len(target_points))
    target_ids = np.asarray(kwargs["target_ids"], dtype=np.int64)
    jacobians = np.eye(6, dtype=np.float64)[:count]
    return {
        "fitness": count / len(source_points),
        "inlier_rmse": 0.01,
        "num_correspondences": count,
        "condition_number": 500.0,
        "effective_rank": 5,
        "hessian": jacobians.T @ jacobians,
        "jacobians": jacobians,
        "weights": np.ones(count, dtype=np.float64),
        "target_ids": target_ids[:count],
    }


class RegistrationRecorder:
    def __init__(self, fail_size=None):
        self.calls = []
        self.fail_size = fail_size

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_size == len(kwargs["target_points"]):
            raise RuntimeError("isolated arm failure")
        pose = np.asarray(kwargs["init_transform"], dtype=np.float64).copy()
        pose[0, 3] += len(kwargs["target_points"]) * 1e-4
        return {
            "transform": pose,
            "fitness": 0.9,
            "inlier_rmse": 0.02,
            "num_correspondences": len(kwargs["source_points"]),
            "condition_number": 10.0,
            "effective_rank": 6,
        }


def make_reserve(config=None, ids=None, supported=None, recorder=None):
    points, normals, scores, default_ids = make_geometry()
    global_ids = default_ids if ids is None else np.asarray(ids, dtype=np.int64)
    cfg = {
        "mode": "shadow",
        "policies": ["runtime_weak", "runtime_corr_weak", "saliency", "random"],
        "random_seeds": [3, 9],
        "budgets": [2, 4, 7],
        "trigger": {"always": True},
        "diversity": {"voxel_cap": None, "voxel_size": 0.01},
        "registration": {"min_correspondences": 1},
    }
    if config:
        cfg.update(config)
    index = FakeIndex(len(points))
    registration = recorder or RegistrationRecorder()
    reserve = PairedShadowReserve(
        cfg, points, normals, scores, global_ids, index,
        LocalSubmapConfig(near=0.1, far=5.0),
        reserve_supported=supported,
        registration_fn=registration,
        diagnostics_fn=fake_diagnostics,
    )
    return reserve, index, registration, normals


def row_ids(evaluation):
    return {
        (row["policy"], row["seed"], row["budget"]): ids.tolist()
        for row, ids in zip(evaluation.rows, evaluation.selected_global_ids)
    }


def test_snapshot_is_immutable_hash_stable_and_has_no_gt_api():
    first = make_snapshot()
    second = make_snapshot()
    changed = make_snapshot(image_delta=1)
    assert first.pair_id == second.pair_id
    assert first.pair_id != changed.pair_id
    assert all("gt" not in field.name.lower() and "ground_truth" not in field.name.lower() for field in dataclasses.fields(first))
    with pytest.raises(ValueError):
        first.target_points[0, 0] = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.frame_id = 8


def test_snapshot_accepts_single_channel_depth():
    snapshot = make_snapshot()
    normalized = BaselineSnapshot(
        frame_id=snapshot.frame_id,
        source=snapshot.source,
        source_points=snapshot.source_points,
        primary_pose=snapshot.primary_pose,
        primary_hessian=snapshot.primary_hessian,
        target_points=snapshot.target_points,
        target_normals=snapshot.target_normals,
        target_global_ids=snapshot.target_global_ids,
        target_tree=snapshot.target_tree,
        intrinsics=snapshot.intrinsics,
        image=snapshot.image,
        depth=snapshot.depth[..., None],
        source_pixel_ids=snapshot.source_pixel_ids,
        timestamp=snapshot.timestamp,
        primary_init_pose=snapshot.primary_init_pose,
        primary_accepted=snapshot.primary_accepted,
    )
    assert normalized.depth.shape == snapshot.depth.shape
    np.testing.assert_array_equal(normalized.depth, snapshot.depth)


def test_snapshot_rejects_duplicate_primary_target_ids():
    with pytest.raises(ValueError, match="unique"):
        make_snapshot(target_ids=(10, 10))


def test_config_requires_shadow_and_forbids_observable_active():
    with pytest.raises(ValueError, match="shadow"):
        make_reserve({"mode": "active"})
    config = {
        "paired_reserve": {"mode": "shadow", "policies": ["saliency"], "budgets": [2]},
        "observable_reserve": {"enabled": True, "mode": "active"},
    }
    points, normals, scores, ids = make_geometry()
    with pytest.raises(ValueError, match="forbidden"):
        PairedShadowReserve(
            config, points, normals, scores, ids, FakeIndex(len(points)),
            LocalSubmapConfig(near=0.1, far=5.0),
        )


def test_rank_diverse_returns_full_order_for_strict_nested_prefixes():
    points, _, scores, ids = make_geometry()
    order = rank_diverse_reserve(
        np.arange(len(points)), scores, points, np.eye(4),
        np.array([[20.0, 0.0, 4.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]]),
        6, 8, 2, 2, 2, 0.01, None, None, None, ids,
    )
    assert len(order) == len(points)
    np.testing.assert_array_equal(order[:2], order[:4][:2])
    np.testing.assert_array_equal(order[:4], order[:7][:4])


def test_query_once_rank_once_and_all_budgets_are_nested_with_shortfall():
    reserve, index, _, _ = make_reserve({"budgets": [4, 8, 12]})
    result = reserve.evaluate(make_snapshot())
    assert result.query_count == 1
    assert index.calls == 1
    assert set(result.ranking_counts.values()) == {1}
    assert len(result.ranking_counts) == 5
    grouped = {}
    for row, ids in zip(result.rows, result.selected_global_ids):
        grouped.setdefault((row["policy"], row["seed"]), []).append((row, ids))
    for arms in grouped.values():
        for (_, smaller), (_, larger) in zip(arms, arms[1:]):
            np.testing.assert_array_equal(smaller, larger[: len(smaller)])
        assert arms[-1][0]["shortfall"] == 2


def test_random_is_deterministic_and_policy_order_independent():
    snapshot = make_snapshot()
    first, _, _, _ = make_reserve()
    second, _, _, _ = make_reserve({"policies": ["random", "saliency", "runtime_corr_weak", "runtime_weak"]})
    assert row_ids(first.evaluate(snapshot)) == row_ids(first.evaluate(snapshot))
    assert row_ids(first.evaluate(snapshot)) == row_ids(second.evaluate(snapshot))


def test_runtime_corr_weak_is_pure_and_penalizes_duplicate_rows():
    means = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    hessian = np.diag([0.01, 1.0, 2.0, 3.0, 4.0, 5.0])
    from utils.fsm_runtime_reserve import reserve_point_jacobians_camera
    rows = reserve_point_jacobians_camera(means, normals, np.eye(4))
    first = runtime_corr_weak_scores(means, normals, np.eye(4), hessian, rows[:1])[0]
    second = runtime_corr_weak_scores(means, normals, np.eye(4), hessian, rows[:1])[0]
    np.testing.assert_array_equal(first, second)
    assert first[0] == pytest.approx(0.0)


def test_normals_are_frozen_concat_and_reruns_start_independently_from_primary(monkeypatch):
    def forbidden_estimate(*args, **kwargs):
        raise AssertionError("normal estimation must not run")

    monkeypatch.setattr("utils.fsm_robust_registration.estimate_normals", forbidden_estimate)
    recorder = RegistrationRecorder()
    reserve, _, _, reserve_normals = make_reserve(
        {"policies": ["saliency"], "budgets": [2, 4]}, recorder=recorder
    )
    snapshot = make_snapshot()
    result = reserve.evaluate(snapshot)
    assert all(row["status"] == "ok" for row in result.rows)
    assert len(recorder.calls) == 2
    for call, selected in zip(recorder.calls, result.selected_global_ids):
        np.testing.assert_array_equal(call["init_transform"], snapshot.primary_pose)
        np.testing.assert_array_equal(call["target_normals"][:2], snapshot.target_normals)
        local = np.flatnonzero(np.isin(reserve.global_ids, selected))
        np.testing.assert_array_equal(call["target_normals"][2:], reserve_normals[local])
    assert recorder.calls[0]["init_transform"] is not recorder.calls[1]["init_transform"]


def test_dedup_supported_pool_and_zero_feedback_state_free():
    ids = np.arange(100, 110)
    supported = np.array([True, False, True, False, True, False, True, False, True, False])
    reserve, index, _, _ = make_reserve(
        {"eligible_pool": "supported", "policies": ["saliency"], "budgets": [7]},
        ids=ids, supported=supported,
    )
    snapshot = make_snapshot(target_ids=(100, 999))
    first = reserve.evaluate(snapshot)
    second = reserve.evaluate(snapshot)
    assert index.calls == 2
    assert first.eligible_global_ids.tolist() == [102, 104, 106, 108]
    assert row_ids(first) == row_ids(second)
    assert all(row["candidate_feedback"] is False for row in first.rows)


def test_arm_exception_isolated_and_fixed_rerun_fields_present():
    recorder = RegistrationRecorder(fail_size=4)
    reserve, _, _, _ = make_reserve(
        {"policies": ["saliency"], "budgets": [2, 4]}, recorder=recorder
    )
    result = reserve.evaluate(make_snapshot())
    assert [row["status"] for row in result.rows] == ["error", "ok"]
    assert result.rows[0]["error_type"] == "RuntimeError"
    assert result.rows[1]["primary_fixed_effective_rank"] == 5
    assert result.rows[1]["augmented_fixed_effective_rank"] == 5
    assert result.rows[1]["rerun_effective_rank"] == 6


def test_artifact_roundtrip_uses_ragged_ids_without_pickle(tmp_path):
    reserve, _, _, _ = make_reserve({"policies": ["saliency", "random"], "random_seeds": [5]})
    evaluation = reserve.evaluate(make_snapshot())
    summary = reserve.save(evaluation, tmp_path)
    with np.load(tmp_path / "paired_reserve_ids.npz", allow_pickle=False) as archive:
        offsets = archive["offsets"]
        ids = archive["global_ids"]
        recovered = [ids[offsets[i]: offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
        assert int(archive["schema_version"]) == 2
        assert archive["selected_p_corr"].shape == ids.shape
        assert archive["selected_actual_correspondence"].shape == ids.shape
        assert archive["actual_reserve_hessian"].shape == (len(evaluation.rows), 6, 6)
        np.testing.assert_array_equal(archive["row_index"], np.arange(len(evaluation.rows)))
    assert recovered == [value.tolist() for value in evaluation.selected_global_ids]
    with open(tmp_path / "paired_reserve_rows.csv", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        assert len(csv_rows) == len(evaluation.rows)
        assert reader.fieldnames[:4] == ["pair_id", "frame_id", "timestamp", "source"]
        assert reader.fieldnames.index("schema_version") > reader.fieldnames.index("latency_amortized_ms")
    with open(tmp_path / "paired_reserve_summary.json", encoding="utf-8") as stream:
        saved_summary = json.load(stream)
        assert saved_summary["query_count"] == 1
        assert saved_summary["paired_no_trigger"] is False
    with open(tmp_path / "paired_reserve_provenance.json", encoding="utf-8") as stream:
        provenance = json.load(stream)
    assert provenance["candidate_feedback"] is False
    assert provenance["ridge_rho"] == pytest.approx(1e-6)
    assert provenance["metric_semantics"]["selection_score"] != provenance["metric_semantics"]["predicted_weak_gain"]
    assert summary["row_count"] == len(evaluation.rows)
    for row in evaluation.rows:
        assert row["reserve_correspondence_row_share"] == row["actual_correspondence_precision"]
        assert row["selected_reserve_hit_rate"] == row["actual_correspondence_fraction"]
    assert summary["reserve_correspondence_row_share_mean"] == summary["actual_correspondence_precision_mean"]
    assert summary["selected_reserve_hit_rate_mean"] == summary["actual_correspondence_fraction_mean"]


def test_trigger_protocol_provenance_and_or_semantics(tmp_path):
    reserve, _, _, _ = make_reserve({
        "policies": ["saliency"], "budgets": [2],
        "trigger_protocol": "absolute_or_scale_normalized_v1",
        "trigger": {
            "condition_number": 120.0,
            "normalized_lambda_min": 0.006,
            "effective_rank_below": 6,
        },
    })
    snapshot = make_snapshot()
    triggered, reasons = reserve._trigger(snapshot, {
        "condition_number": 10.0,
        "effective_rank": 6,
    })
    assert triggered is True
    assert reasons == ("normalized_lambda_min",)
    reserve.save(reserve.evaluate(snapshot), tmp_path)
    provenance = json.loads((tmp_path / "paired_reserve_provenance.json").read_text())
    assert provenance["trigger_protocol"] == "absolute_or_scale_normalized_v1"
    assert provenance["trigger"]["normalized_lambda_min"] == 0.006


def test_ridge_rho_is_configurable_and_validated():
    reserve, _, _, _ = make_reserve({"policies": ["saliency"], "budgets": [2], "ridge_rho": 1e-4})
    result = reserve.evaluate(make_snapshot())
    assert result.rows[0]["ridge_rho"] == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="ridge_rho"):
        make_reserve({"ridge_rho": 0.0})


def test_diagnostics_receive_strict_ordered_target_ids_and_actual_hessian_is_psd():
    calls = []

    def diagnostics(source_points, target_points, target_normals, transform, **kwargs):
        calls.append(np.asarray(kwargs["target_ids"], dtype=np.int64).copy())
        count = min(len(source_points), len(target_points))
        target_ids = np.asarray(kwargs["target_ids"], dtype=np.int64)
        jacobians = np.arange(count * 6, dtype=np.float64).reshape(count, 6) / 10.0
        return {
            "fitness": 1.0, "inlier_rmse": 0.01, "num_correspondences": count,
            "condition_number": 2.0, "effective_rank": min(count, 6),
            "hessian": jacobians.T @ jacobians, "jacobians": jacobians,
            "weights": np.linspace(0.5, 1.0, count), "target_ids": target_ids[:count],
        }

    reserve, _, _, _ = make_reserve({"policies": ["saliency"], "budgets": [2]})
    reserve.diagnostics_fn = diagnostics
    snapshot = make_snapshot()
    result = reserve.evaluate(snapshot)
    np.testing.assert_array_equal(calls[0], snapshot.target_global_ids)
    np.testing.assert_array_equal(
        calls[1], np.concatenate((snapshot.target_global_ids, result.selected_global_ids[0]))
    )
    assert np.unique(calls[1]).size == calls[1].size
    assert np.linalg.eigvalsh(result.actual_reserve_hessian[0])[0] >= -1e-12
    assert result.rows[0]["characteristic_length"] > 0.0
    assert 1 <= result.rows[0]["weak_subspace_dim"] <= 3


def test_runtime_corr_probability_cached_and_unavailable_for_other_policies(monkeypatch):
    probabilities = np.linspace(0.1, 1.0, 10)
    calls = []

    def probability(*args, **kwargs):
        calls.append(1)
        return {"probability": probabilities.copy()}

    monkeypatch.setattr("utils.fsm_paired_reserve.correspondence_probability", probability)
    reserve, _, _, _ = make_reserve({"policies": ["runtime_corr_weak", "saliency"], "budgets": [2, 4]})
    result = reserve.evaluate(make_snapshot())
    assert len(calls) == 1
    for row, ids, selected_probability in zip(result.rows, result.selected_global_ids, result.selected_p_corr):
        if row["policy"] == "runtime_corr_weak":
            local = np.searchsorted(reserve.global_ids, ids)
            np.testing.assert_allclose(selected_probability, probabilities[local])
            assert row["p_corr_available"] is True
            assert "selection_score_sum" in row
            assert "predicted_weak_gain" in row
        else:
            assert np.all(np.isnan(selected_probability))
            assert row["p_corr_available"] is False
            assert row["p_corr_mean"] == ""


def test_augmented_duplicate_global_ids_fail_arm_strictly():
    reserve, _, _, _ = make_reserve({"policies": ["saliency"], "budgets": [2]}, ids=np.arange(10, 20))
    result = reserve.evaluate(make_snapshot(target_ids=(10, 11)))
    assert result.rows[0]["status"] == "ok"
    assert set(result.selected_global_ids[0]).isdisjoint({10, 11})


@pytest.mark.parametrize("length", [0.2, 2.3, 8.0])
def test_normalized_diagnostic_hessians_match_transformed_world_left_rows(length):
    rng = np.random.default_rng(41)
    pose = se3_exp(np.array([0.3, -0.5, 0.2, 1.2, -0.7, 2.0]))
    jacobians = rng.normal(size=(12, 6))
    weights = np.linspace(0.2, 1.4, len(jacobians))
    target_ids = np.array([10, 11, 100, 100, 101, 102, 10, 101, 11, 102, 10, 11])
    selected = np.array([100, 101, 999])
    result = {"jacobians": jacobians, "weights": weights, "target_ids": target_ids}
    adjoint = _camera_geometry(pose)[2]
    coordinate_map = np.diag([1.0 / length] * 3 + [1.0] * 3)
    transformed_rows = jacobians @ adjoint @ coordinate_map
    expected = (transformed_rows.T * weights) @ transformed_rows
    actual = _normalized_diagnostic_hessian(result, length, pose)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    hessian_world = (jacobians.T * weights) @ jacobians
    np.testing.assert_allclose(actual, normalized_hessian_diagnostics(
        hessian_world, pose, characteristic_length=length,
    )["hessian"], atol=1e-12)
    scale_only = jacobians @ coordinate_map
    assert not np.allclose(actual, (scale_only.T * weights) @ scale_only)

    mask = np.isin(target_ids, selected)
    reserve_h, flags, count, unique = _actual_reserve_hessian(result, selected, length, pose)
    expected_reserve = (transformed_rows[mask].T * weights[mask]) @ transformed_rows[mask]
    np.testing.assert_allclose(reserve_h, expected_reserve, atol=1e-12)
    np.testing.assert_array_equal(flags, [True, True, False])
    assert (count, unique) == (4, 2)
    raw_h, raw_flags, raw_count, raw_unique = _actual_reserve_hessian(result, selected, None, pose)
    np.testing.assert_allclose(raw_h, (jacobians[mask].T * weights[mask]) @ jacobians[mask], atol=1e-12)
    np.testing.assert_array_equal(raw_flags, flags)
    assert (raw_count, raw_unique) == (count, unique)
    empty_h, empty_flags, count, unique = _actual_reserve_hessian(result, np.array([999]), length, pose)
    np.testing.assert_array_equal(empty_h, np.zeros((6, 6)))
    np.testing.assert_array_equal(empty_flags, [False])
    assert (count, unique) == (0, 0)


def test_paired_normalized_metrics_are_invariant_to_common_world_reexpression():
    rng = np.random.default_rng(93)
    source = rng.uniform([-1.5, -1.0, 1.0], [1.5, 1.0, 4.0], size=(18, 3))
    normals_camera = rng.normal(size=source.shape)
    normals_camera /= np.linalg.norm(normals_camera, axis=1, keepdims=True)
    pose = se3_exp(np.array([0.2, -0.4, 0.3, 0.8, -1.2, 0.5]))
    world_change = se3_exp(np.array([-0.5, 0.1, -0.2, 2.0, -3.0, 1.0]))
    evaluations = []
    for reference_pose in (pose, world_change @ pose):
        points_world = source @ reference_pose[:3, :3].T + reference_pose[:3, 3]
        normals_world = normals_camera @ reference_pose[:3, :3].T
        primary_ids = np.arange(9)
        reserve_ids = np.arange(100, 109)
        primary = evaluate_point_to_plane_diagnostics(
            source, points_world[:9], normals_world[:9], reference_pose,
            min_correspondences=1, target_ids=primary_ids, return_correspondences=True,
        )
        # Raw diagnostics are world-left, not camera-frame rows.
        expected_rows = np.column_stack((
            np.cross(points_world[primary["source_indices"]], normals_world[primary["target_indices"]]),
            normals_world[primary["target_indices"]],
        ))
        np.testing.assert_allclose(primary["jacobians"], expected_rows, atol=1e-13)
        snapshot = dataclasses.replace(
            make_snapshot(), source_points=source, primary_pose=reference_pose,
            primary_hessian=primary["hessian"], target_points=points_world[:9],
            target_normals=normals_world[:9], target_global_ids=primary_ids,
            source_pixel_ids=np.arange(len(source)),
        )
        reserve = PairedShadowReserve(
            {"policies": ["saliency"], "budgets": [9], "trigger": {"always": True},
             "registration": {"min_correspondences": 1},
             "diversity": {"voxel_cap": None},
             "adaptive_weak": {"characteristic_length_min": 8.0, "characteristic_length_max": 8.0}},
            points_world[9:], normals_world[9:], np.arange(9, dtype=np.float64),
            reserve_ids, FakeIndex(9), LocalSubmapConfig(near=0.1, far=5.0),
            registration_fn=RegistrationRecorder(),
        )
        evaluation = reserve.evaluate(snapshot)
        row = evaluation.rows[0]
        assert row["status"] == "ok", row["error_message"]
        assert row["characteristic_length"] == 8.0
        assert row["actual_correspondence_count"] == 9
        for raw_name, normalized_name in (
            ("primary_hessian", "primary_hessian_normalized"),
            ("augmented_hessian", "augmented_hessian_normalized"),
            ("actual_reserve_hessian", "actual_reserve_hessian_normalized"),
        ):
            expected = normalized_hessian_diagnostics(
                getattr(evaluation, raw_name)[0], reference_pose, characteristic_length=8.0,
            )["hessian"]
            np.testing.assert_allclose(getattr(evaluation, normalized_name)[0], expected, atol=1e-12)
        np.testing.assert_allclose(
            evaluation.augmented_hessian_normalized[0] - evaluation.primary_hessian_normalized[0],
            evaluation.actual_reserve_hessian_normalized[0], atol=1e-12,
        )
        assert row["net_weak_gain"] == pytest.approx(row["actual_reserve_weak_gain"], abs=1e-12)
        evaluations.append(evaluation)

    first, second = evaluations
    for name in ("primary_hessian_normalized", "augmented_hessian_normalized",
                 "actual_reserve_hessian_normalized"):
        np.testing.assert_allclose(getattr(first, name), getattr(second, name), atol=1e-12)
    inverse_adjoint = np.linalg.inv(_camera_geometry(world_change)[2])
    for name in ("primary_hessian", "augmented_hessian", "actual_reserve_hessian"):
        expected = inverse_adjoint.T @ getattr(first, name)[0] @ inverse_adjoint
        np.testing.assert_allclose(getattr(second, name)[0], expected, atol=1e-11)
    for metric in ("characteristic_length", "weak_subspace_dim", "primary_logdet_gain",
                   "augmented_logdet_gain", "net_logdet_gain", "baseline_weak_information",
                   "actual_reserve_weak_gain", "net_weak_gain", "predicted_weak_gain"):
        assert first.rows[0][metric] == pytest.approx(second.rows[0][metric], abs=1e-10)
