"""Synthetic localization integration checks for the CN holdout route."""

import csv

import numpy as np
import pytest

import scripts.fsm_localize as localize
import utils.fsm_runtime_reserve as runtime_reserve
from tests.test_fsm_localize import make_config, make_fake_map, make_frames, successful_result
from utils.fsm_candidate_verification import split_source_validation
from utils.fsm_correspondence_weak import normalized_hessian_diagnostics


@pytest.fixture
def cn_scene(tmp_path, monkeypatch):
    map_path, points = make_fake_map(tmp_path)
    frames = make_frames(points, num_frames=3)
    original_depths = [frame["depth"].copy() for frame in frames]
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
    config["runtime_memory_metrics"] = {"enabled": False}
    config["observable_reserve"] = {
        "enabled": False, "mode": "off",
        "selection": "runtime_corr_weak",
        "spectral_coordinates": "primary_camera_depth",
        "budget": 1, "max_points": 1, "max_visible": 1,
        "trigger": {"iterations": 1},
        "candidate_normal_mode": "frozen_concat",
        "candidate_target_mode": "append",
        "rerun_iterations": 2,
        "accept_only_if": {"mode": "all", "lambda_ratio": 1.5},
        "candidate_verifier": {
            "enabled": True, "block_size": 1, "folds": 2, "fold": 0,
            "min_group_size": 1,
        },
    }
    # Keep the real map, spatial queries and runtime reserve initialization;
    # only sidecar serialization/provenance validation is outside this test.
    load_map = localize.load_fsm_map

    def map_with_normals(path):
        data = load_map(path)
        data.update(
            normals=np.tile([0.0, 0.0, 1.0], (len(points), 1)),
            source_params_sha256=np.asarray("a" * 64),
            selection_controls_sha256=np.asarray("b" * 64),
        )
        return data

    monkeypatch.setattr(localize, "load_fsm_map", map_with_normals)
    geometry_path = tmp_path / "activation_geometry.npz"
    np.savez(geometry_path, schema_version=np.asarray(2))
    monkeypatch.setattr(runtime_reserve, "load_runtime_reserve_geometry", lambda *args, **kwargs: {
        "reserve_global_ids": np.array([100]),
        "reserve_means": np.array([[0.0, 0.0, 1.0]]),
        "reserve_normals": np.array([[0.0, 0.0, 1.0]]),
        "reserve_scores": np.ones(1),
        "reserve_covariances": np.eye(3)[None],
    })
    instances = []

    class ObservedReserve(runtime_reserve.RuntimeObservableReserve):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.plans = []
            self.commits = []
            self.stability_commits = []
            instances.append(self)

        def plan(self, *args, observation=None, **kwargs):
            plan = super().plan(*args, observation=observation, **kwargs)
            self.plans.append((observation, plan))
            return plan

        def commit_branch_adoption(self, frame_id, used, branch):
            self.commits.append((frame_id, used, branch))
            return super().commit_branch_adoption(frame_id, used, branch)

        def commit_stability(self, frame_id, used, twist):
            self.stability_commits.append((frame_id, used, twist))
            return super().commit_stability(frame_id, used, twist)

    monkeypatch.setattr(localize, "RuntimeObservableReserve", ObservedReserve)
    pixel_ids = np.arange(9, dtype=np.int64)
    holdout = split_source_validation(pixel_ids, 3, block_size=1, folds=2, fold=0)
    assert holdout.any() and (~holdout).any()
    return {
        "map_path": map_path, "points": points, "frames": frames, "config": config,
        "depths": original_depths, "holdout": holdout, "instances": instances,
    }


def install_solver(monkeypatch, scene, candidate_dz=None):
    calls = []
    active = candidate_dz is not None
    primary_hessian = np.diag([0.01, 2.0, 3.0, 4.0, 5.0, 6.0])

    def register(**kwargs):
        candidate = active and len(calls) % 2 == 1
        calls.append({
            key: np.array(kwargs[key], copy=True)
            for key in ("source_points", "target_points", "init_transform")
        })
        pose = np.eye(4)
        if candidate:
            pose[2, 3] = candidate_dz
        result = successful_result(pose, len(kwargs["target_points"]))
        hessian = primary_hessian * (2.0 if candidate else 1.0)
        result.update(
            num_source_points=len(kwargs["source_points"]),
            backend="numpy_robust_point_to_plane", num_correspondences=len(kwargs["source_points"]),
            point_to_plane_rmse=0.0, final_mean_robust_cost=0.0,
            hessian=hessian, eigenvalues=np.linalg.eigvalsh(hessian),
            condition_number=600.0, effective_rank=6, iterations=1,
        )
        return result

    monkeypatch.setattr(localize, "register_pointcloud", register)
    return calls, primary_hessian


def assert_training_only(scene, calls):
    training = scene["points"][~scene["holdout"]]
    heldout = scene["points"][scene["holdout"]]
    for call in calls:
        np.testing.assert_array_equal(call["source_points"], training)
        assert not np.any(np.all(call["source_points"][:, None] == heldout[None], axis=2))
    for frame, depth in zip(scene["frames"], scene["depths"]):
        np.testing.assert_array_equal(frame["depth"], depth)


def read_rows(output):
    with (output / "frame_metrics.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_cn_baseline_also_splits_training_source(cn_scene, tmp_path, monkeypatch):
    calls, _ = install_solver(monkeypatch, cn_scene)
    output = tmp_path / "baseline"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    assert len(calls) == 2
    assert_training_only(cn_scene, calls)
    assert all(int(row["num_source_points"]) == 4 for row in read_rows(output))
    assert cn_scene["instances"][0].plans == []


@pytest.mark.parametrize("candidate_dz, accepted", [(0.02, False), (0.0, True)])
def test_cn_real_verifier_controls_candidate_adoption(
    cn_scene, tmp_path, monkeypatch, candidate_dz, accepted,
):
    cn_scene["config"]["observable_reserve"].update(enabled=True, mode="active")
    calls, primary_hessian = install_solver(monkeypatch, cn_scene, candidate_dz)
    output = tmp_path / "active"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    assert len(calls) == 4
    assert_training_only(cn_scene, calls)
    reserve = cn_scene["instances"][0]
    assert reserve.available
    assert len(reserve.plans) == 2
    for observation, plan in reserve.plans:
        np.testing.assert_array_equal(observation.source_pixel_ids, np.flatnonzero(~cn_scene["holdout"]))
        np.testing.assert_array_equal(observation.source_points_camera, calls[0]["source_points"])
        assert np.all(observation.depth.reshape(-1)[cn_scene["holdout"]] == 0)
        assert np.all(observation.depth.reshape(-1)[~cn_scene["holdout"]] == 1)
        assert plan.selected_indices.tolist() == [0]
        assert plan.metrics["reserve_corr_status"] == "ok"
        expected = normalized_hessian_diagnostics(primary_hessian, np.eye(4), observation.source_points_camera)
        assert plan.metrics["reserve_spectral_primary_lambda"] == pytest.approx(expected["eigenvalues"][0])
        assert plan.metrics["reserve_spectral_candidate_lambda"] == pytest.approx(2 * expected["eigenvalues"][0])
    for index in (1, 3):
        assert len(calls[index]["target_points"]) == len(calls[index - 1]["target_points"]) + 1
        np.testing.assert_array_equal(calls[index]["init_transform"], np.eye(4))
    rows = read_rows(output)
    for row in rows[1:]:
        assert row["reserve_tracking_accepted"] == "True"
        assert row["reserve_relative_improvement_accepted"] == "True"
        assert int(row["reserve_verifier_source_count"]) == int(cn_scene["holdout"].sum())
        assert row["reserve_verifier_accepted"] == str(accepted)
        assert row["reserve_used"] == str(accepted)
        assert row["reserve_verifier_reason"] == ("accepted" if accepted else "cost_increase")
        if not accepted:
            assert float(row["reserve_verifier_candidate_cost"]) > float(row["reserve_verifier_primary_cost"])
            assert row["reserve_adoption_reason"] == "verifier_cost_increase"
    assert [used for _, used, _ in reserve.commits] == [accepted, accepted]
    assert [used for _, used, _ in reserve.stability_commits] == [accepted, accepted]
    if not accepted:
        assert all(twist is None for _, _, twist in reserve.stability_commits)
        assert reserve.branch_consecutive_adoptions == reserve.global_consecutive_adoptions == 0
    trajectory = np.loadtxt(output / "trajectory_est.txt")
    np.testing.assert_array_equal(trajectory[:, 1:4], np.zeros((3, 3)))


def test_cn_rerun_attempts_with_empty_selection(cn_scene, tmp_path, monkeypatch):
    cn_scene["config"]["observable_reserve"].update(
        enabled=True, mode="active", candidate_target_mode="rerun_primary",
        attempt_without_selection=True,
    )
    calls, _ = install_solver(monkeypatch, cn_scene, candidate_dz=0.0)

    def forbidden_selection(*args, **kwargs):
        pytest.fail("rerun_primary must not score or select reserve points")

    monkeypatch.setattr(runtime_reserve.RuntimeObservableReserve, "_corr_scores", forbidden_selection)
    monkeypatch.setattr(runtime_reserve, "select_diverse_reserve", forbidden_selection)
    output = tmp_path / "rerun"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    assert len(calls) == 4
    assert_training_only(cn_scene, calls)
    reserve = cn_scene["instances"][0]
    assert len(reserve.plans) == 2
    assert all(plan.selected_indices.size == 0 for _, plan in reserve.plans)
    for index in (1, 3):
        np.testing.assert_array_equal(calls[index]["target_points"], calls[index - 1]["target_points"])
    for row in read_rows(output)[1:]:
        assert row["reserve_selected"] == "0"
        assert row["reserve_triggered"] == row["reserve_attempted"] == "True"
        assert row["reserve_verifier_accepted"] == row["reserve_used"] == "True"
        assert row["reserve_candidate_target_mode"] == "rerun_primary"


def test_cn_all_holdout_baseline_loses_tracking_without_calling_solver(
    cn_scene, tmp_path, monkeypatch,
):
    cn_scene["config"]["observable_reserve"]["candidate_verifier"].update(block_size=16, fold=0)
    assert split_source_validation(np.arange(9), 3, block_size=16, folds=2, fold=0).all()

    def forbidden_solver(**kwargs):
        pytest.fail("empty training source must fail before invoking the solver")

    monkeypatch.setattr(localize, "register_pointcloud", forbidden_solver)
    output = tmp_path / "all_holdout"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    rows = read_rows(output)
    assert len(rows) == 3
    assert rows[0]["tracking_status"] == "initialized"
    assert all(int(row["num_source_points"]) == 0 for row in rows)
    for row in rows[1:]:
        assert row["tracking_status"] == "lost_hold_last_valid"
        assert row["tracking_valid"] == "False"
        assert row["primary_termination_reason"] == "insufficient_training_source"
        assert row["primary_rejection_reason"] == "insufficient_training_source"
    assert cn_scene["instances"][0].plans == []


def test_cn_empty_holdout_explicitly_rejects_candidate(cn_scene, tmp_path, monkeypatch):
    cn_scene["config"]["observable_reserve"].update(enabled=True, mode="active")
    cn_scene["config"]["observable_reserve"]["candidate_verifier"].update(block_size=16, fold=1)
    cn_scene["holdout"] = split_source_validation(np.arange(9), 3, block_size=16, folds=2, fold=1)
    assert not cn_scene["holdout"].any()
    calls, _ = install_solver(monkeypatch, cn_scene, candidate_dz=0.0)
    output = tmp_path / "empty_holdout"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    assert len(calls) == 4
    assert_training_only(cn_scene, calls)
    assert all(len(call["source_points"]) == 9 for call in calls)
    reserve = cn_scene["instances"][0]
    assert len(reserve.plans) == 2
    for observation, plan in reserve.plans:
        np.testing.assert_array_equal(observation.depth, cn_scene["depths"][0])
        assert plan.selected_indices.tolist() == [0]
    for row in read_rows(output)[1:]:
        assert row["reserve_tracking_accepted"] == "True"
        assert row["reserve_relative_improvement_accepted"] == "True"
        assert row["reserve_attempted"] == "True"
        assert row["reserve_verifier_source_count"] == "0"
        assert row["reserve_verifier_accepted"] == row["reserve_used"] == "False"
        assert row["reserve_verifier_reason"] == "empty_holdout"
        assert row["reserve_adoption_reason"] == "verifier_empty_holdout"
    assert [used for _, used, _ in reserve.commits] == [False, False]


def test_cn_estimated_normals_are_shared_by_primary_verifier_and_candidate(
    cn_scene, tmp_path, monkeypatch,
):
    cn_scene["config"]["registration"]["normal_source"] = "estimate"
    cn_scene["config"]["observable_reserve"].update(enabled=True, mode="active")
    calls, _ = install_solver(monkeypatch, cn_scene, candidate_dz=0.0)
    estimate_normals = localize.estimate_normals
    register = localize.register_pointcloud
    verifier_class = localize.FrozenCandidateVerifier
    estimates, solver_normals, verifier_normals = [], [], []

    def observed_estimate(*args, **kwargs):
        normals = estimate_normals(*args, **kwargs)
        assert normals.shape == (9, 3)
        assert np.all(np.isfinite(normals))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        estimates.append(normals)
        return normals

    def observed_register(**kwargs):
        solver_normals.append(kwargs["target_normals"])
        # Estimation must precede the primary solver, not just the verifier.
        assert len(estimates) == len(solver_normals[::2])
        return register(**kwargs)

    def observed_verifier(source, target, normals, **kwargs):
        verifier_normals.append(normals)
        return verifier_class(source, target, normals, **kwargs)

    monkeypatch.setattr(localize, "estimate_normals", observed_estimate)
    monkeypatch.setattr(localize, "register_pointcloud", observed_register)
    monkeypatch.setattr(localize, "FrozenCandidateVerifier", observed_verifier)
    output = tmp_path / "estimated_normals"
    localize.run_localization(
        cn_scene["config"], str(cn_scene["map_path"]), str(output), frames=cn_scene["frames"],
    )
    assert len(calls) == 4
    assert_training_only(cn_scene, calls)
    assert len(estimates) == len(verifier_normals) == 2
    for frame_index, estimated in enumerate(estimates):
        # Object identity proves the attempt context retains the actual
        # estimated array rather than silently falling back to map normals.
        assert solver_normals[2 * frame_index] is estimated
        assert verifier_normals[frame_index] is estimated
        np.testing.assert_array_equal(solver_normals[2 * frame_index + 1][:-1], estimated)
        np.testing.assert_array_equal(solver_normals[2 * frame_index + 1][-1], [0.0, 0.0, 1.0])
    for row in read_rows(output)[1:]:
        assert row["reserve_verifier_source_count"] == "5"
        assert row["reserve_verifier_reason"] == "accepted"
        assert row["reserve_verifier_accepted"] == row["reserve_used"] == "True"
        assert row["reserve_rerun_normal_source"] == "primary_plus_sidecar_frozen"
