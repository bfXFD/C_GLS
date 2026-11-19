from copy import deepcopy

import numpy as np
import pytest

from scripts import fsm_localize
from utils import fsm_runtime_reserve


@pytest.fixture
def replay(tmp_path, monkeypatch):
    points = np.array([[-0.1, -0.1, 1.0], [0.1, 0.1, 1.0], [0.0, 0.0, 1.0]])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    fsm_map = dict(
        means=points, normals=normals, source_params_sha256="source", selection_controls_sha256="controls",
    )
    monkeypatch.setattr(fsm_localize, "load_fsm_map", lambda _: fsm_map)
    sidecar = tmp_path / "activation_geometry.npz"
    sidecar.touch()
    geometry = dict(
        reserve_global_ids=np.array([3]), reserve_means=np.array([[0.05, 0.05, 1.0]]),
        reserve_normals=normals[:1], reserve_scores=np.ones(1), reserve_covariances=np.eye(3)[None],
    )
    monkeypatch.setattr(fsm_runtime_reserve, "load_runtime_reserve_geometry", lambda *_: geometry)

    def query(self, *args, **kwargs):
        count = len(self.points)
        return dict(indices=np.arange(count), query_latency_ms=0.0, candidate_count=count, frustum_count=count)

    monkeypatch.setattr(fsm_localize.MapSpatialIndex, "query_local_submap", query)
    config = {
        "runtime_memory_metrics": {"enabled": False},
        "registration": {
            "backend": "numpy_robust_point_to_plane", "evaluation_protocol": "mvp_strict",
            "first_pose_protocol": "identity", "tracking_acceptance": {"min_fitness": 0.5},
        },
        "local_submap": {"enabled": True, "near": 0.1, "far": 3.0},
        "recovery": {"enabled": True, "max_tier": 0, "verifier_backend": "numpy_point_to_point"},
        "observable_reserve": {
            "enabled": True, "mode": "active", "selection": "random", "budget": 1,
            "geometry_path": str(sidecar), "candidate_normal_mode": "frozen_concat",
            "trigger": {"condition_number": 10.0}, "accept_only_if": {"condition_ratio": 0.75},
        },
    }
    frames = []
    for i in range(4):
        gt = np.eye(4)
        gt[0, 3] = i * 2.0
        frames.append(dict(
            frame_id=10 + i, c2w=gt, depth=np.full((4, 4), 1.0 + i / 10),
            intrinsics=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        ))

    calls = []

    def register(source_points, target_points, init_transform, backend, **kwargs):
        i = round((source_points[0, 2] - 1.0) * 10)
        candidate = len(target_points) > len(points) or kwargs["max_iterations"] == 4
        recovery = backend == "numpy_point_to_point"
        calls.append((i, "candidate" if candidate else "recovery" if recovery else "primary", init_transform.copy()))
        pose = init_transform.copy()
        pose[0, 3] += 0.04 if candidate else 0.02 if recovery else 0.01
        valid = candidate or (recovery and i == 2) or i == 1
        return dict(
            transform=pose, fitness=float(valid), inlier_rmse=0.0,
            optimizer_valid=valid, converged=valid,
            num_source_points=len(source_points), num_target_points=len(target_points), num_correspondences=10,
            hessian=np.eye(6), eigenvalues=np.ones(6), effective_rank=6,
            condition_number=50.0 if candidate and i != 1 else 1.0 if candidate else 20.0,
            latency_ms=3.0 if candidate else 1.0,
        )

    monkeypatch.setattr(fsm_localize, "register_pointcloud", register)

    def run(name, cfg, **kwargs):
        path = tmp_path / name
        return fsm_localize.run_localization(cfg, str(tmp_path / "map.npz"), str(path), frames=frames, **kwargs)

    return config, frames, calls, run


def test_reserve_cutoff_preserves_prefix_and_motion_history(replay):
    config, _, calls, run = replay
    full, cutoff = [], []
    run("full", config, primary_observer=full.append)
    calls.clear()
    run("cutoff", config, reserve_stop_source_idx=2, primary_observer=cutoff.append)
    assert [r["frame_id"] for r in cutoff] == [11, 12, 13]
    assert [(i, kind) for i, kind, _ in calls if kind == "candidate"] == [(1, "candidate")]
    for a, b in zip(full[:2], cutoff[:2]):
        np.testing.assert_array_equal(a["initial_pose"], b["initial_pose"])
        np.testing.assert_array_equal(a["primary_pose"], b["primary_pose"])
    assert cutoff[1]["initial_pose"][0, 3] == pytest.approx(.05)


def test_probe_uses_real_recovery_and_hold_without_committing(replay, monkeypatch):
    config, frames, calls, run = replay

    def unexpected_commit(*args, **kwargs):
        pytest.fail("probe must not commit reserve state")

    monkeypatch.setattr(fsm_runtime_reserve.RuntimeObservableReserve, "commit_branch_adoption", unexpected_commit)
    monkeypatch.setattr(fsm_runtime_reserve.RuntimeObservableReserve, "commit_stability", unexpected_commit)
    rows = []
    run("probe", config, candidate_probe=rows.append)
    probe_calls = [(i, kind, pose) for i, kind, pose in calls if kind != "candidate"]
    calls.clear()
    baseline = deepcopy(config)
    baseline["observable_reserve"]["enabled"] = False
    base_rows = []
    run("baseline", baseline, candidate_probe=base_rows.append)
    assert [row["gated_source"] for row in rows] == ["seed", "reserve", "recovery", "hold_last_valid"]
    assert [row["ungated_source"] for row in rows] == ["seed", "reserve", "reserve", "reserve"]
    assert [row["baseline_source"] for row in rows] == ["seed", "primary", "recovery", "hold_last_valid"]
    assert [row["baseline_tracking_valid"] for row in rows] == [True, True, True, False]
    assert not any(row["candidate_submitted"] for row in rows)
    assert rows[0]["candidate_reserve_count"] is None
    assert rows[2]["candidate_reserve_count"] == 1
    assert rows[2]["candidate_target_count"] == 4
    assert rows[2]["candidate_latency_ms"] == 3.0
    assert rows[2]["gated_raw_translation_error_m"] == pytest.approx(3.97)
    assert rows[3]["gated_raw_translation_error_m"] is None
    assert rows[3]["ungated_raw_translation_error_m"] == pytest.approx(5.92)
    assert [row["frame_id"] for row in rows] == [10, 11, 12, 13]
    assert len(probe_calls) == len(calls)
    for (frame, kind, pose), (base_frame, base_kind, base_pose) in zip(probe_calls, calls):
        assert (frame, kind) == (base_frame, base_kind)
        np.testing.assert_array_equal(pose, base_pose)
    for row, base_row in zip(rows, base_rows):
        for key in ("init_pose", "primary_pose", "baseline_pose"):
            np.testing.assert_array_equal(row[key], base_row[key])

    def mutate(row):
        row["baseline_pose"][:] = 100.0
        row["gt_pose"][:] = 100.0

    calls.clear()
    run("mutation", config, candidate_probe=mutate)
    assert frames[1]["c2w"][0, 3] == 2.0
    assert next(pose[0, 3] for i, kind, pose in calls if (i, kind) == (2, "primary")) == pytest.approx(0.01)


def test_forced_frame_preserves_trigger_and_checks_eligibility(replay, monkeypatch):
    config, _, _, run = replay
    reserve = config["observable_reserve"]
    reserve["trigger"] = {"condition_number": 100.0}
    reserve["candidate_target_mode"] = "rerun_primary"
    reserve["attempt_without_selection"] = True
    reserve["rerun_iterations"] = 4
    original_register = fsm_localize.register_pointcloud

    def register(**kwargs):
        result = original_register(**kwargs)
        if kwargs["source_points"][0, 2] > 1.15:
            result["hessian"] = np.zeros((6, 6))
        return result

    monkeypatch.setattr(fsm_localize, "register_pointcloud", register)
    rows = []
    run("forced", config, candidate_probe=rows.append, candidate_force_frames={10, 11, 12})
    seed, forced, ineligible, unforced = rows
    assert seed["force_requested"] and not seed["force_eligible"]
    assert forced["force_requested"] and forced["force_eligible"] and forced["forced_attempt"]
    assert forced["candidate_attempted"] and forced["candidate_basic_valid"]
    assert forced["candidate_reserve_count"] == 0 and forced["candidate_target_count"] == 3
    assert not forced["natural_triggered"] and not forced["reserve_metrics"]["reserve_triggered"]
    assert not forced["gated_adopted"] and not forced["ungated_adopted"]
    assert ineligible["force_requested"] and not ineligible["force_eligible"] and not ineligible["candidate_attempted"]
    assert ineligible["reserve_metrics"]["reserve_trigger_unavailable_reason"] == "hessian_zero_or_invalid_spectrum"
    assert not unforced["force_requested"] and unforced["candidate_target_count"] is None
    assert reserve["trigger"] == {"condition_number": 100.0}


@pytest.mark.parametrize("invalid", ["optimizer", "nonfinite", "rotation", "last_row", "fitness"])
def test_bypass_keeps_basic_checks_and_updates_trajectory(replay, monkeypatch, invalid):
    config, _, calls, run = replay
    config["observable_reserve"]["bypass_candidate_gate"] = True
    config["observable_reserve"]["candidate_verifier"] = {"enabled": True, "block_size": 1}
    original_register = fsm_localize.register_pointcloud

    def register(**kwargs):
        result = original_register(**kwargs)
        if len(kwargs["target_points"]) > 3 and kwargs["source_points"][0, 2] > 1.25:
            if invalid == "optimizer":
                result["optimizer_valid"] = False
            elif invalid == "nonfinite":
                result["transform"][0, 3] = np.nan
            elif invalid == "rotation":
                result["transform"][0, 0] = -1.0
            elif invalid == "last_row":
                result["transform"][3, 0] = 1.0
            else:
                result["fitness"] = 0.0
        return result

    def unexpected_gate(*args, **kwargs):
        pytest.fail("bypass must not evaluate the candidate gate")

    monkeypatch.setattr(fsm_localize, "register_pointcloud", register)
    monkeypatch.setattr(fsm_localize, "FrozenCandidateVerifier", unexpected_gate)
    monkeypatch.setattr(fsm_runtime_reserve.RuntimeObservableReserve, "evaluate_improvement", unexpected_gate)
    summary = run("bypass", config)
    assert summary["bypass_candidate_gate"]
    assert summary["num_lost_frames"] == 1
    assert (2, "recovery") not in [(i, kind) for i, kind, _ in calls]
    assert next(pose[0, 3] for i, kind, pose in calls if (i, kind) == (3, "primary")) == pytest.approx(0.1)


def test_probe_rejects_midrun_start_and_closed_loop_bypass(replay):
    config, _, _, run = replay
    with pytest.raises(ValueError, match="source index 0"):
        run("invalid_start", config, candidate_probe=lambda row: None, start=1)
    config["observable_reserve"]["bypass_candidate_gate"] = True
    with pytest.raises(ValueError, match="closed-loop only"):
        run("invalid_bypass", config, candidate_probe=lambda row: None)
