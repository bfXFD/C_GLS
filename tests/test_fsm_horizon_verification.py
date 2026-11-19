import dataclasses
import hashlib

import numpy as np
import pytest

from utils.fsm_horizon_verification import (
    TrackStep, TrackerBranchState, TrackingObservation, VerificationConfig,
    assert_separate_output, load_paired_hypotheses, observation_schema,
    reconstruct_branch_state, sha256_file, verify_event,
)


def pose(x=0.0):
    value = np.eye(4)
    value[0, 3] = x
    return value


def observation(frame_id, output_idx):
    return TrackingObservation(
        frame_id, output_idx, float(frame_id), np.array([[0.0, 0.0, 1.0]]),
        np.array([0]), np.ones((2, 2)), np.eye(3),
    )


def step(branch, obs, accepted=True, x=0.0, ids=(1, 2)):
    return TrackStep(
        branch, obs.frame_id, obs.output_idx, pose(x), accepted, False,
        "tracked" if accepted else "lost_hold_last_valid", "previous_pose", "not_applicable",
        "" if accepted else "rejected", "converged", np.asarray(ids),
    )


def test_observation_is_immutable_and_physically_excludes_gt():
    value = observation(1, 1)
    forbidden = ("gt", "ground_truth", "c2w", "w2c", "pose")
    assert all(not any(token in name.lower() for token in forbidden) for name in observation_schema())
    with pytest.raises(ValueError):
        value.depth[0, 0] = 3.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.frame_id = 2


def test_branch_clone_and_candidate_discontinuity_are_independent():
    source_pose = pose(1.0)
    primary = TrackerBranchState("primary", source_pose, [(0, pose(), False), (1, source_pose, False)])
    candidate = primary.clone("candidate")
    candidate.valid_pose_history[-1][1][0, 3] = 9.0
    candidate.cache.local_indices = np.array([4])
    assert primary.valid_pose_history[-1][1][0, 3] == 1.0
    assert primary.cache.local_indices is None
    assert source_pose[0, 3] == 1.0
    rows = [
        {"frame_id": "0", "tracking_valid": "True", "recovered": "False"},
        {"frame_id": "1", "tracking_valid": "True", "recovered": "False"},
    ]
    rebuilt = reconstruct_branch_state(np.stack([pose(), pose(1.0)]), rows, 1, pose(1.2), "candidate")
    assert rebuilt.valid_pose_history[-1][2] is True


def test_lost_step_does_not_enter_history_and_exception_is_branch_local():
    primary = TrackerBranchState("primary", pose(), [(0, pose(), False)])
    candidate = primary.clone("candidate")
    observations = [observation(1, 1)]

    def track(obs, state):
        if state.branch == "candidate":
            raise RuntimeError("candidate failure")
        return step(state.branch, obs, accepted=True)

    def common(*args):
        raise AssertionError("no common evaluation when a branch failed")

    event = {"pair_id": "p", "frame_id": 0, "policy": "runtime_corr_weak", "seed": -1, "budget": 128}
    summary, rows = verify_event(
        event, observations, primary, candidate, track, common,
        VerificationConfig(horizon_frames=1, min_common_valid_frames=1),
    )
    assert len(primary.valid_pose_history) == 1
    assert len(candidate.valid_pose_history) == 1
    assert summary["decision"] == "primary"
    assert rows[1]["tracking_status"] == "exception_hold_last_valid"


def test_common_target_metrics_and_primary_tie_break_are_deterministic():
    def track(obs, state):
        return step(state.branch, obs, x=0.1 if state.branch == "candidate" else 0.0,
                    ids=(2, 3) if state.branch == "candidate" else (1, 2))

    seen = []

    def common(obs, primary, candidate, cap):
        seen.append((tuple(primary.target_global_ids), tuple(candidate.target_global_ids), cap))
        metrics = {"robust_cost_per_corr": 1.0, "point_to_plane_rmse": 1.0, "fitness": 1.0, "corr_count": 10}
        return metrics, dict(metrics), 3

    event = {"pair_id": "p", "frame_id": 0, "policy": "runtime_corr_weak", "seed": -1, "budget": 128}
    config = VerificationConfig(horizon_frames=3, min_common_valid_frames=2)
    outputs = []
    for _ in range(2):
        primary = TrackerBranchState("primary", pose(), [(0, pose(), False)])
        candidate = primary.clone("candidate")
        outputs.append(verify_event(event, [observation(i, i) for i in range(1, 4)], primary, candidate, track, common, config))
    assert outputs[0] == outputs[1]
    assert outputs[0][0]["decision"] == "primary"
    assert outputs[0][0]["decision_reason"] == "candidate_improvement_not_conservative"
    assert seen[0][:2] == ((1, 2), (2, 3))


def test_candidate_requires_conservative_common_target_improvement():
    def track(obs, state):
        return step(state.branch, obs)

    def common(obs, primary, candidate, cap):
        return (
            {"robust_cost_per_corr": 1.0, "point_to_plane_rmse": 1.0, "fitness": 1.0, "corr_count": 100},
            {"robust_cost_per_corr": 0.8, "point_to_plane_rmse": 0.8, "fitness": 0.96, "corr_count": 96}, 2,
        )

    state = TrackerBranchState("primary", pose(), [(0, pose(), False)])
    event = {"pair_id": "p", "frame_id": 0, "policy": "runtime_corr_weak", "seed": -1, "budget": 128}
    summary, _ = verify_event(
        event, [observation(i, i) for i in range(1, 4)], state, state.clone("candidate"),
        track, common, VerificationConfig(),
    )
    assert summary["decision"] == "candidate"
    assert summary["decision_reason"] == "common_target_improvement"


def test_right_censored_horizon_is_inconclusive():
    def track(obs, state):
        return step(state.branch, obs)

    def common(obs, primary, candidate, cap):
        metrics = {"robust_cost_per_corr": 1.0, "point_to_plane_rmse": 1.0, "fitness": 1.0, "corr_count": 10}
        return metrics, metrics, 2

    state = TrackerBranchState("primary", pose(), [(0, pose(), False)])
    event = {"pair_id": "p", "frame_id": 0, "policy": "runtime_corr_weak", "seed": -1, "budget": 128}
    summary, _ = verify_event(event, [observation(1, 1)], state, state.clone("candidate"), track, common)
    assert summary["status"] == "inconclusive"
    assert summary["decision"] == "primary"


def test_right_censored_survival_advantage_still_defaults_to_primary():
    def track(obs, state):
        return step(state.branch, obs, accepted=state.branch == "candidate")

    def common(*args):
        raise AssertionError("no common evaluation when primary is lost")

    state = TrackerBranchState("primary", pose(), [(0, pose(), False)])
    event = {"pair_id": "p", "frame_id": 0, "policy": "runtime_corr_weak", "seed": -1, "budget": 128}
    summary, _ = verify_event(event, [observation(1, 1)], state, state.clone("candidate"), track, common)
    assert summary["status"] == "inconclusive"
    assert summary["decision"] == "primary"
    assert summary["decision_reason"] == "right_censored_horizon"


def test_paired_loader_selects_one_corrweak128_candidate_and_validates_shapes(tmp_path):
    path = tmp_path / "paired.npz"
    poses = np.stack([pose(), pose(1.0)])
    np.savez(
        path, frame_id=np.array([1, 2]), timestamp=np.array([1.0, 2.0]), pair_id=np.array(["a", "b"]),
        policy=np.array(["runtime_corr_weak", "saliency"]), seed=np.array([-1, -1]), budget=np.array([128, 128]),
        primary_pose=poses, candidate_pose=poses, parent_trajectory_sha256=np.asarray("f" * 64),
    )
    events, parent_hash = load_paired_hypotheses(path)
    assert len(events) == 1
    assert events[0]["policy"] == "runtime_corr_weak" and events[0]["budget"] == 128
    assert parent_hash == "f" * 64


def test_parent_hash_and_separate_output_invariants(tmp_path):
    parent = tmp_path / "run"
    parent.mkdir()
    trajectory = parent / "trajectory_est.txt"
    trajectory.write_bytes(b"immutable\n")
    before = sha256_file(trajectory)
    assert before == hashlib.sha256(b"immutable\n").hexdigest()
    with pytest.raises(ValueError, match="outside"):
        assert_separate_output(parent / "shadow", parent)
    assert_separate_output(tmp_path / "shadow", parent)
    assert sha256_file(trajectory) == before
