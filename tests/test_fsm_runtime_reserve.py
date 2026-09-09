import dataclasses
import inspect

import numpy as np
import pytest

from scripts.fsm_localize import STRICT_FRAME_METRIC_FIELDS
from utils.fsm_runtime_reserve import (
    RuntimeObservableReserve,
    RuntimeReserveObservation,
    evaluate_common_target_pose,
    relative_se3_log,
    reserve_point_jacobians_camera,
    runtime_weak_scores,
    select_diverse_reserve,
)
from utils.fsm_local_map import LocalSubmapConfig
from utils.fsm_paired_reserve import BaselineSnapshot, PairedShadowReserve
from utils.fsm_correspondence_weak import (
    SELECTION_SCORE_FUNCTIONS,
    adaptive_weak_subspace,
    correspondence_aware_weak_scores,
    correspondence_probability,
    organized_depth_source_normals,
)


def test_runtime_weak_scoring_prefers_complementary_direction():
    means = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    jacobians = reserve_point_jacobians_camera(means, normals, np.eye(4))
    weak = jacobians[1] / np.linalg.norm(jacobians[1])
    basis = np.linalg.qr(np.column_stack([weak, np.eye(6)]))[0]
    hessian = basis @ np.diag([1e-6, 2.0, 3.0, 4.0, 5.0, 6.0]) @ basis.T
    scores, _, _ = runtime_weak_scores(means, normals, np.eye(4), hessian)
    assert scores[1] > scores[0]


def test_world_normals_are_rotated_to_camera_before_jacobian():
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    jacobian = reserve_point_jacobians_camera(
        np.array([[1.0, 0.0, 2.0]]), np.array([[1.0, 0.0, 0.0]]), pose
    )[0]
    np.testing.assert_allclose(jacobian[3:], [0.0, -1.0, 0.0], atol=1e-12)


def test_tile_depth_voxel_caps_are_deterministic():
    means = np.array([
        [0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.5, 0.0, 2.0], [-0.5, 0.0, 3.0]
    ])
    indices = np.arange(4)
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    intrinsics = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    kwargs = dict(
        indices=indices, scores=scores, means_world=means, candidate_c2w=np.eye(4),
        intrinsics=intrinsics, image_height=10, image_width=10, budget=3,
        tile_rows=2, tile_cols=2, depth_bins=2, voxel_size=0.1,
        tile_cap=2, depth_cap=2, voxel_cap=1, global_ids=np.array([10, 11, 12, 13]),
    )
    first = select_diverse_reserve(**kwargs)
    second = select_diverse_reserve(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert 1 not in first


def make_runtime_reserve(trigger=None, accept_only_if=None, policy="reject"):
    reserve = RuntimeObservableReserve.__new__(RuntimeObservableReserve)
    reserve.trigger = trigger or {}
    reserve.accept_only_if = accept_only_if or {}
    reserve.relative_unavailable_policy = policy
    reserve.stability_active = False
    reserve.stability_mode = "off"
    reserve.branch_policy = reserve.accept_only_if.get("branch_policy", {})
    reserve.branch_consecutive_adoptions = 0
    reserve.last_adoption_branch = None
    reserve.branch_cooldown_until_frame = None
    reserve.global_consecutive_adoptions = 0
    reserve.global_cooldown_until_frame = None
    return reserve


def _v3_reserve():
    from configs.fsm.mvp_observable.common import apply_information_adoption_v3

    config = {"observable_reserve": {"accept_only_if": {}, "stability_gate": {}}}
    acceptance = apply_information_adoption_v3(config)["observable_reserve"]["accept_only_if"]
    return make_runtime_reserve(accept_only_if=acceptance)


def _v3_result(condition, lambda_min):
    return {"condition_number": condition, "eigenvalues": [lambda_min]}


def _v4_reserve():
    from configs.fsm.mvp_observable.common import apply_information_adoption_v4

    config = {"observable_reserve": {"accept_only_if": {}, "stability_gate": {}}}
    acceptance = apply_information_adoption_v4(config)["observable_reserve"]["accept_only_if"]
    reserve = make_runtime_reserve(accept_only_if=acceptance)
    reserve.max_global_consecutive_adoptions = acceptance["max_global_consecutive_adoptions"]
    reserve.global_cooldown_frames = acceptance["global_cooldown_frames"]
    return reserve


def _v4_metrics(frame=1, refinement_translation=0.001, refinement_rotation=0.01,
                candidate_translation=0.001, candidate_rotation=0.01, temporal_rotation=0.1):
    return {
        "frame_id": frame,
        "reserve_refinement_translation_m": refinement_translation,
        "reserve_refinement_rotation_deg": refinement_rotation,
        "reserve_candidate_init_translation_m": candidate_translation,
        "reserve_candidate_init_rotation_deg": candidate_rotation,
        "reserve_candidate_temporal_rotation_deg": temporal_rotation,
    }


@pytest.mark.parametrize(
    ("condition", "lambda_min", "translation", "rotation", "accepted", "branch", "reason"),
    [
        (98.0, 1.02, 0.012, 0.30, True, "both", "information_v3_both"),
        (97.0, 1.01, 0.006, 0.15, True, "condition_only", "information_v3_condition_only"),
        (99.0, 1.03, 0.006, 0.15, True, "lambda_only", "information_v3_lambda_only"),
        (99.0, 1.01, 0.001, 0.01, False, "neither", "information_v3_neither"),
        (97.01, 1.01, 0.001, 0.01, False, "condition_only", "information_v3_condition_threshold"),
        (99.0, 1.029, 0.001, 0.01, False, "lambda_only", "information_v3_lambda_threshold"),
        (98.0, 1.02, 0.01201, 0.30, False, "both", "information_v3_refinement_translation"),
        (98.0, 1.02, 0.012, 0.301, False, "both", "information_v3_refinement_rotation"),
    ],
)
def test_information_v3_truth_table_boundaries_and_refinement(
    condition, lambda_min, translation, rotation, accepted, branch, reason,
):
    reserve = _v3_reserve()
    result = reserve.evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(condition, lambda_min),
        {"frame_id": 1, "reserve_refinement_translation_m": translation,
         "reserve_refinement_rotation_deg": rotation, "unrelated_gt_error": 999.0},
    )
    assert result[0] is accepted
    assert result[1] == reason
    assert result[2]["reserve_information_v3_branch"] == branch


def test_information_v3_state_is_pure_commit_only_and_branch_switch_recounts():
    reserve = _v3_reserve()
    primary = _v3_result(100.0, 1.0)
    condition = _v3_result(97.0, 1.01)
    metrics = {"frame_id": 1, "reserve_refinement_translation_m": 0.001,
               "reserve_refinement_rotation_deg": 0.01}
    first = reserve.evaluate_improvement(primary, condition, metrics)
    assert first[0] and reserve.branch_consecutive_adoptions == 0
    reserve.commit_branch_adoption(1, True, "condition_only")
    second = reserve.evaluate_improvement(primary, condition, dict(metrics, frame_id=2))
    assert second[2]["reserve_information_v3_prospective_streak"] == 2
    reserve.commit_branch_adoption(2, True, "condition_only")
    blocked = reserve.evaluate_improvement(primary, condition, dict(metrics, frame_id=3))
    assert not blocked[0] and "information_v3_cooldown" in blocked[1]
    reserve.commit_branch_adoption(3, False, "condition_only")
    switched = reserve.evaluate_improvement(
        primary, _v3_result(99.0, 1.03), dict(metrics, frame_id=4)
    )
    assert switched[0]
    assert switched[2]["reserve_information_v3_prospective_streak"] == 1
    reserve.commit_branch_adoption(4, True, "lambda_only")
    assert reserve.last_adoption_branch == "lambda_only"
    assert reserve.branch_consecutive_adoptions == 1


def test_information_v3_missing_refinement_fails_closed_without_gt_fields():
    reserve = _v3_reserve()
    accepted, reason, diagnostics = reserve.evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(98.0, 1.02), {"frame_id": 1}
    )
    assert not accepted
    assert reason == "information_v3_refinement_translation|information_v3_refinement_rotation"
    assert not any("gt" in key.lower() for key in diagnostics)


def test_information_v3_summary_counts_branches_adoptions_rejections_and_reasons():
    reserve = _v3_reserve()
    reserve.active = True
    reserve.available = True
    reserve.mode = "active"
    reserve.unavailable_reason = None
    reserve.geometry_path = "activation_geometry.npz"
    reserve.records = []
    for branch, used, reason in (
        ("both", True, "information_v3_both"),
        ("both", False, "information_v3_refinement_rotation"),
        ("condition_only", True, "information_v3_condition_only"),
        ("lambda_only", False, "information_v3_cooldown"),
        ("neither", False, "information_v3_neither"),
    ):
        row = RuntimeObservableReserve.empty_metrics("active")
        row.update({
            "reserve_attempted": True,
            "reserve_information_v3_branch": branch,
            "reserve_information_v3_reason": reason,
            "reserve_used": used,
        })
        reserve.records.append(row)
    summary = reserve.summary()
    assert summary["reserve_information_v3_both_attempt_count"] == 2
    assert summary["reserve_information_v3_both_adopt_count"] == 1
    assert summary["reserve_information_v3_both_reject_count"] == 1
    assert summary["reserve_information_v3_condition_only_adopt_count"] == 1
    assert summary["reserve_information_v3_lambda_only_reject_count"] == 1
    assert summary["reserve_information_v3_neither_reject_count"] == 1
    assert summary["reserve_information_v3_reason_counts"]["information_v3_cooldown"] == 1


@pytest.mark.parametrize(
    ("metrics", "accepted", "reason"),
    [
        (_v4_metrics(candidate_translation=0.015, candidate_rotation=0.50), True, "information_v4_both"),
        (_v4_metrics(candidate_translation=0.01501), False, "information_v4_candidate_init_translation"),
        (_v4_metrics(candidate_rotation=0.501), False, "information_v4_candidate_init_rotation"),
        (_v4_metrics(refinement_rotation=0.20, temporal_rotation=0.701), True, "information_v4_both"),
        (_v4_metrics(refinement_rotation=0.201, temporal_rotation=0.70), True, "information_v4_both"),
        (_v4_metrics(refinement_rotation=0.201, temporal_rotation=0.701), False,
         "information_v4_joint_rotation_spike"),
    ],
)
def test_information_v4_guard_boundaries(metrics, accepted, reason):
    result = _v4_reserve().evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(98.0, 1.02), metrics,
    )
    assert result[:2] == (accepted, reason)
    assert result[2]["reserve_information_v3_branch"] == "both"


def test_information_v4_office0_frames_240_241_253_constructed_filters():
    reserve = _v4_reserve()
    primary = _v3_result(100.0, 1.0)
    candidate = _v3_result(98.0, 1.02)
    frame240 = reserve.evaluate_improvement(
        primary, candidate, _v4_metrics(frame=240, refinement_rotation=0.21, temporal_rotation=0.71)
    )
    assert not frame240[0] and frame240[1] == "information_v4_joint_rotation_spike"
    frame241 = reserve.evaluate_improvement(
        primary, candidate, _v4_metrics(frame=241, candidate_translation=0.016, candidate_rotation=0.51)
    )
    assert not frame241[0]
    assert frame241[1] == "information_v4_candidate_init_translation|information_v4_candidate_init_rotation"
    frame253 = reserve.evaluate_improvement(
        primary, candidate, _v4_metrics(frame=253, refinement_rotation=0.31, temporal_rotation=0.60)
    )
    assert frame253[:2] == (True, "information_v4_both")
    assert frame253[2]["reserve_information_v3_refinement_rotation_pass"] is False
    assert frame253[2]["reserve_information_v4_joint_rotation_pass"] is True


def test_information_v4_global_streak_crosses_branch_switch_and_decision_is_pure():
    reserve = _v4_reserve()
    primary = _v3_result(100.0, 1.0)
    for frame in range(1, 9):
        candidate = _v3_result(98.0, 1.02) if frame % 2 else _v3_result(97.0, 1.01)
        accepted, _, diagnostics = reserve.evaluate_improvement(primary, candidate, _v4_metrics(frame=frame))
        assert accepted
        assert diagnostics["reserve_information_v4_prospective_global_streak"] == frame
        assert reserve.global_consecutive_adoptions == frame - 1
        reserve.commit_branch_adoption(frame, True, diagnostics["reserve_information_v3_branch"])
    blocked = reserve.evaluate_improvement(primary, _v3_result(99.0, 1.03), _v4_metrics(frame=9))
    assert not blocked[0]
    assert "information_v4_global_max_consecutive" in blocked[1]
    assert "information_v4_global_cooldown" in blocked[1]


def test_information_v4_nonadoption_resets_global_streak_but_preserves_cooldown():
    reserve = _v4_reserve()
    reserve.global_consecutive_adoptions = 8
    reserve.global_cooldown_until_frame = 10
    reserve.commit_branch_adoption(9, False, "both")
    assert reserve.global_consecutive_adoptions == 0
    assert reserve.global_cooldown_until_frame == 10
    result = reserve.evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(98.0, 1.02), _v4_metrics(frame=10)
    )
    assert not result[0] and "information_v4_global_cooldown" in result[1]
    assert result[2]["reserve_information_v4_prospective_global_streak"] == 1
    assert result[2]["reserve_information_v4_global_cooldown_remaining_frames"] == 1


def test_information_v4_joint_metrics_fail_closed_and_decision_is_gt_free():
    reserve = _v4_reserve()
    metrics = _v4_metrics()
    metrics.pop("reserve_candidate_temporal_rotation_deg")
    first = reserve.evaluate_improvement(_v3_result(100.0, 1.0), _v3_result(98.0, 1.02), metrics)
    second = reserve.evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(98.0, 1.02), dict(metrics, unrelated_gt_error=999.0)
    )
    assert first[:2] == second[:2] == (False, "information_v4_joint_metrics_unavailable")
    assert not any("gt" in key.lower() for key in first[2])


def test_information_v4_neither_still_reports_prospective_global_diagnostics():
    reserve = _v4_reserve()
    reserve.global_consecutive_adoptions = 3
    reserve.global_cooldown_until_frame = 6
    accepted, reason, diagnostics = reserve.evaluate_improvement(
        _v3_result(100.0, 1.0), _v3_result(99.0, 1.01), _v4_metrics(frame=5)
    )
    assert not accepted and reason == "information_v4_neither"
    assert diagnostics["reserve_information_v4_prospective_global_streak"] == 4
    assert diagnostics["reserve_information_v4_global_cooldown_remaining_frames"] == 2
    assert diagnostics["reserve_information_v4_candidate_init_translation_pass"] is True
    assert diagnostics["reserve_information_v4_joint_rotation_pass"] is True


def test_information_v4_summary_counts_guards_and_split_reasons():
    reserve = _v4_reserve()
    reserve.active = True
    reserve.available = True
    reserve.mode = "active"
    reserve.unavailable_reason = None
    reserve.geometry_path = "activation_geometry.npz"
    reserve.records = []
    for used, reason, init_translation_pass, init_rotation_pass, joint_pass in (
        (True, "information_v4_both", True, True, True),
        (False, "information_v4_candidate_init_translation|information_v4_candidate_init_rotation",
         False, False, True),
        (False, "information_v4_joint_rotation_spike", True, True, False),
    ):
        row = RuntimeObservableReserve.empty_metrics("active")
        row.update({
            "reserve_attempted": True,
            "reserve_information_v3_branch": "both",
            "reserve_information_v4_reason": reason,
            "reserve_information_v4_candidate_init_translation_pass": init_translation_pass,
            "reserve_information_v4_candidate_init_rotation_pass": init_rotation_pass,
            "reserve_information_v4_joint_rotation_pass": joint_pass,
            "reserve_used": used,
        })
        reserve.records.append(row)
    summary = reserve.summary()
    assert summary["reserve_information_v4_attempt_count"] == 3
    assert summary["reserve_information_v4_adopt_count"] == 1
    assert summary["reserve_information_v4_candidate_init_translation_pass_count"] == 2
    assert summary["reserve_information_v4_joint_rotation_pass_count"] == 2
    assert summary["reserve_information_v4_reason_counts"]["information_v4_candidate_init_rotation"] == 1


def _exp_se3(twist):
    from scipy.linalg import expm

    twist = np.asarray(twist, dtype=np.float64)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:3, :3] = np.array([
        [0.0, -twist[2], twist[1]],
        [twist[2], 0.0, -twist[0]],
        [-twist[1], twist[0], 0.0],
    ])
    matrix[:3, 3] = twist[3:]
    return expm(matrix)


@pytest.mark.parametrize("twist", [
    [1e-10, -2e-10, 3e-10, 0.01, -0.02, 0.03],
    [0.2, -0.1, 0.3, 0.04, 0.02, -0.01],
    [np.pi - 1e-7, 0.0, 0.0, 0.02, -0.01, 0.03],
])
def test_relative_se3_log_roundtrip_and_near_pi(twist):
    candidate = _exp_se3(twist)
    result = relative_se3_log(np.eye(4), candidate)
    np.testing.assert_allclose(_exp_se3(result["twist"]), candidate, atol=2e-8)
    assert result["translation_norm_m"] == pytest.approx(np.linalg.norm(result["twist"][3:]))


def test_common_target_pose_evaluation_distinguishes_good_and_bad_pose():
    source = np.array([[x, y, 1.0 + 0.1 * x] for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.0, 0.2)])
    target = source.copy()
    normals = np.tile(np.array([[-0.1, 0.0, 1.0]]) / np.sqrt(1.01), (len(target), 1))
    tree = _Tree(target)
    kwargs = dict(
        source_points=source, target_points=target, target_normals=normals, target_tree=tree,
        max_correspondence_distance=0.2, robust_loss="huber", loss_scale=0.02,
        min_correspondences=6, rank_tolerance=1e-9,
    )
    good = evaluate_common_target_pose(pose=np.eye(4), **kwargs)
    bad_pose = np.eye(4)
    bad_pose[2, 3] = 0.05
    bad = evaluate_common_target_pose(pose=bad_pose, **kwargs)
    assert good["corr_count"] == bad["corr_count"] == len(source)
    assert good["point_to_plane_rmse"] < bad["point_to_plane_rmse"]
    assert good["robust_cost_per_corr"] < bad["robust_cost_per_corr"]


def _stability_reserve(mode="enforce", **overrides):
    reserve = RuntimeObservableReserve.__new__(RuntimeObservableReserve)
    reserve.stability_active = True
    reserve.stability_mode = mode
    reserve.stability_config = {
        "max_refinement_translation_m": 0.012,
        "max_refinement_rotation_deg": 2.0,
        "max_rolling_translation_l1_m": 0.05,
        "common_target": {"max_cost_ratio": 1.0, "max_rmse_ratio": 1.0, "min_corr_retention": 0.95},
        **overrides,
    }
    reserve.rolling_window_frames = 5
    reserve.max_consecutive_adoptions = 8
    reserve.cooldown_frames = 3
    reserve.adopted_twists = __import__("collections").deque()
    reserve.consecutive_adoptions = 0
    reserve.last_adoption_frame = None
    return reserve


def _gate_metrics(frame=10, cost=1.0, rmse=1.0, retention=1.0):
    return {
        "frame_id": frame,
        "reserve_common_target_cost_ratio": cost,
        "reserve_common_target_rmse_ratio": rmse,
        "reserve_common_target_corr_retention": retention,
    }


def test_stability_gate_rejects_refinement_common_rolling_and_streak_without_mutating_state():
    reserve = _stability_reserve()
    candidate = np.eye(4)
    candidate[0, 3] = 0.013
    accepted, reason, _ = reserve.prospective_stability(_gate_metrics(), candidate, np.eye(4))
    assert not accepted and "stability_gate_refinement_translation" in reason
    assert not reserve.adopted_twists and reserve.consecutive_adoptions == 0
    accepted, reason, _ = reserve.prospective_stability(_gate_metrics(cost=1.01), np.eye(4), np.eye(4))
    assert not accepted and reason == "stability_gate_common_target_cost"
    reserve.adopted_twists.append((9, np.array([0, 0, 0, 0.04, 0, 0], dtype=float)))
    accepted, reason, _ = reserve.prospective_stability(_gate_metrics(), candidate, np.eye(4))
    assert not accepted and "stability_gate_rolling_translation" in reason
    reserve.adopted_twists.clear()
    reserve.consecutive_adoptions = 8
    accepted, reason, _ = reserve.prospective_stability(_gate_metrics(), np.eye(4), np.eye(4))
    assert not accepted and "stability_gate_consecutive_adoptions" in reason


def test_stability_commit_only_used_resets_streak_prunes_window_and_applies_cooldown():
    reserve = _stability_reserve()
    twist = np.array([0, 0, 0, 0.005, 0, 0], dtype=float)
    reserve.commit_stability(1, True, twist)
    assert reserve.consecutive_adoptions == 1 and len(reserve.adopted_twists) == 1
    reserve.commit_stability(2, False, None)
    assert reserve.consecutive_adoptions == 0 and len(reserve.adopted_twists) == 1
    accepted, reason, diagnostics = reserve.prospective_stability(_gate_metrics(frame=3), np.eye(4), np.eye(4))
    assert not accepted and reason == "stability_gate_cooldown"
    assert diagnostics["reserve_stability_cooldown_remaining_frames"] == 2
    reserve.commit_stability(7, False, None)
    assert len(reserve.adopted_twists) == 0


def test_report_only_decision_is_gt_independent_and_artifact_has_no_pickle(tmp_path):
    reserve = _stability_reserve(mode="report_only")
    reserve.stability_attempts = []
    metrics = _gate_metrics()
    candidate = np.eye(4)
    candidate[0, 3] = 0.02
    first = reserve.prospective_stability(metrics, candidate, np.eye(4))
    second = reserve.prospective_stability(dict(metrics, unrelated_gt_error=999.0), candidate, np.eye(4))
    assert first[:2] == second[:2]
    refinement = relative_se3_log(np.eye(4), candidate)
    for axis, value in zip(("rx", "ry", "rz", "tx", "ty", "tz"), refinement["twist"]):
        metrics[f"reserve_refinement_twist_{axis}"] = value
    metrics.update(first[2])
    metrics.update(reserve_stability_would_accept=first[0], reserve_stability_reasons=first[1])
    reserve.add_stability_attempt(10, np.eye(4), candidate, np.eye(4), metrics, actual=True)
    path = reserve.save_stability_attempts(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["primary_pose"].shape == (1, 4, 4)
        assert archive["refinement_twist"].shape == (1, 6)
        assert archive["reasons"].dtype.kind == "U"


def test_rank_deficient_optimizer_invalid_result_can_trigger():
    reserve = make_runtime_reserve()
    result = {
        "optimizer_valid": False,
        "transform": np.eye(4),
        "hessian": np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "num_correspondences": 12,
        "effective_rank": 5,
    }
    triggered, reasons, unavailable = reserve._trigger(result)
    assert triggered
    assert reasons == ["effective_rank_below"]
    assert unavailable is None


def test_trigger_reports_unavailable_hessian_reason():
    reserve = make_runtime_reserve()
    triggered, reasons, unavailable = reserve._trigger({
        "transform": np.eye(4), "num_correspondences": 12,
    })
    assert not triggered
    assert reasons == []
    assert unavailable == "hessian_unavailable"


def test_relative_rules_apply_to_rejected_primary_and_fail_closed():
    reserve = make_runtime_reserve(accept_only_if={"condition_ratio": 0.5, "lambda_increase": 0.1, "quality": True})
    primary = {"condition_number": 100.0, "eigenvalues": np.array([0.1]), "fitness": 0.5}
    candidate = {"condition_number": 40.0, "eigenvalues": np.array([0.3]), "fitness": 0.6}
    assert reserve.improvement_accepts(primary, candidate) == (True, "relative_improvement_accepted")
    unavailable = dict(candidate, condition_number=float("inf"))
    assert reserve.improvement_accepts(primary, unavailable) == (False, "relative_metrics_unavailable")


def test_relative_rules_explicit_all_preserves_all_semantics():
    reserve = make_runtime_reserve(accept_only_if={
        "mode": "all", "condition_ratio": 0.98, "lambda_ratio": 1.02,
    })
    primary = {"condition_number": 100.0, "eigenvalues": [1.0]}
    accepted, reason, diagnostics = reserve.evaluate_improvement(
        primary, {"condition_number": 97.0, "eigenvalues": [1.01]}
    )
    assert not accepted
    assert reason == "relative_improvement_lambda_not_improved"
    assert diagnostics["reserve_relative_condition_pass"] is True
    assert diagnostics["reserve_relative_lambda_pass"] is False


@pytest.mark.parametrize(
    ("candidate_condition", "candidate_lambda", "accepted", "reason"),
    [
        (97.0, 1.01, True, "relative_improvement_condition"),
        (99.0, 1.03, True, "relative_improvement_lambda"),
        (97.0, 1.03, True, "relative_improvement_any_accepted"),
        (99.0, 1.01, False, "relative_improvement_condition_not_improved|relative_improvement_lambda_not_improved"),
    ],
)
def test_relative_rules_any_truth_table(candidate_condition, candidate_lambda, accepted, reason):
    reserve = make_runtime_reserve(accept_only_if={
        "protocol": "condition_or_lambda_v2", "mode": "any",
        "condition_ratio": 0.98, "lambda_ratio": 1.02,
    })
    result = reserve.evaluate_improvement(
        {"condition_number": 100.0, "eigenvalues": [1.0]},
        {"condition_number": candidate_condition, "eigenvalues": [candidate_lambda]},
    )
    assert result[:2] == (accepted, reason)


def test_relative_rules_any_fail_closed_per_unavailable_branch():
    reserve = make_runtime_reserve(accept_only_if={
        "mode": "any", "condition_ratio": 0.98, "lambda_ratio": 1.02,
    })
    accepted, reason, diagnostics = reserve.evaluate_improvement(
        {"condition_number": 100.0, "eigenvalues": [1.0]},
        {"condition_number": None, "eigenvalues": [1.03]},
    )
    assert accepted and reason == "relative_improvement_lambda"
    assert diagnostics["reserve_relative_condition_available"] is False
    assert diagnostics["reserve_relative_improvement_reasons"] == "condition_unavailable"
    accepted, reason, diagnostics = reserve.evaluate_improvement(
        {"condition_number": 100.0, "eigenvalues": [0.0]},
        {"condition_number": None, "eigenvalues": [1.0]},
    )
    assert not accepted
    assert reason == "relative_improvement_condition_unavailable|relative_improvement_lambda_unavailable"
    assert diagnostics["reserve_relative_lambda_ratio"] is None


def test_relative_ratio_thresholds_must_be_finite_and_positive(tmp_path):
    from tests.test_mvp_observable_configs import load_config

    _, config = load_config("replica_room0_corr_weak128_info_v2")
    for key, value in (("condition_ratio", 0.0), ("lambda_ratio", float("inf"))):
        invalid = __import__("copy").deepcopy(config)
        invalid["observable_reserve"]["accept_only_if"][key] = value
        with pytest.raises(ValueError, match="finite and positive"):
            RuntimeObservableReserve(
                invalid,
                {"source_params_sha256": "a" * 64, "selection_controls_sha256": "b" * 64},
                str(tmp_path / "fsm_map.npz"), LocalSubmapConfig(near=0.1, far=5.0),
                "mvp_strict", "numpy_robust_point_to_plane",
            )


def test_relative_improvement_summary_counts_are_explicit():
    reserve = make_runtime_reserve(accept_only_if={
        "protocol": "condition_or_lambda_v2", "mode": "any",
        "condition_ratio": 0.98, "lambda_ratio": 1.02,
    })
    reserve.active = True
    reserve.available = True
    reserve.mode = "active"
    reserve.unavailable_reason = None
    reserve.geometry_path = "activation_geometry.npz"
    reserve.records = []
    for condition_available, condition_pass, lambda_available, lambda_pass in (
        (True, True, True, False),
        (True, False, True, True),
        (True, True, True, True),
        (False, False, False, False),
    ):
        row = RuntimeObservableReserve.empty_metrics("active")
        row.update({
            "reserve_attempted": True,
            "reserve_relative_condition_available": condition_available,
            "reserve_relative_condition_pass": condition_pass,
            "reserve_relative_lambda_available": lambda_available,
            "reserve_relative_lambda_pass": lambda_pass,
        })
        reserve.records.append(row)
    summary = reserve.summary()
    assert summary["reserve_relative_improvement_protocol"] == "condition_or_lambda_v2"
    assert summary["reserve_relative_improvement_mode"] == "any"
    assert summary["reserve_relative_condition_pass_count"] == 2
    assert summary["reserve_relative_lambda_pass_count"] == 2
    assert summary["reserve_relative_any_only_count"] == 2
    assert summary["reserve_relative_both_count"] == 1
    assert summary["reserve_relative_condition_unavailable_count"] == 1
    assert summary["reserve_relative_lambda_unavailable_count"] == 1


def test_empty_metrics_are_explicitly_auditable():
    metrics = RuntimeObservableReserve.empty_metrics("active", "unavailable")
    assert set(metrics).issubset(STRICT_FRAME_METRIC_FIELDS)
    for key in (
        "reserve_primary_accepted", "reserve_tracking_accepted", "reserve_consistency_accepted",
        "reserve_relative_improvement_accepted", "reserve_adoption_reason",
        "reserve_relative_improvement_protocol", "reserve_relative_improvement_mode",
        "reserve_relative_condition_available", "reserve_relative_condition_pass",
        "reserve_relative_condition_ratio", "reserve_relative_lambda_available",
        "reserve_relative_lambda_pass", "reserve_relative_lambda_ratio",
        "reserve_relative_improvement_reasons",
        "reserve_information_v3_branch", "reserve_information_v3_condition_base_pass",
        "reserve_information_v3_lambda_base_pass", "reserve_information_v3_condition_strong_pass",
        "reserve_information_v3_lambda_strong_pass",
        "reserve_information_v3_refinement_translation_pass",
        "reserve_information_v3_refinement_rotation_pass",
        "reserve_information_v3_prospective_streak",
        "reserve_information_v3_cooldown_remaining_frames",
        "reserve_information_v3_accepted", "reserve_information_v3_reason",
        "reserve_plan_pose_source", "reserve_rerun_init_source", "reserve_plan_normal_source",
        "reserve_rerun_normal_source", "reserve_trigger_unavailable_reason",
        "reserve_candidate_optimizer_valid", "reserve_candidate_fitness", "reserve_candidate_rmse",
        "reserve_candidate_effective_rank", "reserve_candidate_hessian_trace",
        "reserve_candidate_normalized_lambda_min",
        "reserve_corr_policy", "reserve_corr_status", "reserve_corr_reason",
        "reserve_corr_source_count", "reserve_corr_normal_valid_rate",
        "reserve_corr_distance_valid_rate", "reserve_corr_eligible", "reserve_corr_p_mean",
        "reserve_corr_p_nonzero_rate", "reserve_corr_selected_p_mean",
        "reserve_corr_selected_p10", "reserve_corr_selected_p_min", "reserve_corr_adaptive_k",
        "reserve_corr_characteristic_length", "reserve_corr_latency_ms",
        "reserve_corr_normal_source", "reserve_corr_distance_source",
        "reserve_hessian_rotation_trace", "reserve_hessian_translation_trace",
        "reserve_hessian_cross_frobenius", "reserve_hessian_rotation_lambda_min",
        "reserve_hessian_translation_lambda_min", "reserve_weak_rotation_energy",
        "reserve_weak_translation_energy", "reserve_weak_dominant_block",
    ):
        assert key in metrics


class _Tree:
    def __init__(self, points):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(points)

    def query(self, *args, **kwargs):
        return self.tree.query(*args, **kwargs)


class _Index:
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.backend = "test"
        self.num_points = self.indices.size

    def query_local_submap(self, *args, **kwargs):
        return {"indices": self.indices.copy()}


def _corr_runtime(points, normals, global_ids, supported=None, budget=3):
    reserve = RuntimeObservableReserve.__new__(RuntimeObservableReserve)
    reserve.active = True
    reserve.available = True
    reserve.mode = "active"
    reserve.unavailable_reason = None
    reserve.selection = "runtime_corr_weak"
    reserve.trigger = {"effective_rank_below": 6}
    reserve.points = np.asarray(points, dtype=np.float64)
    reserve.normals = np.asarray(normals, dtype=np.float64)
    reserve.global_ids = np.asarray(global_ids, dtype=np.int64)
    reserve.supported = np.ones(len(points), dtype=bool) if supported is None else np.asarray(supported, dtype=bool)
    reserve.scores = np.ones(len(points), dtype=np.float64)
    reserve.index = _Index(np.arange(len(points)))
    reserve.local_config = LocalSubmapConfig(near=0.1, far=5.0)
    reserve.max_visible = len(points)
    reserve.max_points = budget
    reserve.budget = budget
    reserve.weak_dim = 3
    reserve.plan_normal_source = "sidecar"
    reserve.config = {
        "corr_weak_min_dim": 1,
        "corr_weak_max_dim": 3,
        "correspondence": {},
        "diversity": {"voxel_cap": None, "voxel_size": 0.01},
    }
    return reserve


def _corr_result():
    return {
        "transform": np.eye(4),
        "hessian": np.diag([1e-8, 2e-8, 1.0, 2.0, 3.0, 4.0]),
        "num_correspondences": 4,
        "effective_rank": 5,
        "condition_number": 1e9,
    }


def test_runtime_observation_is_frozen_normalizes_depth_and_has_no_gt_fields():
    depth, _, points = _organized_plane()
    ids = np.array([40, 41, 49, 31])
    observation = RuntimeReserveObservation(
        source_points_camera=points[ids], source_pixel_ids=ids, depth=depth[..., None],
        primary_target_tree=_Tree(points[ids]), primary_target_global_ids=np.array([1, 2]),
    )
    assert observation.depth.shape == depth.shape
    assert all("gt" not in field.name.lower() and "ground_truth" not in field.name.lower()
               for field in dataclasses.fields(observation))
    with pytest.raises(ValueError):
        observation.depth[0, 0] = 3.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.depth = depth


def test_corr_trigger_without_observation_fails_closed_and_old_policy_ignores_observation():
    points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    reserve = _corr_runtime(points, normals, [10, 11], budget=1)
    plan = reserve.plan(1, _corr_result(), np.eye(3), 10, 10)
    assert plan.selected_indices.size == 0
    assert plan.metrics["reserve_corr_status"] == "unavailable"
    assert plan.metrics["reserve_corr_reason"] == "missing_observation"

    reserve.selection = "saliency"
    old_signature = inspect.signature(RuntimeObservableReserve.plan)
    assert old_signature.parameters["observation"].kind is inspect.Parameter.KEYWORD_ONLY
    plan = reserve.plan(1, _corr_result(), np.eye(3), 10, 10, observation=None)
    assert plan.selected_indices.tolist() == [0]


def test_active_corr_ranking_prefix_matches_paired_same_snapshot_and_config():
    depth, intrinsics, all_points = _organized_plane()
    source_ids = np.array([40, 41, 49, 31])
    source = all_points[source_ids]
    target = source[:2].copy()
    tree = _Tree(target)
    reserve_points = np.array([
        [0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [-0.1, 0.0, 2.0],
        [0.0, 0.1, 2.0], [0.0, -0.1, 2.0], [0.2, 0.0, 2.0],
    ])
    reserve_normals = np.tile([0.0, 0.0, -1.0], (len(reserve_points), 1))
    global_ids = np.arange(100, 100 + len(reserve_points))
    supported = np.array([True, True, False, True, True, True])
    runtime = _corr_runtime(reserve_points, reserve_normals, global_ids, supported=supported, budget=3)
    observation = RuntimeReserveObservation(
        source, source_ids, depth, tree, np.array([10, 11])
    )
    active = runtime.plan(7, _corr_result(), intrinsics, *depth.shape, observation=observation)
    assert active.metrics["reserve_corr_status"] == "ok"
    assert active.metrics["reserve_corr_eligible"] == 5

    paired = PairedShadowReserve(
        {
            "mode": "shadow", "policies": ["runtime_corr_weak"], "budgets": [3],
            "eligible_pool": "supported", "weak_subspace_dim": 3,
            "trigger": {"always": True}, "correspondence": {},
            "corr_weak_min_dim": 1, "corr_weak_max_dim": 3,
            "diversity": {"voxel_cap": None, "voxel_size": 0.01},
            "registration": {"min_correspondences": 1},
        },
        reserve_points, reserve_normals, np.ones(len(reserve_points)), global_ids,
        runtime.index, runtime.local_config, reserve_supported=supported,
        registration_fn=lambda **kwargs: {"transform": np.eye(4)},
        diagnostics_fn=lambda *args, **kwargs: {
            "fitness": 1.0, "inlier_rmse": 0.0, "num_correspondences": 4,
            "condition_number": 1e9, "effective_rank": 5,
            "jacobians": np.ones((4, 6)), "hessian": _corr_result()["hessian"],
            "correspondence_target_ids": np.array([10, 11, 10, 11]),
            "correspondence_jacobians": np.ones((4, 6)), "correspondence_weights": np.ones(4),
        },
    )
    snapshot = BaselineSnapshot(
        frame_id=7, source="frame", source_points=source, primary_pose=np.eye(4),
        primary_hessian=_corr_result()["hessian"], target_points=target,
        target_normals=np.tile([0.0, 0.0, -1.0], (2, 1)), target_global_ids=np.array([10, 11]),
        target_tree=tree, intrinsics=intrinsics, image=np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth=depth, source_pixel_ids=source_ids,
    )
    evaluation = paired.evaluate(snapshot)
    np.testing.assert_array_equal(runtime.global_ids[active.selected_indices], evaluation.selected_global_ids[0])


def test_summary_uses_all_tracking_frames_and_reports_conditional_rescue_metrics():
    reserve = make_runtime_reserve()
    reserve.active = True
    reserve.enabled = True
    reserve.mode = "active"
    reserve.available = True
    reserve.unavailable_reason = None
    reserve.geometry_path = "geometry.npz"
    reserve.records = [
        dict(RuntimeObservableReserve.empty_metrics("active"), reserve_primary_accepted=True),
        dict(
            RuntimeObservableReserve.empty_metrics("active"),
            reserve_triggered=True, reserve_attempted=True, reserve_used=True,
            reserve_primary_accepted=True, reserve_selected=64, reserve_correspondence=80,
            reserve_primary_condition=100.0, reserve_augmented_condition=50.0,
            reserve_primary_lambda_min=2.0, reserve_augmented_lambda_min=3.0,
            reserve_latency_ms=4.0,
        ),
        dict(
            RuntimeObservableReserve.empty_metrics("active"),
            reserve_triggered=True, reserve_attempted=True, reserve_used=True,
            reserve_primary_accepted=False, reserve_selected=32, reserve_correspondence=40,
            reserve_primary_condition=80.0, reserve_augmented_condition=60.0,
            reserve_primary_lambda_min=1.0, reserve_augmented_lambda_min=2.0,
            reserve_latency_ms=8.0,
        ),
        dict(RuntimeObservableReserve.empty_metrics("active"), reserve_primary_accepted=False),
    ]
    summary = reserve.summary()
    assert summary["num_reserve_attempt_frames"] == 2
    assert summary["num_reserve_adopted_frames"] == 2
    assert summary["num_reserve_rescue_frames"] == 1
    assert summary["reserve_attempt_rate"] == 0.5
    assert summary["reserve_adoption_rate"] == 0.5
    assert summary["reserve_conditional_adoption_rate"] == 1.0
    assert summary["reserve_rescue_rate"] == 0.5
    assert summary["reserve_selected_attempted_mean"] == 48.0
    assert summary["reserve_correspondence_attempted_mean"] == 60.0
    assert summary["reserve_attempted_primary_condition_mean"] == 90.0
    assert summary["reserve_attempted_final_condition_mean"] == 55.0
    assert summary["reserve_attempted_condition_reduction_abs_mean"] == 35.0
    assert summary["reserve_attempted_condition_reduction_percent_mean"] == 37.5
    assert summary["reserve_attempted_primary_lambda_min_mean"] == 1.5
    assert summary["reserve_attempted_final_lambda_min_mean"] == 2.5
    assert summary["reserve_attempted_lambda_min_increase_abs_mean"] == 1.0
    assert summary["reserve_latency_all_tracking_frames_mean_ms"] == 3.0
    assert summary["reserve_latency_triggered_mean_ms"] == 6.0


def _organized_plane(height=9, width=9, z=2.0):
    depth = np.full((height, width), z, dtype=np.float64)
    intrinsics = np.array([[20.0, 0.0, width // 2], [0.0, 20.0, height // 2], [0.0, 0.0, 1.0]])
    rows, cols = np.indices(depth.shape)
    points = np.column_stack((
        (cols.ravel() - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (rows.ravel() - intrinsics[1, 2]) * z / intrinsics[1, 1],
        depth.ravel(),
    ))
    return depth, intrinsics, points


def test_organized_depth_normals_are_camera_frame_and_confident():
    depth, intrinsics, _ = _organized_plane()
    indices = np.array([4 * 9 + 4, 0])
    normals, valid, confidence = organized_depth_source_normals(depth, intrinsics, indices)
    np.testing.assert_allclose(normals[0], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_allclose(confidence, [1.0, 0.0], atol=1e-12)


def _adjoint(c2w):
    rotation = c2w[:3, :3]
    t = c2w[:3, 3]
    skew = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, :3] = skew @ rotation
    result[3:, 3:] = rotation
    return result


def test_adaptive_subspace_preserves_world_camera_quadratic_form():
    angle = 0.4
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    c2w[:3, 3] = [0.3, -0.2, 0.5]
    hessian_world = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = adaptive_weak_subspace(hessian_world, c2w, np.array([[0.0, 0.0, 2.0]]))
    camera_twist = np.array([0.1, -0.2, 0.05, 0.3, 0.2, -0.1])
    world_twist = _adjoint(c2w) @ camera_twist
    np.testing.assert_allclose(
        world_twist @ hessian_world @ world_twist,
        camera_twist @ result["camera_hessian"] @ camera_twist,
        rtol=1e-12,
    )


@pytest.mark.parametrize("weak_eigenvalues, expected_k", [
    ([1e-8, 1.0, 1.0, 1.0, 1.0, 1.0], 1),
    ([1e-8, 2e-8, 1.0, 1.0, 1.0, 1.0], 2),
    ([1e-8, 2e-8, 3e-8, 1.0, 1.0, 1.0], 3),
])
def test_adaptive_subspace_k_and_hessian_scale_invariance(weak_eigenvalues, expected_k):
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]])
    hessian = np.diag(weak_eigenvalues)
    first = adaptive_weak_subspace(hessian, np.eye(4), points)
    scalar_scaled = adaptive_weak_subspace(17.0 * hessian, np.eye(4), points)
    scene_scale = 2.0
    hessian_scale = np.diag([scene_scale] * 3 + [1.0] * 3)
    geometry_scaled = adaptive_weak_subspace(
        hessian_scale @ hessian @ hessian_scale, np.eye(4), scene_scale * points
    )
    assert first["k"] == scalar_scaled["k"] == geometry_scaled["k"] == expected_k
    assert first["length"] == first["characteristic_length"]
    np.testing.assert_allclose(first["eigs"], first["eigenvalues"])
    np.testing.assert_allclose(np.abs(first["basis"]), np.abs(scalar_scaled["basis"]), atol=1e-12)
    np.testing.assert_allclose(np.abs(first["basis"]), np.abs(geometry_scaled["basis"]), atol=1e-12)


def _probability_fixture():
    depth, intrinsics, all_points = _organized_plane()
    indices = np.array([4 * 9 + 4, 4 * 9 + 5, 5 * 9 + 4, 3 * 9 + 4])
    source = all_points[indices]
    normals = np.tile([0.0, 0.0, -1.0], (source.shape[0], 1))
    valid = np.ones(source.shape[0], dtype=bool)
    confidence = np.ones(source.shape[0])
    return depth, intrinsics, indices, source, normals, valid, confidence


def _probability(reserve, reserve_normals=None, **kwargs):
    depth, intrinsics, indices, source, normals, valid, confidence = _probability_fixture()
    return correspondence_probability(
        np.asarray(reserve, dtype=np.float64), source, np.eye(4), depth, intrinsics, indices,
        reserve_normals_world=None if reserve_normals is None else np.asarray(reserve_normals, dtype=np.float64),
        source_normals_camera=normals, source_normal_valid=valid, source_normal_confidence=confidence,
        **kwargs,
    )


def test_correspondence_distance_and_hard_cutoff_terms():
    result = _probability(
        [[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], dmax=0.15, normal_missing_prior=1.0
    )
    assert result["probability"][0] > 0.9
    assert result["probability"][1] == 0.0


def test_correspondence_depth_is_asymmetric_and_missing_uses_prior():
    behind = _probability([[0.0, 0.0, 2.06]], depth_sigma_front=0.2, depth_sigma_back=0.02, distance_sigma=1.0, dmax=1.0)
    front = _probability([[0.0, 0.0, 1.94]], depth_sigma_front=0.2, depth_sigma_back=0.02, distance_sigma=1.0, dmax=1.0)
    missing = _probability([[5.0, 0.0, 2.0]], missing_depth_prior=0.23, distance_sigma=10.0, dmax=10.0)
    assert front["terms"]["depth"][0] > behind["terms"]["depth"][0]
    assert missing["terms"]["depth"][0] == pytest.approx(0.23)


def test_correspondence_normal_sign_invariant_but_mismatch_penalized():
    aligned = _probability([[0.0, 0.0, 2.0]], [[0.0, 0.0, 1.0]])
    flipped = _probability([[0.0, 0.0, 2.0]], [[0.0, 0.0, -1.0]])
    mismatch = _probability([[0.0, 0.0, 2.0]], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(aligned["probability"], flipped["probability"])
    assert mismatch["probability"][0] < aligned["probability"][0]


def test_correspondence_competition_penalizes_source_already_won_by_base():
    losing = _probability([[0.0, 0.0, 2.0]], primary_source_target_distances=np.zeros(4))
    winning = _probability([[0.0, 0.0, 2.0]], primary_source_target_distances=np.full(4, 0.2))
    assert winning["probability"][0] > losing["probability"][0]


def test_correspondence_outputs_terms_calibration_and_no_nan():
    result = _probability([[0.0, 0.0, 2.0], [5.0, 0.0, 2.0]], reserve_normals=None)
    assert set(result) == {"probability", "terms", "diagnostics", "calibration"}
    arrays = [result["probability"]] + list(result["terms"].values()) + list(result["calibration"].values())
    assert all(np.all(np.isfinite(value)) for value in arrays)


def test_corr_weak_prefers_matchable_complement_over_unmatchable_high_weak_gain():
    points = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    normals = np.array([[1.0, 0.0, 0.2], [0.0, 0.0, 1.0]])
    jacobians = reserve_point_jacobians_camera(points, normals, np.eye(4))
    weak = jacobians[1] / np.linalg.norm(jacobians[1])
    basis = np.linalg.qr(np.column_stack([weak, np.eye(6)]))[0]
    hessian = basis @ np.diag([1e-8, 1.0, 1.0, 1.0, 1.0, 1.0]) @ basis.T
    scores, details = correspondence_aware_weak_scores(
        points, normals, np.eye(4), hessian, np.array([[0.0, 0.0, 2.0]]),
        np.array([0.95, 0.001]), visible_indices=np.array([0, 1]),
    )
    assert details["weak_score"][1] > details["weak_score"][0]
    assert scores[0] > scores[1]
    assert SELECTION_SCORE_FUNCTIONS["runtime_corr_weak"] is correspondence_aware_weak_scores


def test_corr_weak_scores_only_visible_and_has_no_gt_parameters():
    import inspect

    signature = inspect.signature(correspondence_probability)
    assert not any("gt" in name.lower() or "ground_truth" in name.lower() for name in signature.parameters)
    points = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    scores, _ = correspondence_aware_weak_scores(
        points, normals, np.eye(4), np.diag([1e-8, 1, 1, 1, 1, 1]),
        np.array([[0.0, 0.0, 2.0]]), np.array([0.8]), visible_indices=np.array([1]),
    )
    assert scores[0] == 0.0
    assert np.all(np.isfinite(scores))
