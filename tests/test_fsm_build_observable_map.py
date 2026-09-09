import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.fsm_build_observable_map import (
    CANDIDATE_SOURCE_REMOVED,
    CANDIDATE_SOURCE_SPATIAL,
    build_forced_replacement_candidates,
    build_observable_selection,
)
from utils.fsm_map import load_activation_geometry


ROOT = os.path.dirname(os.path.dirname(__file__))


def make_params(path, count=300, logits=None):
    means = np.column_stack([np.linspace(-0.2, 0.2, count), np.zeros(count), np.full(count, 2.0)])
    if logits is None:
        logits = np.linspace(-1.0, 1.0, count)
    np.savez(
        path,
        means3D=means,
        normals=np.tile(np.array([[0.0, 0.0, 1.0]]), (count, 1)),
        log_scales=np.log(np.full((count, 1), 0.02)),
        logit_opacities=np.asarray(logits)[:, None],
        unnorm_rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (count, 1)),
        cam_unnorm_rots=np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:, None], (1, 2))[None],
        cam_trans=np.zeros((1, 3, 2)),
        intrinsics=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        org_width=100,
        org_height=100,
        keyframe_time_indices=np.array([0, 1]),
        gt_w2c_all_frames=np.repeat((10.0 * np.eye(4))[None], 2, axis=0),
    )


def test_builder_writes_selection_artifact_for_synthetic_params(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection" / "observable.npz"
    make_params(params_path)
    plan = build_observable_selection(
        str(params_path), str(output_path), count=2, ratio=None, pair_cap=20,
        greedy_core_cap=2, replacement_count=2,
    )
    loaded = np.load(output_path)
    assert plan["uses_query_future_ground_truth"] is False
    assert plan["reference_pose_role"] == "mapping_estimated_keyframes"
    assert plan["held_out_query"] is False
    assert plan["uses_future_estimated_query_poses"] is True
    assert plan["selection_mode"] == "observable_forced_replacement"
    assert plan["objective"] == "sum_per_pose_logdet"
    assert plan["selected_count"] == 2
    assert loaded["selected_indices"].shape == (2,)
    assert loaded["factor_gaussian_id_space"] == "candidate_local"
    assert plan["changed_count"] == 2
    assert set(loaded["replacement_indices"]).isdisjoint(loaded["baseline_indices"])
    assert loaded["per_pose_information"].shape == (2, 6, 6)
    assert loaded["per_pose_logdet"].shape == (2,)
    assert "per_pose_information" not in plan
    assert "pose_ptr" not in loaded
    assert "jacobians" not in loaded
    assert "cheap_scores" not in loaded
    assert (output_path.parent / "observable.npz.done.json").is_file()


def test_cli_dry_run_is_strictly_write_free_and_prints_json_plan(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "must_not_exist" / "observable.npz"
    map_path = tmp_path / "also_absent" / "fsm_map.npz"
    make_params(params_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fsm_build_observable_map.py",
            "--params",
            str(params_path),
            "--output",
            str(output_path),
            "--map-output",
            str(map_path),
            "--count",
            "2",
            "--replacement-count",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["estimated_factor_memory_bytes"] > 0
    assert plan["uses_query_future_ground_truth"] is False
    assert plan["candidate_count"] == 300
    assert not output_path.parent.exists()
    assert not map_path.parent.exists()


def test_builder_rejects_insufficient_greedy_fraction(tmp_path):
    params_path = tmp_path / "params.npz"
    make_params(params_path, count=10)
    with pytest.raises(ValueError, match="observable_core_fraction"):
        build_observable_selection(
            str(params_path), str(tmp_path / "out.npz"), count=4, ratio=None,
            greedy_core_cap=1, replacement_count=1, min_greedy_fraction=0.5,
        )


def test_candidate_pool_cap_includes_removed_and_challengers(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    make_params(params_path, count=12)
    plan = build_observable_selection(
        str(params_path), str(output_path), count=8, ratio=None, candidate_cap=4, greedy_core_cap=2,
        replacement_count=2, fill_strategy="opacity"
    )
    loaded = np.load(output_path)
    assert plan["candidate_count"] == 4
    assert plan["observable_core_count"] == 2
    assert plan["cheap_fill_count"] == 6
    assert loaded["candidate_global_ids"].shape == (4,)
    assert np.count_nonzero(loaded["candidate_source_codes"] == CANDIDATE_SOURCE_REMOVED) == 2
    assert loaded["selected_indices"].shape == (8,)
    assert np.unique(loaded["selected_indices"]).size == 8


def test_cheap_fill_is_stable_and_full_budget_is_exact(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    make_params(params_path, count=6, logits=np.zeros(6))
    plan = build_observable_selection(
        str(params_path), str(output_path), count=5, ratio=None, candidate_cap=2, greedy_core_cap=0, fill_strategy="opacity"
    )
    loaded = np.load(output_path)
    np.testing.assert_array_equal(loaded["selected_indices"], np.arange(5))
    assert plan["selection_mode"] == "cheap_baseline"
    assert plan["cheap_fill_count"] == 5
    assert plan["actual_visibility_pairs"] == 0


def test_dry_run_memory_uses_capped_candidate_not_large_budget(tmp_path):
    params_path = tmp_path / "params.npz"
    make_params(params_path, count=100)
    plan = build_observable_selection(
        str(params_path), str(tmp_path / "out.npz"), count=90, ratio=None, candidate_cap=8,
        greedy_core_cap=4, replacement_count=4, dry_run=True
    )
    assert plan["candidate_count"] == 8
    assert plan["possible_visibility_pairs"] == 16
    assert plan["bounded_visibility_pairs"] == 16


def test_trace_score_scatter_and_map_scores_use_correct_global_ids(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "map.npz"
    make_params(params_path, count=8, logits=[0, 3, -2, 2, -1, 1, -3, -4])
    build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path), count=4, ratio=None,
        candidate_cap=4, greedy_core_cap=2, replacement_count=2, fill_strategy="opacity",
    )
    loaded = np.load(output_path)
    candidate_ids = loaded["candidate_global_ids"]
    fsm_map = np.load(map_path)
    logits = np.array([0, 3, -2, 2, -1, 1, -3, -4], dtype=np.float64)
    expected_scores = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(fsm_map["scores"], expected_scores[fsm_map["indices"]].astype(np.float32))
    assert fsm_map["score_name"] == "opacity"
    assert "full_trace_scores" not in loaded
    assert candidate_ids.shape == loaded["candidate_trace_scores"].shape


def test_save_factors_is_explicit_opt_in(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    make_params(params_path, count=8)
    build_observable_selection(
        str(params_path), str(output_path), count=2, ratio=None, pair_cap=20,
        greedy_core_cap=1, replacement_count=1, candidate_cap=3, save_factors=True
    )
    loaded = np.load(output_path)
    assert loaded["pose_ptr"].shape == (3,)
    assert loaded["jacobians"].shape[1] == 6
    assert loaded["cheap_scores"].shape == (8,)
    assert loaded["full_trace_scores"].shape == (8,)


def test_skip_existing_requires_matching_provenance_and_loadable_outputs(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "map.npz"
    make_params(params_path, count=10)
    first = build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path), count=2, ratio=None,
        greedy_core_cap=1, replacement_count=1
    )
    skipped = build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path), count=2, ratio=None,
        greedy_core_cap=1, replacement_count=1, skip_existing=True
    )
    assert first.get("skipped_existing") is None
    assert skipped["skipped_existing"] is True
    rerun = build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path), count=3, ratio=None,
        greedy_core_cap=1, replacement_count=1, skip_existing=True
    )
    assert rerun.get("skipped_existing") is None
    assert rerun["selected_count"] == 3


def test_forced_replacement_is_exact_outside_and_map_matches(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "map.npz"
    make_params(params_path, count=12)
    plan = build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path), count=6, ratio=None,
        replacement_count=2, greedy_core_cap=2, candidate_cap=6,
    )
    artifact = np.load(output_path)
    fsm_map = np.load(map_path)
    baseline = set(artifact["baseline_indices"].tolist())
    selected = set(artifact["selected_indices"].tolist())
    assert len(baseline - selected) == len(selected - baseline) == plan["changed_count"] == 2
    assert set(artifact["replacement_indices"].tolist()).isdisjoint(baseline)
    np.testing.assert_array_equal(fsm_map["indices"], artifact["selected_indices"])
    assert fsm_map["strategy"] == "observable_forced_replacement"
    assert artifact["preservation_trace_ratio"].shape == (2,)
    assert artifact["replacement_guarantee"]


def test_candidate_construction_has_spatial_sources_and_is_deterministic():
    means = np.column_stack([np.arange(20), np.zeros(20), np.zeros(20)]).astype(np.float64)
    scores = np.linspace(20.0, 1.0, 20)
    baseline = np.arange(6)
    first = build_forced_replacement_candidates(means, scores, baseline, 2, 10)
    second = build_forced_replacement_candidates(means, scores, baseline, 2, 10)
    for lhs, rhs in zip(first, second):
        np.testing.assert_array_equal(lhs, rhs)
    removed, challengers, factor_ids, source_codes = first
    assert factor_ids.size == 10
    assert challengers.size == 8
    spatial_ids = factor_ids[source_codes == CANDIDATE_SOURCE_SPATIAL]
    assert spatial_ids.size > 0
    assert np.ptp(means[spatial_ids, 0]) > 0
    assert set(challengers).isdisjoint(baseline)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"count": 5, "replacement_count": 3, "candidate_cap": 5}, "at least 2"),
        ({"count": 9, "replacement_count": 2, "candidate_cap": 4}, "outside"),
    ],
)
def test_replacement_boundaries_are_rejected(tmp_path, kwargs, message):
    params_path = tmp_path / "params.npz"
    make_params(params_path, count=10)
    with pytest.raises(ValueError, match=message):
        build_observable_selection(str(params_path), str(tmp_path / "out.npz"), ratio=None, **kwargs)


def test_provenance_replacement_count_prevents_stale_skip(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    make_params(params_path, count=12)
    build_observable_selection(
        str(params_path), str(output_path), count=5, ratio=None,
        replacement_count=1, greedy_core_cap=2, candidate_cap=6,
    )
    rerun = build_observable_selection(
        str(params_path), str(output_path), count=5, ratio=None,
        replacement_count=2, greedy_core_cap=2, candidate_cap=6, skip_existing=True,
    )
    assert rerun.get("skipped_existing") is None
    assert rerun["replacement_count"] == 2


def test_activation_geometry_removed_values_match_source_params(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "fsm_map.npz"
    geometry_path = tmp_path / "activation_geometry.npz"
    make_params(params_path, count=12)
    build_observable_selection(
        str(params_path),
        str(output_path),
        map_output=str(map_path),
        activation_geometry_output=str(geometry_path),
        count=6,
        ratio=None,
        replacement_count=2,
        greedy_core_cap=2,
        candidate_cap=6,
        fill_strategy="opacity",
    )
    params = dict(np.load(params_path))
    geometry = load_activation_geometry(str(geometry_path), str(map_path))
    removed = geometry["removed_global_ids"]
    np.testing.assert_allclose(geometry["removed_means"], params["means3D"][removed])
    np.testing.assert_allclose(geometry["removed_log_scales"], params["log_scales"][removed])
    np.testing.assert_allclose(geometry["removed_scales"], np.exp(np.repeat(params["log_scales"][removed], 3, axis=1)))
    np.testing.assert_allclose(geometry["removed_normals"], params["normals"][removed])
    np.testing.assert_allclose(geometry["removed_scores"], 1.0 / (1.0 + np.exp(-params["logit_opacities"][removed, 0])))
    assert geometry["normal_source"] == "params:normals"
    assert set(geometry["replacement_global_ids"]).issubset(geometry["selected_global_ids"])
    assert set(geometry["removed_global_ids"]).isdisjoint(geometry["selected_global_ids"])


def test_explicit_cheap_activation_geometry_has_empty_removed_set(tmp_path):
    params_path = tmp_path / "params.npz"
    map_path = tmp_path / "fsm_map.npz"
    geometry_path = tmp_path / "activation_geometry.npz"
    make_params(params_path, count=8)
    build_observable_selection(
        str(params_path),
        str(tmp_path / "selection.npz"),
        map_output=str(map_path),
        activation_geometry_output=str(geometry_path),
        count=4,
        ratio=None,
        greedy_core_cap=0,
        replacement_count=0,
    )
    geometry = load_activation_geometry(str(geometry_path), str(map_path))
    assert geometry["removed_global_ids"].shape == (0,)
    assert geometry["removed_covariances"].shape == (0, 3, 3)
    assert geometry["normal_source"] == "empty_removed_set"


def test_activation_geometry_is_opt_in_by_default(tmp_path):
    params_path = tmp_path / "params.npz"
    make_params(params_path, count=8)
    build_observable_selection(
        str(params_path), str(tmp_path / "selection.npz"), map_output=str(tmp_path / "fsm_map.npz"),
        count=4, ratio=None, greedy_core_cap=0, replacement_count=0,
    )
    assert not (tmp_path / "activation_geometry.npz").exists()
    assert not list(tmp_path.glob("*activation*"))


def test_activation_geometry_atomic_write_and_corruption_prevents_skip(tmp_path, monkeypatch):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "map" / "fsm_map.npz"
    geometry_path = tmp_path / "map" / "activation_geometry.npz"
    make_params(params_path, count=10)
    replacements = []
    original_replace = os.replace

    def recording_replace(source, destination):
        replacements.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)
    build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path),
        activation_geometry_output=str(geometry_path), count=4, ratio=None,
        replacement_count=1, greedy_core_cap=1, candidate_cap=4,
    )
    geometry_replaces = [(source, destination) for source, destination in replacements if destination == str(geometry_path)]
    assert len(geometry_replaces) == 1
    assert os.path.dirname(geometry_replaces[0][0]) == str(geometry_path.parent)
    with np.load(geometry_path) as geometry:
        corrupted = dict(geometry)
    corrupted["map_selected_ids_sha256"] = np.asarray("0" * 64)
    np.savez(geometry_path, **corrupted)
    rerun = build_observable_selection(
        str(params_path), str(output_path), map_output=str(map_path),
        activation_geometry_output=str(geometry_path), count=4, ratio=None,
        replacement_count=1, greedy_core_cap=1, candidate_cap=4, skip_existing=True,
    )
    assert rerun.get("skipped_existing") is None
    load_activation_geometry(str(geometry_path), str(map_path))


def test_reserve_is_outside_base_deterministic_and_matches_params(tmp_path):
    params_path = tmp_path / "params.npz"
    make_params(params_path, count=20)
    outputs = []
    for suffix in ("first", "second"):
        selection_path = tmp_path / f"{suffix}_selection.npz"
        map_path = tmp_path / f"{suffix}_map.npz"
        geometry_path = tmp_path / f"{suffix}_geometry.npz"
        plan = build_observable_selection(
            str(params_path), str(selection_path), map_output=str(map_path),
            activation_geometry_output=str(geometry_path), count=6, ratio=None,
            replacement_count=2, greedy_core_cap=2, candidate_cap=10,
            reserve_count=7, fill_strategy="opacity",
        )
        outputs.append((np.load(selection_path), load_activation_geometry(str(geometry_path), str(map_path))))
        assert plan["reserve_count"] == 7
        assert plan["reserve_policy"]
        assert sum(plan["reserve_source_coverage"].values()) == 7
    first_artifact, first_geometry = outputs[0]
    second_artifact, second_geometry = outputs[1]
    reserve = first_geometry["reserve_global_ids"]
    assert set(reserve).isdisjoint(first_artifact["selected_indices"])
    assert np.unique(reserve).size == reserve.size
    np.testing.assert_array_equal(reserve, second_geometry["reserve_global_ids"])
    np.testing.assert_array_equal(first_geometry["reserve_source_codes"], second_geometry["reserve_source_codes"])
    params = dict(np.load(params_path))
    np.testing.assert_allclose(first_geometry["reserve_means"], params["means3D"][reserve])
    np.testing.assert_allclose(first_geometry["reserve_normals"], params["normals"][reserve])
    np.testing.assert_allclose(
        first_geometry["reserve_scales"], np.exp(np.repeat(params["log_scales"][reserve], 3, axis=1))
    )
    np.testing.assert_allclose(first_geometry["reserve_rotations"], params["unnorm_rotations"][reserve])
    np.testing.assert_allclose(
        first_geometry["reserve_covariances"],
        np.eye(3)[None] * (np.exp(params["log_scales"][reserve, 0]) ** 2)[:, None, None]
        + np.eye(3)[None] * 1e-6,
    )
    np.testing.assert_allclose(
        first_geometry["reserve_scores"], 1.0 / (1.0 + np.exp(-params["logit_opacities"][reserve, 0]))
    )
    assert int(first_geometry["reserve_count"]) == reserve.size
    assert str(first_geometry["reserve_policy"])


def test_reserve_zero_disables_sidecar_reserve(tmp_path):
    params_path = tmp_path / "params.npz"
    map_path = tmp_path / "map.npz"
    geometry_path = tmp_path / "geometry.npz"
    make_params(params_path, count=10)
    plan = build_observable_selection(
        str(params_path), str(tmp_path / "selection.npz"), map_output=str(map_path),
        activation_geometry_output=str(geometry_path), count=4, ratio=None,
        replacement_count=1, greedy_core_cap=1, candidate_cap=4, reserve_count=0,
    )
    geometry = load_activation_geometry(str(geometry_path), str(map_path))
    assert plan["reserve_count"] == 0
    assert geometry["reserve_global_ids"].shape == (0,)
    assert geometry["reserve_covariances"].shape == (0, 3, 3)


def test_reserve_count_is_part_of_skip_provenance(tmp_path):
    params_path = tmp_path / "params.npz"
    output_path = tmp_path / "selection.npz"
    map_path = tmp_path / "map.npz"
    geometry_path = tmp_path / "geometry.npz"
    make_params(params_path, count=12)
    common = dict(
        map_output=str(map_path), activation_geometry_output=str(geometry_path),
        count=4, ratio=None, replacement_count=1, greedy_core_cap=1, candidate_cap=4,
    )
    build_observable_selection(str(params_path), str(output_path), reserve_count=2, **common)
    skipped = build_observable_selection(
        str(params_path), str(output_path), reserve_count=2, skip_existing=True, **common
    )
    assert skipped["skipped_existing"] is True
    rerun = build_observable_selection(
        str(params_path), str(output_path), reserve_count=3, skip_existing=True, **common
    )
    assert rerun.get("skipped_existing") is None
    assert rerun["reserve_count"] == 3
