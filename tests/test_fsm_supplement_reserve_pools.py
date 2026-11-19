import json
from pathlib import Path

import numpy as np
import pytest

from scripts import fsm_supplement_reserve_pools as pools
from scripts.fsm_build_observable_map import _build_activation_geometry, _cheap_scores
from utils.fsm_map import build_fsm_map, load_fsm_map, load_runtime_reserve_geometry, save_fsm_map


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pools, "RESERVE_COUNT", 12)
    rng = np.random.default_rng(19)
    centers = np.repeat(np.arange(4)[:, None], 3, axis=1)
    means = (np.repeat(centers, 20, axis=0) + rng.uniform(0, 0.2, size=(80, 3))).astype(np.float32)
    normal = np.tile([0.0, 0.0, 1.0], (80, 1))
    params = dict(means3D=means, log_scales=np.full((80, 1), -1.0),
                  unnorm_rotations=np.tile([1.0, 0.0, 0.0, 0.0], (80, 1)),
                  logit_opacities=np.linspace(-2, 2, 80), normals=normal,
                  cam_unnorm_rots=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                  cam_trans=np.zeros((10, 3)), intrinsics=np.eye(3), org_width=80, org_height=60,
                  keyframe_time_indices=np.array([0, 4, 8]))
    paths = {name: tmp_path / f"{name}.npz" for name in
             ("params", "map_path", "geometry_path", "selection_path", "full_map")}
    np.savez(paths["params"], **params)
    np.savez(paths["selection_path"], reference_frame_ids=params["keyframe_time_indices"])
    ids = np.array([0, 1, 20, 40, 60])
    cheap = np.array([0, 2, 20, 40, 60])
    preserve = np.array([0, 20, 40, 60])
    reserve = np.arange(3, 15)
    params_hash = pools._sha256(paths["params"])
    control_hash = "a" * 64
    base = build_fsm_map(params, ids, source_params_sha256=params_hash, selection_controls_sha256=control_hash)
    save_fsm_map(base, paths["map_path"])
    geometry = _build_activation_geometry(
        params, ids, cheap, preserve, np.array([1]), np.array([2]), normal[[2]], "params:normals",
        reserve, normal[reserve], np.ones(12, dtype=np.int64), np.ones(12), "params:normals",
        _cheap_scores(params, "scale_opacity"), dict(params_sha256=params_hash, controls_sha256=control_hash),
    )
    np.savez(paths["geometry_path"], **geometry)
    return paths


def test_preflight_is_read_only_and_uses_original_universe(tmp_path, inputs):
    out = tmp_path / "out"
    manifest = pools.pool_manifest("office4", out, inputs=inputs)
    assert not out.exists()
    assert manifest["reference_views"] == dict(status="verified", count=3, role="mapping_estimated_keyframes",
                                              archived_count=3, archived_ids_match=True)
    assert manifest["candidate_universe"]["source_count"] == 80
    assert manifest["candidate_universe"]["eligible_count"] == 74
    assert manifest["candidate_universe"]["archived_reserve_subset_verified"] is True
    assert not manifest["inputs"]["full_map"]["present"]
    assert [row["status"] for row in manifest["pools"]] == ["planned"] * 4 + ["ready"]
    assert all(row["map_path"] == str(inputs["map_path"]) for row in manifest["pools"])
    assert manifest["pools"][-1]["geometry_path"] == str(inputs["geometry_path"])
    assert manifest["pools"][-1]["positive_trace_count"] == 12

    np.savez(inputs["full_map"], indices=np.r_[np.arange(79), 0])
    full = pools.pool_manifest("office4", out, inputs=inputs)["inputs"]["full_map"]
    assert full["present"] and full["count"] == 80 and not full["covers_source_ids"]


def test_missing_or_mismatched_params_cannot_use_archive_as_full_universe(tmp_path, inputs):
    inputs["params"].unlink()
    manifest = pools.pool_manifest("room2", tmp_path / "out", inputs=inputs)
    assert [row["status"] for row in manifest["pools"]] == ["blocked"] * 4 + ["ready"]
    assert not manifest["inputs"]["params"]["present"]
    assert "candidate_universe" not in manifest
    np.savez(inputs["params"], means3D=np.zeros((80, 3)))
    with pytest.raises(ValueError, match="SHA-256"):
        pools.build_pools("room2", tmp_path / "out", inputs=inputs)
    assert not (tmp_path / "out").exists()


def test_toy_sidecars_keep_frozen_map_and_match_control_quotas(tmp_path, inputs):
    out = tmp_path / "out"
    frozen = [pools._sha256(inputs[key]) for key in ("map_path", "geometry_path")]
    manifest = pools.build_pools("fr3", out, inputs=inputs)
    assert all(row["status"] == "ready" for row in manifest["pools"])
    assert Path(manifest["manifest_path"]) == out / "fr3/manifest.json"
    assert pools.pool_manifest("fr3", out, inputs=inputs) == manifest
    assert len({row["point_ids_sha256"] for row in manifest["pools"][:3]}) == 3
    assert [pools._sha256(inputs[key]) for key in ("map_path", "geometry_path")] == frozen
    assert {tuple(row) for row in manifest["control_quota"]["voxel_quotas"]} == {
        (0, 0, 0, 3), (2, 2, 2, 3), (4, 4, 4, 3), (6, 6, 6, 3),
    }

    base = load_fsm_map(inputs["map_path"])
    archive = load_runtime_reserve_geometry(inputs["geometry_path"], base)
    for row in manifest["pools"][:-1]:
        geometry = load_runtime_reserve_geometry(row["geometry_path"], base)
        assert np.array_equal(geometry["replacement_global_ids"], archive["replacement_global_ids"])
        ids = geometry["reserve_global_ids"]
        assert np.unique(ids).size == 12
        assert set(ids).isdisjoint(archive["baseline_global_ids"])
        assert set(ids).isdisjoint(base["indices"])
        assert np.array_equal(np.bincount(ids // 20), [3, 3, 3, 3])
        assert row["quality_count"] == (6 if row["arm"] == "quality_spatial" else 0)
        assert row["point_ids_sha256"] == str(geometry["e3_point_ids_sha256"])

    second = pools.build_pools("fr3", tmp_path / "again", inputs=inputs)
    assert [row["point_ids_sha256"] for row in manifest["pools"]] == [row["point_ids_sha256"] for row in second["pools"]]
    with pytest.raises(ValueError, match="existing"):
        pools.build_pools("fr3", out, inputs=inputs)
    proof = Path(manifest["manifest_path"])
    raw = json.loads(proof.read_text())
    raw["pools"][0]["point_ids_sha256"] = "forged"
    proof.write_text(json.dumps(raw))
    rejected = pools.pool_manifest("fr3", out, inputs=inputs)
    assert "Pool point IDs differ" in rejected["lookup_error"]
    assert rejected["pools"][0]["status"] == "blocked"


def test_build_refuses_linked_output(tmp_path, inputs):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        pools.build_pools("room2", alias, inputs=inputs)
    assert not list(target.iterdir())


def test_config_factory_keeps_frozen_replay_and_closed_loop_controls_identical():
    config = dict(registration={"backend": "test"}, observable_reserve={
        "selection": "runtime_corr_weak", "candidate_verifier": {"enabled": True}, "seed": 7,
        "geometry_path": "stale", "budget": 128,
    }, activation_diagnostics={"enabled": True, "geometry_path": "stale"})
    pool = dict(status="ready", geometry_path="/pool/activation_geometry.npz")
    closed = pools.make_config(config, pool, mode="closed_loop")
    assert pools.make_config(config, pool, mode="replay") == closed
    assert closed["observable_reserve"]["geometry_path"] == pool["geometry_path"]
    assert closed["observable_reserve"]["mode"] == "active"
    assert closed["activation_diagnostics"]["geometry_path"] == pool["geometry_path"]
    assert not closed["paired_reserve_evaluation"]["enabled"]
    baseline = pools.make_config(config, pool, mode="baseline")
    assert not baseline["observable_reserve"]["enabled"]
    assert baseline["observable_reserve"]["candidate_verifier"]["enabled"]
    shadow = pools.make_config(config, pool, mode="shadow")
    assert not shadow["observable_reserve"]["enabled"]
    assert shadow["paired_reserve_evaluation"]["policies"] == ["runtime_corr_weak"]
    assert shadow["paired_reserve_evaluation"]["geometry_path"] == pool["geometry_path"]
    assert config["observable_reserve"]["geometry_path"] == "stale"
    with pytest.raises(ValueError, match="blocked"):
        pools.make_config(config, {**pool, "status": "blocked"}, mode="closed_loop")
