import copy
import csv
import json
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import fsm_supplement_static as static


def write_map(path, params, means, strategy, percent, *, strategy_override=None):
    count = (len(means) * percent + 99) // 100
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = dict.fromkeys(static.REQUIRED_FSM_FIELDS, np.array(0))
    fields.update(indices=np.arange(count), means=means[:count], strategy=strategy_override or strategy,
                  compression_ratio=count / len(means), source_params_path=str(params))
    np.savez(path, **fields)


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    mvp = tmp_path / "mvp"
    mvp.mkdir()
    monkeypatch.setattr(static, "ROOT", tmp_path)
    monkeypatch.setattr(static, "MVP", mvp)
    monkeypatch.setattr(static, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(static, "HISTORY", mvp / "history")
    monkeypatch.setattr(static, "RECOVERY", mvp / "recovery")
    monkeypatch.setattr(static, "query_assets", lambda _: (range(2), dict(frame_count=2, metadata=[]), []))
    output = mvp / "supplement_static_mock"
    history = dict(configs={}, map_paths={}, source_and_input_hashes=[], end_exclusive=None,
                   thread_env=dict.fromkeys(static.THREADS, "1"))
    n = 2000
    means = np.arange(n * 3).reshape(n, 3)
    for scene in static.SCENES:
        params = tmp_path / f"{scene}.npz"
        np.savez(params, means3D=means, log_scales=means, unnorm_rotations=means, logit_opacities=means,
                 cam_unnorm_rots=np.array([[1], [0], [0], [0]]), cam_trans=np.array([[0], [0], [0]]),
                 intrinsics=np.eye(3), org_width=640, org_height=480, keyframe_time_indices=[0])
        full, observable, pareto, proof = static.scene_assets(scene, None)
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(json.dumps(dict(provenance=dict(params_path=str(params),
                              params_sha256=static.file_record(params)["sha256"]))))
        if scene != "office4":
            write_map(full, params, means, "full", 100)
        if scene != "fr3":
            write_map(observable, params, means, static.OBSERVABLE, 20)
        if scene == "room2":
            variants = ((static.FPS, 20), (static.VOXEL, 20))
        elif scene == "fr3":
            variants = ((static.FPS, 20), (static.FPS, 10), (static.VOXEL, 10))
        else:
            variants = ()
        for strategy, percent in variants:
            path = pareto / f"{strategy}_ratio_0p{percent // 10}" / "fsm_map.npz"
            write_map(path, params, means, strategy, percent)
        if scene in ("room2", "office0"):
            history["configs"][scene] = static.scene_config(scene)
            history["map_paths"][scene] = dict(full=str(full), compact=str(observable))
            history["source_and_input_hashes"].extend(map(static.file_record, (full, observable)))
            for key in ("full", "compact"):
                root = static.RECOVERY if scene == "office0" and key == "full" else static.HISTORY
                directory = root / scene / key
                directory.mkdir(parents=True)
                for filename in static.RUN_FILES:
                    (directory / filename).touch()
    static.write_new(static.HISTORY / "protocol.json", history)
    score = Mock(side_effect=lambda *args: (dict(raw_ate_rmse_m=0.01, all_frame_joint_success={
        "2cm_1deg": 1.0, "5cm_2deg": 1.0}), [(0, 0.0), (1, 1.0)], np.zeros((2, 2)), b"gt"))
    monkeypatch.setattr(static, "score", score)
    return output, means, score


def test_preflight_reuses_only_same_protocol_results_and_plans_full_export(inventory):
    output, _, score = inventory
    plan, inputs = static.make_plan(output)
    assert not output.exists()
    room2, fr3, office0, office4 = (plan["scenes"][name] for name in static.SCENES)
    assert [pair["status"] for pair in room2["comparisons"]] == ["pending", "reuse_result", "pending", "pending"]
    assert room2["arms"][static.arm_id(static.OBSERVABLE, 30)]["build"]["reserve_count"] == 0
    assert room2["arms"][static.arm_id(static.OBSERVABLE, 10)]["build"]["count"] == 200
    assert fr3["arms"][static.arm_id(static.FPS, 30)]["build"]["count"] == 600
    for scene, percent in ((room2, 20), (fr3, 10)):
        comparison = scene["comparisons"][-1]
        assert comparison["experiment"] == "E2"
        assert comparison["matched_global_points"] == 20 * percent
        for method, side in ((static.FPS, "reference"), (static.VOXEL, "candidate")):
            assert comparison[side] == static.arm_id(method, percent)
            assert scene["arms"][comparison[side]]["status"] == "reuse_map"
    recovery = static.RECOVERY / "office0/full"
    assert office0["arms"]["full_100"]["result_path"] == str(recovery)
    assert recovery / "trajectory_gt.txt" in inputs
    assert [call.args[0] for call in score.call_args_list].count(recovery) == 1
    assert office4["checkpoint"]["verified"]
    assert office4["arms"]["full_100"]["status"] == "build_map"
    assert office4["arms"]["full_100"]["build"]["count"] == 2000
    assert office4["arms"][static.arm_id(static.OBSERVABLE, 20)]["status"] == "reuse_map"


def test_full_office4_override_requires_same_checkpoint_geometry(inventory):
    output, means, _ = inventory
    params = static.ROOT / "office4.npz"
    full = static.ROOT / "verified_full.npz"
    write_map(full, params, means + 1, "full", 100)
    plan, _ = static.make_plan(output, full)
    assert "geometry differs" in plan["scenes"]["office4"]["prerequisites"][0]
    write_map(full, params, means, "full", 100)
    plan, _ = static.make_plan(output, full)
    assert {a["status"] for a in plan["scenes"]["office4"]["arms"].values()} == {"reuse_map"}


def test_voxel_alias_and_missing_checkpoint_do_not_pass_preflight(inventory):
    output, means, _ = inventory
    params = static.ROOT / "room2.npz"
    _, _, pareto, _ = static.scene_assets("room2", None)
    path = pareto / "voxel_grid_pure_ratio_0p2/fsm_map.npz"
    write_map(path, params, means, static.VOXEL, 20, strategy_override="voxel_scale_opacity")
    (static.ROOT / "fr3.npz").unlink()
    plan, _ = static.make_plan(output)
    assert plan["scenes"]["room2"]["comparisons"][-1]["status"] == "blocked"
    assert "missing original checkpoint" in plan["scenes"]["fr3"]["prerequisites"][0]


def test_missing_keyframes_block_new_observable_builds_only(inventory):
    output, _, _ = inventory
    checkpoint = static.ROOT / "room2.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files if key != "keyframe_time_indices"}
    np.savez(checkpoint, **fields)
    proof = static.scene_assets("room2", None)[3]
    metadata = static.read_json(proof)
    metadata["provenance"]["params_sha256"] = static.file_record(checkpoint)["sha256"]
    proof.write_text(json.dumps(metadata))
    plan, _ = static.make_plan(output)
    arms = plan["scenes"]["room2"]["arms"]
    assert arms["full_100"]["status"] == "reuse_result"
    assert arms[static.arm_id(static.FPS, 20)]["status"] == "reuse_map"
    for percent in (10, 30):
        arm = arms[static.arm_id(static.OBSERVABLE, percent)]
        assert arm["status"] == "blocked"
        assert "original estimated mapping keyframes" in arm["blockers"][0]


def test_metrics_pairing_and_lost_frames(tmp_path):
    cfg = static.scene_config("room2")
    scene = dict(config=cfg, query=dict(frame_count=3))
    arm = dict(map="map.npz", asset=dict(global_points=4))
    summary = {k: cfg["registration"][k] for k in ("backend", "evaluation_protocol", "first_pose_protocol", "motion_model")}
    summary.update(map_path=arm["map"], N_star=4, num_frames=3, num_lost_frames=1, num_recovered_frames=0,
                   observable_reserve_enabled=False, uses_gt_current_frame_initialization=False, uses_gt_fallback=False,
                   num_reserve_adopted_frames=0, num_fallbacks=0, recovery_enabled=True, raw_ate_rmse_m=0.02)
    static.write_new(tmp_path / "summary.json", summary)
    frames = [dict(frame_id=i, timestamp=i, tracking_valid=i != 1, tracking_status="mock", recovered=False,
                   recovery_stage="", total_frame_latency_ms=10, local_query_latency_ms=5, latency_ms=2)
              for i in range(3)]
    errors = [dict(frame_id=i, timestamp=i, tracking_valid=i != 1, raw_translation_error_m=e,
                   raw_rotation_error_deg=r) for i, (e, r) in enumerate(((0, 0), ("", ""), (0.02, 1)))]
    for filename, rows in (("frame_metrics.csv", frames), ("per_frame_errors.csv", errors)):
        with (tmp_path / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (tmp_path / "trajectory_gt.txt").write_bytes(b"0 gt\n1 gt\n2 gt\n")
    score = static.score(tmp_path, arm, scene)
    assert score[0]["lost"] == score[0]["longest_lost_run"] == 1
    assert score[0]["all_frame_joint_success"]["2cm_1deg"] == pytest.approx(2 / 3)
    assert score[0]["timed_frames"] == 2
    result = static.paired(score, score)
    assert result["common_valid_frames"] == 2
    assert result["common_valid_raw_rmse"]["reference"]["translation_m"] == pytest.approx(0.02 / np.sqrt(2))
    with pytest.raises(ValueError, match="timestamps and GT"):
        static.paired(score, (*score[:3], b"different"))
    with pytest.raises(ValueError, match="timestamps and GT"):
        static.paired(score, (score[0], [(0, 0), (1, 0.5), (2, 2)], *score[2:]))


def test_freeze_no_overwrite_and_execution_mock(inventory, monkeypatch):
    output, _, _ = inventory
    from scripts import fsm_build_observable_map, fsm_export, fsm_localize

    builders = Mock()
    monkeypatch.setattr(fsm_build_observable_map, "build_observable_selection", builders.observable)
    monkeypatch.setattr(fsm_export, "export_fsm_maps", builders.export)
    monkeypatch.setattr(fsm_localize, "run_localization", builders.localize)
    for key in static.THREADS:
        monkeypatch.setenv(key, "1")
    plan, _ = static.make_plan(output)
    source = static.ROOT / "room2.npz"
    static.freeze(output, plan, {source})
    manifest_path = output / "protocol_manifest.json"
    manifest_before = manifest_path.read_bytes()
    assert not builders.mock_calls
    with pytest.raises(ValueError, match="no overwrite"):
        static.freeze(output, plan, {source})
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="frozen dependency changed"):
        static.execute(output)
    assert not builders.mock_calls
    assert not (output / "run").exists()
    with source.open("r+b") as stream:
        stream.truncate(source.stat().st_size - len(b"changed"))
    check_map = Mock(return_value=dict(global_points=200, map_bytes=100, retained_fraction=0.1))
    monkeypatch.setattr(static, "check_map", check_map)
    static.execute(output)
    assert builders.observable.call_count == 2
    assert all(call.kwargs["reserve_count"] == 0 for call in builders.observable.call_args_list)
    assert builders.export.call_count == 2
    assert [c.args[2] for c in builders.export.call_args_list] == [[static.FPS], ["full"]]
    assert builders.export.call_args.kwargs["counts"] == [2000]
    assert builders.localize.call_count == 11
    assert all(call.kwargs["start"] == 0 and call.kwargs["end"] == 2 for call in builders.localize.call_args_list)
    assert (output / "run/summary.json").is_file()
    assert manifest_path.read_bytes() == manifest_before
    with pytest.raises(ValueError, match="no overwrite"):
        static.execute(output)


def test_protocol_rejects_score_bias_and_active_reserve():
    config = static.scene_config("fr3")
    for section, key in (("local_submap", "use_map_scores"), ("observable_reserve", "enabled")):
        changed = copy.deepcopy(config)
        changed[section][key] = True
        with pytest.raises(ValueError, match="fixed-map strict robust"):
            static.validate_protocol(changed)
