#!/usr/bin/env python3
"""Preflight and freeze E1/E2 static-map comparisons; execute only a frozen plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_cn_consistency_experiments import MVP, file_record, resolved_config
from utils.fsm_map import REQUIRED_FSM_FIELDS, REQUIRED_SPLATAM_FIELDS

FR3 = "freiburg3_long_office_household"
AUDIT = MVP.parent / "audit_v2_full_rerun/main"
HISTORY = MVP / "cn_consistency_compression_loss_full_v1"
RECOVERY = HISTORY.with_name(HISTORY.name + "_office0_recovery_v1")
OBSERVABLE = "observable_forced_replacement"
FPS = "fps_scale_opacity"
VOXEL = "voxel_grid_pure"
SCENES = {
    "room2": (("full", 100), (OBSERVABLE, 30), (OBSERVABLE, 20), (OBSERVABLE, 10), (FPS, 20), (VOXEL, 20)),
    "fr3": (("full", 100), (FPS, 30), (FPS, 20), (FPS, 10), (VOXEL, 10)),
    "office0": (("full", 100), (OBSERVABLE, 20)),
    "office4": (("full", 100), (OBSERVABLE, 20)),
}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
RUN_FILES = ("summary.json", "frame_metrics.csv", "per_frame_errors.csv", "trajectory_gt.txt")
OBS_CONTROLS = dict(max_reference_poses=64, candidate_cap=4096, pair_cap=200000,
                    greedy_core_cap=128, replacement_count=128, fill_strategy="scale_opacity",
                    normal_k=20, regularization=1e-6, min_eigen_every=0, min_eigen_bonus=0.0)
LABELS = {"full": "Full", OBSERVABLE: "Observable", FPS: "FPS-C-GLS", VOXEL: "Voxel-only"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arm_id(strategy, percent):
    return f"{strategy}_{percent}"


def scene_config(scene):
    cfg = resolved_config("tum", FR3, "baseline") if scene == "fr3" else resolved_config("replica", scene, "baseline")
    cfg["local_submap"].update(frustum_culling="voxel_blocks", coverage_representatives="spatial_strata",
                               projection_pre_cap=None, oracle_mode="off", use_map_scores=False)
    cfg["observable_reserve"]["candidate_verifier"] = {"enabled": False}
    validate_protocol(cfg)
    return cfg


def validate_protocol(cfg):
    reg, submap = cfg["registration"], cfg["local_submap"]
    if (tuple(reg[k] for k in ("backend", "evaluation_protocol", "first_pose_protocol", "motion_model"))
            != ("numpy_robust_point_to_plane", "mvp_strict", "gt_first_frame", "constant_velocity")
            or cfg["motion_model"]["type"] != "constant_velocity" or reg.get("fallback_on_divergence")
            or any(submap.get(k) != v for k, v in dict(enabled=True, frustum_culling="voxel_blocks",
                      coverage_representatives="spatial_strata", use_map_scores=False,
                      projection_pre_cap=None, oracle_mode="off", max_points=10000).items())
            or cfg["dataset"]["max_points"] != 10000
            or any(cfg.get(key, {}).get("enabled") for key in ("observable_reserve", "activation_diagnostics",
                      "paired_reserve_evaluation", "final_target_oracle"))
            or cfg["observable_reserve"].get("mode") != "off"
            or cfg["observable_reserve"].get("candidate_verifier", {}).get("enabled")):
        raise ValueError("not the fixed-map strict robust protocol")
    if cfg["data"] != cfg["dataset"] or tuple(cfg["data"][k] for k in ("start", "end", "stride")) != (0, -1, 1):
        raise ValueError("not the complete unsliced query protocol")


def query_assets(cfg):
    """Resolve the runner's frame order without decoding RGB-D or localizing."""
    from datasets.gradslam_datasets import ReplicaDataset, TUMDataset, load_dataset_config

    data = cfg["data"]
    folder = Path(data["basedir"]) / data["sequence"]
    yaml = Path(data["gradslam_data_cfg"])
    replica = data["dataset_name"] == "replica"
    pose = "groundtruth.txt" if (folder / "groundtruth.txt").is_file() else "pose.txt"
    metadata = [yaml, *(folder / name for name in (("traj.txt",) if replica else ("rgb.txt", "depth.txt", pose)))]
    if missing := [str(path) for path in metadata if not path.is_file()]:
        raise ValueError(f"missing query metadata: {missing}")
    dataset = (ReplicaDataset if replica else TUMDataset)(
        load_dataset_config(str(yaml)), data["basedir"], data["sequence"],
        start=data["start"], end=data["end"], stride=data["stride"], device=cfg["primary_device"],
        desired_height=data["desired_image_height"], desired_width=data["desired_image_width"],
        relative_pose=data.get("relative_pose", True), ignore_bad=False,
        use_train_split=data.get("use_train_split", True),
    )
    pairs = list(zip(dataset.color_paths, dataset.depth_paths))
    if len(pairs) < 2 or len(pairs) != len(dataset) or len(dataset.poses) != len(pairs):
        raise ValueError("query has incomplete RGB-D/GT frame associations")
    images = [Path(path) for pair in pairs for path in pair]
    if missing := [str(path) for path in images if not path.is_file()]:
        raise ValueError(f"missing query images ({len(missing)}): {missing[:3]}")
    digest = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
    return dataset, dict(frame_count=len(pairs), rgbd_order_sha256=digest,
                         metadata=list(map(str, metadata))), metadata + images


def scene_assets(scene, office4_full):
    name = f"tum_{FR3}" if scene == "fr3" else f"replica_{scene}_0"
    base = AUDIT / name
    full = base / "full/maps/full_ratio_1/fsm_map.npz"
    if scene == "office4" and office4_full is not None:
        full = office4_full.resolve()
    suite = "replica_office1_4_untouched_v1" if scene == "office4" else "runtime_reserve_v1"
    observable = MVP / suite / "maps" / name / "logdet/fsm_map.npz"
    proof = (MVP / "tum_fps_hybrid_v1/maps" / FR3 / "activation_geometry.npz.done.json" if scene == "fr3"
             else observable.with_name("selection.npz.done.json"))
    return full, observable, base / "pareto/maps", proof


def check_map(path, strategy, percent, params, source_means, params_hash):
    total = len(source_means)
    with np.load(path, allow_pickle=False) as data:
        if not set(REQUIRED_FSM_FIELDS).issubset(data.files):
            raise ValueError("incomplete FSM map")
        if (str(data["strategy"]) != strategy or Path(str(data["source_params_path"])).resolve() != params
                or ("source_params_sha256" in data and str(data["source_params_sha256"]) != params_hash)):
            raise ValueError("map strategy or checkpoint provenance mismatch")
        ids, means = data["indices"], data["means"]
        budget = (total * percent + 99) // 100
        if (ids.shape != (budget,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != budget
                or np.any(ids < 0) or np.any(ids >= total) or means.shape != (budget, 3)
                or not np.isclose(float(data["compression_ratio"]), budget / total, rtol=1e-7, atol=1e-9)):
            raise ValueError("map does not match the exact full-map point budget/ID space")
        if "reserve_global_ids" in data:
            raise ValueError("reserve payload must not be included in the static map")
        for start in range(0, budget, 65536):
            span = slice(start, start + 65536)
            if not np.allclose(means[span], source_means[ids[span]], atol=1e-7, rtol=0):
                raise ValueError(f"map geometry differs from {params} at source IDs")
    return dict(global_points=budget, map_bytes=path.stat().st_size, retained_fraction=budget / total)


def score(path, arm, scene):
    from scripts.fsm_bench_compression_loss import score_run

    summary = read_json(path / "summary.json")
    cfg = scene["config"]
    expected = {k: cfg["registration"][k] for k in
                ("backend", "evaluation_protocol", "first_pose_protocol", "motion_model")}
    expected.update(map_path=arm["map"], N_star=arm["asset"]["global_points"],
                    num_frames=scene["query"]["frame_count"], observable_reserve_enabled=False,
                    uses_gt_current_frame_initialization=False, uses_gt_fallback=False,
                    num_reserve_adopted_frames=0, num_fallbacks=0, recovery_enabled=cfg["recovery"]["enabled"])
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{path}: result does not match map/protocol/frame count")
    metrics, identities, errors = score_run(path, summary)
    if [i for i, _ in identities] != list(range(expected["num_frames"])):
        raise ValueError(f"{path}: incomplete or unordered frame set")
    metrics.update(arm["asset"], peak_rss_mb=summary.get("peak_rss_mb"))
    gt = (path / "trajectory_gt.txt").read_bytes()
    if len(gt.splitlines()) != expected["num_frames"]:
        raise ValueError(f"{path}: incomplete GT trajectory")
    return metrics, identities, errors, gt


def paired(reference, candidate):
    from scripts.fsm_bench_compression_loss import compare

    a, ids, x, gt = reference
    b, other_ids, y, other_gt = candidate
    if ids != other_ids or gt != other_gt:
        raise ValueError("comparison arms do not have identical frame IDs, timestamps and GT")
    result = compare(a, b, x, y)
    result["candidate_only_lost"] = result.pop("compact_only_lost")
    result["reference_only_lost"] = result.pop("full_only_lost")
    common = np.isfinite(x[:, 0]) & np.isfinite(y[:, 0])
    result["common_valid_raw_rmse"] = {}
    for name, values in (("reference", x), ("candidate", y)):
        rmse = None
        if common.any():
            translation, rotation = np.sqrt(np.mean(values[common] ** 2, axis=0))
            rmse = dict(translation_m=float(translation), rotation_deg=float(rotation))
        result["common_valid_raw_rmse"][name] = rmse
    return result


def make_plan(output, office4_full=None):
    history_path = HISTORY / "protocol.json"
    history = read_json(history_path) if history_path.is_file() else None
    old_hashes = {item["path"]: item for item in history["source_and_input_hashes"]} if history else {}
    dependencies = {Path(__file__).resolve(), *ROOT.glob("utils/fsm_*.py"),
                    *ROOT.glob("datasets/gradslam_datasets/*.py")}
    dependencies.update(ROOT / "scripts" / name for name in ("splatam.py", "fsm_localize.py", "fsm_export.py",
                        "fsm_build_observable_map.py", "fsm_bench_compression_loss.py", "fsm_bench_closed_loop_queries.py",
                        "fsm_check_pose_update_preservation.py", "fsm_cn_consistency_experiments.py"))
    dependencies.update(ROOT / "configs/fsm/mvp_observable" / name for name in ("common.py", "cn_consistency_case.py"))
    if history:
        dependencies.add(history_path)
    scenes = {}
    for scene, matrix in SCENES.items():
        cfg = scene_config(scene)
        value = scenes[scene] = dict(config=cfg, arms={}, comparisons=[], replay_not_independent_revisit=True)
        full, observable, pareto, proof = scene_assets(scene, office4_full)
        checkpoint = None
        source_means = None
        prerequisites = []
        checkpoint_error = "missing checkpoint provenance"
        reference_error = None
        try:
            provenance = read_json(proof)["provenance"]
            dependencies.add(proof)
            checkpoint = Path(provenance["params_path"]).resolve()
            value["checkpoint"] = dict(path=str(checkpoint), expected_sha256=provenance["params_sha256"],
                                       available=checkpoint.is_file())
            if checkpoint.is_file():
                record = file_record(checkpoint)
                if record["sha256"] != provenance["params_sha256"]:
                    raise ValueError("checkpoint SHA-256 differs from frozen map provenance")
                value["checkpoint"]["verified"] = True
                dependencies.add(checkpoint)
                with np.load(checkpoint, allow_pickle=False) as data:
                    required = {*REQUIRED_SPLATAM_FIELDS, "logit_opacities"}
                    if not required.issubset(data.files):
                        raise ValueError(f"missing checkpoint fields: {sorted(required - set(data.files))}")
                    source_means = data["means3D"]
                    if source_means.ndim != 2 or source_means.shape[1] != 3 or not len(source_means):
                        raise ValueError("checkpoint means3D must have shape [N, 3] with N > 0")
                    if scene == "room2":
                        from utils.fsm_observability import load_reference_views

                        fields = ("cam_unnorm_rots", "cam_trans", "intrinsics", "org_width", "org_height", "keyframe_time_indices")
                        try:
                            load_reference_views({key: data[key] for key in fields}, max_reference_poses=64)
                        except (ValueError, KeyError, IndexError) as exc:
                            reference_error = f"original estimated mapping keyframes unavailable: {exc}"
                checkpoint_error = None
            else:
                checkpoint_error = f"missing original checkpoint: {checkpoint}"
        except (OSError, ValueError, KeyError, IndexError) as exc:
            checkpoint_error = str(exc)
            source_means = None
        value["checkpoint_issue"] = checkpoint_error

        try:
            _, value["query"], query_paths = query_assets(cfg)
            dependencies.update(query_paths)
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            prerequisites.append(f"query unavailable: {exc}")
        n_full = len(source_means) if source_means is not None else 0
        export_full = scene == "office4" and office4_full is None and not full.is_file() and n_full > 0
        if not full.is_file() and not export_full:
            prerequisites.append("missing exported original full FSM map; supply a verified same-checkpoint full map; no reconstruction/proxy")
        elif not n_full:
            prerequisites.append(f"cannot verify full map against original checkpoint: {checkpoint_error}")
        elif not export_full:
            try:
                full_asset = check_map(full, "full", 100, checkpoint, source_means, provenance["params_sha256"])
            except (OSError, ValueError, KeyError) as exc:
                prerequisites.append(f"invalid full map: {exc}")
        value["full_reference_path"] = str(full)
        value["prerequisites"] = prerequisites
        saved = {}
        for strategy, percent in matrix:
            name = arm_id(strategy, percent)
            ratio = percent / 100
            if strategy == "full":
                existing = None if export_full else full
            elif strategy == OBSERVABLE:
                existing = observable if percent == 20 else None
            else:
                existing = pareto / f"{strategy}_ratio_{str(ratio).replace('.', 'p')}" / "fsm_map.npz"
            dest = output / "run/maps" / scene / name
            arm = value["arms"][name] = dict(strategy=strategy, label=LABELS[strategy], percent=percent,
                map=str(existing) if existing else str(dest / "fsm_map.npz"),
                existing_map_present=bool(existing and existing.is_file()), status="blocked")
            if prerequisites:
                arm["blockers"] = prerequisites
                continue
            if arm["existing_map_present"]:
                try:
                    arm["asset"] = (full_asset if strategy == "full" else
                                    check_map(existing, strategy, percent, checkpoint, source_means, provenance["params_sha256"]))
                except (OSError, ValueError, KeyError) as exc:
                    arm["blockers"] = [f"invalid existing map: {exc}"]
                    continue
                dependencies.add(existing)
                arm["status"] = "reuse_map"
            else:
                count = (n_full * percent + 99) // 100
                if strategy == OBSERVABLE and (reference_error or min(count, n_full - count) < OBS_CONTROLS["replacement_count"]):
                    arm["blockers"] = [reference_error or "not enough points for the frozen 128 replacements"]
                    continue
                arm.update(status="build_map", build=dict(params=str(checkpoint), count=count))
                if strategy == OBSERVABLE:
                    arm["build"].update(OBS_CONTROLS, reserve_count=0)
                else:
                    arm["build"].update(seed=0, fps_candidate_cap=20000)
                    arm["map"] = str(dest / f"{strategy}_count_{arm['build']['count']}" / "fsm_map.npz")

            historical = scene in ("room2", "office0") and (strategy, percent) in (("full", 100), (OBSERVABLE, 20))
            if not historical:
                continue
            key = "full" if strategy == "full" else "compact"
            old_path = (RECOVERY if scene == "office0" and key == "full" else HISTORY) / scene / key
            try:
                if (not history or history["configs"][scene] != cfg or history["end_exclusive"] is not None
                        or history["thread_env"] != dict.fromkeys(THREADS, "1")
                        or history["map_paths"][scene][key] != arm["map"] or file_record(existing) != old_hashes[arm["map"]]
                        or any(file_record(Path(path)) != old_hashes[path] for path in value["query"]["metadata"])):
                    raise ValueError("frozen historical map/config/query does not match")
                saved[name] = score(old_path, arm, value)
            except (OSError, ValueError, KeyError) as exc:
                arm["historical_reuse_issue"] = str(exc)
            else:
                dependencies.update(old_path / filename for filename in RUN_FILES)
                arm.update(status="reuse_result", result_path=str(old_path), metrics=saved[name][0])

        full_id = arm_id("full", 100)
        for strategy, percent in matrix[1:]:
            candidate = arm_id(strategy, percent)
            if strategy == (FPS if scene == "fr3" else OBSERVABLE):
                value["comparisons"].append(dict(experiment="E1", reference=full_id, candidate=candidate))
            if strategy == VOXEL:
                value["comparisons"].append(dict(experiment="E2", reference=arm_id(FPS, percent), candidate=candidate))
        for pair in value["comparisons"]:
            a, b = pair["reference"], pair["candidate"]
            pair["status"] = "blocked" if any(value["arms"][name]["status"] == "blocked" for name in (a, b)) else "pending"
            if a in saved and b in saved:
                pair.update(status="reuse_result", metrics=paired(saved[a], saved[b]))
            if pair["experiment"] == "E2" and pair["status"] != "blocked":
                pair["matched_global_points"] = (n_full * value["arms"][b]["percent"] + 99) // 100
    return dict(schema_version=1, output_root=str(output), scenes=scenes,
                thread_env=dict.fromkeys(THREADS, "1"),
                metrics_protocol=dict(scope="full query, frame zero included in accuracy, excluded from timing",
                    joint_thresholds={"2cm_1deg": [0.02, 1], "5cm_2deg": [0.05, 2]},
                    joint_rule="inclusive thresholds; lost is failure; JTS uses the historical strict grid",
                    raw_ate_scope="own valid frames plus separate paired common-valid RMSE",
                    pairing="exact frame IDs, timestamps and GT bytes; pairwise valid intersection for error deltas",
                    ate_reference_line="+50% is descriptive only, not acceptance or a ranking threshold",
                    timing="historical and new timing not controlled repeats; descriptive only", missing="null, not zero")), dependencies


def safe_output(path):
    path = path.expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or path != path.resolve():
        raise ValueError("output path must not traverse symlinks or parent components")
    if path.is_relative_to(ROOT):
        if path.parent != MVP or not path.name.startswith("supplement_static_"):
            raise ValueError("workspace output must be a new supplement_static_* directory under the MVP root")
    elif not path.is_relative_to(Path("/tmp/kilo")) or path == Path("/tmp/kilo"):
        raise ValueError("external output must be a new directory under /tmp/kilo")
    return path


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def freeze(output, plan, dependencies):
    output = safe_output(output)
    if output.exists() or plan["output_root"] != str(output):
        raise ValueError("freeze requires its own new output directory; no overwrite")
    frozen = dict(plan, inputs=[file_record(path) for path in sorted(dependencies)])
    output.mkdir()
    write_new(output / "protocol_manifest.json", frozen)


def execute(output):
    from scripts.fsm_build_observable_map import build_observable_selection
    from scripts.fsm_export import export_fsm_maps
    from scripts.fsm_localize import run_localization

    output = safe_output(output)
    plan = read_json(output / "protocol_manifest.json")
    if plan["schema_version"] != 1 or plan["output_root"] != str(output):
        raise ValueError("not this output directory's frozen plan")
    if any(os.environ.get(key) != plan["thread_env"][key] for key in THREADS):
        raise ValueError("set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 before execution")

    def verify_inputs():
        for record in plan["inputs"]:
            if file_record(record["path"]) != record:
                raise ValueError(f"frozen dependency changed: {record['path']}")

    run = output / "run"
    if run.exists() or run.is_symlink():
        raise ValueError("run directory already exists; no overwrite or resume")
    verify_inputs()
    run.mkdir()
    for scene, value in plan["scenes"].items():
        cfg = value["config"]
        validate_protocol(cfg)
        if not any(arm["status"] in ("reuse_map", "build_map") for arm in value["arms"].values()):
            continue
        dataset, query, _ = query_assets(cfg)
        if query != value["query"]:
            raise ValueError(f"{scene}: query frame order changed after freeze")
        checkpoint = Path(value["checkpoint"]["path"])
        with np.load(checkpoint, allow_pickle=False) as data:
            source_means = data["means3D"]
        scored = {}
        for name, arm in value["arms"].items():
            status = arm["status"]
            if status == "blocked":
                continue
            path = Path(arm["map"])
            if status == "build_map":
                dest = run / "maps" / scene / name
                if not path.is_relative_to(dest) or path != path.resolve():
                    raise ValueError(f"invalid build output: {path}")
                dest.mkdir(parents=True)
                options = dict(arm["build"])
                params = options.pop("params")
                if arm["strategy"] == OBSERVABLE:
                    build_observable_selection(params, str(dest / "selection.npz"), map_output=str(path), **options)
                else:
                    count = options.pop("count")
                    export_fsm_maps(params, str(dest), [arm["strategy"]], counts=[count], **options)
                arm["asset"] = check_map(path, arm["strategy"], arm["percent"], checkpoint, source_means,
                                         value["checkpoint"]["expected_sha256"])
            result = Path(arm["result_path"]) if status == "reuse_result" else run / "localize" / scene / name
            if status != "reuse_result":
                result.mkdir(parents=True)
                run_localization(cfg, str(path), str(result), dataset=dataset, start=0, end=len(dataset))
                arm.update(status="completed", result_path=str(result))
            scored[name] = score(result, arm, value)
            arm["metrics"] = scored[name][0]
        for pair in value["comparisons"]:
            if pair["status"] == "pending":
                pair.update(status="completed", metrics=paired(scored[pair["reference"]], scored[pair["candidate"]]))
    verify_inputs()
    write_new(run / "summary.json", plan["scenes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MVP / "supplement_static_v1")
    parser.add_argument("--office4-full-map", type=Path, help="Explicit previously exported Office4 full FSM map")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true", help="Write a new manifest, do not build or localize")
    modes.add_argument("--execute-frozen", action="store_true", help="Run a previously frozen plan once")
    args = parser.parse_args(argv)
    try:
        output = safe_output(args.output_root)
        if args.execute_frozen:
            if args.office4_full_map:
                parser.error("Office4 input must be set before freezing")
            execute(output)
            return
        plan, dependencies = make_plan(output, args.office4_full_map)
        if args.freeze:
            freeze(output, plan, dependencies)
            print(output / "protocol_manifest.json")
        else:
            print(json.dumps(plan, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
