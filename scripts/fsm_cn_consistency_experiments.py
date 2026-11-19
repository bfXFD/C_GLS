"""Plan, freeze, execute serial CN experiments, or collect existing results."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs/fsm/mvp_observable/cn_consistency_case.py"
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
ARMS = (
    "baseline", "rerun_frozen", "rerun_pca", "corr_frozen", "corr_pca",
    "saliency_pca", *(f"random_pca_s{seed}" for seed in range(5)),
)
STAGES = {
    "dev": (("replica", "room2"), ("replica", "office0")),
    "regression": tuple(("replica", f"office{i}") for i in range(1, 5))
    + (("tum", "freiburg3_long_office_household"),),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="dev")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--conda-env", default="splatam")
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = MVP / f"cn_consistency_{args.stage}_v1"
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        parser.error("require 0 <= start < end (or omit end for full horizon)")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms are not allowed")
    if not args.conda_env or args.conda_env.startswith("-"):
        parser.error("invalid conda environment")
    return args


def safe_output(path, *, collect=False):
    original = path.expanduser().absolute()
    if any(part.is_symlink() for part in (original, *original.parents)):
        raise ValueError("output path must not traverse symlinks")
    root = original.resolve()
    historical = ROOT / "experiments/FSM"
    allowed_study = root.parent == MVP and root.name.startswith("cn_consistency_")
    if root == ROOT or root in ROOT.parents or (root.is_relative_to(historical) and not allowed_study):
        raise ValueError("refusing workspace ancestor or historical experiment root")
    if any(part in {"audit_v2_full_rerun", "configs", "scripts", "tests", "utils", "data"} for part in root.parts):
        raise ValueError("refusing protected output path")
    if root.exists() and not root.is_dir():
        raise ValueError("output root must be a directory")
    if not collect and root.exists() and any(root.iterdir()):
        raise ValueError("output root is nonempty; no overwrite or skip-existing is allowed")
    return root


def input_paths(dataset, scene):
    base = MVP
    if dataset == "replica":
        suite = "replica_office1_4_untouched_v1" if scene in {f"office{i}" for i in range(1, 5)} else "runtime_reserve_v1"
        base = base / suite / "maps" / f"replica_{scene}_0" / "logdet"
    else:
        base = base / "tum_fps_hybrid_v1/maps" / scene
    return base / "fsm_map.npz", base / "activation_geometry.npz"


def resolved_config(dataset, scene, arm):
    env = {"CGLS_CN_DATASET": dataset, "CGLS_CN_SCENE": scene, "CGLS_CN_ARM": arm}
    with patch.dict(os.environ, env):
        return runpy.run_path(str(CASE))["config"]


def build_plan(args):
    output = safe_output(args.output_root)
    runs = []
    for dataset, scene in STAGES[args.stage]:
        map_path, sidecar = input_paths(dataset, scene)
        for arm in args.arms:
            name = f"{dataset}_{scene}_{arm}"
            destination = output / "runs" / name
            env = {"CGLS_CN_DATASET": dataset, "CGLS_CN_SCENE": scene, "CGLS_CN_ARM": arm}
            command = ["env", *(f"{key}={value}" for key, value in env.items()),
                       "conda", "run", "-n", args.conda_env, "python",
                       str(ROOT / "scripts/fsm_eval_long_horizon.py"),
                       "--output-root", str(destination), "--start", str(args.start)]
            if args.end is not None:
                command += ["--end", str(args.end)]
            command += ["--case", f"{name},{CASE},{map_path},{sidecar}"]
            if any("," in str(path) or "\n" in str(path) for path in (CASE, map_path, sidecar, destination)):
                raise ValueError("case and output paths cannot contain commas or newlines")
            runs.append(dict(
                name=name, dataset=dataset, scene=scene, arm=arm, output=str(destination),
                map=str(map_path), sidecar=str(sidecar), command=command,
                command_shell=shlex.join(command), config=resolved_config(dataset, scene, arm),
            ))
    partial = args.start != 0 or args.end is not None
    subset = set(args.arms) != set(ARMS)
    return dict(
        schema_version=1, stage=args.stage, output_root=str(output),
        scene_exposure="development_exposed" if args.stage == "dev" else "historically_exposed_regression_not_untouched",
        start=args.start, end=args.end, partial=partial, arm_subset=subset,
        run_count=len(runs), serial_execution=True,
        interpretation="descriptive_research_only_no_safety_guarantee_no_composite_conclusions_from_subsets",
        evaluation=dict(
            primary=["lost", "valid", "jts_percent"],
            jts_protocol="100 * mean joint success over k=1..100, strict raw translation < .005*k m and rotation < .1*k deg; all frames including first; invalid/nonfinite fail; exact frame/timestamp/validity matching required",
            secondary=["verifier_decision_counts", "latency", "raw_ate_rmse_m", "raw_rotation_abs_rmse_deg"],
            missing_values="null_not_zero", threshold_status="global_research_defaults_not_calibrated",
            contrasts=["rerun_frozen-baseline: extra optimization", "rerun_pca-rerun_frozen: primary normal rebuild",
                       "corr_frozen-rerun_frozen: appended support with frozen normals",
                       "corr_pca-corr_frozen: augmented normal rebuild",
                       "corr_pca vs saliency_pca and all five random_pca seeds"],
            no_replace=True, no_inferential_or_safety_claims=True,
        ),
        runs=runs,
    )


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def preflight(run, plan):
    """Exercise the runner loader and actual verifier helpers without localization."""
    code = '''import json, sys
import numpy as np
from scripts.fsm_eval_long_horizon import validate_case
from scripts.fsm_localize import validate_verifier_config, split_source_validation
run, start, end = json.loads(sys.argv[1])
case = dict(name=run['name'], config=sys.argv[2], map=run['map'], activation_geometry=run['sidecar'])
actual = validate_case(case, start, end)['config_object']
expected = run['config']
for section in ('activation_diagnostics', 'observable_reserve', 'paired_reserve_evaluation'):
    cfg = expected.get(section, {})
    if cfg.get('enabled') and cfg.get('geometry_path') in (None, 'auto'):
        cfg['geometry_path'] = run['sidecar']
if actual != expected:
    raise ValueError('runner resolved config differs from frozen plan')
registration = actual['registration']
if registration['backend'] != 'numpy_robust_point_to_plane' or registration['evaluation_protocol'] != 'mvp_strict':
    raise ValueError('verifier requires strict robust point-to-plane')
cfg = actual['observable_reserve']['candidate_verifier']
if cfg.get('enabled') is not True:
    raise ValueError('all arms including baseline require enabled verifier')
options = {k: v for k, v in cfg.items() if k not in ('enabled', 'block_size', 'folds', 'fold', 'max_translation_m')}
if 'max_translation_m' in cfg:
    options['max_translation'] = cfg['max_translation_m']
validate_verifier_config(options)
split_source_validation(np.empty(0, dtype=np.int64), 1, block_size=cfg['block_size'], folds=cfg['folds'], fold=cfg['fold'])
'''
    command = run["command"][:run["command"].index("python") + 1]
    command += ["-c", code, json.dumps([run, plan["start"], plan["end"]]), str(CASE)]
    subprocess.run(command, cwd=ROOT, check=True)


def verify_frozen(manifest):
    for record in manifest["source_hashes"] + manifest["input_hashes"]:
        if file_record(record["path"]) != record:
            raise ValueError(f"frozen source/input changed: {record['path']}")


def freeze_and_execute(plan):
    root = safe_output(Path(plan["output_root"]))
    sources = {
        CASE, Path(__file__).resolve(), CASE.with_name("common.py"),
        ROOT / "scripts/fsm_localize.py", ROOT / "scripts/fsm_eval_long_horizon.py",
        ROOT / "scripts/splatam.py",
        *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py"),
        *ROOT.glob("configs/data/**/*.yaml"),
    }
    inputs = {Path(run[key]) for run in plan["runs"] for key in ("map", "sidecar")}
    # Hash every dependency before creating outputs or starting any localization.
    manifest = dict(plan, frozen_before_any_run=True,
                    source_hashes=[file_record(path) for path in sorted(sources)],
                    input_hashes=[file_record(path) for path in sorted(inputs)])
    for run in plan["runs"]:
        preflight(run, plan)
    verify_frozen(manifest)
    root.mkdir(parents=True, exist_ok=True)
    safe_output(root)
    with (root / "protocol_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for run in plan["runs"]:
        verify_frozen(manifest)
        preflight(run, plan)
        verify_frozen(manifest)
        subprocess.run(run["command"], cwd=ROOT, check=True)
        verify_frozen(manifest)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def matched_jts(local, frames, summary):
    path = local / "per_frame_errors.csv"
    if not path.is_file() or not frames:
        return None, "missing_frame_or_raw_errors", None
    with path.open(newline="", encoding="utf-8") as stream:
        errors = list(csv.DictReader(stream))
    required = {"frame_id", "timestamp", "tracking_valid"}
    if not errors or any(not required.issubset(row) for row in frames + errors):
        return None, "missing_frame_identity_or_validity", None
    if any(not {"raw_translation_error_m", "raw_rotation_error_deg"}.issubset(row) for row in errors):
        return None, "missing_raw_error_columns", None
    if len(errors) != len(frames):
        raise ValueError(f"{local}: raw/frame row count mismatch")
    if summary.get("num_frames") is not None and summary["num_frames"] != len(frames):
        raise ValueError(f"{local}: summary/frame row count mismatch")
    identities, successes = [], 0
    for frame, error in zip(frames, errors):
        try:
            identity = (int(frame["frame_id"]), float(frame["timestamp"]))
            raw_identity = (int(error["frame_id"]), float(error["timestamp"]))
        except (TypeError, ValueError):
            raise ValueError(f"{local}: invalid frame identity") from None
        if not math.isfinite(identity[1]) or identity != raw_identity:
            raise ValueError(f"{local}: raw/frame ID or timestamp mismatch")
        identities.append(identity)
        valid = frame["tracking_valid"].lower()
        if valid not in ("true", "false") or valid != error["tracking_valid"].lower():
            raise ValueError(f"{local}: raw/frame validity mismatch")
        if valid == "false":
            continue
        try:
            translation = float(error["raw_translation_error_m"] or "nan")
            rotation = float(error["raw_rotation_error_deg"] or "nan")
        except (TypeError, ValueError):
            return None, "unparseable_raw_errors", None
        if math.isfinite(translation) and math.isfinite(rotation):
            if translation < 0 or rotation < 0:
                raise ValueError(f"{local}: negative raw errors")
            successes += sum(translation < .005 * k and rotation < .1 * k for k in range(1, 101))
    if len({item[0] for item in identities}) != len(identities) or any(
        a[0] >= b[0] or a[1] >= b[1] for a, b in zip(identities, identities[1:])
    ):
        raise ValueError(f"{local}: duplicate or unordered frame identities")
    return successes / len(frames), "matched_all_frames_first_included", identities


def collect(output):
    root = safe_output(output, collect=True)
    manifest = read_json(root / "protocol_manifest.json")
    if not manifest or not manifest.get("frozen_before_any_run"):
        raise ValueError("collection requires a frozen protocol_manifest.json")
    rows, scene_identities = [], {}
    for run in manifest["runs"]:
        directory = root / "runs" / run["name"]
        local = directory / "localize" / run["name"]
        summary = read_json(local / "summary.json")
        done = directory / ".state" / f"{run['name']}.done.json"
        frame_path = local / "frame_metrics.csv"
        frames, fields = [], []
        if frame_path.is_file():
            with frame_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                frames = list(reader)
        decisions, latencies = {}, {}
        for field in fields:
            values = [frame[field] for frame in frames if frame.get(field) not in (None, "")]
            if "verifier" in field and any(token in field for token in ("decision", "reason", "accepted", "passed")):
                decisions[field] = dict(counts=dict(Counter(values)), observed=len(values), missing=len(frames) - len(values))
            if "latency" in field:
                numbers = []
                for value in values:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        numbers.append(number)
                latencies[field] = dict(mean_ms=sum(numbers) / len(numbers) if numbers else None,
                                        observed=len(numbers), missing=len(frames) - len(numbers))
        metrics = summary or {}
        jts, jts_status, identities = matched_jts(local, frames, metrics)
        scene_key = (run["dataset"], run["scene"])
        if identities is not None:
            if scene_key in scene_identities and scene_identities[scene_key] != identities:
                raise ValueError(f"{scene_key}: arm frame/timestamp mismatch")
            scene_identities[scene_key] = identities
        horizon = read_json(local / "long_horizon_metrics.json") or {}
        raw_absolute = horizon.get("raw_absolute", {})
        rows.append(dict(
            case=run["name"], dataset=run["dataset"], scene=run["scene"], arm=run["arm"],
            status="complete" if summary is not None and done.is_file() and frame_path.is_file() else "incomplete",
            partial=manifest["partial"], arm_subset=manifest["arm_subset"],
            frames=metrics.get("num_frames"), valid=metrics.get("num_valid_frames"), lost=metrics.get("num_lost_frames"),
            jts={"jts_percent": jts} if jts is not None else None, jts_percent=jts, jts_status=jts_status,
            raw_ate_rmse_m=metrics.get("raw_ate_rmse_m", raw_absolute.get("translation_rmse_m")),
            raw_rotation_abs_rmse_deg=metrics.get("raw_rotation_abs_rmse_deg", raw_absolute.get("rotation_rmse_deg")),
            latency_mean_ms=metrics.get("production_equivalent_total_frame_latency_mean_ms"),
            verifier_decision_counts=decisions or None, frame_latency=latencies or None,
            frame_rows=len(frames) if frame_path.is_file() else None,
        ))
    result = dict(stage=manifest["stage"], partial=manifest["partial"], arm_subset=manifest["arm_subset"],
                  interpretation=manifest["interpretation"], scene_exposure=manifest["scene_exposure"], runs=rows)
    destination = root / "collection"
    destination.mkdir(exist_ok=True)
    (destination / "runs.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (destination / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()})
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.collect:
            result = collect(args.output_root)
        else:
            result = build_plan(args)
            if args.execute:
                freeze_and_execute(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
