"""Paired diagnostic timing with unchanged voxel-block tracking and recovery."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_bench_closed_loop_queries import arm_config, run_metrics
from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import load_dataset_from_config


def distribution(values):
    return dict(count=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95))) if values else None


def summarize(rows):
    result = {}
    keys = ("query_ms", "gather_ms", "tree_ms", "normal_prep_ms", "registration_ms")
    for name in ("normal", "recovery"):
        group = [r for r in rows[1:] if r["recovery_attempted"] == (name == "recovery")]
        if not group:
            continue
        per_frame = []
        for row in group:
            entry = {key: row[key] for key in ("frame_load_ms", "pointcloud_ms", "total_frame_ms")}
            entry.update({key: sum(a.get(key, 0) for a in row["attempts"]) for key in keys})
            entry["duplicate_query_ms"] = sum(a.get("query_ms", 0) for a in row["attempts"]
                                               if a["same_query_as_earlier_attempt"])
            entry["other_ms"] = entry["total_frame_ms"] - sum(entry[k] for k in (*keys, "frame_load_ms", "pointcloud_ms"))
            per_frame.append(entry)
        attempts = [a for row in group for a in row["attempts"]]
        by_attempt = {}
        for tier, backend in sorted({(a["tier"], a["backend"]) for a in attempts}):
            selected = [a for a in attempts if (a["tier"], a["backend"]) == (tier, backend)]
            by_attempt[f"{tier}:{backend}"] = dict(
                count=len(selected), duplicate_queries=sum(a["same_query_as_earlier_attempt"] for a in selected),
                tree_reused=sum(a.get("tree_reused", False) for a in selected),
                normals_reused=sum(a.get("normals_reused", False) for a in selected),
                **{k: distribution([a[k] for a in selected if k in a]) for k in keys})
        result[name] = dict(stages={k: distribution([r[k] for r in per_frame]) for k in per_frame[0]},
                            by_attempt=by_attempt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use complete sequences instead of diagnostic windows")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = safe_output(args.output_root)
    windows = {"room2": (0, 80), "office0": (1700, 1850)}
    if args.full:
        windows = {scene: (0, None) for scene in windows}
    old = MVP / "cn_consistency_query_closed_loop_full_v1"
    plan = {scene: dict(start=start, end=end, config=arm_config(scene, "voxel_blocks"),
                        map=str(input_paths("replica", scene)[0]),
                        restart=str(old / f"{scene}_r1/voxel_blocks/trajectory_est.txt") if start else None)
            for scene, (start, end) in windows.items()}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    paths = {Path(__file__).resolve(), ROOT / "scripts/fsm_bench_closed_loop_queries.py", CASE,
             CASE.with_name("common.py"), ROOT / "scripts/fsm_localize.py", ROOT / "scripts/splatam.py",
             ROOT / "scripts/fsm_cn_consistency_experiments.py", *ROOT.glob("utils/fsm_*.py"),
             *ROOT.glob("datasets/gradslam_datasets/*.py")}
    for case in plan.values():
        paths.add(Path(case["map"]))
        if case["restart"]:
            paths.add(Path(case["restart"]))
        data = case["config"]["data"]
        paths.update((Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    hashes = [file_record(p) for p in sorted(paths)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(cases=plan, source_and_input_hashes=hashes, frame_zero_excluded=True,
                       thread_env={k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
                  stream, indent=2, allow_nan=False)
    results = {}
    for scene, case in plan.items():
        os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
        dataset = load_dataset_from_config(str(CASE))
        options = {}
        if case["restart"]:
            row = np.loadtxt(case["restart"])[case["start"]]
            if row[0] != case["start"]:
                raise ValueError("restart pose does not match window index")
            initial = np.eye(4)
            initial[:3, :3] = Rotation.from_quat(row[4:]).as_matrix()
            initial[:3, 3] = row[1:4]
            options = dict(first_pose_protocol_override="explicit", explicit_initial_pose=initial)
        profile = []
        metrics = {}
        for variant in (("baseline", "profile") if scene == "room2" else ("profile", "baseline")):
            destination = output / scene / variant
            print(f"{scene} {variant}: {case['start']} to {case['end']}", flush=True)
            run = run_localization(case["config"], case["map"], str(destination), dataset=dataset,
                                   start=case["start"], end=case["end"], **options,
                                   stage_profile=profile.append if variant == "profile" else None)
            metrics[variant] = run_metrics(destination, run, 100.0)
        ref, ref_ids, ref_state = metrics["baseline"]
        measured, ids, state = metrics["profile"]
        if (ids != ref_ids or ids != [(r["frame_id"], t) for r, (_, t) in zip(profile, ids)]
                or len(profile) != len(ids)):
            raise ValueError("profile and baseline frames differ")
        trajectory_equal = ((output / scene / "baseline/trajectory_est.txt").read_bytes()
                            == (output / scene / "profile/trajectory_est.txt").read_bytes())
        with (output / scene / "stages.jsonl").open("x") as stream:
            for row in profile:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        results[scene] = dict(baseline=ref, profile=measured, trajectory_equal=trajectory_equal,
                              tracking_state_equal=state == ref_state, **summarize(profile))
        print(f"{scene}: equal trajectory={trajectory_equal}, equal state={state == ref_state}, "
              f"recovery frames={sum(r['recovery_attempted'] for r in profile)}", flush=True)
        del dataset
    if hashes != [file_record(Path(r["path"])) for r in hashes]:
        raise RuntimeError("Inputs changed during experiment")
    with (output / "summary.json").open("x") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
