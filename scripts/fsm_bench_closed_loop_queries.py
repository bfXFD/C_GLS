"""Compare radius and uncapped voxel-block culling in closed-loop localization."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsm_cn_consistency_experiments import CASE, MVP, file_record, input_paths, matched_jts, resolved_config, safe_output
from scripts.fsm_localize import run_localization
from utils.fsm_dataset import load_dataset_from_config


SCENES = ("room2", "office0")
ARMS = ("radius", "voxel_blocks")
TIMING_COLUMNS = ("total_frame_latency_ms", "local_query_latency_ms", "latency_ms")


def arm_config(scene, arm):
    cfg = resolved_config("replica", scene, "baseline")
    cfg["observable_reserve"]["candidate_verifier"] = {"enabled": False}
    cfg["local_submap"].update(frustum_culling=arm, projection_pre_cap=None, oracle_mode="off")
    return cfg


def run_metrics(output, summary, deadline):
    with (output / "frame_metrics.csv").open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    jts, reason, identities = matched_jts(output, frames, summary)
    if not identities:
        raise ValueError(f"{output}: cannot match trajectory output: {reason}")
    timing = {}
    for field in TIMING_COLUMNS:
        # Frame zero is initialized, not tracked. Lost and recovered frames stay in the sample.
        values = np.array([float(row[field]) for row in frames[1:]])
        if not len(values) or not np.all(np.isfinite(values) & (values >= 0)):
            raise ValueError(f"{output}: invalid timing: {field}")
        timing[field] = dict(p50_ms=float(np.percentile(values, 50)),
                             p95_ms=float(np.percentile(values, 95)), max_ms=float(values.max()),
                             over_deadline=int(np.count_nonzero(values > deadline)))
    state = [(r["tracking_valid"], r["tracking_status"], r["recovered"], r["recovery_stage"])
             for r in frames]
    lost = sum(r[0] == "False" for r in state)
    if lost != summary["num_lost_frames"]:
        raise ValueError(f"{output}: summary lost count differs from frame data")
    metrics = dict(frames=len(frames), timed_frames=len(frames)-1, lost=lost,
                   recovered=summary["num_recovered_frames"], jts_percent=jts,
                   raw_ate_rmse_m=summary.get("raw_ate_rmse_m"),
                   local_index_build_ms=summary.get("local_index_build_ms"),
                   block_index_build_ms=summary.get("block_index_build_ms"),
                   block_index_memory_bytes=summary.get("block_index_memory_bytes"), timing=timing)
    return metrics, identities, state


def compare(output, results, deadline):
    result = {}
    for arm in ARMS:
        result[arm], ids, state = run_metrics(output / arm, results[arm], deadline)
        if arm == "radius":
            ref_ids, ref_state = ids, state
        elif ids != ref_ids:
            raise ValueError(f"{output}: arms use different frames")
    result["tracking_state_equal"] = state == ref_state
    result["trajectory_bytes_equal"] = ((output / "radius/trajectory_est.txt").read_bytes()
                                         == (output / "voxel_blocks/trajectory_est.txt").read_bytes())
    if ((output / "radius/trajectory_gt.txt").read_bytes()
            != (output / "voxel_blocks/trajectory_gt.txt").read_bytes()):
        raise ValueError(f"{output}: ground truth differs between arms")
    p95 = [result[arm]["timing"]["total_frame_latency_ms"]["p95_ms"] for arm in ARMS]
    result["whole_frame_p95_speedup"] = p95[0] / p95[1] if p95[1] > 0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--end", type=int, help="Exclusive end frame; omit for full sequences")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (args.end is not None and args.end < 2 or args.repeats < 1
            or not np.isfinite(args.deadline_ms) or args.deadline_ms <= 0):
        parser.error("require end >= 2, repeats >= 1, and a positive finite deadline")
    root = safe_output(args.output_root)
    configs = {s: {a: arm_config(s, a) for a in ARMS} for s in SCENES}
    maps = {s: input_paths("replica", s)[0] for s in SCENES}
    plan = dict(scenes=SCENES, map_paths={s: str(p) for s,p in maps.items()}, configs=configs,
                end_exclusive=args.end, repeats=args.repeats, deadline_ms=args.deadline_ms,
                initialized_frame_in_jts=True, initialized_frame_in_latency=False,
                thread_env={k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    inputs = set(maps.values())
    for scene in SCENES:
        data = configs[scene]["radius"]["data"]
        inputs.update((Path(data["gradslam_data_cfg"]), Path(data["basedir"]) / data["sequence"] / "traj.txt"))
    sources = {Path(__file__).resolve(), CASE, CASE.with_name("common.py"), ROOT / "scripts/fsm_localize.py",
               ROOT / "scripts/splatam.py", ROOT / "scripts/fsm_cn_consistency_experiments.py",
               *ROOT.glob("utils/fsm_*.py"), *ROOT.glob("datasets/gradslam_datasets/*.py")}
    frozen = [file_record(p) for p in sorted(inputs | sources)]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol.json").open("x") as stream:
        json.dump(dict(plan, source_and_input_hashes=frozen), stream, indent=2, allow_nan=False)
    all_results = []
    for scene_index, scene in enumerate(SCENES):
        os.environ.update(CGLS_CN_DATASET="replica", CGLS_CN_SCENE=scene, CGLS_CN_ARM="baseline")
        dataset = load_dataset_from_config(str(CASE))
        if args.end is not None and args.end > len(dataset):
            raise ValueError(f"{scene}: end exceeds the dataset length")
        for repeat in range(args.repeats):
            pair = root / f"{scene}_r{repeat+1}"
            results = {}
            order = ARMS if (scene_index + repeat) % 2 == 0 else ARMS[::-1]
            for arm in order:
                print(f"{scene} repeat {repeat+1}: {arm}", flush=True)
                results[arm] = run_localization(configs[scene][arm], str(maps[scene]), str(pair / arm),
                                                dataset=dataset, start=0, end=args.end)
            compared = dict(scene=scene, repeat=repeat+1, order=order,
                            **compare(pair, results, args.deadline_ms))
            all_results.append(compared)
            print(json.dumps(compared), flush=True)
        del dataset
    if [file_record(Path(r["path"])) for r in frozen] != frozen:
        raise RuntimeError("A recorded input or source changed during the experiment")
    with (root / "comparison.json").open("x") as stream:
        json.dump(all_results, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
