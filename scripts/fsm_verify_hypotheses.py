import argparse
import csv
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_dataset import get_frame, load_dataset_from_config
from utils.fsm_horizon_verification import (
    DEFAULT_BUDGET, DEFAULT_HORIZON, DEFAULT_POLICY, SCHEMA_VERSION,
    StrictReplayTracker, VerificationConfig, assert_separate_output,
    load_paired_hypotheses, observation_schema, reconstruct_branch_state,
    save_results, sha256_file, verify_event,
)


def _load_tum_estimate(path):
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 8 or np.any(~np.isfinite(values)):
        raise ValueError(f"{path} must contain finite TUM rows")
    quaternions = values[:, 4:8]
    lengths = np.linalg.norm(quaternions, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError(f"{path} contains zero quaternions")
    x, y, z, w = (quaternions / lengths[:, None]).T
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], values.shape[0], axis=0)
    poses[:, :3, 3] = values[:, 1:4]
    poses[:, :3, :3] = np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)
    return values[:, 0], poses


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline GT-free short-horizon paired-hypothesis verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--localize-dir", required=True)
    parser.add_argument("--paired-npz", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, choices=[DEFAULT_BUDGET])
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--min-common-valid-frames", type=int, default=2)
    parser.add_argument("--max-cost-ratio", type=float, default=0.90)
    parser.add_argument("--min-fitness-retention", type=float, default=0.95)
    parser.add_argument("--min-corr-retention", type=float, default=0.95)
    parser.add_argument("--common-target-max-points", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args(argv)


def run(args):
    config_path = Path(args.config).resolve()
    map_path = Path(args.map).resolve()
    localize_dir = Path(args.localize_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    paired_path = Path(args.paired_npz).resolve() if args.paired_npz else localize_dir / "paired_reserve/paired_reserve_ids.npz"
    trajectory_path = localize_dir / "trajectory_est.txt"
    metrics_path = localize_dir / "frame_metrics.csv"
    for path in (config_path, map_path, paired_path, trajectory_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    assert_separate_output(output_dir, localize_dir)
    parent_hash_before = sha256_file(trajectory_path)
    events, artifact_parent_hash = load_paired_hypotheses(
        paired_path, policy=args.policy, budget=args.budget, seed=None,
    )
    if artifact_parent_hash != parent_hash_before:
        raise ValueError("paired artifact parent trajectory hash mismatch")
    if args.max_events is not None:
        if args.max_events < 0:
            raise ValueError("max-events must be non-negative")
        events = events[:args.max_events]
    module = SourceFileLoader(config_path.stem, str(config_path)).load_module()
    config = module.config
    dataset = load_dataset_from_config(str(config_path))
    timestamps, trajectory = _load_tum_estimate(trajectory_path)
    with open(metrics_path, newline="", encoding="utf-8") as stream:
        frame_rows = list(csv.DictReader(stream))
    if timestamps.shape[0] != len(frame_rows):
        raise ValueError("trajectory and frame_metrics row counts differ")
    frame_to_output = {int(row["frame_id"]): index for index, row in enumerate(frame_rows)}
    tracker = StrictReplayTracker(config, map_path)
    decision = VerificationConfig(
        horizon_frames=args.horizon, min_common_valid_frames=args.min_common_valid_frames,
        max_cost_ratio=args.max_cost_ratio, min_fitness_retention=args.min_fitness_retention,
        min_corr_retention=args.min_corr_retention, common_target_max_points=args.common_target_max_points,
    )
    event_rows = []
    step_rows = []
    pose_rows = []
    for event in events:
        if event["frame_id"] not in frame_to_output:
            raise ValueError(f"paired frame {event['frame_id']} is absent from frame_metrics")
        event_output = frame_to_output[event["frame_id"]]
        primary = reconstruct_branch_state(trajectory, frame_rows, event["frame_id"], event["primary_pose"], "primary")
        candidate = reconstruct_branch_state(trajectory, frame_rows, event["frame_id"], event["candidate_pose"], "candidate")
        observations = []
        for output_idx in range(event_output + 1, min(event_output + 1 + args.horizon, len(frame_rows))):
            dataset_index = int(frame_rows[output_idx]["frame_id"])
            frame = get_frame(dataset, dataset_index)
            frame["timestamp"] = float(frame_rows[output_idx]["timestamp"])
            observations.append(tracker.make_observation(frame, output_idx))
        summary, steps = verify_event(
            event, observations, primary, candidate, tracker.track, tracker.common_metrics, decision,
            pose_rows,
        )
        event_rows.append(summary)
        step_rows.extend(steps)
    parent_hash_after = sha256_file(trajectory_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("parent trajectory changed during horizon verification")
    provenance = {
        "schema_version": SCHEMA_VERSION, "mode": "offline_shadow", "uses_ground_truth": False,
        "candidate_feedback": False, "policy": args.policy, "budget": args.budget,
        "horizon_frames": args.horizon, "decision": {
            "primary_default": True, "min_common_valid_frames": args.min_common_valid_frames,
            "max_cost_ratio": args.max_cost_ratio, "min_fitness_retention": args.min_fitness_retention,
            "min_corr_retention": args.min_corr_retention,
        },
        "observation_fields": observation_schema(), "parent_trajectory_sha256": parent_hash_before,
        "parent_trajectory_sha256_after": parent_hash_after,
        "paired_artifact_sha256": sha256_file(paired_path), "map_sha256": sha256_file(map_path),
        "config_path": str(config_path), "map_path": str(map_path), "paired_path": str(paired_path),
        "localize_dir": str(localize_dir), "event_count": len(event_rows),
    }
    save_results(output_dir, event_rows, step_rows, provenance, pose_rows)
    return {"event_count": len(event_rows), "step_count": len(step_rows), "output_dir": str(output_dir)}


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
