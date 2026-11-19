import argparse
import json
import csv
import os
import sys
from pathlib import Path

import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.fsm_score_paired_reserve_gt import _load_tum
from utils.fsm_horizon_evaluation import (
    JOINT_RULE, dimension_label, joint_label, load_pose_artifact, pose_errors,
    sha256_file, write_csv, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GT-only offline labeler for short-horizon branch poses.")
    parser.add_argument("--horizon-dir", required=True)
    parser.add_argument("--trajectory-gt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--translation-neutral-m", type=float, default=0.0)
    parser.add_argument("--rotation-neutral-deg", type=float, default=0.0)
    parser.add_argument("--joint-rule", default=JOINT_RULE, choices=[JOINT_RULE])
    return parser.parse_args(argv)


def label(args):
    horizon = Path(args.horizon_dir).resolve()
    poses_path = horizon / "horizon_branch_poses.npz"
    events_path = horizon / "horizon_events.csv"
    gt_path = Path(args.trajectory_gt).resolve()
    output = Path(args.output_dir).resolve()
    for path in (poses_path, events_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.translation_neutral_m < 0 or args.rotation_neutral_deg < 0:
        raise ValueError("neutral margins must be non-negative")
    values = load_pose_artifact(poses_path)
    gt_timestamps, gt_poses = _load_tum(gt_path)
    with open(events_path, newline="", encoding="utf-8") as stream:
        event_rows = list(csv.DictReader(stream))
    event_by_pair = {row["pair_id"]: row for row in event_rows}
    if len(event_by_pair) != len(event_rows):
        raise ValueError("horizon events must have unique pair IDs")
    grouped = {}
    for index, pair_id in enumerate(values["pair_id"].astype(str)):
        grouped.setdefault(pair_id, {}).setdefault(int(values["offset"][index]), {})[str(values["branch"][index])] = index
    rows = []
    extra_pose_pairs = set(grouped) - set(event_by_pair)
    if extra_pose_pairs:
        raise ValueError(f"branch poses contain unknown event pair IDs: {sorted(extra_pose_pairs)[:3]}")
    for pair_id in sorted(event_by_pair):
        translation_deltas = []
        rotation_deltas = []
        for offset in sorted(grouped.get(pair_id, {})):
            branches = grouped[pair_id][offset]
            if set(branches) != {"primary", "candidate"}:
                raise ValueError(f"{pair_id} offset {offset} must contain both branches")
            p_index, c_index = branches["primary"], branches["candidate"]
            if not values["accepted"][p_index] or not values["accepted"][c_index]:
                continue
            timestamp = float(values["timestamp"][p_index])
            matches = np.flatnonzero(np.isclose(gt_timestamps, timestamp, rtol=0.0, atol=5e-9))
            if matches.size != 1:
                raise ValueError(f"horizon timestamp {timestamp:.9f} has {matches.size} GT matches")
            truth = gt_poses[int(matches[0])]
            p_trans, p_rot = pose_errors(values["pose"][p_index], truth)
            c_trans, c_rot = pose_errors(values["pose"][c_index], truth)
            translation_deltas.append(c_trans - p_trans)
            rotation_deltas.append(c_rot - p_rot)
        if translation_deltas:
            translation_delta = float(np.median(translation_deltas))
            rotation_delta = float(np.median(rotation_deltas))
            translation_label = dimension_label(translation_delta, args.translation_neutral_m)
            rotation_label = dimension_label(rotation_delta, args.rotation_neutral_deg)
            combined = joint_label(translation_label, rotation_label, args.joint_rule)
        else:
            translation_delta = rotation_delta = ""
            translation_label = rotation_label = combined = "neutral"
        rows.append({
            "pair_id": pair_id, "common_accepted_steps": len(translation_deltas),
            "median_candidate_minus_primary_translation_error_m": translation_delta,
            "median_candidate_minus_primary_rotation_error_deg": rotation_delta,
            "translation_label": translation_label, "rotation_label": rotation_label,
            "joint_label": combined,
            "label_status": "gt_comparable" if translation_deltas else "no_joint_evidence",
        })
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "horizon_gt_labels.csv"
    fields = list(rows[0]) if rows else [
        "pair_id", "common_accepted_steps", "median_candidate_minus_primary_translation_error_m",
        "median_candidate_minus_primary_rotation_error_deg", "translation_label", "rotation_label", "joint_label", "label_status",
    ]
    write_csv(labels_path, rows, fields)
    provenance = {
        "schema_version": 1, "mode": "offline_gt_only", "uses_ground_truth": True,
        "joint_rule": args.joint_rule, "translation_neutral_m": args.translation_neutral_m,
        "rotation_neutral_deg": args.rotation_neutral_deg, "row_count": len(rows),
        "horizon_events_sha256": sha256_file(events_path),
        "horizon_branch_poses_sha256": sha256_file(poses_path),
        "trajectory_gt_sha256": sha256_file(gt_path), "labels_sha256": sha256_file(labels_path),
    }
    write_json(output / "horizon_gt_labels_provenance.json", provenance)
    return provenance


def main(argv=None):
    print(json.dumps(label(parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
