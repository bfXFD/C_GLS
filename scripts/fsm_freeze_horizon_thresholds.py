import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_horizon_evaluation import (
    GRID, HORIZONS, evaluate_rows, freeze_score, read_csv, sha256_file,
    threshold_grid, validate_label_hashes, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Freeze short-horizon thresholds from development outputs.")
    parser.add_argument("--development", action="append", required=True, help="HORIZON:HORIZON_DIR:LABEL_DIR")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def freeze(args):
    inputs = {}
    selected = {}
    for value in args.development:
        horizon_text, horizon_dir_text, label_dir_text = value.split(":", 2)
        horizon = int(horizon_text)
        if horizon in inputs:
            raise ValueError(f"duplicate development horizon: {horizon}")
        horizon_dir, label_dir = Path(horizon_dir_text).resolve(), Path(label_dir_text).resolve()
        events_path = horizon_dir / "horizon_events.csv"
        poses_path = horizon_dir / "horizon_branch_poses.npz"
        labels_path = label_dir / "horizon_gt_labels.csv"
        provenance_path = label_dir / "horizon_gt_labels_provenance.json"
        with open(provenance_path, encoding="utf-8") as stream:
            provenance = json.load(stream)
        validate_label_hashes(provenance, events_path, poses_path)
        if provenance.get("labels_sha256") != sha256_file(labels_path):
            raise ValueError("label provenance labels hash mismatch")
        events, labels = read_csv(events_path), read_csv(labels_path)
        candidates = []
        for thresholds in threshold_grid():
            metrics = evaluate_rows(events, labels, thresholds)
            candidates.append((freeze_score(metrics, thresholds), thresholds, metrics))
        _, thresholds, metrics = max(candidates, key=lambda item: item[0])
        selected[str(horizon)] = {"thresholds": thresholds, "development_metrics": metrics}
        inputs[str(horizon)] = {
            "horizon_events_sha256": sha256_file(events_path),
            "horizon_branch_poses_sha256": sha256_file(poses_path),
            "horizon_gt_labels_sha256": sha256_file(labels_path),
            "horizon_gt_labels_provenance_sha256": sha256_file(provenance_path),
        }
    if set(inputs) != {str(value) for value in HORIZONS}:
        raise ValueError(f"development inputs must contain exactly horizons {HORIZONS}")
    document = {
        "schema_version": 1, "immutable": True,
        "selection_priority": ["harmful_rejection_recall", "beneficial_retention", "primary_default"],
        "primary_default_tie_break": "lowest max_cost_ratio, then highest fitness and correspondence retention",
        "grid": {name: list(values) for name, values in GRID.items()},
        "inputs": inputs, "horizons": selected,
    }
    write_json(Path(args.output).resolve(), document, exclusive=True)
    return document


def main(argv=None):
    args = parse_args(argv)
    result = freeze(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "horizons": sorted(result["horizons"])}, sort_keys=True))


if __name__ == "__main__":
    main()
