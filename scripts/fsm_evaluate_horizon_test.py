import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_horizon_evaluation import evaluate_rows, read_csv, sha256_file, validate_label_hashes, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Apply frozen short-horizon thresholds to held-out test outputs.")
    parser.add_argument("--frozen-json", required=True)
    parser.add_argument("--test", action="append", required=True, help="HORIZON:HORIZON_DIR:LABEL_DIR")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def evaluate(args):
    frozen_path = Path(args.frozen_json).resolve()
    with open(frozen_path, encoding="utf-8") as stream:
        frozen = json.load(stream)
    if frozen.get("immutable") is not True:
        raise ValueError("threshold JSON is not marked immutable")
    results = {}
    input_hashes = {}
    for value in args.test:
        horizon_text, horizon_dir_text, label_dir_text = value.split(":", 2)
        horizon = str(int(horizon_text))
        if horizon not in frozen.get("horizons", {}) or horizon in results:
            raise ValueError(f"unexpected or duplicate test horizon: {horizon}")
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
        thresholds = frozen["horizons"][horizon]["thresholds"]
        results[horizon] = {"thresholds": thresholds, **evaluate_rows(read_csv(events_path), read_csv(labels_path), thresholds)}
        input_hashes[horizon] = {
            "horizon_events_sha256": sha256_file(events_path),
            "horizon_branch_poses_sha256": sha256_file(poses_path),
            "horizon_gt_labels_sha256": sha256_file(labels_path),
        }
    document = {
        "schema_version": 1, "frozen_json_sha256": sha256_file(frozen_path),
        "input_hashes": input_hashes, "horizons": results,
    }
    write_json(Path(args.output).resolve(), document)
    return document


def main(argv=None):
    result = evaluate(parse_args(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
