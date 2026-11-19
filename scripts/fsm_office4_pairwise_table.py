import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2/runs"
CASES = {
    "baseline": SUITE / "replica_office4_baseline/localize/replica_office4_baseline/per_frame_errors.csv",
    "corrweak128": SUITE / "replica_office4_corrweak128/localize/replica_office4_corrweak128/per_frame_errors.csv",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = {name: load(path) for name, path in CASES.items()}
    common = np.logical_and.reduce([rows["tracking_valid"].astype(bool) for rows in data.values()])
    rows = []
    for name, values in data.items():
        rows.append({
            "scene": "office4", "method": name, "pairwise_common": int(common.sum()),
            "valid": int(values["tracking_valid"].astype(bool).sum()),
            "lost": int(len(values) - values["tracking_valid"].astype(bool).sum()),
            "common_translation_rmse_m": float(np.sqrt(np.mean(values["raw_translation_error_m"][common] ** 2))),
            "common_rotation_rmse_deg": float(np.sqrt(np.mean(values["raw_rotation_error_deg"][common] ** 2))),
            "per_frame_errors_sha256": sha256(CASES[name]),
        })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "schema_version": 1, "support": "pairwise_common_valid", "common": int(common.sum()),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in CASES.items()},
        "output_sha256": sha256(output),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
