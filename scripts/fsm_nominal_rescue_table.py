import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "room2": ROOT / "experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2/runs/replica_room2_corrweak128/localize/replica_room2_corrweak128/summary.json",
    "office0": ROOT / "experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2/runs/replica_office0_corrweak128/localize/replica_office0_corrweak128/summary.json",
    "office4": ROOT / "experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2/runs/replica_office4_corrweak128/localize/replica_office4_corrweak128/summary.json",
    "fr3": ROOT / "experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2_tum/runs/tum_freiburg3_long_office_household_corrweak128/localize/tum_freiburg3_long_office_household_corrweak128/summary.json",
}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    rows = []
    for scene, path in RUNS.items():
        summary = json.loads(path.read_text())
        rows.append({
            "scene": scene,
            "reserve_attempts": summary["num_reserve_attempt_frames"],
            "reserve_adoptions": summary["num_reserve_adopted_frames"],
            "candidate_adoptions_after_primary_rejection": summary["num_reserve_rescue_frames"],
            "reserve_rescue_rate": summary["reserve_rescue_rate"],
            "summary_sha256": sha256(path),
        })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
