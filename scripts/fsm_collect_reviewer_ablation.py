import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "dataset", "scene", "case", "status", "frames", "valid", "lost",
    "raw_ate_rmse_m", "raw_rotation_abs_rmse_deg", "latency_mean_ms",
    "reserve_attempts", "reserve_adoptions", "config_sha256", "resolved_config_sha256",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect reviewer-ablation terminal summaries.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for manifest_path in sorted(root.glob("runs/*/manifest.json")):
        case_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        case = manifest["cases"][0]
        case_name = case["name"]
        state_path = case_dir / ".state" / f"{case_name}.done.json"
        summary_path = case_dir / "localize" / case_name / "summary.json"
        if not summary_path.is_file() or not state_path.is_file():
            rows.append({"case": case_name, "status": "incomplete"})
            continue
        summary = json.loads(summary_path.read_text())
        state = json.loads(state_path.read_text())
        provenance = state.get("provenance", {})
        config = json.loads(Path(case["resolved_config_snapshot"]).read_text())
        review = config.get("reviewer_ablation", {}) if isinstance(config, dict) else {}
        rows.append({
            "dataset": review.get("dataset", ""),
            "scene": review.get("scene", ""),
            "case": case_name,
            "status": "complete",
            "frames": summary.get("num_frames", ""),
            "valid": summary.get("num_valid_frames", ""),
            "lost": summary.get("num_lost_frames", ""),
            "raw_ate_rmse_m": summary.get("raw_ate_rmse_m", ""),
            "raw_rotation_abs_rmse_deg": summary.get("raw_rotation_abs_rmse_deg", ""),
            "latency_mean_ms": summary.get("production_equivalent_total_frame_latency_mean_ms", ""),
            "reserve_attempts": summary.get("num_reserve_attempt_frames", ""),
            "reserve_adoptions": summary.get("num_reserve_adopted_frames", ""),
            "config_sha256": provenance.get("config_sha256", ""),
            "resolved_config_sha256": provenance.get("resolved_config_sha256", ""),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    print(json.dumps({"runs": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
