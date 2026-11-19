import argparse
import csv
import hashlib
import json
from pathlib import Path


ARMS = ("full", "compact", "corrweak", "replace")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures, rows, records = [], [], {}
    for scene in args.scenes:
        for arm in ARMS:
            run = root / scene / arm
            summary_path = run / "localize" / f"{scene}_{arm}" / "summary.json"
            manifest_path = run / "manifest.json"
            if not summary_path.is_file() or not manifest_path.is_file():
                failures.append(f"missing {scene}/{arm}")
                continue
            summary = json.loads(summary_path.read_text())
            frames = int(summary["num_frames"])
            valid = int(summary["num_valid_frames"])
            lost = int(summary["num_lost_frames"])
            if valid + lost != frames:
                failures.append(f"accounting {scene}/{arm}")
            rows.append({
                "scene": scene, "arm": arm, "frames": frames, "valid": valid, "lost": lost,
                "raw_ate_rmse_m": summary["raw_ate_rmse_m"],
                "raw_rotation_rmse_deg": summary["raw_rotation_abs_rmse_deg"],
                "latency_mean_ms": summary["production_equivalent_total_frame_latency_mean_ms"],
                "attempts": summary["num_reserve_attempt_frames"], "adoptions": summary["num_reserve_adopted_frames"],
                "summary_sha256": sha256(summary_path), "manifest_sha256": sha256(manifest_path),
            })
            records[f"{scene}/{arm}"] = rows[-1]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    document = {"schema_version": 1, "accepted": not failures, "scenes": args.scenes, "arms": list(ARMS), "records": records, "failures": failures}
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": not failures, "runs": len(rows), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
