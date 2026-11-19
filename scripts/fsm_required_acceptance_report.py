import argparse
import hashlib
import json
from pathlib import Path


SCENES = ("office0", "room2", "office4", "fr3", "reverse")
HORIZONS = (1, 3, 5, 10)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Audit the required acceptance experiment chain.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = []
    records = {}
    reverse_acceptance = root / "reverse_revisit/acceptance.json"
    frozen = root / "frozen/horizon_thresholds_office0.json"
    required = [reverse_acceptance, frozen]
    for scene in SCENES:
        paired = root / f"paired/{scene}/localize/paired"
        required.extend([
            paired / "summary.json",
            paired / "trajectory_gt.txt",
            paired / "trajectory_est.txt",
            paired / "paired_reserve/paired_reserve_ids.npz",
        ])
        for horizon in HORIZONS:
            horizon_dir = root / f"horizon_gtfree/{scene}/h{horizon}"
            label_dir = root / f"horizon_labels/{scene}/h{horizon}"
            required.extend([
                horizon_dir / "horizon_events.csv",
                horizon_dir / "horizon_branch_poses.npz",
                horizon_dir / "provenance.json",
                label_dir / "horizon_gt_labels.csv",
                label_dir / "horizon_gt_labels_provenance.json",
            ])
        if scene != "office0":
            required.append(root / f"evaluation/{scene}.json")
    for path in required:
        if not path.is_file():
            failures.append(f"missing: {path}")
        else:
            records[str(path.relative_to(root))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    if reverse_acceptance.is_file() and not json.loads(reverse_acceptance.read_text()).get("accepted"):
        failures.append("reverse revisit acceptance rejected")
    if frozen.is_file():
        document = json.loads(frozen.read_text())
        if document.get("immutable") is not True or set(document.get("horizons", {})) != {"1", "3", "5", "10"}:
            failures.append("frozen threshold document is incomplete")
    result = {
        "schema_version": 1,
        "accepted": not failures,
        "development_scene": "office0",
        "test_scenes": ["room2", "office4", "fr3", "reverse_desk2_to_desk"],
        "horizons": list(HORIZONS),
        "records": records,
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": result["accepted"], "checks": len(records), "failures": failures}, indent=2))
    if args.require_all and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
