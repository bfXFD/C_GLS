"""Inventory selected historical exports without deleting experiment evidence."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FSM = ROOT / "experiments/FSM"
HISTORICAL_MAPS = (
    "replica_room0_0/batch_500_pareto_v1_1/maps",
    "replica_room1_0/batch_500_pareto_v1_1/maps",
    "replica_room2_0/batch_500_pareto_v1_1/maps",
    "replica_office0_0/batch_500_pareto_v1_1/maps",
    "replica_room0_0/batch_500_multiscene_core_v1_0_e/maps",
    "replica_room1_0/batch_500_multiscene_core_v1_0_e/maps",
    "replica_room2_0/batch_500_multiscene_core_v1_0_e/maps",
    "replica_office0_0/batch_500_multiscene_core_v1_0_e/maps",
    "replica_room0_0/batch_500_coverage_gls_v1_0_e/maps",
    "replica_room0_0/batch_500_baselines_v1_0_c/maps",
    "tum_freiburg2_xyz/batch_500_tum_pareto_v1_1/maps",
    "compression_cost_v1_0/maps",
)
SMOKE_PAYLOADS = (
    "replica_room0_0/batch_smoke/maps",
    "replica_room0_0/batch_smoke/fisher",
    "replica_room0_0/batch_smoke_compressed/maps",
    "replica_room0_0/batch_smoke_compressed/fisher",
)
DUPLICATE_MAPS = (
    "replica_room0_0/batch_100_full/maps/full_ratio_1/fsm_map.npz",
    "replica_room0_0/batch_100_full_posefix/maps/full_ratio_1/fsm_map.npz",
    "replica_room0_0/batch_500_full_posefix/maps/full_ratio_1/fsm_map.npz",
    "replica_room0_0/batch_2000_full_posefix/maps/full_ratio_1/fsm_map.npz",
)
CANONICAL_MAP = "replica_room0_0/export_full/full_ratio_1/fsm_map.npz"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(relative, group, canonical_hash):
    path = FSM / relative
    row = {
        "relative_path": relative,
        "path": str(path),
        "group": group,
        "exists": path.exists(),
        "deletion_approved": False,
        "dependency_audit_complete": False,
    }
    if not path.exists():
        return row
    row["allocated_bytes"] = int(subprocess.check_output(
        ["du", "-s", "-B1", "--", str(path)], text=True
    ).split()[0])
    if path.is_file():
        row["files"] = [{
            "path": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256(path), "hardlink_count": path.stat().st_nlink,
        }]
        row["identical_to_retained_canonical"] = row["files"][0]["sha256"] == canonical_hash
    else:
        files = []
        for directory, _, names in os.walk(path, followlinks=False):
            for name in sorted(names):
                item = Path(directory) / name
                stat = item.lstat()
                record = {
                    "path": str(item.relative_to(path)), "size_bytes": stat.st_size,
                    "is_symlink": item.is_symlink(), "hardlink_count": stat.st_nlink,
                }
                # Preserve inexpensive reconstruction metadata hashes in the inventory.
                if item.suffix in (".json", ".py", ".yaml", ".yml") and not item.is_symlink():
                    record["sha256"] = sha256(item)
                files.append(record)
        row["files"] = files
    row["file_count"] = len(row["files"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    canonical = FSM / CANONICAL_MAP
    canonical_hash = sha256(canonical) if canonical.is_file() else None
    records = []
    for group, paths in (
        ("verified_duplicate_candidates", DUPLICATE_MAPS),
        ("short_diagnostic_payload_candidates", SMOKE_PAYLOADS),
        ("historical_export_review_only", HISTORICAL_MAPS),
    ):
        records.extend(inspect(path, group, canonical_hash) for path in paths)
    totals = {}
    for record in records:
        totals[record["group"]] = totals.get(record["group"], 0) + record.get("allocated_bytes", 0)
    document = {
        "schema_version": 1, "operation": "inventory_only", "deleted_files": 0,
        "retained_canonical": {"path": str(canonical), "sha256": canonical_hash},
        "protected": [
            str(FSM / "audit_v2_full_rerun"),
            str(FSM / "mvp_observable_v1"),
            str(ROOT / "data"), str(ROOT / "experiments/TUM"),
            str(ROOT.parent / "SplaTAM/experiments"),
            str(ROOT.parent / "C_GLS/ralsubmission"),
        ],
        "warning": "Candidate sizes are not safe-to-delete totals. No complete transitive dependency audit has been performed.",
        "allocated_bytes_by_group": totals, "records": records,
    }
    output = Path(args.output).resolve()
    if output.exists() or output.suffix != ".json":
        raise ValueError("output must be a new JSON file, never an existing artifact")
    if "audit_v2_full_rerun" in output.parts or "mvp_observable_v1" in output.parts:
        raise ValueError("inventory output cannot be written inside frozen experiment trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    document["allocation_semantics"] = "Per-path du totals can count shared inodes repeatedly; totals are inventory sizes, not reclaim estimates."
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "deleted_files": 0,
                      "candidate_gib_by_group": {key: round(value / 1024 ** 3, 3) for key, value in totals.items()}}, indent=2))


if __name__ == "__main__":
    main()
