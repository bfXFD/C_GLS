"""Explicitly deduplicate four verified immutable map exports, preserving paths."""

import argparse
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path


FSM = Path(__file__).resolve().parents[1] / "experiments/FSM"
CANONICAL = FSM / "replica_room0_0/export_full/full_ratio_1/fsm_map.npz"
CANDIDATES = [FSM / f"replica_room0_0/{name}/maps/full_ratio_1/fsm_map.npz" for name in (
    "batch_100_full", "batch_100_full_posefix", "batch_500_full_posefix", "batch_2000_full_posefix",
)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def info(path):
    value = path.stat()
    return {"inode": value.st_ino, "device": value.st_dev, "links": value.st_nlink,
            "size_bytes": value.st_size, "allocated_bytes": value.st_blocks * 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if CANONICAL.is_symlink() or not CANONICAL.is_file():
        raise ValueError("canonical must be a regular non-symlink file")
    source_stat = CANONICAL.stat()
    digest = sha256(CANONICAL)
    records = []
    for path in CANDIDATES:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate must be a regular non-symlink file: {path}")
        value = path.stat()
        if (value.st_dev, value.st_size, value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode)) != (
            source_stat.st_dev, source_stat.st_size, source_stat.st_uid, source_stat.st_gid,
            stat.S_IMODE(source_stat.st_mode),
        ) or sha256(path) != digest:
            raise ValueError(f"candidate content or permission mismatch: {path}")
        records.append({"path": str(path), "before": info(path)})
    report = Path(args.report).resolve()
    if report.exists() or report.suffix != ".json":
        raise ValueError("report must be a new JSON file, never an existing artifact")
    if "audit_v2_full_rerun" in report.parts or "mvp_observable_v1" in report.parts:
        raise ValueError("report cannot be written into frozen experiment trees")
    report.parent.mkdir(parents=True, exist_ok=True)
    document = {"applied": args.apply, "canonical": str(CANONICAL), "sha256": digest,
                "warning": "These exports now share content; replace paths atomically when regenerating, never write in-place.",
                "records": records}
    # Persist the pre-operation evidence even if a later operation fails.
    with report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
    for record in records:
        path = Path(record["path"])
        if args.apply and not os.path.samefile(CANONICAL, path):
            if info(path) != record["before"] or sha256(path) != digest:
                raise ValueError(f"candidate changed after preflight: {path}")
            temporary = path.with_name(f".dedupe-{uuid.uuid4().hex}.npz")
            try:
                os.link(CANONICAL, temporary)
                linked_stat = temporary.stat()
                if (linked_stat.st_dev, linked_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino) or sha256(temporary) != digest:
                    raise ValueError("canonical changed during operation; no replacement performed")
                if info(path) != record["before"] or sha256(path) != digest:
                    raise ValueError(f"candidate changed during operation: {path}")
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        record["after"] = info(path)
        record["verified_sha256"] = sha256(path)
        if record["verified_sha256"] != digest:
            raise ValueError(f"post-operation hash mismatch: {path}")
        report_temporary = report.with_name(f".report-{uuid.uuid4().hex}.json")
        try:
            with report_temporary.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(document, indent=2) + "\n")
            os.replace(report_temporary, report)
        finally:
            if report_temporary.exists():
                report_temporary.unlink()
    print(json.dumps({"applied": args.apply, "verified_paths": len(records), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
