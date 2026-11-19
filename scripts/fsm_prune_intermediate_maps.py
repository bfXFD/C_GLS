"""Archive selection records, then optionally prune allowlisted historical map arrays."""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from fsm_storage_inventory import FSM, HISTORICAL_MAPS


ROOT = FSM.parents[1]
ALLOWED_DIRS = (*HISTORICAL_MAPS, "replica_room0_0/batch_smoke/maps", "replica_room0_0/batch_smoke_compressed/maps")
KEPT_FIELDS = {
    "indices.npy", "strategy.npy", "compression_ratio.npy", "lambda_min.npy", "delta.npy",
    "source_params_path.npy", "metadata_json.npy", "schema_version.npy", "seed.npy",
}
TEXT_SUFFIXES = {".json", ".csv", ".py", ".sh", ".tex", ".md", ".txt", ".yaml", ".yml"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def reference_texts():
    roots = [
        FSM / "mvp_observable_v1", ROOT / "scripts", ROOT / "configs",
        ROOT.parent / "C_GLS/ralsubmission", ROOT.parent / "C_GLS/docs",
    ]
    result = []
    for root in roots:
        if not root.exists():
            continue
        for parent, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            for name in names:
                path = Path(parent) / name
                if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > 25 * 1024 ** 2:
                    continue
                # This script's explicit cleanup allowlist is not a scientific dependency.
                if path.resolve() in {Path(__file__).resolve(), ROOT / "scripts/fsm_storage_inventory.py"}:
                    continue
                result.append((str(path), path.read_text(errors="replace")))
    return result


def references(path, texts):
    relative = str(path.relative_to(FSM))
    directory = str(path.parent.relative_to(FSM))
    return [source for source, text in texts if relative in text or directory in text]


def original_params(source_value):
    path = Path(source_value)
    candidates = [path]
    # Historical exports predate the checkout rename. Preserve that recorded path,
    # but verify the original map parameters still exist at the migrated location.
    for old in ("/home/ras/xbf/SplaTAM_modify/",):
        if source_value.startswith(old):
            candidates.append(ROOT / source_value[len(old):])
    return next((p.resolve() for p in candidates if p.is_file()), None)


def archive_selection(path, destination):
    with np.load(path, allow_pickle=False) as data:
        if "indices" not in data.files or "source_params_path" not in data.files:
            raise ValueError("missing selection indices or source parameters")
        source_value = str(data["source_params_path"].item())
        params = original_params(source_value)
        if params is None:
            raise ValueError(f"original params unavailable: {source_value}")
        indices = data["indices"]
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError("invalid selection index array")
        count = int(indices.size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as target:
        kept = []
        for member in source.infolist():
            if member.filename in KEPT_FIELDS or (
                member.filename.endswith("_json.npy") and member.file_size <= 1024 ** 2
            ):
                target.writestr(member.filename, source.read(member.filename))
                kept.append(member.filename)
    with np.load(destination, allow_pickle=False) as data:
        if not np.array_equal(data["indices"], indices):
            raise ValueError("archived indices do not match original")
    return {"selected_count": count, "source_params_recorded": source_value,
            "source_params_existing": str(params), "kept_members": kept,
            "selection_archive": str(destination), "selection_archive_sha256": sha256(destination),
            "selection_archive_bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    archive = Path(args.archive_dir).resolve()
    if archive.exists() or archive.is_symlink():
        raise ValueError("archive-dir must be a new directory")
    if archive.parent != FSM.resolve():
        raise ValueError("archive-dir must be directly below experiments/FSM")
    archive.mkdir()
    texts = reference_texts()
    manifest = {"schema_version": 1, "applied": args.apply, "operation": "prune_historical_map_arrays",
                "note": "Not an exact file backup. Selection indices/scalars and outputs are retained; full byte-identical reconstruction is not guaranteed.",
                "scanned_text_files": len(texts), "deleted": [], "skipped": [], "params_hashes": {}}
    report = archive / "manifest.json"

    def persist():
        temporary = archive / "manifest.pending.json"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, report)

    persist()
    for relative in ALLOWED_DIRS:
        directory = FSM / relative
        if directory.is_symlink() or not directory.exists():
            continue
        if directory.resolve() != directory.absolute():
            raise ValueError(f"symlinked ancestor: {directory}")
        parent = directory.parent
        if not (parent / "summary_all.csv").is_file():
            manifest["skipped"].append({"path": str(directory), "reason": "missing summary_all.csv"})
            continue
        # Copy small batch-level records to the cleanup ledger as well as leaving originals.
        for record in (parent / "task_plan.csv", parent / "summary_all.csv", directory / "export_summary.csv"):
            if record.is_file():
                destination = archive / "records" / record.relative_to(FSM)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(record.read_bytes())
        for path in sorted(directory.rglob("fsm_map.npz")):
            if path.is_symlink() or not path.is_file():
                manifest["skipped"].append({"path": str(path), "reason": "not regular map"})
                continue
            refs = references(path, texts)
            if refs:
                manifest["skipped"].append({"path": str(path), "reason": "referenced by retained text", "references": refs})
                continue
            before = fingerprint(path)
            destination = archive / "selections" / path.relative_to(FSM)
            try:
                metadata = archive_selection(path, destination)
            except (ValueError, KeyError) as exc:
                manifest["skipped"].append({"path": str(path), "reason": str(exc)})
                persist()
                continue
            source_params = metadata["source_params_existing"]
            if source_params not in manifest["params_hashes"]:
                manifest["params_hashes"][source_params] = sha256(Path(source_params))
            record = {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
                      "allocated_bytes": path.stat().st_blocks * 512, "hardlink_count": path.stat().st_nlink,
                      "fingerprint": before, **metadata, "deleted": False}
            manifest["deleted"].append(record)
            persist()
            if fingerprint(path) != before:
                raise ValueError(f"map changed during archival: {path}")
            if args.apply:
                path.unlink()
                record["deleted"] = True
                persist()
    persist()
    print(json.dumps({"applied": args.apply, "archived_maps": len(manifest["deleted"]),
                      "deleted_maps": sum(r["deleted"] for r in manifest["deleted"]),
                      "skipped": len(manifest["skipped"]),
                      "removed_single_link_gib": sum(r["allocated_bytes"] for r in manifest["deleted"] if r["deleted"] and r["hardlink_count"] == 1) / 1024 ** 3,
                      "manifest": str(report)}, indent=2))


if __name__ == "__main__":
    main()
