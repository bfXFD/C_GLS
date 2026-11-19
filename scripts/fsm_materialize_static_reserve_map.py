#!/usr/bin/env python3
"""Materialize a compact FSM map with its entire independent reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.fsm_map import load_activation_geometry, load_fsm_map, save_fsm_map, validate_fsm_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--activation-geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy-name", default="static_all_reserve_v1")
    parser.add_argument("--expected-reserve-count", type=int, default=4096)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_plan(args: argparse.Namespace, base_count: int, reserve_count: int) -> dict:
    return {
        "schema_version": 1,
        "builder": "static_all_reserve_materializer_v1",
        "base_map": str(Path(args.base_map).resolve()),
        "base_map_sha256": sha256(args.base_map),
        "activation_geometry": str(Path(args.activation_geometry).resolve()),
        "activation_geometry_sha256": sha256(args.activation_geometry),
        "strategy_name": args.strategy_name,
        "expected_reserve_count": args.expected_reserve_count,
        "base_count": base_count,
        "reserve_count": reserve_count,
        "merged_count": base_count + reserve_count,
    }


def atomic_save(map_value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save_fsm_map(map_value, str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.expected_reserve_count <= 0:
        raise ValueError("expected-reserve-count must be positive")
    base = load_fsm_map(args.base_map)
    geometry = load_activation_geometry(args.activation_geometry, base)
    reserve_count = int(np.asarray(geometry["reserve_count"]).item())
    if reserve_count != args.expected_reserve_count:
        raise ValueError(f"Expected {args.expected_reserve_count} reserve rows, got {reserve_count}")
    base_ids = np.asarray(base["indices"], dtype=np.int64)
    reserve_ids = np.asarray(geometry["reserve_global_ids"], dtype=np.int64)
    if np.intersect1d(base_ids, reserve_ids, assume_unique=True).size:
        raise ValueError("Base and reserve global IDs overlap")
    plan = canonical_plan(args, base_ids.size, reserve_ids.size)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **plan}, indent=2, sort_keys=True))
        return

    output = Path(args.output)
    done_path = Path(f"{output}.done.json")
    if args.skip_existing and output.is_file() and done_path.is_file():
        stored = json.loads(done_path.read_text())
        if stored.get("provenance") != plan:
            raise RuntimeError(f"Existing output provenance differs: {output}")
        loaded = load_fsm_map(str(output))
        if int(np.asarray(loaded["indices"]).size) != plan["merged_count"] or sha256(output) != stored.get("output_sha256"):
            raise RuntimeError(f"Existing output failed count/hash validation: {output}")
        print(json.dumps({"status": "skipped", **stored}, indent=2, sort_keys=True))
        return
    if output.exists() or done_path.exists():
        raise FileExistsError(f"Refusing existing incomplete or unverified output: {output}")

    merged = dict(base)
    field_pairs = {
        "means": "reserve_means",
        "covariances": "reserve_covariances",
        "scales": "reserve_scales",
        "rotations": "reserve_rotations",
        "indices": "reserve_global_ids",
        "scores": "reserve_scores",
    }
    for map_field, reserve_field in field_pairs.items():
        base_values = np.asarray(base[map_field])
        reserve_values = np.asarray(geometry[reserve_field])
        if map_field == "scores" and base_values.size == 0:
            if reserve_values.size:
                raise ValueError("Cannot append reserve scores to a score-free base map")
            merged[map_field] = base_values
        else:
            merged[map_field] = np.concatenate([base_values, reserve_values], axis=0)
    merged["strategy"] = np.asarray(args.strategy_name)
    merged["compression_ratio"] = np.asarray(float(merged["indices"].size) / max(1, int(np.max(merged["indices"])) + 1), dtype=np.float32)
    merged["static_base_map_sha256"] = np.asarray(plan["base_map_sha256"])
    merged["static_activation_geometry_sha256"] = np.asarray(plan["activation_geometry_sha256"])
    merged["static_base_count"] = np.asarray(plan["base_count"], dtype=np.int64)
    merged["static_reserve_count"] = np.asarray(plan["reserve_count"], dtype=np.int64)
    validate_fsm_map(merged)
    atomic_save(merged, output)
    done = {"provenance": plan, "output": str(output.resolve()), "output_size_bytes": output.stat().st_size, "output_sha256": sha256(output)}
    done_path.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", **done}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
