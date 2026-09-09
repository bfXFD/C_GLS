import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


GIB = 1024**3


def available_ram_bytes() -> int:
    values = {}
    with open("/proc/meminfo") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    return values["MemAvailable"]


def free_disk_bytes(path: str) -> int:
    probe = Path(path).expanduser().resolve()
    while not probe.exists():
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def check_scipy() -> str:
    from scipy.spatial import cKDTree

    return f"{cKDTree.__module__}.{cKDTree.__name__}"


def inspect_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as values:
        fields = sorted(values.files)
        if "means3D" in values:
            count = int(np.asarray(values["means3D"]).shape[0])
        elif "means" in values:
            count = int(np.asarray(values["means"]).shape[0])
        else:
            count = None
    return {"path": path, "fields": fields, "gaussian_count": count, "size_bytes": os.path.getsize(path)}


def run_preflight(
    inputs: list[str], npz_inputs: list[str], output_root: str, estimated_output_bytes: int,
    min_free_bytes: int = 15 * GIB, min_available_ram_bytes: int = 12 * GIB,
) -> dict[str, Any]:
    missing = [path for path in inputs + npz_inputs if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    scipy_backend = check_scipy()
    metadata = [inspect_npz(path) for path in npz_inputs]
    free = free_disk_bytes(output_root)
    available = available_ram_bytes()
    required_disk = int(min_free_bytes) + int(estimated_output_bytes)
    if free < required_disk:
        raise RuntimeError(f"insufficient free disk: {free} < {required_disk}")
    if available < int(min_available_ram_bytes):
        raise RuntimeError(f"insufficient available RAM: {available} < {min_available_ram_bytes}")
    return {
        "ok": True,
        "scipy_backend": scipy_backend,
        "free_disk_bytes": free,
        "available_ram_bytes": available,
        "estimated_output_bytes": int(estimated_output_bytes),
        "disk_safety_bytes": int(min_free_bytes),
        "npz_metadata": metadata,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource and input preflight for the MVP observable runner.")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--npz-input", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--estimated-output-bytes", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=15.0)
    parser.add_argument("--min-available-ram-gib", type=float, default=12.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_preflight(
        args.input,
        args.npz_input,
        args.output_root,
        args.estimated_output_bytes,
        int(args.min_free_gib * GIB),
        int(args.min_available_ram_gib * GIB),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
