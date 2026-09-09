import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_map import fsm_map_field_sizes, load_fsm_map


def parse_args():
    parser = argparse.ArgumentParser(description="Report field-level storage for an FSM map.")
    parser.add_argument("--map", required=True, help="Input fsm_map.npz path.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def summarize_storage(map_path: str) -> dict:
    fsm_map = load_fsm_map(map_path)
    field_bytes = fsm_map_field_sizes(fsm_map)
    means = np.asarray(fsm_map["means"])
    return {
        "map_path": map_path,
        "serialized_bytes": Path(map_path).stat().st_size,
        "field_bytes": field_bytes,
        "array_bytes_total": int(sum(field_bytes.values())),
        "means_only_fp32_bytes": int(means.astype(np.float32).nbytes),
        "means_only_fp16_bytes": int(means.astype(np.float16).nbytes),
    }


def main() -> None:
    args = parse_args()
    summary = summarize_storage(args.map)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
