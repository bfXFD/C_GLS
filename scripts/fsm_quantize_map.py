import argparse
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_map import load_fsm_map, save_fsm_map


FLOAT_FIELDS = ("means", "covariances", "scales", "rotations", "scores")
DEFAULT_QUANTIZE_FIELDS = ("means",)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize an FSM map for localization compression experiments.")
    parser.add_argument("--map", required=True, help="Input fsm_map.npz path.")
    parser.add_argument("--output", required=True, help="Output quantized fsm_map.npz path.")
    parser.add_argument("--float-dtype", default="float16", choices=["float16", "float32"], help="Storage dtype for floating fields.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_QUANTIZE_FIELDS),
        choices=FLOAT_FIELDS,
        help="Floating fields to quantize. Keep rotations/scales/covariances FP32 unless explicitly testing aggressive quantization.",
    )
    parser.add_argument("--drop-scores", action="store_true", help="Replace full-length score array with an empty array.")
    return parser.parse_args()


def quantize_fsm_map(
    fsm_map: dict[str, Any],
    float_dtype: str = "float16",
    fields: list[str] | tuple[str, ...] = DEFAULT_QUANTIZE_FIELDS,
    drop_scores: bool = False,
) -> dict[str, Any]:
    if float_dtype not in ("float16", "float32"):
        raise ValueError("float_dtype must be float16 or float32")
    unknown = sorted(set(fields) - set(FLOAT_FIELDS))
    if unknown:
        raise ValueError(f"Unknown quantization fields: {unknown}")
    dtype = np.dtype(float_dtype)
    quantized = dict(fsm_map)
    for key in fields:
        if key not in quantized:
            continue
        quantized[key] = np.asarray(quantized[key]).astype(dtype)
    if drop_scores and "scores" in quantized:
        quantized["scores"] = np.asarray([], dtype=dtype)
    return quantized


def quantize_file(
    input_path: str,
    output_path: str,
    float_dtype: str = "float16",
    fields: list[str] | tuple[str, ...] = DEFAULT_QUANTIZE_FIELDS,
    drop_scores: bool = False,
) -> dict[str, Any]:
    fsm_map = load_fsm_map(input_path)
    quantized = quantize_fsm_map(fsm_map, float_dtype=float_dtype, fields=fields, drop_scores=drop_scores)
    save_fsm_map(quantized, output_path)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_size_mb": Path(input_path).stat().st_size / (1024 * 1024),
        "output_size_mb": Path(output_path).stat().st_size / (1024 * 1024),
        "float_dtype": float_dtype,
        "fields": list(fields),
        "drop_scores": bool(drop_scores),
    }


def main() -> None:
    args = parse_args()
    summary = quantize_file(args.map, args.output, float_dtype=args.float_dtype, fields=args.fields, drop_scores=args.drop_scores)
    print(
        "Quantized FSM map: "
        f"{summary['input_size_mb']:.3f} MB -> {summary['output_size_mb']:.3f} MB "
        f"({summary['input_size_mb'] / max(summary['output_size_mb'], 1e-12):.2f}x smaller)"
    )


if __name__ == "__main__":
    main()
