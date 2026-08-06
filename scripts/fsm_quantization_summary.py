import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "dataset",
    "scene",
    "method",
    "strategy",
    "compression_ratio",
    "base_ate_rmse",
    "quantized_ate_rmse",
    "ate_delta",
    "ate_relative",
    "base_rpe_rmse",
    "quantized_rpe_rmse",
    "base_latency_mean_ms",
    "quantized_latency_mean_ms",
    "latency_relative",
    "base_map_size_mb",
    "quantized_map_size_mb",
    "extra_size_reduction",
    "num_fallbacks",
    "base_summary_path",
    "quantized_summary_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize safe quantization localization ablations.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiment",
        nargs=5,
        action="append",
        metavar=("DATASET", "SCENE", "BASE_SUMMARY_ALL", "QUANT_ROOT", "METHOD"),
        required=True,
        help="Dataset label, scene label, base summary_all.csv, quantization root, and method name. Repeat as needed.",
    )
    return parser.parse_args()


def _read_summary_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _ratio_from_method(method: str) -> float:
    token = method.rsplit("_ratio_", 1)[1]
    return float(token.replace("p", "."))


def _strategy_from_method(method: str) -> str:
    return method.rsplit("_ratio_", 1)[0]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _find_base_row(rows: list[dict[str, str]], strategy: str, ratio: float) -> dict[str, str]:
    for row in rows:
        if row.get("strategy") != strategy:
            continue
        if abs(float(row.get("compression_ratio", "nan")) - ratio) < 1e-9:
            return row
    raise KeyError(f"No base row for {strategy} ratio {ratio}")


def build_quantization_rows(experiments: list[list[str]]) -> list[dict[str, Any]]:
    output_rows = []
    for dataset, scene, base_summary_all, quant_root, method in experiments:
        strategy = _strategy_from_method(method)
        ratio = _ratio_from_method(method)
        base_rows = _read_summary_rows(base_summary_all)
        base_row = _find_base_row(base_rows, strategy=strategy, ratio=ratio)
        quant_dir = Path(quant_root) / f"{method}_fp16_drop_scores"
        quant_summary_path = quant_dir / "localize" / "summary.json"
        quant_map_path = quant_dir / "fsm_map.npz"
        quant_summary = _read_json(quant_summary_path)

        base_ate = _to_float(base_row.get("ate_rmse"))
        quant_ate = _to_float(quant_summary.get("ate_rmse"))
        base_rpe = _to_float(base_row.get("rpe_rmse", base_row.get("rpe_trans_rmse")))
        quant_rpe = _to_float(quant_summary.get("rpe_trans_rmse"))
        base_latency = _to_float(base_row.get("latency_mean_ms"))
        quant_latency = _to_float(quant_summary.get("latency_mean_ms"))
        base_map_size = _to_float(base_row.get("map_size_mb"))
        quant_map_size = quant_map_path.stat().st_size / 1_000_000

        output_rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "method": method,
                "strategy": strategy,
                "compression_ratio": f"{ratio:.12g}",
                "base_ate_rmse": f"{base_ate:.12g}",
                "quantized_ate_rmse": f"{quant_ate:.12g}",
                "ate_delta": f"{quant_ate - base_ate:.12g}",
                "ate_relative": f"{_safe_ratio(quant_ate, base_ate):.12g}",
                "base_rpe_rmse": f"{base_rpe:.12g}",
                "quantized_rpe_rmse": f"{quant_rpe:.12g}",
                "base_latency_mean_ms": f"{base_latency:.12g}",
                "quantized_latency_mean_ms": f"{quant_latency:.12g}",
                "latency_relative": f"{_safe_ratio(quant_latency, base_latency):.12g}",
                "base_map_size_mb": f"{base_map_size:.12g}",
                "quantized_map_size_mb": f"{quant_map_size:.12g}",
                "extra_size_reduction": f"{_safe_ratio(base_map_size, quant_map_size):.12g}",
                "num_fallbacks": quant_summary.get("num_fallbacks", ""),
                "base_summary_path": str(base_summary_all),
                "quantized_summary_path": str(quant_summary_path),
            }
        )
    return output_rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_quantization_rows(args.experiment)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} quantization rows to {args.output}")


if __name__ == "__main__":
    main()
