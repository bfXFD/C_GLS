import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "scene",
    "run_name",
    "strategy",
    "compression_ratio",
    "ate_rmse",
    "ate_over_full",
    "rpe_rmse",
    "latency_mean_ms",
    "speedup_over_full",
    "map_size_mb",
    "size_reduction_over_full",
    "fitness",
    "registration_rmse",
    "summary_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge FSM localization summaries into a Pareto-ready table.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--summary",
        nargs=2,
        action="append",
        metavar=("SCENE", "SUMMARY_ALL_CSV"),
        required=True,
        help="Scene label and summary_all.csv path. Repeat for multiple scenes.",
    )
    return parser.parse_args()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def _find_full_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = []
    for row in rows:
        if row.get("strategy") != "full":
            continue
        ratio = _to_float(row, "compression_ratio")
        candidates.append((abs(ratio - 1.0), row))
    if not candidates:
        raise KeyError("No full-map localization row found")
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def build_pareto_rows(summaries: list[list[str]]) -> list[dict[str, Any]]:
    output_rows = []
    for scene, summary_path in summaries:
        rows = _read_rows(summary_path)
        full = _find_full_row(rows)
        full_ate = _to_float(full, "ate_rmse")
        full_latency = _to_float(full, "latency_mean_ms")
        full_map_size = _to_float(full, "map_size_mb")
        for row in rows:
            ate = _to_float(row, "ate_rmse")
            latency = _to_float(row, "latency_mean_ms")
            map_size = _to_float(row, "map_size_mb")
            output_rows.append(
                {
                    "scene": scene,
                    "run_name": row.get("run_name", ""),
                    "strategy": row.get("strategy", ""),
                    "compression_ratio": row.get("compression_ratio", ""),
                    "ate_rmse": row.get("ate_rmse", ""),
                    "ate_over_full": f"{_safe_ratio(ate, full_ate):.12g}",
                    "rpe_rmse": row.get("rpe_rmse", row.get("rpe_trans_rmse", "")),
                    "latency_mean_ms": row.get("latency_mean_ms", ""),
                    "speedup_over_full": f"{_safe_ratio(full_latency, latency):.12g}",
                    "map_size_mb": row.get("map_size_mb", ""),
                    "size_reduction_over_full": f"{_safe_ratio(full_map_size, map_size):.12g}",
                    "fitness": row.get("fitness", ""),
                    "registration_rmse": row.get("registration_rmse", ""),
                    "summary_path": str(summary_path),
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
    rows = build_pareto_rows(args.summary)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} Pareto rows to {args.output}")


if __name__ == "__main__":
    main()
