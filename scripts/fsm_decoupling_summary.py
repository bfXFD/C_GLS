import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "scene",
    "method",
    "ratio",
    "localization_strategy",
    "localization_ratio",
    "ate_rmse",
    "rpe_trans_rmse",
    "fitness",
    "registration_rmse",
    "latency_mean_ms",
    "map_size_mb",
    "psnr_mean",
    "ms_ssim_mean",
    "depth_l1_mean",
    "num_render_gaussians",
    "num_eval_frames",
    "render_summary_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge FSM rendering and localization metrics for decoupling analysis.")
    parser.add_argument("--scene", required=True, help="Scene label written to the output table.")
    parser.add_argument("--localization-summary", required=True, help="summary_all.csv from fsm_eval.py.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--entry",
        nargs=4,
        action="append",
        metavar=("METHOD", "STRATEGY", "RATIO", "RENDER_SUMMARY"),
        required=True,
        help="One row: display method, localization strategy, ratio, render summary.json.",
    )
    return parser.parse_args()


def _read_localization_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_localization_row(rows: list[dict[str, str]], strategy: str, ratio: str) -> dict[str, str]:
    ratio_value = float(ratio)
    matches = []
    for row in rows:
        if row.get("strategy") != strategy:
            continue
        try:
            row_ratio = float(row.get("compression_ratio", "nan"))
        except ValueError:
            continue
        if np.isclose(row_ratio, ratio_value, rtol=0.0, atol=1e-9):
            matches.append(row)
    if not matches:
        raise KeyError(f"No localization row for strategy={strategy}, ratio={ratio}")
    return matches[0]


def _load_render_summary(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_decoupling_rows(
    scene: str,
    localization_summary: str,
    entries: list[list[str]],
) -> list[dict[str, Any]]:
    localization_rows = _read_localization_rows(localization_summary)
    rows = []
    for method, strategy, ratio, render_summary_path in entries:
        loc = _find_localization_row(localization_rows, strategy=strategy, ratio=ratio)
        render = _load_render_summary(render_summary_path)
        rows.append(
            {
                "scene": scene,
                "method": method,
                "ratio": str(ratio),
                "localization_strategy": strategy,
                "localization_ratio": loc.get("compression_ratio", ratio),
                "ate_rmse": loc.get("ate_rmse", ""),
                "rpe_trans_rmse": loc.get("rpe_rmse", loc.get("rpe_trans_rmse", "")),
                "fitness": loc.get("fitness", ""),
                "registration_rmse": loc.get("registration_rmse", ""),
                "latency_mean_ms": loc.get("latency_mean_ms", ""),
                "map_size_mb": loc.get("map_size_mb", ""),
                "psnr_mean": render.get("psnr_mean", ""),
                "ms_ssim_mean": render.get("ms_ssim_mean", ""),
                "depth_l1_mean": render.get("depth_l1_mean", ""),
                "num_render_gaussians": render.get("num_render_gaussians", ""),
                "num_eval_frames": render.get("num_eval_frames", ""),
                "render_summary_path": str(render_summary_path),
            }
        )
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_decoupling_rows(
        scene=args.scene,
        localization_summary=args.localization_summary,
        entries=args.entry,
    )
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} decoupling rows to {args.output}")


if __name__ == "__main__":
    main()
