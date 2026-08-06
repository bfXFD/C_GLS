import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MAIN_METHODS = [
    ("full", 1.0),
    ("voxel_scale_opacity", 0.25),
    ("fps_scale_opacity", 0.25),
    ("voxel_scale_opacity", 0.20),
    ("fps_scale_opacity", 0.20),
    ("voxel_scale_opacity", 0.15),
    ("voxel_grid_pure", 0.10),
    ("voxel_scale_opacity", 0.10),
    ("scale_large", 0.05),
    ("voxel_scale_opacity", 0.05),
]

ABLATION_METHODS = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
ABLATION_RATIOS = [0.25, 0.20, 0.15, 0.10, 0.05]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready FSM tables and lightweight figures.")
    parser.add_argument("--replica-pareto", default="/home/ras/xbf/C_GLS_code/experiments/FSM/pareto_v1_1_summary.csv")
    parser.add_argument("--tum-pareto", default="/home/ras/xbf/C_GLS_code/experiments/FSM/tum_pareto_v1_1_summary.csv")
    parser.add_argument("--quantization", default="/home/ras/xbf/C_GLS_code/experiments/FSM/quantization_v1_1_summary.csv")
    parser.add_argument("--output-dir", default="/home/ras/xbf/C_GLS_code/experiments/FSM/paper_artifacts_v1_2")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_means(rows: list[dict[str, str]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for key_values, group in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        for metric in metrics:
            item[metric] = f"{_mean([_to_float(row, metric) for row in group]):.6g}"
        item["num_rows"] = len(group)
        output.append(item)
    return output


def build_replica_main_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for strategy, ratio in MAIN_METHODS:
        selected = [
            row
            for row in rows
            if row.get("strategy") == strategy and abs(_to_float(row, "compression_ratio") - ratio) < 1e-9
        ]
        if not selected:
            continue
        output.append(
            {
                "strategy": strategy,
                "compression_ratio": f"{ratio:.2f}",
                "mean_ate_rmse": f"{_mean([_to_float(row, 'ate_rmse') for row in selected]):.6g}",
                "mean_ate_over_full": f"{_mean([_to_float(row, 'ate_over_full') for row in selected]):.6g}",
                "mean_rpe_rmse": f"{_mean([_to_float(row, 'rpe_rmse') for row in selected]):.6g}",
                "mean_speedup_over_full": f"{_mean([_to_float(row, 'speedup_over_full') for row in selected]):.6g}",
                "mean_size_reduction_over_full": f"{_mean([_to_float(row, 'size_reduction_over_full') for row in selected]):.6g}",
                "num_scenes": len(selected),
            }
        )
    return output


def build_ablation_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for ratio in ABLATION_RATIOS:
        for strategy in ABLATION_METHODS:
            selected = [
                row
                for row in rows
                if row.get("strategy") == strategy and abs(_to_float(row, "compression_ratio") - ratio) < 1e-9
            ]
            if not selected:
                continue
            output.append(
                {
                    "strategy": strategy,
                    "compression_ratio": f"{ratio:.2f}",
                    "mean_ate_over_full": f"{_mean([_to_float(row, 'ate_over_full') for row in selected]):.6g}",
                    "mean_speedup_over_full": f"{_mean([_to_float(row, 'speedup_over_full') for row in selected]):.6g}",
                    "mean_size_reduction_over_full": f"{_mean([_to_float(row, 'size_reduction_over_full') for row in selected]):.6g}",
                    "num_scenes": len(selected),
                }
            )
    return output


def build_tum_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keep = []
    for row in rows:
        ratio = _to_float(row, "compression_ratio")
        strategy = row.get("strategy", "")
        if strategy == "full" and abs(ratio - 1.0) < 1e-9:
            keep.append(row)
        elif strategy in {"fps_scale_opacity", "random", "voxel_scale_opacity", "voxel_grid_pure", "scale_large"} and abs(ratio - 0.10) < 1e-9:
            keep.append(row)
        elif strategy == "voxel_scale_opacity" and abs(ratio - 0.25) < 1e-9:
            keep.append(row)
    return [
        {
            "scene": row.get("scene", ""),
            "strategy": row.get("strategy", ""),
            "compression_ratio": row.get("compression_ratio", ""),
            "ate_rmse": row.get("ate_rmse", ""),
            "ate_over_full": row.get("ate_over_full", ""),
            "rpe_rmse": row.get("rpe_rmse", ""),
            "speedup_over_full": row.get("speedup_over_full", ""),
            "size_reduction_over_full": row.get("size_reduction_over_full", ""),
            "map_size_mb": row.get("map_size_mb", ""),
        }
        for row in keep
    ]


def _write_markdown_table(path: str | Path, title: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_replica(rows: list[dict[str, str]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        (output_dir / "plot_warning.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    selected = [row for row in rows if row.get("strategy") in {"random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"}]
    by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_strategy[row.get("strategy", "")].append(row)
    plt.figure(figsize=(7, 5))
    for strategy, group in sorted(by_strategy.items()):
        means = _group_means(group, ["compression_ratio"], ["ate_over_full", "speedup_over_full"])
        points = sorted((float(row["speedup_over_full"]), float(row["ate_over_full"])) for row in means)
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", label=strategy)
    plt.xlabel("Speedup over full")
    plt.ylabel("ATE / full")
    plt.title("Replica Pareto: localization accuracy vs speed")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "replica_pareto_speedup.png", dpi=200)
    plt.close()


def generate_artifacts(replica_pareto: str, tum_pareto: str, quantization: str, output_dir: str, make_plots: bool = True) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replica_rows = _read_rows(replica_pareto)
    tum_rows = _read_rows(tum_pareto)

    replica_main = build_replica_main_table(replica_rows)
    ablation = build_ablation_table(replica_rows)
    tum_table = build_tum_table(tum_rows)

    replica_fields = ["strategy", "compression_ratio", "mean_ate_rmse", "mean_ate_over_full", "mean_rpe_rmse", "mean_speedup_over_full", "mean_size_reduction_over_full", "num_scenes"]
    ablation_fields = ["strategy", "compression_ratio", "mean_ate_over_full", "mean_speedup_over_full", "mean_size_reduction_over_full", "num_scenes"]
    tum_fields = ["scene", "strategy", "compression_ratio", "ate_rmse", "ate_over_full", "rpe_rmse", "speedup_over_full", "size_reduction_over_full", "map_size_mb"]

    _write_csv(output_root / "table_replica_main.csv", replica_main, replica_fields)
    _write_csv(output_root / "table_replica_ablation.csv", ablation, ablation_fields)
    _write_csv(output_root / "table_tum_main.csv", tum_table, tum_fields)
    _write_markdown_table(output_root / "table_replica_main.md", "Replica Main Results", replica_main, replica_fields)
    _write_markdown_table(output_root / "table_replica_ablation.md", "Replica Coverage-Saliency Ablation", ablation, ablation_fields)
    _write_markdown_table(output_root / "table_tum_main.md", "TUM RGB-D Main Results", tum_table, tum_fields)

    if Path(quantization).exists():
        quant_rows = _read_rows(quantization)
        quant_fields = list(quant_rows[0].keys()) if quant_rows else []
        if quant_fields:
            _write_markdown_table(output_root / "table_quantization.md", "Safe Quantization Ablation", quant_rows, quant_fields)
    if make_plots:
        _plot_replica(replica_rows, output_root)
    return {
        "output_dir": str(output_root),
        "replica_main": str(output_root / "table_replica_main.csv"),
        "replica_ablation": str(output_root / "table_replica_ablation.csv"),
        "tum_main": str(output_root / "table_tum_main.csv"),
    }


def main() -> None:
    args = parse_args()
    outputs = generate_artifacts(
        replica_pareto=args.replica_pareto,
        tum_pareto=args.tum_pareto,
        quantization=args.quantization,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
