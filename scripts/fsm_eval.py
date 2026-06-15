import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from scripts.fsm_export import export_fsm_maps
from scripts.fsm_fisher import compute_and_save_fisher
from scripts.fsm_localize import load_config, run_localization
from utils.fsm_dataset import load_dataset_from_config


SUMMARY_ALL_FIELDS = [
    "dataset",
    "sequence",
    "run_name",
    "strategy",
    "compression_ratio",
    "N_full",
    "N_star",
    "lambda_min",
    "delta_trace",
    "delta_spectral",
    "trace_inv_full",
    "pred_delta",
    "residual_valid",
    "sample_retained_count",
    "sample_pruned_count",
    "ate_rmse",
    "rpe_rmse",
    "fitness",
    "registration_rmse",
    "latency_mean_ms",
    "latency_p95_ms",
    "memory_mb",
    "map_size_mb",
    "backend",
    "voxel_size",
    "max_corr_dist",
    "init_mode",
    "seed",
]

TASK_PLAN_FIELDS = [
    "task_id",
    "strategy",
    "compression_ratio",
    "map_dir",
    "map_path",
    "localize_dir",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run or plan FSM batch evaluation experiments.")
    parser.add_argument("--config", required=True, help="Path to FSM config.")
    parser.add_argument("--params", required=True, help="Path to SplaTAM params.npz.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--strategies", nargs="+", default=None, help="Pruning strategies.")
    parser.add_argument("--compression-ratios", nargs="+", type=float, default=None, help="Compression ratios.")
    parser.add_argument("--start", type=int, default=None, help="Start frame index, forwarded to localization.")
    parser.add_argument("--end", type=int, default=None, help="End frame index, forwarded to localization.")
    parser.add_argument("--dry-run", action="store_true", help="Only write task plan and empty summary schema.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip localization tasks with existing summary.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite exported map variant directories.")
    parser.add_argument("--fisher-max-points", type=int, default=200000, help="Deterministic point subsample used for Fisher on large maps.")
    parser.add_argument("--fisher-seed", type=int, default=None, help="Seed for Fisher subsampling. Defaults to config seed + 10007.")
    return parser.parse_args()


def _format_ratio(ratio: float) -> str:
    return f"{ratio:.6g}".replace(".", "p")


def variant_name(strategy: str, compression_ratio: float) -> str:
    return f"{strategy}_ratio_{_format_ratio(float(compression_ratio))}"


def _write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_task_plan(
    output_dir: str,
    strategies: list[str],
    compression_ratios: list[float],
) -> list[dict[str, str]]:
    output_root = Path(output_dir)
    rows = []
    task_id = 0
    for strategy in strategies:
        for ratio in compression_ratios:
            name = variant_name(strategy, ratio)
            rows.append(
                {
                    "task_id": str(task_id),
                    "strategy": strategy,
                    "compression_ratio": f"{float(ratio):.12g}",
                    "map_dir": str(output_root / "maps" / name),
                    "map_path": str(output_root / "maps" / name / "fsm_map.npz"),
                    "localize_dir": str(output_root / "localize" / name),
                    "status": "planned",
                }
            )
            task_id += 1
    return rows


def empty_summary_row(
    config: dict[str, Any],
    params_path: str,
    strategy: str,
    compression_ratio: str | float,
) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    registration_cfg = config.get("registration", {})
    return {
        "dataset": dataset_cfg.get("dataset_name", dataset_cfg.get("name", "")),
        "sequence": dataset_cfg.get("sequence", ""),
        "run_name": Path(params_path).parent.name,
        "strategy": strategy,
        "compression_ratio": str(compression_ratio),
        "N_full": "",
        "N_star": "",
        "lambda_min": "",
        "delta_trace": "",
        "delta_spectral": "",
        "trace_inv_full": "",
        "pred_delta": "",
        "residual_valid": "",
        "sample_retained_count": "",
        "sample_pruned_count": "",
        "ate_rmse": "",
        "rpe_rmse": "",
        "fitness": "",
        "registration_rmse": "",
        "latency_mean_ms": "",
        "latency_p95_ms": "",
        "memory_mb": "",
        "map_size_mb": "",
        "backend": registration_cfg.get("backend", ""),
        "voxel_size": registration_cfg.get("voxel_size", ""),
        "max_corr_dist": registration_cfg.get("max_correspondence_distance", ""),
        "init_mode": registration_cfg.get("init_mode", ""),
        "seed": config.get("seed", ""),
    }


def _load_npz_scalar(data: Any, key: str, default: Any = "") -> Any:
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return value


def _summary_from_outputs(
    config: dict[str, Any],
    params_path: str,
    task: dict[str, str],
    export_row: dict[str, str],
    localize_summary: dict[str, Any],
    fisher_path: str | Path,
) -> dict[str, Any]:
    row = empty_summary_row(config, params_path, task["strategy"], task["compression_ratio"])
    fisher = np.load(fisher_path, allow_pickle=True)
    map_path = Path(task["map_path"])
    row.update(
        {
            "N_full": export_row.get("N_full", ""),
            "N_star": export_row.get("N_star", ""),
            "lambda_min": _load_npz_scalar(fisher, "lambda_min", ""),
            "delta_trace": _load_npz_scalar(fisher, "delta_trace", ""),
            "delta_spectral": _load_npz_scalar(fisher, "delta_spectral", ""),
            "trace_inv_full": _load_npz_scalar(fisher, "trace_inv_full", ""),
            "pred_delta": _load_npz_scalar(fisher, "pred_delta_spectral", ""),
            "residual_valid": _load_npz_scalar(fisher, "residual_valid", ""),
            "sample_retained_count": _load_npz_scalar(fisher, "sample_retained_count", ""),
            "sample_pruned_count": _load_npz_scalar(fisher, "sample_pruned_count", ""),
            "ate_rmse": localize_summary.get("ate_rmse", ""),
            "rpe_rmse": localize_summary.get("rpe_trans_rmse", ""),
            "latency_mean_ms": localize_summary.get("latency_mean_ms", ""),
            "latency_p95_ms": localize_summary.get("latency_p95_ms", ""),
            "map_size_mb": f"{map_path.stat().st_size / (1024 * 1024):.12g}" if map_path.exists() else "",
        }
    )
    frame_metrics = Path(task["localize_dir"]) / "frame_metrics.csv"
    if frame_metrics.exists():
        with open(frame_metrics, newline="") as f:
            frames = list(csv.DictReader(f))
        if frames:
            row["fitness"] = np.mean([float(frame["fitness"]) for frame in frames])
            row["registration_rmse"] = np.mean([float(frame["inlier_rmse"]) for frame in frames])
    return row


def run_eval(
    config_path: str,
    params_path: str,
    output_dir: str,
    strategies: list[str] | None = None,
    compression_ratios: list[float] | None = None,
    start: int | None = None,
    end: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
    fisher_max_points: int | None = 200000,
    fisher_seed: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    strategies = strategies or config.get("fsm", {}).get("strategies", ["full"])
    compression_ratios = compression_ratios or config.get("fsm", {}).get("compression_ratios", [1.0])
    fisher_seed = int(config.get("seed", 0)) + 10007 if fisher_seed is None else int(fisher_seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task_plan = build_task_plan(str(output_root), strategies, compression_ratios)
    _write_csv(output_root / "task_plan.csv", TASK_PLAN_FIELDS, task_plan)

    if dry_run:
        summary_rows = [empty_summary_row(config, params_path, task["strategy"], task["compression_ratio"]) for task in task_plan]
        _write_csv(output_root / "summary_all.csv", SUMMARY_ALL_FIELDS, summary_rows)
        return {"tasks": task_plan, "summary_rows": summary_rows, "dry_run": True}

    print(f"[fsm_eval] exporting {len(task_plan)} FSM map variants to {output_root / 'maps'}", flush=True)
    export_rows = export_fsm_maps(
        params_path=params_path,
        output_dir=str(output_root / "maps"),
        strategies=strategies,
        compression_ratios=compression_ratios,
        seed=int(config.get("seed", 0)),
        overwrite=overwrite,
    )
    export_by_map = {row["map_path"]: row for row in export_rows}

    fisher_full_path = output_root / "fisher" / "fisher_scores.npz"
    print(f"[fsm_eval] computing full-map Fisher statistics at {fisher_full_path}", flush=True)
    compute_and_save_fisher(
        str(fisher_full_path),
        params_path=params_path,
        max_points=fisher_max_points,
        seed=fisher_seed,
    )
    print("[fsm_eval] loading dataset", flush=True)
    dataset = load_dataset_from_config(config_path)

    summary_rows = []
    for task in task_plan:
        print(f"[fsm_eval] running localization for {task['strategy']} ratio={task['compression_ratio']}", flush=True)
        localize_dir = Path(task["localize_dir"])
        local_summary_path = localize_dir / "summary.json"
        if skip_existing and local_summary_path.exists():
            localize_summary = _read_json(local_summary_path)
        else:
            localize_summary = run_localization(
                config=config,
                map_path=task["map_path"],
                output_dir=str(localize_dir),
                dataset=dataset,
                start=start,
                end=end,
            )
        retained_fisher_path = output_root / "fisher" / f"{variant_name(task['strategy'], float(task['compression_ratio']))}.npz"
        print(f"[fsm_eval] computing retained Fisher residuals at {retained_fisher_path}", flush=True)
        compute_and_save_fisher(
            str(retained_fisher_path),
            params_path=params_path,
            retained_indices=np.load(task["map_path"], allow_pickle=True)["indices"],
            max_points=fisher_max_points,
            seed=fisher_seed,
        )
        summary_rows.append(
            _summary_from_outputs(
                config=config,
                params_path=params_path,
                task=task,
                export_row=export_by_map.get(task["map_path"], {}),
                localize_summary=localize_summary,
                fisher_path=retained_fisher_path,
            )
        )

    _write_csv(output_root / "summary_all.csv", SUMMARY_ALL_FIELDS, summary_rows)
    return {"tasks": task_plan, "summary_rows": summary_rows, "dry_run": False}


def main() -> None:
    args = parse_args()
    result = run_eval(
        config_path=args.config,
        params_path=args.params,
        output_dir=args.output_dir,
        strategies=args.strategies,
        compression_ratios=args.compression_ratios,
        start=args.start,
        end=args.end,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        fisher_max_points=args.fisher_max_points,
        fisher_seed=args.fisher_seed,
    )
    mode = "planned" if result["dry_run"] else "completed"
    print(f"FSM eval {mode}: {len(result['tasks'])} tasks, output={args.output_dir}")


if __name__ == "__main__":
    main()
