import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from scripts.fsm_localize import load_config
from utils.fsm_dataset import get_frame, load_dataset_from_config
from utils.fsm_map import load_fsm_map, load_splatam_params
from utils.fsm_pruning import score_opacity


FIELDS = [
    "scene",
    "method",
    "strategy",
    "compression_ratio",
    "num_selected",
    "num_full",
    "occupied_voxels",
    "occupied_voxel_ratio",
    "coverage_entropy",
    "mean_nn_distance",
    "median_nn_distance",
    "scale_mean",
    "scale_median",
    "scale_p90",
    "opacity_mean",
    "opacity_median",
    "trajectory_distance_mean",
    "trajectory_distance_median",
    "map_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze coverage, saliency, and trajectory proximity of FSM skeleton maps.")
    parser.add_argument("--scene", required=True, help="Scene label written to CSV.")
    parser.add_argument("--config", required=True, help="FSM config path used to load trajectory poses.")
    parser.add_argument("--params", required=True, help="Full SplaTAM params.npz path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for occupancy statistics in world units.")
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument("--trajectory-end", type=int, default=500)
    parser.add_argument("--trajectory-every", type=int, default=10)
    parser.add_argument(
        "--entry",
        nargs=3,
        action="append",
        metavar=("METHOD", "STRATEGY", "FSM_MAP"),
        required=True,
        help="One map to analyze: display method, pruning strategy, fsm_map.npz.",
    )
    return parser.parse_args()


def _decode_scalar(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(array)


def _coverage_stats(points: np.ndarray, voxel_size: float) -> dict[str, float]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    grid = np.floor(points / float(voxel_size)).astype(np.int64)
    _, counts = np.unique(grid, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    return {
        "occupied_voxels": int(counts.size),
        "coverage_entropy": entropy,
    }


def _sample_rows(values: np.ndarray, max_rows: int, seed: int = 0) -> np.ndarray:
    if values.shape[0] <= max_rows:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(values.shape[0], size=max_rows, replace=False))
    return values[indices]


def _nearest_distance_stats(points: np.ndarray, max_points: int = 5000) -> dict[str, float]:
    sampled = _sample_rows(np.asarray(points, dtype=np.float64), max_rows=max_points)
    if sampled.shape[0] < 2:
        return {"mean_nn_distance": float("nan"), "median_nn_distance": float("nan")}
    diff = sampled[:, None, :] - sampled[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return {
        "mean_nn_distance": float(np.mean(nearest)),
        "median_nn_distance": float(np.median(nearest)),
    }


def _trajectory_points(config_path: str, start: int, end: int, every: int) -> np.ndarray:
    if every <= 0:
        raise ValueError("trajectory_every must be positive")
    dataset = load_dataset_from_config(config_path)
    selected = range(int(start), min(int(end), len(dataset)), int(every))
    points = [get_frame(dataset, idx)["c2w"][:3, 3] for idx in selected]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _trajectory_distance_stats(points: np.ndarray, trajectory: np.ndarray, max_points: int = 10000) -> dict[str, float]:
    if trajectory.size == 0:
        return {"trajectory_distance_mean": float("nan"), "trajectory_distance_median": float("nan")}
    sampled = _sample_rows(np.asarray(points, dtype=np.float64), max_rows=max_points, seed=1)
    diff = sampled[:, None, :] - trajectory[None, :, :]
    nearest = np.min(np.linalg.norm(diff, axis=2), axis=1)
    return {
        "trajectory_distance_mean": float(np.mean(nearest)),
        "trajectory_distance_median": float(np.median(nearest)),
    }


def build_analysis_rows(
    scene: str,
    config_path: str,
    params_path: str,
    entries: list[list[str]],
    voxel_size: float = 0.1,
    trajectory_start: int = 0,
    trajectory_end: int = 500,
    trajectory_every: int = 10,
) -> list[dict[str, Any]]:
    params = load_splatam_params(params_path)
    num_full = int(np.asarray(params["means3D"]).shape[0])
    opacity = score_opacity(params) if "logit_opacities" in params else np.full(num_full, np.nan, dtype=np.float64)
    trajectory = _trajectory_points(config_path, start=trajectory_start, end=trajectory_end, every=trajectory_every)

    rows = []
    for method, strategy, map_path in entries:
        fsm_map = load_fsm_map(map_path)
        points = np.asarray(fsm_map["means"], dtype=np.float64)
        scales = np.linalg.norm(np.asarray(fsm_map["scales"], dtype=np.float64), axis=1)
        indices = np.asarray(fsm_map["indices"], dtype=np.int64)
        coverage = _coverage_stats(points, voxel_size=voxel_size)
        nn_stats = _nearest_distance_stats(points)
        traj_stats = _trajectory_distance_stats(points, trajectory)
        selected_opacity = opacity[indices]
        rows.append(
            {
                "scene": scene,
                "method": method,
                "strategy": strategy,
                "compression_ratio": _decode_scalar(fsm_map["compression_ratio"]),
                "num_selected": int(points.shape[0]),
                "num_full": num_full,
                "occupied_voxels": coverage["occupied_voxels"],
                "occupied_voxel_ratio": f"{coverage['occupied_voxels'] / max(points.shape[0], 1):.12g}",
                "coverage_entropy": f"{coverage['coverage_entropy']:.12g}",
                "mean_nn_distance": f"{nn_stats['mean_nn_distance']:.12g}",
                "median_nn_distance": f"{nn_stats['median_nn_distance']:.12g}",
                "scale_mean": f"{float(np.mean(scales)):.12g}",
                "scale_median": f"{float(np.median(scales)):.12g}",
                "scale_p90": f"{float(np.percentile(scales, 90)):.12g}",
                "opacity_mean": f"{float(np.nanmean(selected_opacity)):.12g}",
                "opacity_median": f"{float(np.nanmedian(selected_opacity)):.12g}",
                "trajectory_distance_mean": f"{traj_stats['trajectory_distance_mean']:.12g}",
                "trajectory_distance_median": f"{traj_stats['trajectory_distance_median']:.12g}",
                "map_path": str(map_path),
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
    load_config(args.config)
    rows = build_analysis_rows(
        scene=args.scene,
        config_path=args.config,
        params_path=args.params,
        entries=args.entry,
        voxel_size=args.voxel_size,
        trajectory_start=args.trajectory_start,
        trajectory_end=args.trajectory_end,
        trajectory_every=args.trajectory_every,
    )
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} skeleton analysis rows to {args.output}")


if __name__ == "__main__":
    main()
