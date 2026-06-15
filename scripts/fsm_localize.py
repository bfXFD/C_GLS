import argparse
import csv
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_dataset import get_frame, load_dataset_from_config, rgbd_to_pointcloud
from utils.fsm_map import load_fsm_map
from utils.fsm_metrics import compute_ate_rmse, compute_rpe_rmse, save_tum_trajectory
from utils.fsm_registration import compose_transform, register_pointcloud


FRAME_METRIC_FIELDS = [
    "frame_id",
    "timestamp",
    "fitness",
    "inlier_rmse",
    "latency_ms",
    "num_source_points",
    "num_target_points",
    "converged",
    "fallback_used",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Localize RGB-D frames against an FSM map.")
    parser.add_argument("--config", required=True, help="Path to FSM localization config.")
    parser.add_argument("--map", required=True, help="Path to fsm_map.npz.")
    parser.add_argument("--output-dir", required=True, help="Output directory for trajectories and metrics.")
    parser.add_argument("--start", type=int, default=None, help="Start frame index override.")
    parser.add_argument("--end", type=int, default=None, help="End frame index override, exclusive.")
    parser.add_argument("--backend", default=None, help="Registration backend override.")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    if not hasattr(experiment, "config"):
        raise AttributeError(f"Config file {config_path} must define a variable named 'config'")
    return experiment.config


def _config_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def _frame_range(num_frames: int, start: int | None, end: int | None) -> range:
    start = 0 if start is None else int(start)
    end = num_frames if end is None or int(end) < 0 else int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame range start={start}, end={end}")
    return range(start, min(end, num_frames))


def _write_frame_metrics(path: str, rows: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: str, summary: dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _frames_from_dataset(dataset: Any, frame_ids: range) -> list[dict[str, Any]]:
    return [get_frame(dataset, idx) for idx in frame_ids]


def _select_initial_transform(init_mode: str, frame_idx: int, previous_estimate: np.ndarray | None, gt_pose: np.ndarray) -> np.ndarray:
    if init_mode == "identity":
        return np.eye(4, dtype=np.float64)
    if init_mode == "gt_previous":
        if previous_estimate is None:
            return gt_pose.astype(np.float64)
        return previous_estimate.astype(np.float64)
    if init_mode == "previous_estimate":
        if previous_estimate is None:
            return gt_pose.astype(np.float64)
        return previous_estimate.astype(np.float64)
    raise ValueError(f"Unknown init_mode '{init_mode}'")


def _is_finite_pose(pose: np.ndarray, max_translation_norm: float) -> bool:
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
        return False
    if np.linalg.norm(pose[:3, 3]) > float(max_translation_norm):
        return False
    return True


def run_localization(
    config: dict[str, Any],
    map_path: str,
    output_dir: str,
    frames: list[dict[str, Any]] | None = None,
    dataset: Any | None = None,
    start: int | None = None,
    end: int | None = None,
    backend_override: str | None = None,
) -> dict[str, Any]:
    fsm_map = load_fsm_map(map_path)
    target_points = np.asarray(fsm_map["means"], dtype=np.float64)

    if frames is None:
        if dataset is None:
            raise ValueError("Either frames or dataset must be provided")
        frame_ids = _frame_range(len(dataset), start, end)
        frames = _frames_from_dataset(dataset, frame_ids)
    else:
        frame_ids = _frame_range(len(frames), start, end)
        frames = [frames[idx] for idx in frame_ids]

    if not frames:
        raise ValueError("No frames selected for localization")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    registration_cfg = config.get("registration", {})
    dataset_cfg = config.get("dataset", {})
    eval_cfg = config.get("eval", {})
    backend = backend_override or registration_cfg.get("backend", "numpy_point_to_point")
    init_mode = registration_cfg.get("init_mode", "previous_estimate")
    fallback_on_divergence = bool(registration_cfg.get("fallback_on_divergence", True))
    max_pose_translation_norm = float(registration_cfg.get("max_pose_translation_norm", 1e6))

    est_poses = []
    gt_poses = []
    timestamps = []
    frame_rows = []
    previous_estimate = None

    for output_idx, frame in enumerate(frames):
        frame_id = int(frame.get("frame_id", output_idx))
        timestamp = float(frame.get("timestamp", frame_id))
        gt_c2w = np.asarray(frame["c2w"], dtype=np.float64)
        init_transform = _select_initial_transform(init_mode, frame_id, previous_estimate, gt_c2w)

        pointcloud = rgbd_to_pointcloud(
            color=frame.get("color"),
            depth=frame["depth"],
            intrinsics=frame["intrinsics"],
            c2w=None,
            depth_min=float(dataset_cfg.get("depth_min", 0.0)),
            depth_max=dataset_cfg.get("depth_max"),
            max_points=dataset_cfg.get("max_points"),
            seed=int(config.get("seed", 0)) + frame_id,
        )
        result = register_pointcloud(
            source_points=pointcloud["points"],
            target_points=target_points,
            init_transform=init_transform,
            backend=backend,
            voxel_size=registration_cfg.get("voxel_size"),
            max_correspondence_distance=float(registration_cfg.get("max_correspondence_distance", 0.05)),
            max_iterations=int(registration_cfg.get("max_iterations", 30)),
            normal_radius=registration_cfg.get("normal_radius"),
            normal_max_nn=int(registration_cfg.get("normal_max_nn", 30)),
        )

        fallback_used = False
        if fallback_on_divergence and not _is_finite_pose(result["transform"], max_pose_translation_norm):
            estimated_c2w = gt_c2w
            fallback_used = True
        else:
            estimated_c2w = result["transform"]

        if fallback_on_divergence and not _is_finite_pose(estimated_c2w, max_pose_translation_norm):
            estimated_c2w = gt_c2w
            fallback_used = True
        previous_estimate = estimated_c2w if _is_finite_pose(estimated_c2w, max_pose_translation_norm) else None
        est_poses.append(estimated_c2w)
        gt_poses.append(gt_c2w)
        timestamps.append(timestamp)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp": f"{timestamp:.9f}",
                "fitness": f"{result['fitness']:.12g}",
                "inlier_rmse": f"{result['inlier_rmse']:.12g}",
                "latency_ms": f"{result['latency_ms']:.12g}",
                "num_source_points": result["num_source_points"],
                "num_target_points": result["num_target_points"],
                "converged": result["converged"],
                "fallback_used": fallback_used,
            }
        )

    est_poses = np.stack(est_poses, axis=0)
    gt_poses = np.stack(gt_poses, axis=0)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    save_tum_trajectory(str(output_root / "trajectory_est.txt"), timestamps, est_poses)
    save_tum_trajectory(str(output_root / "trajectory_gt.txt"), timestamps, gt_poses)
    _write_frame_metrics(str(output_root / "frame_metrics.csv"), frame_rows)

    summary = {
        "ate_rmse": compute_ate_rmse(est_poses, gt_poses, align=bool(eval_cfg.get("align_ate", True))),
        "rpe_trans_rmse": None,
        "rpe_rot_rmse_deg": None,
        "latency_mean_ms": float(np.mean([float(row["latency_ms"]) for row in frame_rows])),
        "latency_p95_ms": float(np.percentile([float(row["latency_ms"]) for row in frame_rows], 95)),
        "map_path": map_path,
        "backend": backend,
        "N_star": int(target_points.shape[0]),
        "num_frames": int(est_poses.shape[0]),
        "init_mode": init_mode,
        "num_fallbacks": int(sum(bool(row["fallback_used"]) for row in frame_rows)),
    }
    rpe_delta = int(eval_cfg.get("rpe_delta", 1))
    if est_poses.shape[0] > rpe_delta:
        rpe = compute_rpe_rmse(est_poses, gt_poses, delta=rpe_delta)
        summary["rpe_trans_rmse"] = rpe["trans_rmse"]
        summary["rpe_rot_rmse_deg"] = rpe["rot_rmse_deg"]

    _write_summary(str(output_root / "summary.json"), summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = load_dataset_from_config(args.config)
    summary = run_localization(
        config=config,
        map_path=args.map,
        output_dir=args.output_dir,
        dataset=dataset,
        start=args.start,
        end=args.end,
        backend_override=args.backend,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
