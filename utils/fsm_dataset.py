import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Any

import numpy as np


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth with shape [H, W], [H, W, 1], or [1, H, W], got {depth.shape}")
    return depth.astype(np.float64, copy=False)


def _normalize_color(color: np.ndarray | None, height: int, width: int) -> np.ndarray | None:
    if color is None:
        return None
    color = np.asarray(color)
    if color.ndim == 3 and color.shape[0] in (3, 4) and color.shape[1] == height and color.shape[2] == width:
        color = np.transpose(color, (1, 2, 0))
    if color.ndim != 3 or color.shape[0] != height or color.shape[1] != width:
        raise ValueError(f"Expected color with shape [H, W, C] or [C, H, W], got {color.shape}")
    if color.shape[2] not in (3, 4):
        raise ValueError(f"Expected color with 3 or 4 channels, got {color.shape[2]}")
    color = color[..., :3]
    if np.issubdtype(color.dtype, np.integer):
        color = color.astype(np.float32) / 255.0
    else:
        color = color.astype(np.float32, copy=False)
    return color


def _normalize_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape[0] < 3 or intrinsics.shape[1] < 3:
        raise ValueError(f"Expected intrinsics with at least shape [3, 3], got {intrinsics.shape}")
    return intrinsics[:3, :3]


def _validate_transform(transform: np.ndarray | None, name: str) -> np.ndarray | None:
    if transform is None:
        return None
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected {name} with shape [4, 4], got {transform.shape}")
    return transform


def _valid_depth_mask(
    depth: np.ndarray,
    mask: np.ndarray | None,
    depth_min: float,
    depth_max: float | None,
) -> np.ndarray:
    if depth_min < 0:
        raise ValueError("depth_min must be non-negative")
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        if depth_max <= depth_min:
            raise ValueError("depth_max must be greater than depth_min")
        valid &= depth <= depth_max
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth.shape:
            raise ValueError(f"Expected mask with shape {depth.shape}, got {mask.shape}")
        valid &= mask.astype(bool)
    return valid


def backproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray | None = None,
    depth_min: float = 0.0,
    depth_max: float | None = None,
) -> np.ndarray:
    depth = _squeeze_depth(depth)
    intrinsics = _normalize_intrinsics(intrinsics)
    valid = _valid_depth_mask(depth, mask=mask, depth_min=depth_min, depth_max=depth_max)

    ys, xs = np.nonzero(valid)
    z = depth[ys, xs]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("intrinsics fx and fy must be non-zero")

    x = (xs.astype(np.float64) - cx) / fx * z
    y = (ys.astype(np.float64) - cy) / fy * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _sample_indices(num_points: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=int(max_points), replace=False)).astype(np.int64)


def rgbd_to_pointcloud(
    color: np.ndarray | None,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray | None = None,
    depth_min: float = 0.0,
    depth_max: float | None = None,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    depth = _squeeze_depth(depth)
    height, width = depth.shape
    intrinsics = _normalize_intrinsics(intrinsics)
    c2w = _validate_transform(c2w, "c2w")
    color = _normalize_color(color, height=height, width=width)

    valid = _valid_depth_mask(depth, mask=None, depth_min=depth_min, depth_max=depth_max)
    pixel_indices = np.flatnonzero(valid.reshape(-1)).astype(np.int64)
    points = backproject_depth(depth, intrinsics, mask=valid, depth_min=depth_min, depth_max=depth_max)

    if c2w is not None and points.size > 0:
        points_h = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
        points = (c2w @ points_h.T).T[:, :3].astype(np.float32)

    colors = None
    if color is not None:
        flat_colors = color.reshape(-1, 3)
        colors = flat_colors[pixel_indices]

    keep = _sample_indices(points.shape[0], max_points=max_points, seed=seed)
    result = {
        "points": points[keep],
        "colors": None if colors is None else colors[keep].astype(np.float32),
        "indices": pixel_indices[keep],
        "intrinsics": intrinsics.astype(np.float32),
        "c2w": None if c2w is None else c2w.astype(np.float32),
    }
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_frame(dataset: Any, frame_idx: int) -> dict[str, Any]:
    color, depth, intrinsics, pose = dataset[frame_idx]
    color = _to_numpy(color)
    depth = _to_numpy(depth)
    intrinsics = _normalize_intrinsics(_to_numpy(intrinsics))
    c2w = _validate_transform(_to_numpy(pose), "pose")
    return {
        "frame_id": int(frame_idx),
        "color": color,
        "depth": depth,
        "intrinsics": intrinsics.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "w2c": np.linalg.inv(c2w).astype(np.float32),
    }


def load_dataset_from_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from datasets.gradslam_datasets import load_dataset_config
    from scripts.splatam import get_dataset

    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    config = experiment.config
    data_config = config.get("data", config.get("dataset"))
    if data_config is None:
        raise KeyError("FSM config must contain a 'data' or 'dataset' section")
    if "gradslam_data_cfg" in data_config:
        gradslam_data_cfg = load_dataset_config(data_config["gradslam_data_cfg"])
    else:
        gradslam_data_cfg = {"dataset_name": data_config["dataset_name"]}
    return get_dataset(
        gradslam_data_cfg,
        basedir=data_config["basedir"],
        sequence=data_config["sequence"],
        start=data_config.get("start", 0),
        end=data_config.get("end", -1),
        stride=data_config.get("stride", 1),
        desired_height=data_config.get("desired_image_height"),
        desired_width=data_config.get("desired_image_width"),
        device=config.get("primary_device", "cpu"),
        relative_pose=True,
        ignore_bad=False,
        use_train_split=True,
    )
