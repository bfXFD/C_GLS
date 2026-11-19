import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Any

import numpy as np


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tum_first_associated_pose(sequence_dir: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    def load_list(name: str, skiprows: int = 0) -> np.ndarray:
        return np.atleast_2d(np.loadtxt(
            os.path.join(sequence_dir, name), delimiter=" ", dtype=np.str_,
            skiprows=skiprows, comments="#",
        ))

    pose_name = "groundtruth.txt" if os.path.isfile(os.path.join(sequence_dir, "groundtruth.txt")) else "pose.txt"
    rgb = load_list("rgb.txt")
    depth = load_list("depth.txt")
    poses = load_list(pose_name, skiprows=1)
    rgb_t = rgb[:, 0].astype(np.float64)
    depth_t = depth[:, 0].astype(np.float64)
    pose_t = poses[:, 0].astype(np.float64)
    for timestamp in rgb_t:
        depth_index = int(np.argmin(np.abs(depth_t - timestamp)))
        pose_index = int(np.argmin(np.abs(pose_t - timestamp)))
        if abs(depth_t[depth_index] - timestamp) < 0.08 and abs(pose_t[pose_index] - timestamp) < 0.08:
            values = poses[pose_index, 1:].astype(np.float64)
            result = np.eye(4, dtype=np.float64)
            result[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
            result[:3, 3] = values[:3]
            return result
    raise ValueError(f"No associated RGB-D/GT frames in {sequence_dir}")


def dataset_pose_provenance(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", config.get("dataset", {}))
    dataset_name = str(data.get("dataset_name", "")).lower()
    sequence = data.get("sequence")
    reference = data.get("pose_reference_sequence")
    result = {
        "dataset": dataset_name,
        "query_sequence": sequence,
        "pose_reference_sequence": reference,
        "pose_protocol": "first_query_relative" if reference is None else "frozen_reference_first_pose_v1",
    }
    if dataset_name != "tum" or not sequence:
        return result
    basedir = data["basedir"]
    query_dir = os.path.join(basedir, sequence)
    result["query_sources"] = {
        name: {"path": os.path.join(query_dir, name), "sha256": _sha256_file(os.path.join(query_dir, name))}
        for name in ("groundtruth.txt", "rgb.txt", "depth.txt")
    }
    if reference is None:
        return result
    reference_dir = os.path.join(basedir, reference)
    map0_abs = _tum_first_associated_pose(reference_dir)
    query0_abs = _tum_first_associated_pose(query_dir)
    transform = np.linalg.inv(map0_abs) @ query0_abs
    result.update(
        reference_sources={
            name: {"path": os.path.join(reference_dir, name), "sha256": _sha256_file(os.path.join(reference_dir, name))}
            for name in ("groundtruth.txt", "rgb.txt", "depth.txt")
        },
        map0_abs=map0_abs.tolist(),
        query0_abs=query0_abs.tolist(),
        fixed_frame_transform_A=transform.tolist(),
        transformed_pose_definition="A @ inv(P_query0_abs) @ P_query_abs = inv(P_map0_abs) @ P_query_abs",
        first_pose_protocol="query GT is used once to initialize A; later query GT is evaluation-only",
        map_construction_query_frame_count=0,
    )
    return result


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
        relative_pose=data_config.get("relative_pose", True),
        pose_reference_sequence=data_config.get("pose_reference_sequence"),
        ignore_bad=False,
        use_train_split=data_config.get("use_train_split", True),
        query_only_map_frame=data_config.get("query_only_map_frame", False),
    )
