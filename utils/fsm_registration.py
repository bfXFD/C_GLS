import time
from typing import Any

import numpy as np


SUPPORTED_BACKENDS = ("numpy_point_to_point", "open3d_point_to_point", "open3d_point_to_plane")


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for ICP registration backends") from exc
    return o3d


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected {name} with shape [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _validate_transform(transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected init_transform with shape [4, 4], got {transform.shape}")
    if np.any(~np.isfinite(transform)):
        raise ValueError("init_transform must contain only finite values")
    return transform


def compose_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _validate_transform(a)
    b = _validate_transform(b)
    return (a @ b).astype(np.float64)


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = _validate_transform(T)
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = _validate_points(points, "points")
    transform = _validate_transform(transform)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def _nearest_neighbor_indices(
    source: np.ndarray,
    target: np.ndarray,
    target_tree: Any | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if target_tree is not None:
        distances, indices = target_tree.query(source, k=1, workers=-1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)

    indices = np.empty(source.shape[0], dtype=np.int64)
    distances = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), source.shape[0])
        diff = source[start:end, None, :] - target[None, :, :]
        squared_distances = np.sum(diff * diff, axis=2)
        local_indices = np.argmin(squared_distances, axis=1)
        indices[start:end] = local_indices
        distances[start:end] = np.sqrt(squared_distances[np.arange(end - start), local_indices])
    return indices.astype(np.int64), distances


def _best_fit_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _is_valid_transform(transform: np.ndarray, max_translation_norm: float = 1e6) -> bool:
    transform = np.asarray(transform)
    if transform.shape != (4, 4) or np.any(~np.isfinite(transform)):
        return False
    if np.linalg.norm(transform[:3, 3]) > max_translation_norm:
        return False
    return True


def _register_numpy_point_to_point(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    max_correspondence_distance: float,
    max_iterations: int,
) -> dict[str, Any]:
    transform = init_transform.copy()
    inlier_rmse = float("inf")
    fitness = 0.0
    try:
        from scipy.spatial import cKDTree

        target_tree = cKDTree(target_points)
    except Exception:
        target_tree = None

    for _ in range(max_iterations):
        transformed = transform_points(source_points, transform)
        nn_indices, distances = _nearest_neighbor_indices(transformed, target_points, target_tree=target_tree)
        inlier_mask = distances <= max_correspondence_distance
        if np.count_nonzero(inlier_mask) < 3:
            break

        src_corr = transformed[inlier_mask]
        tgt_corr = target_points[nn_indices[inlier_mask]]
        delta = _best_fit_transform(src_corr, tgt_corr)
        if not _is_valid_transform(delta):
            break
        next_transform = delta @ transform
        if not _is_valid_transform(next_transform):
            break
        transform = next_transform

        prev_rmse = inlier_rmse
        inlier_rmse = float(np.sqrt(np.mean(distances[inlier_mask] ** 2)))
        fitness = float(np.count_nonzero(inlier_mask) / source_points.shape[0])
        if abs(prev_rmse - inlier_rmse) < 1e-9:
            break

    return {
        "transform": transform,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "backend": "numpy_point_to_point",
        "converged": bool(np.isfinite(inlier_rmse) and fitness > 0.0),
        "num_source_points": int(source_points.shape[0]),
        "num_target_points": int(target_points.shape[0]),
    }


def _make_point_cloud(points: np.ndarray):
    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def estimate_normals(points: np.ndarray, radius: float = 0.05, max_nn: int = 20) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    if max_nn <= 0:
        raise ValueError("max_nn must be positive")
    points = _validate_points(points, "points")
    o3d = _require_open3d()
    pcd = _make_point_cloud(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape != points.shape:
        raise RuntimeError("Open3D did not return one normal per point")
    return normals


def _voxel_downsample(pcd: Any, voxel_size: float | None):
    if voxel_size is None:
        return pcd
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return pcd.voxel_down_sample(float(voxel_size))


def register_pointcloud(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray | None = None,
    backend: str = "open3d_point_to_plane",
    voxel_size: float | None = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 30,
    normal_radius: float | None = None,
    normal_max_nn: int = 30,
) -> dict[str, Any]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown registration backend '{backend}'. Supported: {SUPPORTED_BACKENDS}")
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    source_points = _validate_points(source_points, "source_points")
    target_points = _validate_points(target_points, "target_points")
    init_transform = _validate_transform(init_transform)

    if backend == "numpy_point_to_point":
        start = time.perf_counter()
        result = _register_numpy_point_to_point(
            source_points=source_points,
            target_points=target_points,
            init_transform=init_transform,
            max_correspondence_distance=float(max_correspondence_distance),
            max_iterations=int(max_iterations),
        )
        result["latency_ms"] = float((time.perf_counter() - start) * 1000.0)
        return result

    o3d = _require_open3d()
    source = _voxel_downsample(_make_point_cloud(source_points), voxel_size)
    target = _voxel_downsample(_make_point_cloud(target_points), voxel_size)

    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("source and target point clouds must remain non-empty after downsampling")

    if backend == "open3d_point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        radius = float(normal_radius if normal_radius is not None else max_correspondence_distance * 2.0)
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(normal_max_nn)))
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(normal_max_nn)))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iterations))
    start = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(max_correspondence_distance),
        init_transform,
        estimation,
        criteria,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    transform = np.asarray(result.transformation, dtype=np.float64)
    return {
        "transform": transform,
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "latency_ms": float(latency_ms),
        "backend": backend,
        "converged": bool(np.isfinite(result.inlier_rmse) and result.fitness > 0.0),
        "num_source_points": int(len(source.points)),
        "num_target_points": int(len(target.points)),
    }
