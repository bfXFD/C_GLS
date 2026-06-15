from typing import Any

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"Expected vector with shape [3], got {v.shape}")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if np.any(~np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    return points


def _validate_normals(normals: np.ndarray, num_points: int) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != (num_points, 3):
        raise ValueError(f"Expected normals with shape [{num_points}, 3], got {normals.shape}")
    if np.any(~np.isfinite(normals)):
        raise ValueError("normals must contain only finite values")
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("normal vectors must be non-zero")
    return normals / norms


def _validate_weights(weights: np.ndarray | None, num_points: int) -> np.ndarray:
    if weights is None:
        return np.ones(num_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (num_points,):
        raise ValueError(f"Expected weights with shape [{num_points}], got {weights.shape}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    return weights


def point_to_plane_pose_jacobian(point: np.ndarray, normal: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if point.shape != (3,) or normal.shape != (3,):
        raise ValueError(f"Expected point and normal with shape [3], got {point.shape} and {normal.shape}")
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        raise ValueError("normal vector must be non-zero")
    normal = normal / normal_norm
    if R is None:
        R = np.eye(3, dtype=np.float64)
    else:
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"Expected R with shape [3, 3], got {R.shape}")

    rotation_part = normal @ (-R @ skew(point))
    translation_part = normal
    return np.concatenate([rotation_part, translation_part], axis=0).astype(np.float64)


def estimate_map_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    points = _validate_points(points)
    if k <= 2:
        raise ValueError("k must be greater than 2")
    num_points = points.shape[0]
    k = min(int(k), num_points)
    if k < 3:
        raise ValueError("At least three points are required to estimate normals")

    normals = np.zeros_like(points, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        _, all_neighbor_indices = tree.query(points, k=k, workers=-1)
        if k == 1:
            all_neighbor_indices = all_neighbor_indices[:, None]
    except Exception:
        all_neighbor_indices = None

    for idx, point in enumerate(points):
        if all_neighbor_indices is None:
            diff = points - point[None, :]
            distances = np.sum(diff * diff, axis=1)
            neighbor_indices = np.argpartition(distances, kth=k - 1)[:k]
        else:
            neighbor_indices = all_neighbor_indices[idx]
        neighborhood = points[neighbor_indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        covariance = centered.T @ centered / max(neighborhood.shape[0], 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        normal = eigvecs[:, np.argmin(eigvals)]
        if normal[2] < 0:
            normal = -normal
        normals[idx] = normal
    return _validate_normals(normals, num_points)


def compute_geometric_fisher(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    normal_k: int = 20,
) -> dict[str, Any]:
    points = _validate_points(points)
    num_points = points.shape[0]
    if normals is None:
        normals = estimate_map_normals(points, k=min(int(normal_k), num_points))
    else:
        normals = _validate_normals(normals, num_points)
    weights = _validate_weights(weights, num_points)

    cross_terms = np.cross(points, normals)
    jacobians = np.concatenate([cross_terms, normals], axis=1).astype(np.float64)
    per_gaussian_fim = weights[:, None, None] * jacobians[:, :, None] * jacobians[:, None, :]

    scores = np.trace(per_gaussian_fim, axis1=1, axis2=2)
    full_fim = np.sum(per_gaussian_fim, axis=0)
    eigvals = np.linalg.eigvalsh(full_fim)
    return {
        "per_gaussian_fim": per_gaussian_fim,
        "scores": scores,
        "full_fim": full_fim,
        "lambda_min": float(np.min(eigvals)),
        "eigvals": eigvals,
        "normals": normals,
        "jacobians": jacobians,
    }


def compute_pruning_residuals(
    per_gaussian_fim: np.ndarray,
    retained_indices: np.ndarray,
) -> dict[str, Any]:
    per_gaussian_fim = np.asarray(per_gaussian_fim, dtype=np.float64)
    if per_gaussian_fim.ndim != 3 or per_gaussian_fim.shape[1:] != (6, 6):
        raise ValueError(f"Expected per_gaussian_fim with shape [N, 6, 6], got {per_gaussian_fim.shape}")
    num_points = per_gaussian_fim.shape[0]
    retained_indices = np.asarray(retained_indices, dtype=np.int64)
    if retained_indices.ndim != 1:
        raise ValueError(f"Expected retained_indices with shape [N*], got {retained_indices.shape}")
    if retained_indices.size == 0:
        raise ValueError("retained_indices must not be empty")
    if np.any(retained_indices < 0) or np.any(retained_indices >= num_points):
        raise IndexError("retained_indices contain values outside the Gaussian range")
    if np.unique(retained_indices).size != retained_indices.size:
        raise ValueError("retained_indices must be unique")

    keep = np.zeros(num_points, dtype=bool)
    keep[retained_indices] = True
    pruned_indices = np.flatnonzero(~keep).astype(np.int64)
    full_fim = np.sum(per_gaussian_fim, axis=0)
    retained_fim = np.sum(per_gaussian_fim[retained_indices], axis=0)
    pruned_fim = full_fim - retained_fim
    return {
        "retained_fim": retained_fim,
        "pruned_fim": pruned_fim,
        "pruned_indices": pruned_indices,
        "delta_trace": float(np.trace(pruned_fim)),
        "delta_spectral": float(np.linalg.norm(pruned_fim, ord=2)),
        "full_fim": full_fim,
    }


def _safe_trace_inverse(matrix: np.ndarray, regularization: float = 1e-9) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError(f"Expected matrix with shape [6, 6], got {matrix.shape}")
    regularized = matrix + float(regularization) * np.eye(6, dtype=np.float64)
    return float(np.trace(np.linalg.pinv(regularized)))


def predict_error_increment(
    full_fim: np.ndarray,
    pruned_fim: np.ndarray,
    regularization: float = 1e-9,
) -> dict[str, Any]:
    full_fim = np.asarray(full_fim, dtype=np.float64)
    pruned_fim = np.asarray(pruned_fim, dtype=np.float64)
    if full_fim.shape != (6, 6) or pruned_fim.shape != (6, 6):
        raise ValueError(f"Expected full_fim and pruned_fim with shape [6, 6], got {full_fim.shape} and {pruned_fim.shape}")

    regularized_full = full_fim + float(regularization) * np.eye(6, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(regularized_full)
    lambda_min = float(np.min(eigvals))
    trace_inv_full = _safe_trace_inverse(full_fim, regularization=regularization)
    delta_trace = float(np.trace(pruned_fim))
    delta_spectral = float(np.linalg.norm(pruned_fim, ord=2))

    def compute_delta(delta: float) -> float:
        if lambda_min <= 0 or delta >= lambda_min:
            return float("nan")
        return float(delta / (lambda_min * (lambda_min - delta)) * trace_inv_full)

    return {
        "lambda_min": lambda_min,
        "trace_inv_full": trace_inv_full,
        "delta_trace": delta_trace,
        "delta_spectral": delta_spectral,
        "pred_delta_trace": compute_delta(delta_trace),
        "pred_delta_spectral": compute_delta(delta_spectral),
        "valid_trace_bound": bool(lambda_min > 0 and delta_trace < lambda_min),
        "valid_spectral_bound": bool(lambda_min > 0 and delta_spectral < lambda_min),
    }
