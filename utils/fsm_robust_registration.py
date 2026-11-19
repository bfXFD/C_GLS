"""Pure NumPy robust point-to-plane registration.

The optimizer uses left-multiplicative SE(3) perturbations::

    T_next = Exp([rotation, translation]) @ T

Consequently, the point-to-plane Jacobian is evaluated at the already
transformed source point and has columns ``[p x n, n]``.
"""

from typing import Any

import numpy as np


_BACKEND = "numpy_robust_point_to_plane"


def skew(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected vector with shape [3], got {vector.shape}")
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=np.float64,
    )


def so3_exp(rotation: np.ndarray) -> np.ndarray:
    """Exponential map from an axis-angle vector to SO(3)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3,) or np.any(~np.isfinite(rotation)):
        raise ValueError("rotation must be a finite vector with shape [3]")
    theta = float(np.linalg.norm(rotation))
    omega = skew(rotation)
    if theta < 1e-8:
        a = 1.0 - theta * theta / 6.0 + theta**4 / 120.0
        b = 0.5 - theta * theta / 24.0 + theta**4 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * omega + b * (omega @ omega)


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Exponential map for a twist ordered as ``[rotation, translation]``."""
    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (6,) or np.any(~np.isfinite(twist)):
        raise ValueError("twist must be a finite vector with shape [6]")
    rotation = twist[:3]
    translation = twist[3:]
    theta = float(np.linalg.norm(rotation))
    omega = skew(rotation)
    if theta < 1e-8:
        b = 0.5 - theta * theta / 24.0 + theta**4 / 720.0
        c = 1.0 / 6.0 - theta * theta / 120.0 + theta**4 / 5040.0
    else:
        b = (1.0 - np.cos(theta)) / (theta * theta)
        c = (theta - np.sin(theta)) / (theta**3)
    left_jacobian = np.eye(3, dtype=np.float64) + b * omega + c * (omega @ omega)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = so3_exp(rotation)
    transform[:3, 3] = left_jacobian @ translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = _validate_points(points, "points")
    transform = _validate_transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def point_to_plane_residual_jacobian(
    transformed_source: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals and the Jacobian for a left SE(3) perturbation."""
    transformed_source = _validate_points(transformed_source, "transformed_source")
    target_points = _validate_points(target_points, "target_points")
    if target_points.shape != transformed_source.shape:
        raise ValueError("transformed_source and target_points must have matching shapes")
    target_normals = _validate_normals(target_normals, target_points.shape[0])
    residuals = np.sum(target_normals * (transformed_source - target_points), axis=1)
    jacobian = np.concatenate(
        [np.cross(transformed_source, target_normals), target_normals], axis=1
    )
    return residuals.astype(np.float64), jacobian.astype(np.float64)


def robust_weights(residuals: np.ndarray, loss: str = "huber", scale: float = 0.02) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 1 or np.any(~np.isfinite(residuals)):
        raise ValueError("residuals must be a finite one-dimensional array")
    loss = str(loss).lower()
    if loss not in ("none", "huber", "cauchy"):
        raise ValueError("loss must be one of: 'none', 'huber', 'cauchy'")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    if loss == "none":
        return np.ones_like(residuals)
    scaled_abs = np.abs(residuals) / float(scale)
    if loss == "huber":
        return np.where(scaled_abs <= 1.0, 1.0, 1.0 / np.maximum(scaled_abs, 1e-15))
    return 1.0 / (1.0 + scaled_abs * scaled_abs)


def aggregate_correspondence_hessian(
    jacobians: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
    rank_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Aggregate weighted correspondence Jacobians into pose observability metrics."""
    jacobians = np.asarray(jacobians, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if jacobians.ndim != 2 or jacobians.shape[1] != 6 or np.any(~np.isfinite(jacobians)):
        raise ValueError("jacobians must be a finite array with shape [N, 6]")
    if weights.shape != (jacobians.shape[0],) or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"weights must be finite and non-negative with shape [{jacobians.shape[0]}]")
    if rank_tolerance <= 0.0 or not np.isfinite(rank_tolerance):
        raise ValueError("rank_tolerance must be positive and finite")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (jacobians.shape[0],):
            raise ValueError(f"mask must be boolean with shape [{jacobians.shape[0]}]")
        jacobians = jacobians[mask]
        weights = weights[mask]

    hessian = jacobians.T @ (jacobians * weights[:, None])
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    lambda_max = max(float(eigenvalues[-1]), 0.0)
    threshold = max(float(rank_tolerance) * lambda_max, np.finfo(np.float64).eps)
    rank = int(np.count_nonzero(eigenvalues > threshold))
    return {
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace": float(np.trace(hessian)),
        "rank": rank,
        "effective_rank": rank,
        "rank_threshold": threshold,
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
    }


def estimate_normals(
    points: np.ndarray,
    k: int = 20,
    target_tree: Any | None = None,
    numpy_fallback_max_points: int = 10000,
) -> np.ndarray:
    """Estimate unoriented normals with local PCA and vectorized kNN queries."""
    points = _validate_points(points, "points")
    if points.shape[0] < 3:
        raise ValueError("At least three target points are required to estimate normals")
    if k < 3:
        raise ValueError("normal_k must be at least 3")
    k = min(int(k), points.shape[0])
    tree = target_tree
    if tree is None:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
        except (ImportError, ModuleNotFoundError):
            tree = None

    if tree is not None:
        try:
            _, neighbor_indices = tree.query(points, k=k, workers=-1)
        except TypeError:
            _, neighbor_indices = tree.query(points, k=k)
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, None]
    else:
        if points.shape[0] > int(numpy_fallback_max_points):
            raise RuntimeError(
                "Normal estimation for more than 10000 points requires scipy.spatial.cKDTree "
                "or a compatible target_tree"
            )
        squared_distances = np.sum(
            (points[:, None, :] - points[None, :, :]) ** 2,
            axis=2,
        )
        neighbor_indices = np.argpartition(squared_distances, kth=k - 1, axis=1)[:, :k]

    neighborhood = points[neighbor_indices]
    centered = neighborhood - np.mean(neighborhood, axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariances)
    normals = eigenvectors[:, :, 0]
    dominant = np.argmax(np.abs(normals), axis=1)
    signs = np.where(normals[np.arange(normals.shape[0]), dominant] < 0.0, -1.0, 1.0)
    normals *= signs[:, None]
    return _validate_normals(normals, points.shape[0])


def find_correspondences(
    transformed_source: np.ndarray,
    target_points: np.ndarray,
    max_correspondence_distance: float,
    target_tree: Any | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, target indices, and Euclidean distances."""
    transformed_source = _validate_points(transformed_source, "transformed_source")
    target_points = _validate_points(target_points, "target_points")
    if max_correspondence_distance <= 0 or not np.isfinite(max_correspondence_distance):
        raise ValueError("max_correspondence_distance must be positive and finite")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if target_tree is not None:
        try:
            distances, target_indices = target_tree.query(transformed_source, k=1, workers=-1)
        except TypeError:
            distances, target_indices = target_tree.query(transformed_source, k=1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    else:
        distances = np.empty(transformed_source.shape[0], dtype=np.float64)
        target_indices = np.empty(transformed_source.shape[0], dtype=np.int64)
        for start in range(0, transformed_source.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), transformed_source.shape[0])
            diff = transformed_source[start:end, None, :] - target_points[None, :, :]
            squared_distances = np.sum(diff * diff, axis=2)
            local_indices = np.argmin(squared_distances, axis=1)
            target_indices[start:end] = local_indices
            distances[start:end] = np.sqrt(
                squared_distances[np.arange(end - start), local_indices]
            )
    mask = np.isfinite(distances) & (distances <= float(max_correspondence_distance))
    source_indices = np.flatnonzero(mask).astype(np.int64)
    return source_indices, target_indices[mask], distances[mask]


def register_robust_point_to_plane(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 30,
    loss: str = "huber",
    loss_scale: float = 0.02,
    lm_damping: float = 1e-3,
    min_correspondences: int = 6,
    normal_k: int = 20,
    target_tree: Any | None = None,
    step_tolerance: float | None = None,
    cost_tolerance: float | None = None,
    rotation_step_tolerance_deg: float = 0.01,
    translation_step_tolerance_m: float = 1e-4,
    relative_cost_tolerance: float = 1e-5,
    convergence_patience: int = 2,
    min_iterations: int = 2,
    rank_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Register source to target using robust point-to-plane ICP.

    This is the main entry point intended to be wrapped by ``register_pointcloud``.
    All optimization and diagnostics are implemented with NumPy; ``target_tree``
    is optional and only needs to provide a nearest-neighbor ``query`` method.
    """
    source_points = _validate_points(source_points, "source_points")
    target_points = _validate_points(target_points, "target_points")
    transform = _validate_transform(init_transform).copy()
    if step_tolerance is not None:
        rotation_step_tolerance_deg = float(np.degrees(step_tolerance))
        translation_step_tolerance_m = float(step_tolerance)
    if cost_tolerance is not None:
        relative_cost_tolerance = float(cost_tolerance)
    _validate_options(
        max_correspondence_distance,
        max_iterations,
        loss,
        loss_scale,
        lm_damping,
        min_correspondences,
        rank_tolerance,
        rotation_step_tolerance_deg,
        translation_step_tolerance_m,
        relative_cost_tolerance,
        convergence_patience,
        min_iterations,
    )
    if target_normals is None:
        if target_points.shape[0] < min_correspondences:
            return _failure_result(
                transform, source_points, target_points, "insufficient_correspondences"
            )
        target_normals = estimate_normals(target_points, k=normal_k, target_tree=target_tree)
    else:
        target_normals = _validate_normals(target_normals, target_points.shape[0])

    damping = float(lm_damping)
    converged = False
    failure_reason: str | None = None
    termination_reason: str | None = None
    iterations = 0
    convergence_streak = 0
    accepted_step_count = 0
    rejected_lm_trials = 0
    final_step_rotation_deg = float("nan")
    final_step_translation_m = float("nan")
    last_relative_reduction = float("nan")
    initial_mean_robust_cost = float("nan")

    for iteration in range(int(max_iterations)):
        transformed = transform_points(source_points, transform)
        source_indices, target_indices, _ = find_correspondences(
            transformed,
            target_points,
            max_correspondence_distance,
            target_tree=target_tree,
        )
        if source_indices.size < min_correspondences:
            failure_reason = "insufficient_correspondences"
            break
        corr_source = transformed[source_indices]
        corr_target = target_points[target_indices]
        corr_normals = target_normals[target_indices]
        residuals, jacobian = point_to_plane_residual_jacobian(
            corr_source, corr_target, corr_normals
        )
        weights = robust_weights(residuals, loss=loss, scale=loss_scale)
        hessian, gradient = _normal_equations(jacobian, residuals, weights)
        current_cost = _robust_cost(residuals, loss, loss_scale)
        if not np.isfinite(initial_mean_robust_cost):
            initial_mean_robust_cost = current_cost / source_indices.size

        diagonal = np.maximum(np.diag(hessian), 1e-12)
        # Use the undamped gradient: large LM damping can make any step tiny.
        stationary = bool(
            np.isfinite(current_cost)
            and np.all(np.isfinite(hessian))
            and np.all(np.isfinite(gradient))
            and np.max(np.abs(gradient) / np.sqrt(diagonal))
            <= np.sqrt(np.finfo(np.float64).eps) * max(1.0, np.sqrt(2.0 * current_cost))
        )
        if stationary:
            iterations = iteration + 1
            final_step_rotation_deg = 0.0
            final_step_translation_m = 0.0
            last_relative_reduction = 0.0
            convergence_streak += 1
            if iterations >= min_iterations and convergence_streak >= convergence_patience:
                converged = True
                termination_reason = "converged"
                break
            continue
        accepted = False
        step = np.zeros(6, dtype=np.float64)
        candidate_cost = current_cost
        for _ in range(10):
            damped_hessian = hessian + damping * np.diag(diagonal)
            try:
                step = np.linalg.solve(damped_hessian, -gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(damped_hessian, -gradient, rcond=None)[0]
            if np.any(~np.isfinite(step)):
                failure_reason = "non_finite_step"
                break
            candidate_transform = se3_exp(step) @ transform
            candidate_source = transform_points(source_points[source_indices], candidate_transform)
            candidate_residuals, _ = point_to_plane_residual_jacobian(
                candidate_source, corr_target, corr_normals
            )
            candidate_cost = _robust_cost(candidate_residuals, loss, loss_scale)
            if candidate_cost < current_cost:
                transform = candidate_transform
                damping = max(damping * 0.3, 1e-12)
                accepted = True
                break
            rejected_lm_trials += 1
            damping = min(damping * 10.0, 1e12)
        iterations = iteration + 1
        if failure_reason == "non_finite_step":
            termination_reason = failure_reason
            break
        if not accepted:
            failure_reason = "lm_step_rejected"
            termination_reason = failure_reason
            break
        accepted_step_count += 1
        relative_reduction = (current_cost - candidate_cost) / max(current_cost, 1e-15)
        final_step_rotation_deg = float(np.degrees(np.linalg.norm(step[:3])))
        final_step_translation_m = float(np.linalg.norm(step[3:]))
        last_relative_reduction = float(relative_reduction)
        meets_stop = (
            final_step_rotation_deg <= rotation_step_tolerance_deg
            and final_step_translation_m <= translation_step_tolerance_m
            and relative_reduction <= relative_cost_tolerance
            and np.linalg.norm(gradient[:3] / diagonal[:3])
            <= np.radians(rotation_step_tolerance_deg)
            and np.linalg.norm(gradient[3:] / diagonal[3:]) <= translation_step_tolerance_m
        )
        convergence_streak = convergence_streak + 1 if meets_stop else 0
        if iterations >= min_iterations and convergence_streak >= convergence_patience:
            converged = True
            termination_reason = "converged"
            break

    diagnostics = evaluate_point_to_plane_diagnostics(
        source_points,
        target_points,
        target_normals,
        transform,
        max_correspondence_distance,
        target_tree,
        loss,
        loss_scale,
        min_correspondences,
        rank_tolerance,
    )
    diagnostics_finite = _diagnostics_are_finite(transform, diagnostics)
    if diagnostics["num_correspondences"] < min_correspondences:
        converged = False
        failure_reason = "insufficient_correspondences"
        termination_reason = failure_reason
    elif diagnostics["effective_rank"] < 6:
        converged = False
        failure_reason = "degenerate_geometry"
        termination_reason = failure_reason
    elif not diagnostics_finite:
        converged = False
        failure_reason = "non_finite_diagnostics"
        termination_reason = failure_reason
    elif not converged and failure_reason is None:
        termination_reason = "max_iterations"
    elif failure_reason is not None and termination_reason is None:
        termination_reason = failure_reason

    optimizer_valid = bool(
        diagnostics_finite
        and diagnostics["num_correspondences"] >= min_correspondences
        and diagnostics["effective_rank"] == 6
        and failure_reason is None
    )
    final_mean_robust_cost = (
        float(diagnostics["robust_cost"] / diagnostics["num_correspondences"])
        if diagnostics["num_correspondences"] > 0
        else float("nan")
    )
    if not optimizer_valid and failure_reason is None:
        failure_reason = termination_reason or "optimizer_invalid"

    diagnostics.update(
        {
            "transform": transform,
            "backend": _BACKEND,
            "converged": bool(converged),
            "optimizer_valid": optimizer_valid,
            "termination_reason": termination_reason,
            "failure_reason": failure_reason,
            "iterations": int(iterations),
            "num_source_points": int(source_points.shape[0]),
            "num_target_points": int(target_points.shape[0]),
            "loss": str(loss).lower(),
            "loss_scale": float(loss_scale),
            "lm_damping": float(damping),
            "final_step_rotation_deg": final_step_rotation_deg,
            "final_step_translation_m": final_step_translation_m,
            "initial_mean_robust_cost": initial_mean_robust_cost,
            "final_mean_robust_cost": final_mean_robust_cost,
            "last_relative_reduction": last_relative_reduction,
            "accepted_step_count": int(accepted_step_count),
            "rejected_lm_trials": int(rejected_lm_trials),
        }
    )
    return diagnostics


# A descriptive alias for integration code that names NumPy backends explicitly.
register_numpy_robust_point_to_plane = register_robust_point_to_plane


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected {name} with shape [N, 3], got {points.shape}")
    if points.shape[0] == 0 or np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must be non-empty and finite")
    return points


def _validate_normals(normals: np.ndarray, count: int) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != (count, 3) or np.any(~np.isfinite(normals)):
        raise ValueError(f"Expected finite target_normals with shape [{count}, 3]")
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("target_normals must be non-zero")
    return normals / norms


def _validate_transform(transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or np.any(~np.isfinite(transform)):
        raise ValueError("init_transform must be finite with shape [4, 4]")
    return transform


def _validate_options(
    max_distance: float,
    max_iterations: int,
    loss: str,
    loss_scale: float,
    lm_damping: float,
    min_correspondences: int,
    rank_tolerance: float,
    rotation_step_tolerance_deg: float,
    translation_step_tolerance_m: float,
    relative_cost_tolerance: float,
    convergence_patience: int,
    min_iterations: int,
) -> None:
    if max_distance <= 0 or not np.isfinite(max_distance):
        raise ValueError("max_correspondence_distance must be positive and finite")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if str(loss).lower() not in ("none", "huber", "cauchy"):
        raise ValueError("loss must be one of: 'none', 'huber', 'cauchy'")
    if loss_scale <= 0 or not np.isfinite(loss_scale):
        raise ValueError("loss_scale must be positive and finite")
    if lm_damping <= 0 or not np.isfinite(lm_damping):
        raise ValueError("lm_damping must be positive and finite")
    if min_correspondences < 1:
        raise ValueError("min_correspondences must be positive")
    if rank_tolerance <= 0 or not np.isfinite(rank_tolerance):
        raise ValueError("rank_tolerance must be positive and finite")
    for value, name in (
        (rotation_step_tolerance_deg, "rotation_step_tolerance_deg"),
        (translation_step_tolerance_m, "translation_step_tolerance_m"),
        (relative_cost_tolerance, "relative_cost_tolerance"),
    ):
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be positive")
    if min_iterations < 1:
        raise ValueError("min_iterations must be positive")


def _diagnostics_are_finite(transform: np.ndarray, diagnostics: dict[str, Any]) -> bool:
    finite_keys = (
        "fitness",
        "inlier_rmse",
        "point_to_plane_rmse",
        "robust_cost",
        "hessian",
        "gradient",
        "eigenvalues",
        "residual_variance",
    )
    return bool(
        np.all(np.isfinite(transform))
        and all(np.all(np.isfinite(np.asarray(diagnostics[key]))) for key in finite_keys)
        and np.isfinite(float(diagnostics["condition_number"]))
        and np.all(np.isfinite(np.asarray(diagnostics["pose_covariance"])))
    )


def _normal_equations(
    jacobian: np.ndarray, residuals: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weighted_jacobian = jacobian * weights[:, None]
    hessian = jacobian.T @ weighted_jacobian
    gradient = jacobian.T @ (weights * residuals)
    return 0.5 * (hessian + hessian.T), gradient


def _robust_cost(residuals: np.ndarray, loss: str, scale: float) -> float:
    squared = residuals * residuals
    loss = str(loss).lower()
    if loss == "none":
        return float(0.5 * np.sum(squared))
    absolute = np.abs(residuals)
    if loss == "huber":
        values = np.where(
            absolute <= scale,
            0.5 * squared,
            scale * (absolute - 0.5 * scale),
        )
        return float(np.sum(values))
    return float(0.5 * scale * scale * np.sum(np.log1p(squared / (scale * scale))))


def evaluate_point_to_plane_diagnostics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    transform: np.ndarray,
    max_correspondence_distance: float = 0.05,
    target_tree: Any | None = None,
    loss: str = "huber",
    loss_scale: float = 0.02,
    min_correspondences: int = 6,
    rank_tolerance: float = 1e-9,
    source_ids: np.ndarray | None = None,
    target_ids: np.ndarray | None = None,
    return_correspondences: bool = False,
) -> dict[str, Any]:
    """Evaluate point-to-plane diagnostics at a fixed source-to-target pose."""
    source_points = _validate_points(source_points, "source_points")
    target_points = _validate_points(target_points, "target_points")
    target_normals = _validate_normals(target_normals, target_points.shape[0])
    transform = _validate_transform(transform)
    if max_correspondence_distance <= 0.0 or not np.isfinite(max_correspondence_distance):
        raise ValueError("max_correspondence_distance must be positive and finite")
    if min_correspondences < 1:
        raise ValueError("min_correspondences must be positive")
    if rank_tolerance <= 0.0 or not np.isfinite(rank_tolerance):
        raise ValueError("rank_tolerance must be positive and finite")
    if not isinstance(return_correspondences, (bool, np.bool_)):
        raise TypeError("return_correspondences must be a boolean")
    source_ids = _validate_ids(source_ids, source_points.shape[0], "source_ids")
    target_ids = _validate_ids(target_ids, target_points.shape[0], "target_ids")

    transformed = transform_points(source_points, transform)
    source_indices, target_indices, distances = find_correspondences(
        transformed, target_points, max_correspondence_distance, target_tree=target_tree
    )
    count = int(source_indices.size)
    if count < min_correspondences:
        result = _empty_diagnostics(count, source_points.shape[0], distances)
        if return_correspondences:
            result.update(_empty_correspondences(source_ids.dtype, target_ids.dtype))
        return result
    residuals, jacobian = point_to_plane_residual_jacobian(
        transformed[source_indices], target_points[target_indices], target_normals[target_indices]
    )
    weights = robust_weights(residuals, loss=loss, scale=loss_scale)
    hessian_metrics = aggregate_correspondence_hessian(
        jacobian, weights, rank_tolerance=rank_tolerance
    )
    hessian = hessian_metrics["hessian"]
    eigenvalues = hessian_metrics["eigenvalues"]
    eigenvectors = hessian_metrics["eigenvectors"]
    threshold = hessian_metrics["rank_threshold"]
    observable = eigenvalues > threshold
    rank = hessian_metrics["effective_rank"]
    gradient = jacobian.T @ (weights * residuals)
    condition = (
        float(eigenvalues[-1] / eigenvalues[0])
        if rank == 6 and eigenvalues[0] > 0
        else float("inf")
    )
    inverse_eigenvalues = np.zeros(6, dtype=np.float64)
    inverse_eigenvalues[observable] = 1.0 / eigenvalues[observable]
    hessian_pinv = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    degrees_of_freedom = max(count - rank, 1)
    residual_variance = float(np.sum(weights * residuals * residuals) / degrees_of_freedom)
    pose_covariance = residual_variance * hessian_pinv
    result = {
        "fitness": float(count / source_points.shape[0]),
        "inlier_rmse": float(np.sqrt(np.mean(distances * distances))),
        "point_to_plane_rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "robust_cost": _robust_cost(residuals, loss, loss_scale),
        "num_correspondences": count,
        "hessian": hessian,
        "gradient": gradient,
        "eigenvalues": eigenvalues,
        "condition_number": condition,
        "effective_rank": rank,
        "pose_covariance": pose_covariance,
        "pose_covariance_valid": bool(rank == 6),
        "residual_variance": residual_variance,
    }
    if return_correspondences:
        result.update(
            {
                "source_indices": source_indices,
                "target_indices": target_indices,
                "source_ids": source_ids[source_indices],
                "target_ids": target_ids[target_indices],
                "distances": distances,
                "residuals": residuals,
                "weights": weights,
                "jacobians": jacobian,
            }
        )
    return result


def _validate_ids(ids: np.ndarray | None, count: int, name: str) -> np.ndarray:
    if ids is None:
        return np.arange(count, dtype=np.int64)
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.shape[0] != count:
        raise ValueError(f"Expected {name} with shape [{count}]")
    return ids


def _empty_correspondences(source_dtype: np.dtype, target_dtype: np.dtype) -> dict[str, np.ndarray]:
    return {
        "source_indices": np.empty(0, dtype=np.int64),
        "target_indices": np.empty(0, dtype=np.int64),
        "source_ids": np.empty(0, dtype=source_dtype),
        "target_ids": np.empty(0, dtype=target_dtype),
        "distances": np.empty(0, dtype=np.float64),
        "residuals": np.empty(0, dtype=np.float64),
        "weights": np.empty(0, dtype=np.float64),
        "jacobians": np.empty((0, 6), dtype=np.float64),
    }


def _empty_diagnostics(count: int, source_count: int, distances: np.ndarray) -> dict[str, Any]:
    return {
        "fitness": float(count / source_count),
        "inlier_rmse": (
            float(np.sqrt(np.mean(distances * distances))) if count else float("inf")
        ),
        "point_to_plane_rmse": float("inf"),
        "robust_cost": float("inf"),
        "num_correspondences": int(count),
        "hessian": np.zeros((6, 6), dtype=np.float64),
        "gradient": np.zeros(6, dtype=np.float64),
        "eigenvalues": np.zeros(6, dtype=np.float64),
        "condition_number": float("inf"),
        "effective_rank": 0,
        "pose_covariance": np.full((6, 6), np.nan, dtype=np.float64),
        "pose_covariance_valid": False,
        "residual_variance": float("nan"),
    }


def _failure_result(
    transform: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    reason: str,
) -> dict[str, Any]:
    result = _empty_diagnostics(0, source_points.shape[0], np.empty(0))
    result.update(
        {
            "transform": transform,
            "backend": _BACKEND,
            "converged": False,
            "optimizer_valid": False,
            "termination_reason": reason,
            "failure_reason": reason,
            "iterations": 0,
            "num_source_points": int(source_points.shape[0]),
            "num_target_points": int(target_points.shape[0]),
            "final_step_rotation_deg": float("nan"),
            "final_step_translation_m": float("nan"),
            "initial_mean_robust_cost": float("nan"),
            "final_mean_robust_cost": float("nan"),
            "last_relative_reduction": float("nan"),
            "accepted_step_count": 0,
            "rejected_lm_trials": 0,
        }
    )
    return result
