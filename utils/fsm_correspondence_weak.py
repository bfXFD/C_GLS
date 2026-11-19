from typing import Any

import numpy as np


def _rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or np.any(~np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(rotation)
    normalized = u @ vt
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vt
    return normalized


def _camera_geometry(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4) or np.any(~np.isfinite(c2w)):
        raise ValueError("c2w must be a finite 4x4 transform")
    rotation = _rotation_matrix(c2w[:3, :3])
    translation = c2w[:3, 3]
    skew = np.array(
        [[0.0, -translation[2], translation[1]],
         [translation[2], 0.0, -translation[0]],
         [-translation[1], translation[0], 0.0]],
        dtype=np.float64,
    )
    adjoint = np.zeros((6, 6), dtype=np.float64)
    adjoint[:3, :3] = rotation
    adjoint[3:, :3] = skew @ rotation
    adjoint[3:, 3:] = rotation
    return rotation, translation, adjoint


def organized_depth_source_normals(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    sample_flat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate camera-frame normals at organized depth samples without GT geometry."""
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    indices = np.asarray(sample_flat_indices, dtype=np.int64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if intrinsics.shape != (3, 3) or np.any(~np.isfinite(intrinsics)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= depth.size):
        raise ValueError("sample_flat_indices must be valid flat depth indices")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")

    height, width = depth.shape
    rows, cols = np.divmod(indices, width)
    interior = (rows > 0) & (rows + 1 < height) & (cols > 0) & (cols + 1 < width)
    normals = np.zeros((indices.size, 3), dtype=np.float64)
    confidence = np.zeros(indices.size, dtype=np.float64)
    if not np.any(interior):
        return normals, interior, confidence

    local = np.flatnonzero(interior)
    r, c = rows[local], cols[local]
    zc = depth[r, c]
    zl, zr = depth[r, c - 1], depth[r, c + 1]
    zu, zd = depth[r - 1, c], depth[r + 1, c]
    finite = np.isfinite(zc) & np.isfinite(zl) & np.isfinite(zr) & np.isfinite(zu) & np.isfinite(zd)
    finite &= (zc > 0.0) & (zl > 0.0) & (zr > 0.0) & (zu > 0.0) & (zd > 0.0)

    def backproject(pixel_col: np.ndarray, pixel_row: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack(((pixel_col - cx) * z / fx, (pixel_row - cy) * z / fy, z))

    horizontal = backproject(c + 1, r, zr) - backproject(c - 1, r, zl)
    vertical = backproject(c, r + 1, zd) - backproject(c, r - 1, zu)
    raw = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(raw, axis=1)
    finite &= np.isfinite(lengths) & (lengths > np.finfo(np.float64).eps)
    valid_local = local[finite]
    normals[valid_local] = raw[finite] / lengths[finite, None]
    flip = normals[valid_local, 2] > 0.0
    normals[valid_local[flip]] *= -1.0
    relative_jump = np.maximum.reduce(
        (np.abs(zl - zc), np.abs(zr - zc), np.abs(zu - zc), np.abs(zd - zc))
    ) / np.maximum(zc, np.finfo(np.float64).eps)
    confidence[valid_local] = np.exp(-8.0 * relative_jump[finite])
    valid = np.zeros(indices.size, dtype=bool)
    valid[valid_local] = True
    return normals, valid, np.clip(confidence, 0.0, 1.0)


def normalized_hessian_diagnostics(
    hessian_world: np.ndarray,
    reference_pose: np.ndarray,
    source_points_camera: np.ndarray | None = None,
    characteristic_length: float | None = None,
) -> dict[str, Any]:
    """Diagnose a world Hessian in depth-normalized reference-camera coordinates.

    Twists are ordered [rotation, translation]. Length defaults to the median
    positive camera depth clipped to [0.5, 5.0], or 1.0 without positive depths.
    An explicit finite positive length is used unchanged, allowing comparisons
    with a shared reference and scale. Rank uses 6 * eps * lambda_max; singular
    matrices have infinite condition number and zero trace has lambda/trace 0.
    Negative eigenvalues within 64 * eps * spectral scale are roundoff-clipped.
    """
    hessian = np.asarray(hessian_world, dtype=np.float64)
    pose = np.asarray(reference_pose, dtype=np.float64)
    if hessian.shape != (6, 6) or np.any(~np.isfinite(hessian)):
        raise ValueError("hessian_world must be a finite 6x6 matrix")
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
        raise ValueError("reference_pose must be a finite 4x4 rigid transform")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6)):
        raise ValueError("reference_pose must be a rigid transform with a proper rotation")
    median_depth = 1.0
    if source_points_camera is not None:
        points = np.asarray(source_points_camera, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("source_points_camera must be finite with shape [N, 3]")
        valid_depth = points[:, 2][points[:, 2] > 0.0]
        if valid_depth.size:
            median_depth = float(np.median(valid_depth))
    if characteristic_length is None:
        length = float(np.clip(median_depth, 0.5, 5.0))
    else:
        value = np.asarray(characteristic_length, dtype=np.float64)
        if value.ndim != 0 or not np.isfinite(value) or value <= 0.0:
            raise ValueError("characteristic_length must be a finite positive scalar")
        length = float(value)

    _, _, adjoint = _camera_geometry(pose)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        symmetric = 0.5 * (hessian + hessian.T)
        camera_hessian = adjoint.T @ symmetric @ adjoint
        coordinate_map = np.diag([1.0 / length] * 3 + [1.0] * 3)
        normalized_hessian = coordinate_map.T @ camera_hessian @ coordinate_map
        normalized_hessian = 0.5 * (normalized_hessian + normalized_hessian.T)
    if np.any(~np.isfinite(normalized_hessian)):
        raise ValueError("normalized hessian must be finite")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError("normalized hessian eigendecomposition failed") from exc
    if np.any(~np.isfinite(eigenvalues)):
        raise ValueError("normalized hessian eigenvalues must be finite")
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -64.0 * np.finfo(np.float64).eps * spectral_scale:
        raise ValueError("hessian_world must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rank_threshold = 6.0 * np.finfo(np.float64).eps * float(eigenvalues[-1])
    rank = int(np.count_nonzero(eigenvalues > rank_threshold))
    # Scale before summation to avoid overflowing the trace of finite spectra.
    scaled = eigenvalues / eigenvalues[-1] if eigenvalues[-1] > 0.0 else eigenvalues
    return {
        "hessian": normalized_hessian,
        "eigenvalues": eigenvalues,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]) if rank == 6 else float("inf"),
        "effective_rank": rank,
        "characteristic_length": length,
        "normalized_lambda_min": float(scaled[0] / np.sum(scaled)) if rank > 0 else 0.0,
        "eigenvectors": eigenvectors,
        "camera_hessian": camera_hessian,
        "adjoint_c2w": adjoint,
        "coordinate_map": coordinate_map,
    }


def adaptive_weak_subspace(
    hessian_world: np.ndarray,
    c2w: np.ndarray,
    source_points_camera: np.ndarray,
    min_dim: int = 1,
    max_dim: int = 3,
    trace_threshold: float = 1e-3,
    relative_threshold: float = 1e-2,
    characteristic_depth_min: float = 0.5,
    characteristic_depth_max: float = 5.0,
) -> dict[str, Any]:
    """Return an adaptive weak eigenspace in depth-normalized camera twist coordinates."""
    points = np.asarray(source_points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
        raise ValueError("source_points_camera must be finite with shape [N, 3]")
    if not 1 <= int(min_dim) <= int(max_dim) <= 3:
        raise ValueError("weak dimensions must satisfy 1 <= min_dim <= max_dim <= 3")
    if trace_threshold <= 0.0 or relative_threshold <= 0.0:
        raise ValueError("weak thresholds must be positive")
    if (not np.isfinite(characteristic_depth_min) or not np.isfinite(characteristic_depth_max)
            or not 0.0 < characteristic_depth_min <= characteristic_depth_max):
        raise ValueError("characteristic depth bounds must be finite, positive and ordered")
    valid_depth = points[:, 2][np.isfinite(points[:, 2]) & (points[:, 2] > 0.0)]
    median_depth = float(np.median(valid_depth)) if valid_depth.size else 1.0
    length = float(np.clip(median_depth, characteristic_depth_min, characteristic_depth_max))

    diagnostics = normalized_hessian_diagnostics(hessian_world, c2w, points, length)
    eigenvalues = diagnostics["eigenvalues"]
    eigenvectors = diagnostics["eigenvectors"]
    trace = max(float(np.sum(eigenvalues)), np.finfo(np.float64).eps)
    threshold = max(float(trace_threshold) * trace, float(relative_threshold) * float(eigenvalues[-1]))
    requested = int(np.count_nonzero(eigenvalues < threshold))
    k = int(np.clip(requested, min_dim, max_dim))
    deficits = np.maximum(threshold - eigenvalues[:k], 0.0) / threshold
    if not np.any(deficits > 0.0):
        deficits = threshold / np.maximum(eigenvalues[:k] + threshold, threshold)
    return {
        "basis": eigenvectors[:, :k],
        "eigs": eigenvalues,
        "eigenvalues": eigenvalues,
        "deficit_weights": deficits,
        "k": k,
        "length": length,
        "characteristic_length": length,
        "threshold": threshold,
        "truncated": bool(requested > max_dim),
        "camera_hessian": diagnostics["camera_hessian"],
        "normalized_hessian": diagnostics["hessian"],
        "adjoint_c2w": diagnostics["adjoint_c2w"],
        "coordinate_map": diagnostics["coordinate_map"],
    }


def _query(tree: Any, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        distances, indices = tree.query(points, k=k, workers=-1)
    except TypeError:
        distances, indices = tree.query(points, k=k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    return distances, indices


def _median_depth_3x3(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    observed = np.zeros(u.size, dtype=np.float64)
    valid = np.zeros(u.size, dtype=bool)
    for index, (col, row) in enumerate(zip(u, v)):
        if not np.isfinite(col) or not np.isfinite(row):
            continue
        x, y = int(np.rint(col)), int(np.rint(row))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        patch = depth[max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)]
        values = patch[np.isfinite(patch) & (patch > 0.0)]
        if values.size:
            observed[index] = float(np.median(values))
            valid[index] = True
    return observed, valid


def correspondence_probability(
    reserve_points_world: np.ndarray,
    source_points_camera: np.ndarray,
    c2w: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    source_flat_indices: np.ndarray,
    reserve_normals_world: np.ndarray | None = None,
    source_normals_camera: np.ndarray | None = None,
    source_normal_valid: np.ndarray | None = None,
    source_normal_confidence: np.ndarray | None = None,
    primary_source_target_distances: np.ndarray | None = None,
    base_target_tree: Any | None = None,
    k: int = 4,
    distance_sigma: float = 0.05,
    dmax: float = 0.15,
    depth_sigma_front: float = 0.10,
    depth_sigma_back: float = 0.03,
    missing_depth_prior: float = 0.35,
    normal_missing_prior: float = 0.5,
    normal_power: float = 2.0,
    competition_sigma: float = 0.02,
) -> dict[str, Any]:
    """Estimate candidate correspondence probability from current-frame observations only."""
    reserve = np.asarray(reserve_points_world, dtype=np.float64)
    source_camera = np.asarray(source_points_camera, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    flat_indices = np.asarray(source_flat_indices, dtype=np.int64)
    if reserve.ndim != 2 or reserve.shape[1] != 3 or np.any(~np.isfinite(reserve)):
        raise ValueError("reserve_points_world must be finite with shape [N, 3]")
    if source_camera.ndim != 2 or source_camera.shape[1] != 3 or np.any(~np.isfinite(source_camera)):
        raise ValueError("source_points_camera must be finite with shape [M, 3]")
    if source_camera.shape[0] == 0:
        raise ValueError("source_points_camera must not be empty")
    if flat_indices.shape != (source_camera.shape[0],):
        raise ValueError("source_flat_indices must have one entry per source point")
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("depth and intrinsics must have shapes [H, W] and [3, 3]")
    if k <= 0 or min(distance_sigma, dmax, depth_sigma_front, depth_sigma_back, competition_sigma) <= 0.0:
        raise ValueError("distance, depth, competition scales and k must be positive")
    for prior in (missing_depth_prior, normal_missing_prior):
        if not 0.0 <= prior <= 1.0:
            raise ValueError("missing-data priors must be in [0, 1]")

    rotation, translation, _ = _camera_geometry(c2w)
    source_world = source_camera @ rotation.T + translation
    try:
        from scipy.spatial import cKDTree
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("correspondence_probability requires scipy.spatial.cKDTree") from exc
    source_tree = cKDTree(source_world)
    query_k = min(int(k), source_world.shape[0])
    distances, neighbor_indices = _query(source_tree, reserve, query_k)
    distance_term = np.exp(-0.5 * (distances / float(distance_sigma)) ** 2)
    distance_term *= distances <= float(dmax)

    reserve_camera = (reserve - translation) @ rotation
    z = reserve_camera[:, 2]
    u = intrinsics[0, 0] * reserve_camera[:, 0] / np.where(z != 0.0, z, 1.0) + intrinsics[0, 2]
    v = intrinsics[1, 1] * reserve_camera[:, 1] / np.where(z != 0.0, z, 1.0) + intrinsics[1, 2]
    observed_depth, observed_valid = _median_depth_3x3(depth, u, v)
    depth_delta = z - observed_depth
    depth_sigma = np.where(depth_delta >= 0.0, depth_sigma_back, depth_sigma_front)
    depth_term = np.full(reserve.shape[0], float(missing_depth_prior), dtype=np.float64)
    projectable = observed_valid & np.isfinite(z) & (z > 0.0)
    depth_term[projectable] = np.exp(-0.5 * (depth_delta[projectable] / depth_sigma[projectable]) ** 2)

    if source_normals_camera is None:
        source_normals_camera, source_normal_valid, source_normal_confidence = organized_depth_source_normals(
            depth, intrinsics, flat_indices
        )
    else:
        source_normals_camera = np.asarray(source_normals_camera, dtype=np.float64)
        if source_normals_camera.shape != source_camera.shape:
            raise ValueError("source_normals_camera must match source_points_camera")
        lengths = np.linalg.norm(source_normals_camera, axis=1)
        inferred_valid = np.all(np.isfinite(source_normals_camera), axis=1) & (lengths > 0.0)
        source_normals_camera = np.divide(
            source_normals_camera, lengths[:, None], out=np.zeros_like(source_normals_camera), where=lengths[:, None] > 0.0
        )
        if source_normal_valid is None:
            source_normal_valid = inferred_valid
        if source_normal_confidence is None:
            source_normal_confidence = np.asarray(source_normal_valid, dtype=np.float64)
    source_normal_valid = np.asarray(source_normal_valid, dtype=bool)
    source_normal_confidence = np.asarray(source_normal_confidence, dtype=np.float64)
    if source_normal_valid.shape != (source_camera.shape[0],) or source_normal_confidence.shape != source_normal_valid.shape:
        raise ValueError("source normal validity and confidence must have shape [M]")
    source_normal_confidence = np.clip(source_normal_confidence, 0.0, 1.0) * source_normal_valid

    normal_term = np.full(distances.shape, float(normal_missing_prior), dtype=np.float64)
    if reserve_normals_world is not None:
        reserve_normals = np.asarray(reserve_normals_world, dtype=np.float64)
        if reserve_normals.shape != reserve.shape:
            raise ValueError("reserve_normals_world must match reserve_points_world")
        reserve_normals_camera = reserve_normals @ rotation
        reserve_lengths = np.linalg.norm(reserve_normals_camera, axis=1)
        reserve_valid = np.all(np.isfinite(reserve_normals_camera), axis=1) & (reserve_lengths > 0.0)
        reserve_normals_camera = np.divide(
            reserve_normals_camera, reserve_lengths[:, None], out=np.zeros_like(reserve_normals_camera),
            where=reserve_lengths[:, None] > 0.0,
        )
        neighbor_normals = source_normals_camera[neighbor_indices]
        compatibility = np.abs(np.sum(reserve_normals_camera[:, None, :] * neighbor_normals, axis=2)) ** float(normal_power)
        confidence = source_normal_confidence[neighbor_indices] * reserve_valid[:, None]
        normal_term = confidence * compatibility + (1.0 - confidence) * float(normal_missing_prior)

    if primary_source_target_distances is not None:
        base_distances = np.asarray(primary_source_target_distances, dtype=np.float64)
        if base_distances.shape != (source_camera.shape[0],) or np.any(base_distances < 0.0):
            raise ValueError("primary_source_target_distances must be non-negative with shape [M]")
    elif base_target_tree is not None:
        base_distances, _ = _query(base_target_tree, source_world, 1)
        base_distances = base_distances[:, 0]
    else:
        base_distances = np.full(source_camera.shape[0], np.inf, dtype=np.float64)
    neighbor_base_distances = base_distances[neighbor_indices]
    competition_term = np.ones(distances.shape, dtype=np.float64)
    finite_base = np.isfinite(neighbor_base_distances)
    competition_term[finite_base] = 1.0 / (
        1.0 + np.exp(np.clip((distances[finite_base] - neighbor_base_distances[finite_base]) / competition_sigma, -60.0, 60.0))
    )

    joint = distance_term * depth_term[:, None] * normal_term * competition_term
    joint = np.clip(np.nan_to_num(joint, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    probability = 1.0 - np.prod(1.0 - joint, axis=1)
    return {
        "probability": np.clip(probability, 0.0, 1.0),
        "terms": {
            "distance": distance_term,
            "depth": depth_term,
            "normal": normal_term,
            "competition": competition_term,
            "joint": joint,
        },
        "diagnostics": {
            "neighbor_indices": neighbor_indices,
            "neighbor_distances": distances,
            "primary_distances": np.where(finite_base, neighbor_base_distances, 0.0),
            "primary_distance_valid": finite_base,
            "observed_depth": observed_depth,
            "observed_depth_valid": observed_valid,
            "projected_depth": z,
            "depth_delta": depth_delta,
            "source_normal_valid": source_normal_valid,
        },
        "calibration": {
            "distance_sigma": np.full(reserve.shape[0], float(distance_sigma)),
            "dmax": np.full(reserve.shape[0], float(dmax)),
            "depth_sigma": depth_sigma,
            "depth_prior": np.full(reserve.shape[0], float(missing_depth_prior)),
            "normal_confidence": source_normal_confidence[neighbor_indices],
            "competition_sigma": np.full(reserve.shape[0], float(competition_sigma)),
        },
    }


def correspondence_aware_weak_scores(
    reserve_points_world: np.ndarray,
    reserve_normals_world: np.ndarray,
    c2w: np.ndarray,
    hessian_world: np.ndarray,
    source_points_camera: np.ndarray,
    correspondence_probabilities: np.ndarray,
    visible_indices: np.ndarray | None = None,
    alpha: float = 1.0,
    min_dim: int = 1,
    max_dim: int = 3,
    saliency_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score only visible reserve rows by correspondence probability and normalized weak gain."""
    points = np.asarray(reserve_points_world, dtype=np.float64)
    normals = np.asarray(reserve_normals_world, dtype=np.float64)
    probabilities = np.asarray(correspondence_probabilities, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("reserve points and normals must have matching shape [N, 3]")
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    visible = np.arange(points.shape[0], dtype=np.int64) if visible_indices is None else np.asarray(visible_indices, dtype=np.int64)
    if visible.ndim != 1 or np.any(visible < 0) or np.any(visible >= points.shape[0]):
        raise ValueError("visible_indices contains invalid reserve indices")
    if probabilities.shape == (points.shape[0],):
        visible_probability = probabilities[visible]
    elif probabilities.shape == (visible.size,):
        visible_probability = probabilities
    else:
        raise ValueError("correspondence_probabilities must have shape [N] or [num_visible]")
    if np.any(~np.isfinite(visible_probability)) or np.any((visible_probability < 0.0) | (visible_probability > 1.0)):
        raise ValueError("correspondence probabilities must be finite and in [0, 1]")

    weak = adaptive_weak_subspace(
        hessian_world, c2w, source_points_camera, min_dim=min_dim, max_dim=max_dim
    )
    rotation, translation, _ = _camera_geometry(c2w)
    camera_points = (points[visible] - translation) @ rotation
    camera_normals = normals[visible] @ rotation
    normal_lengths = np.linalg.norm(camera_normals, axis=1)
    camera_normals = np.divide(
        camera_normals, normal_lengths[:, None], out=np.zeros_like(camera_normals), where=normal_lengths[:, None] > 0.0
    )
    jacobians = np.concatenate((np.cross(camera_points, camera_normals), camera_normals), axis=1)
    jacobians[:, :3] /= weak["characteristic_length"]
    jacobian_lengths = np.linalg.norm(jacobians, axis=1)
    normalized = np.divide(
        jacobians, jacobian_lengths[:, None], out=np.zeros_like(jacobians), where=jacobian_lengths[:, None] > 0.0
    )
    projected = normalized @ weak["basis"]
    deficit = weak["deficit_weights"]
    weak_score = np.sum(projected * projected * deficit[None, :], axis=1) / max(float(np.sum(deficit)), np.finfo(float).eps)
    visible_scores = np.power(visible_probability, float(alpha)) * weak_score
    if saliency_weights is not None:
        saliency = np.asarray(saliency_weights, dtype=np.float64)
        if saliency.shape != (points.shape[0],) or np.any(~np.isfinite(saliency)) or np.any(saliency < 0.0):
            raise ValueError("saliency_weights must be finite and non-negative with shape [N]")
        visible_scores *= saliency[visible]
    scores = np.zeros(points.shape[0], dtype=np.float64)
    scores[visible] = np.nan_to_num(visible_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores, {"weak": weak, "weak_score": weak_score, "visible_indices": visible}


SELECTION_SCORE_FUNCTIONS = {"runtime_corr_weak": correspondence_aware_weak_scores}
