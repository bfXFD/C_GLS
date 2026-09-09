from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferenceViews:
    w2c: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    frame_ids: np.ndarray
    source: str = "params_keyframes"


@dataclass(frozen=True)
class VisibilityFactors:
    pose_ptr: np.ndarray
    gaussian_ids: np.ndarray
    jacobians: np.ndarray
    weights: np.ndarray
    num_gaussians: int
    visible_pose_count: int = 0

    @property
    def num_poses(self) -> int:
        return int(self.pose_ptr.size - 1)

    @property
    def num_pairs(self) -> int:
        return int(self.gaussian_ids.size)

    @property
    def nonempty_pose_count(self) -> int:
        return int(np.count_nonzero(np.diff(self.pose_ptr)))

    @property
    def pose_coverage(self) -> float:
        return float(self.nonempty_pose_count / self.num_poses) if self.num_poses else 0.0


@dataclass(frozen=True)
class ObservableSelection:
    indices: np.ndarray
    trace_scores: np.ndarray
    greedy_indices: np.ndarray
    fill_indices: np.ndarray
    information_matrix: np.ndarray
    logdet: float
    lambda_min: float
    candidate_indices: np.ndarray
    per_pose_information: np.ndarray
    per_pose_logdet: np.ndarray
    per_pose_lambda_min: np.ndarray
    objective: float


def _as_params(params_or_path: dict[str, Any] | str) -> tuple[dict[str, Any], str]:
    if isinstance(params_or_path, str):
        return dict(np.load(params_or_path, allow_pickle=True)), params_or_path
    return params_or_path, "in_memory_params"


def _time_major(values: Any, component_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] == component_size:
        array = array[0].T
    else:
        array = array.squeeze()
    if array.ndim == 1 and array.shape[0] == component_size:
        array = array[None, :]
    elif array.ndim == 2 and array.shape[1] == component_size:
        pass
    elif array.ndim == 2 and array.shape[0] == component_size:
        array = array.T
    else:
        raise ValueError(f"Expected {name} with a component dimension of {component_size}, got {array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _quaternion_rotations(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Camera quaternions must be non-zero")
    w, x, y, z = (quaternions / norms).T
    rotations = np.empty((quaternions.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - w * z)
    rotations[:, 0, 2] = 2.0 * (x * z + w * y)
    rotations[:, 1, 0] = 2.0 * (x * y + w * z)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - w * x)
    rotations[:, 2, 0] = 2.0 * (x * z - w * y)
    rotations[:, 2, 1] = 2.0 * (y * z + w * x)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def load_reference_views(
    params_or_path: dict[str, Any] | str,
    keyframes_only: bool = True,
    max_reference_poses: int | None = None,
) -> ReferenceViews:
    """Load estimated reference poses from params, never query/future ground truth."""
    params, source_path = _as_params(params_or_path)
    required = ("cam_unnorm_rots", "cam_trans", "intrinsics", "org_width", "org_height")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing estimated reference-view fields: {missing}")

    rotations = _time_major(params["cam_unnorm_rots"], 4, "cam_unnorm_rots")
    translations = _time_major(params["cam_trans"], 3, "cam_trans")
    if rotations.shape[0] != translations.shape[0]:
        raise ValueError("cam_unnorm_rots and cam_trans must contain the same number of poses")

    num_frames = rotations.shape[0]
    if keyframes_only:
        if "keyframe_time_indices" not in params:
            raise KeyError("params keyframe loading requires 'keyframe_time_indices'")
        frame_ids = np.asarray(params["keyframe_time_indices"], dtype=np.int64).reshape(-1)
    else:
        frame_ids = np.arange(num_frames, dtype=np.int64)
    if frame_ids.size == 0:
        raise ValueError("Reference view set must not be empty")
    if np.any(frame_ids < 0) or np.any(frame_ids >= num_frames):
        raise IndexError("keyframe_time_indices contain values outside the estimated camera trajectory")
    if np.unique(frame_ids).size != frame_ids.size:
        raise ValueError("keyframe_time_indices must be unique")
    if max_reference_poses is not None:
        if max_reference_poses <= 0:
            raise ValueError("max_reference_poses must be positive or None")
        if frame_ids.size > max_reference_poses:
            positions = np.linspace(0, frame_ids.size - 1, int(max_reference_poses), dtype=np.int64)
            frame_ids = frame_ids[positions]

    intrinsics = np.asarray(params["intrinsics"], dtype=np.float64)
    if intrinsics.shape == (4, 4):
        intrinsics = intrinsics[:3, :3]
    if intrinsics.shape != (3, 3) or np.any(~np.isfinite(intrinsics)):
        raise ValueError(f"Expected finite intrinsics with shape [3, 3], got {intrinsics.shape}")
    width = int(np.asarray(params["org_width"]).reshape(()))
    height = int(np.asarray(params["org_height"]).reshape(()))
    if width <= 0 or height <= 0:
        raise ValueError("org_width and org_height must be positive")

    all_w2c = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], num_frames, axis=0)
    all_w2c[:, :3, :3] = _quaternion_rotations(rotations)
    all_w2c[:, :3, 3] = translations
    return ReferenceViews(
        w2c=all_w2c[frame_ids],
        intrinsics=intrinsics,
        width=width,
        height=height,
        frame_ids=frame_ids,
        source=f"params_keyframes:{source_path}" if keyframes_only else f"params_estimated_trajectory:{source_path}",
    )


def build_frustum_center_factors(
    means: np.ndarray,
    normals: np.ndarray,
    views: ReferenceViews,
    gaussian_weights: np.ndarray | None = None,
    pair_cap: int | None = 2_000_000,
    per_pose_cap: int | None = None,
    chunk_size: int = 65_536,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    frustum_margin: float = 0.0,
) -> VisibilityFactors:
    """Build robust-tracker-compatible point-to-plane factors for visible map points."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3 or np.any(~np.isfinite(means)):
        raise ValueError(f"Expected finite means with shape [N, 3], got {means.shape}")
    if means.shape[0] == 0:
        raise ValueError("means must not be empty")
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != means.shape or np.any(~np.isfinite(normals)):
        raise ValueError(f"Expected finite normals with shape {means.shape}, got {normals.shape}")
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(normal_norms <= 1e-12):
        raise ValueError("normals must be non-zero")
    normals = normals / normal_norms
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if pair_cap is not None and pair_cap <= 0:
        raise ValueError("pair_cap must be positive or None")
    if per_pose_cap is not None and per_pose_cap <= 0:
        raise ValueError("per_pose_cap must be positive or None")
    if min_depth <= 0 or (max_depth is not None and max_depth <= min_depth):
        raise ValueError("Depth limits must satisfy 0 < min_depth < max_depth")

    num_gaussians = means.shape[0]
    if pair_cap is not None:
        derived_per_pose_cap = int(pair_cap) // views.w2c.shape[0]
        if per_pose_cap is None:
            per_pose_cap = derived_per_pose_cap
        else:
            per_pose_cap = min(int(per_pose_cap), derived_per_pose_cap)
    if gaussian_weights is None:
        gaussian_weights = np.ones(num_gaussians, dtype=np.float64)
    else:
        gaussian_weights = np.asarray(gaussian_weights, dtype=np.float64)
        if gaussian_weights.shape != (num_gaussians,) or np.any(~np.isfinite(gaussian_weights)) or np.any(gaussian_weights < 0):
            raise ValueError(f"Expected finite non-negative gaussian_weights with shape [{num_gaussians}]")

    fx, fy = float(views.intrinsics[0, 0]), float(views.intrinsics[1, 1])
    cx, cy = float(views.intrinsics[0, 2]), float(views.intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Focal lengths must be positive")
    x_min = -float(frustum_margin) * views.width
    x_max = (1.0 + float(frustum_margin)) * views.width
    y_min = -float(frustum_margin) * views.height
    y_max = (1.0 + float(frustum_margin)) * views.height

    pose_ptr = [0]
    all_ids: list[np.ndarray] = []
    all_jacobians: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    total_pairs = 0
    visible_pose_count = 0
    for pose in views.w2c:
        pose_ids: list[np.ndarray] = []
        pose_jacobians: list[np.ndarray] = []
        pose_weights: list[np.ndarray] = []
        pose_count = 0
        pose_has_visible = False
        for start in range(0, num_gaussians, int(chunk_size)):
            if per_pose_cap is not None and per_pose_cap > 0 and pose_count >= per_pose_cap:
                break
            stop = min(start + int(chunk_size), num_gaussians)
            camera_points = means[start:stop] @ pose[:3, :3].T + pose[:3, 3]
            z = camera_points[:, 2]
            safe_z = np.where(np.abs(z) > 1e-15, z, 1.0)
            u = fx * camera_points[:, 0] / safe_z + cx
            v = fy * camera_points[:, 1] / safe_z + cy
            visible = (z > min_depth) & (u >= x_min) & (u < x_max) & (v >= y_min) & (v < y_max)
            if max_depth is not None:
                visible &= z <= max_depth
            local_ids = np.flatnonzero(visible)
            pose_has_visible |= bool(local_ids.size)
            remaining = local_ids.size
            if per_pose_cap is not None:
                remaining = min(remaining, per_pose_cap - pose_count)
            local_ids = local_ids[:remaining]
            if local_ids.size == 0:
                continue

            points = camera_points[local_ids]
            ids = local_ids.astype(np.int64) + start
            camera_normals = normals[ids] @ pose[:3, :3].T
            jacobian = np.concatenate([np.cross(points, camera_normals), camera_normals], axis=1)
            weights = gaussian_weights[ids]
            pose_ids.append(ids)
            pose_jacobians.append(jacobian)
            pose_weights.append(weights)
            pose_count += ids.size
            total_pairs += ids.size

        if pose_ids:
            all_ids.append(np.concatenate(pose_ids))
            all_jacobians.append(np.concatenate(pose_jacobians))
            all_weights.append(np.concatenate(pose_weights))
        visible_pose_count += int(pose_has_visible)
        pose_ptr.append(total_pairs)

    return VisibilityFactors(
        pose_ptr=np.asarray(pose_ptr, dtype=np.int64),
        gaussian_ids=np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int64),
        jacobians=np.concatenate(all_jacobians) if all_jacobians else np.empty((0, 6), dtype=np.float64),
        weights=np.concatenate(all_weights) if all_weights else np.empty(0, dtype=np.float64),
        num_gaussians=num_gaussians,
        visible_pose_count=visible_pose_count,
    )


def trajectory_trace_scores(factors: VisibilityFactors, num_gaussians: int | None = None) -> np.ndarray:
    num_gaussians = factors.num_gaussians if num_gaussians is None else int(num_gaussians)
    if num_gaussians <= 0:
        raise ValueError("num_gaussians must be positive")
    contributions = factors.weights * np.sum(factors.jacobians * factors.jacobians, axis=1)
    return np.bincount(factors.gaussian_ids, weights=contributions, minlength=num_gaussians).astype(np.float64)


def aggregate_information_for_indices(
    factors: VisibilityFactors,
    indices: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Return one regularized 6x6 information matrix per reference pose."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"Expected indices with shape [N], got {indices.shape}")
    if np.any(indices < 0) or np.any(indices >= factors.num_gaussians):
        raise IndexError("indices contain values outside the factor Gaussian range")
    if np.unique(indices).size != indices.size:
        raise ValueError("indices must be unique")

    information = np.repeat(
        (float(regularization) * np.eye(6, dtype=np.float64))[None], factors.num_poses, axis=0
    )
    eligible = np.zeros(factors.num_gaussians, dtype=bool)
    eligible[indices] = True
    for pose_id, (start, stop) in enumerate(zip(factors.pose_ptr[:-1], factors.pose_ptr[1:])):
        pair_ids = np.arange(start, stop, dtype=np.int64)
        if pair_ids.size == 0:
            continue
        pair_ids = pair_ids[eligible[factors.gaussian_ids[pair_ids]]]
        if pair_ids.size == 0:
            continue
        weighted = factors.jacobians[pair_ids] * np.sqrt(factors.weights[pair_ids])[:, None]
        information[pose_id] += weighted.T @ weighted
    return information


def _distribution_summary(values: np.ndarray, preserved_threshold: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "preserved_fraction": float(np.mean(values >= preserved_threshold)),
    }


def compare_replacement_information(
    factors: VisibilityFactors,
    removed_indices: np.ndarray,
    replacement_indices: np.ndarray,
    regularization: float = 1e-6,
    psd_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compare replacement-only and removed-only information at every pose."""
    if psd_tolerance < 0:
        raise ValueError("psd_tolerance must be non-negative")
    removed = aggregate_information_for_indices(factors, removed_indices, regularization)
    replacement = aggregate_information_for_indices(factors, replacement_indices, regularization)
    removed_trace = np.trace(removed, axis1=1, axis2=2)
    replacement_trace = np.trace(replacement, axis1=1, axis2=2)
    trace_ratio = replacement_trace / removed_trace
    removed_logdet = np.linalg.slogdet(removed)[1]
    replacement_logdet = np.linalg.slogdet(replacement)[1]
    logdet_delta = replacement_logdet - removed_logdet
    removed_lambda_min = np.linalg.eigvalsh(removed)[:, 0]
    replacement_lambda_min = np.linalg.eigvalsh(replacement)[:, 0]
    lambda_min_delta = replacement_lambda_min - removed_lambda_min
    difference_lambda_min = np.linalg.eigvalsh(replacement - removed)[:, 0]
    psd_preserved = difference_lambda_min >= -float(psd_tolerance)
    return {
        "scope": "per_pose_replacement_set_vs_removed_set_regularized_information",
        "removed_per_pose_information": removed,
        "replacement_per_pose_information": replacement,
        "trace_ratio": trace_ratio,
        "logdet_delta": logdet_delta,
        "lambda_min_delta": lambda_min_delta,
        "difference_lambda_min": difference_lambda_min,
        "psd_preserved": psd_preserved,
        "summary": {
            "trace_ratio": _distribution_summary(trace_ratio, 1.0),
            "logdet_delta": _distribution_summary(logdet_delta, 0.0),
            "lambda_min_delta": _distribution_summary(lambda_min_delta, 0.0),
            "difference_lambda_min": _distribution_summary(difference_lambda_min, -float(psd_tolerance)),
            "psd_preserved_fraction": float(np.mean(psd_preserved)),
        },
    }


def rank1_logdet_gain(inverse: np.ndarray, jacobian: np.ndarray, weight: float) -> float:
    quadratic = float(jacobian @ inverse @ jacobian)
    return float(np.log1p(max(float(weight) * quadratic, 0.0)))


def sherman_morrison_rank1(inverse: np.ndarray, jacobian: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return inverse.copy()
    vector = np.sqrt(float(weight)) * np.asarray(jacobian, dtype=np.float64)
    projected = inverse @ vector
    denominator = 1.0 + float(vector @ projected)
    return inverse - np.outer(projected, projected) / denominator


def weak_direction_score(
    information_matrix: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
) -> float:
    _, eigenvectors = np.linalg.eigh(np.asarray(information_matrix, dtype=np.float64))
    weak_direction = eigenvectors[:, 0]
    projections = np.asarray(jacobians, dtype=np.float64) @ weak_direction
    return float(np.sum(np.asarray(weights, dtype=np.float64) * projections * projections))


def _stable_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    ids = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((ids, -np.asarray(scores, dtype=np.float64)))
    return order[:count].astype(np.int64)


def _candidate_pose_contributions(
    factors: VisibilityFactors, candidate_indices: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pre-aggregate each candidate's rank factors into per-pose 6x6 blocks."""
    candidate_set = set(int(value) for value in candidate_indices)
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {value: [] for value in candidate_set}
    for pose_id, (start, stop) in enumerate(zip(factors.pose_ptr[:-1], factors.pose_ptr[1:])):
        pose_ids = factors.gaussian_ids[start:stop]
        if pose_ids.size == 0:
            continue
        order = np.argsort(pose_ids, kind="mergesort")
        sorted_ids = pose_ids[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            gaussian_id = int(sorted_ids[left])
            if gaussian_id not in candidate_set:
                continue
            pair_ids = start + order[left:right]
            weighted = factors.jacobians[pair_ids] * np.sqrt(factors.weights[pair_ids])[:, None]
            grouped[gaussian_id].append((pose_id, weighted.T @ weighted))
    result = {}
    for gaussian_id, entries in grouped.items():
        result[gaussian_id] = (
            np.asarray([entry[0] for entry in entries], dtype=np.int64),
            np.asarray([entry[1] for entry in entries], dtype=np.float64).reshape(-1, 6, 6),
        )
    return result


def _weak_vectors(information: np.ndarray) -> np.ndarray:
    """Compute one weak eigenvector per pose for a whole bonus step."""
    _, eigenvectors = np.linalg.eigh(np.asarray(information, dtype=np.float64))
    return eigenvectors[:, :, 0]


def _candidate_gain(
    inverse: np.ndarray,
    pose_ids: np.ndarray,
    contributions: np.ndarray,
    weak_vectors: np.ndarray | None = None,
    weak_bonus: float = 0.0,
) -> float:
    gain = 0.0
    identity = np.eye(6, dtype=np.float64)
    for pose_id, contribution in zip(pose_ids, contributions):
        sign, value = np.linalg.slogdet(identity + inverse[pose_id] @ contribution)
        if sign > 0:
            gain += float(value)
        if weak_vectors is not None:
            vector = weak_vectors[pose_id]
            gain += float(weak_bonus) * float(vector @ contribution @ vector)
    return gain


def select_observable_gaussians(
    factors: VisibilityFactors,
    budget: int,
    candidate_cap: int | None = 4096,
    greedy_core_cap: int | None = 4096,
    regularization: float = 1e-6,
    min_eigen_every: int = 0,
    min_eigen_bonus: float = 0.0,
    eligible_indices: np.ndarray | None = None,
) -> ObservableSelection:
    num_gaussians = int(factors.num_gaussians)
    if budget <= 0:
        raise ValueError("budget must be positive")
    budget = min(int(budget), num_gaussians)
    if candidate_cap is not None and candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive or None")
    if greedy_core_cap is not None and greedy_core_cap < 0:
        raise ValueError("greedy_core_cap must be non-negative or None")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if min_eigen_every < 0 or min_eigen_bonus < 0:
        raise ValueError("min-eigen controls must be non-negative")

    if eligible_indices is None:
        eligible = np.arange(num_gaussians, dtype=np.int64)
    else:
        eligible = np.asarray(eligible_indices, dtype=np.int64)
        if eligible.ndim != 1:
            raise ValueError(f"Expected eligible_indices with shape [N], got {eligible.shape}")
        if np.any(eligible < 0) or np.any(eligible >= num_gaussians):
            raise IndexError("eligible_indices contain values outside the Gaussian range")
        if np.unique(eligible).size != eligible.size:
            raise ValueError("eligible_indices must be unique")
        eligible = np.sort(eligible)

    trace_scores = trajectory_trace_scores(factors, num_gaussians)
    configured_candidates = eligible.size if candidate_cap is None else int(candidate_cap)
    candidate_count = min(eligible.size, configured_candidates)
    if budget > candidate_count:
        raise ValueError("budget must not exceed the bounded candidate count")
    eligible_order = np.lexsort((eligible, -trace_scores[eligible]))
    candidates = eligible[eligible_order[:candidate_count]].astype(np.int64)
    core_budget = budget if greedy_core_cap is None else min(int(greedy_core_cap), budget, candidate_count)
    contributions = _candidate_pose_contributions(factors, candidates)
    information = np.repeat(
        (float(regularization) * np.eye(6, dtype=np.float64))[None], factors.num_poses, axis=0
    )
    inverse = np.repeat(
        (np.eye(6, dtype=np.float64) / float(regularization))[None], factors.num_poses, axis=0
    )
    selected: list[int] = []
    available = set(int(idx) for idx in candidates)
    heap: list[tuple[float, int, int]] = []
    state_version = 0

    def rebuild_heap(weak_vectors: np.ndarray | None = None) -> None:
        heap.clear()
        for gaussian_id in sorted(available):
            pose_ids, blocks = contributions[gaussian_id]
            gain = _candidate_gain(inverse, pose_ids, blocks, weak_vectors, min_eigen_bonus)
            heapq.heappush(heap, (-gain, gaussian_id, state_version))

    rebuild_heap()

    for step in range(core_budget):
        use_weak_bonus = min_eigen_every > 0 and min_eigen_bonus > 0 and step % min_eigen_every == 0
        weak_vectors = _weak_vectors(information) if use_weak_bonus else None
        if use_weak_bonus:
            rebuild_heap(weak_vectors)
        best_id = None
        while heap:
            _, gaussian_id, version = heapq.heappop(heap)
            if gaussian_id not in available:
                continue
            pose_ids, blocks = contributions[gaussian_id]
            gain = _candidate_gain(inverse, pose_ids, blocks, weak_vectors, min_eigen_bonus)
            next_bound = -heap[0][0] if heap else -np.inf
            if version == state_version or gain >= next_bound - 1e-12:
                best_id = gaussian_id
                break
            heapq.heappush(heap, (-gain, gaussian_id, state_version))
        if best_id is None:
            break
        selected.append(best_id)
        available.remove(best_id)
        pose_ids, blocks = contributions[best_id]
        for pose_id, contribution in zip(pose_ids, blocks):
            information[pose_id] += contribution
            inverse[pose_id] = np.linalg.inv(information[pose_id])
        state_version += 1

    indices = np.asarray(selected, dtype=np.int64)
    signs, per_pose_logdet = np.linalg.slogdet(information)
    per_pose_logdet = np.where(signs > 0, per_pose_logdet, -np.inf)
    per_pose_eigvals = np.linalg.eigvalsh(information)
    aggregate_information = np.sum(information, axis=0)
    aggregate_sign, aggregate_logdet = np.linalg.slogdet(aggregate_information)
    aggregate_eigvals = np.linalg.eigvalsh(aggregate_information)
    return ObservableSelection(
        indices=indices,
        trace_scores=trace_scores,
        greedy_indices=np.asarray(selected, dtype=np.int64),
        fill_indices=np.empty(0, dtype=np.int64),
        information_matrix=aggregate_information,
        logdet=float(aggregate_logdet) if aggregate_sign > 0 else float("-inf"),
        lambda_min=float(aggregate_eigvals[0]),
        candidate_indices=candidates,
        per_pose_information=information,
        per_pose_logdet=per_pose_logdet,
        per_pose_lambda_min=per_pose_eigvals[:, 0],
        objective=float(np.sum(per_pose_logdet)),
    )
