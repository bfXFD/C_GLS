"""Frozen same-frame holdout consistency checks, not safety/generalization claims.

Poses map source coordinates into target coordinates. Costs are means over ALL
frozen source points, including unmatched points at the truncation penalty.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


_DEFAULT_CONFIG = {
    "relative_tolerance": 0.0,
    "absolute_tolerance": 1e-12,
    "euclidean_relative_tolerance": 0.0,
    "euclidean_absolute_tolerance": 1e-12,
    "min_coverage_drop": 0.0,
    "max_group_cost_increase": 1e-12,
    "min_group_size": 6,
    "max_translation": 0.1,
    "max_rotation_deg": 5.0,
}

VERIFIER_METRIC_FIELDS = tuple(
    "reserve_verifier_" + name
    for name in (
        "primary_cost", "candidate_cost", "primary_euclidean_cost",
        "candidate_euclidean_cost", "primary_coverage", "candidate_coverage",
        "primary_count", "candidate_count", "primary_valid", "candidate_valid",
        "worst_group_delta", "worst_group_euclidean_delta",
        "translation", "rotation_deg", "source_count",
        "accepted", "reason", "primary_reason", "candidate_reason",
    )
)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def validate_verifier_config(config=None) -> dict:
    """Validate pair options before constructing a verifier or starting a run.

    Accept None or a mapping of optional overrides; return a fresh dictionary
    containing every default and normalized Python float/int value. Unknown
    keys, nonfinite/negative tolerances, and invalid bounds raise ValueError.
    min_group_size must be a positive integer, min_coverage_drop <= 1, and
    max_rotation_deg <= 180. max_group_cost_increase bounds BOTH point-to-plane
    and Euclidean group cost increases for groups meeting min_group_size.
    The input mapping and module defaults are never mutated.
    """
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("config must be a mapping or None")
    options = dict(_DEFAULT_CONFIG)
    if config is not None:
        if set(config) - set(options):
            raise ValueError("unknown verifier configuration key")
        options.update(config)
    for name, value in options.items():
        options[name] = (_integer(value, name) if name == "min_group_size"
                         else _number(value, name))
    if options["min_coverage_drop"] > 1 or options["max_rotation_deg"] > 180:
        raise ValueError("min_coverage_drop <= 1 and max_rotation_deg <= 180 required")
    return options


def _readonly(array):
    # A bytes backing store prevents callers from re-enabling WRITEABLE.
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def split_source_validation(pixel_ids, width, block_size=16, folds=4, fold=0):
    """Return a deterministic bool mask (True = validation), without GT.

    ``pixel_ids`` are nonnegative flattened row-major integer pixel indices.
    Entire (block row, block column) pairs share a fixed integer hash/fold.
    Duplicates and input reordering preserve membership; folds partition input.
    Empty input yields an empty mask. No image height is needed.
    """
    width = _integer(width, "width")
    block_size = _integer(block_size, "block_size")
    folds = _integer(folds, "folds", 2)
    fold = _integer(fold, "fold", 0)
    if fold >= folds:
        raise ValueError("fold must be smaller than folds")
    ids = np.asarray(pixel_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("pixel_ids must be a one-dimensional nonnegative integer array")
    # Python integers deliberately avoid platform-dependent overflow behavior.
    return np.fromiter(
        (((int(i) // width // block_size) * 73856093
          ^ (int(i) % width // block_size) * 19349663) % folds == fold
         for i in ids), dtype=bool, count=ids.size,
    )


def _pose(value):
    try:
        pose = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError):
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    rotation = pose[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        defect = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    # Match the spectral diagnostics' input tolerance for float32 poses.
    if not (np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            and defect <= 1e-6 and abs(np.linalg.det(rotation) - 1) <= 1e-6):
        return None
    if defect > 8 * np.finfo(np.float64).eps:
        left, _, right = np.linalg.svd(rotation)
        pose[:3, :3] = left @ right
    return pose


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FrozenCandidateVerifier:
    """Immutable copied geometry and fixed KD-tree for holdout comparison.

    Constructor errors raise ValueError. Invalid poses/overflow/insufficient
    correspondences produce finite evaluations with valid=False. Normals are
    normalized once; empty/nonfinite geometry and zero normals are rejected.
    Spatial groups, when provided, are nonnegative integer labels per source.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    target_normals: np.ndarray
    max_distance: float
    loss_scale: float
    robust_loss: str
    min_correspondences: int
    spatial_groups: np.ndarray
    _tree: cKDTree = field(repr=False)
    _max_cost: float
    _group_ids: np.ndarray = field(repr=False)
    _group_inverse: np.ndarray = field(repr=False)
    _group_sizes: np.ndarray = field(repr=False)

    def __init__(self, source_points, target_points, target_normals, max_distance,
                 loss_scale, robust_loss="huber", min_correspondences=6,
                 spatial_groups=None):
        distance = _number(max_distance, "max_distance", positive=True)
        scale = _number(loss_scale, "loss_scale", positive=True)
        minimum = _integer(min_correspondences, "min_correspondences")
        if not isinstance(robust_loss, str) or robust_loss not in ("none", "huber", "cauchy"):
            raise ValueError("robust_loss must be 'none', 'huber', or 'cauchy'")
        for name, value in (("source_points", source_points), ("target_points", target_points),
                            ("target_normals", target_normals)):
            array = np.array(value, dtype=np.float64, copy=True)
            if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be nonempty finite [N, 3]")
            object.__setattr__(self, name, _readonly(array))
        if self.target_normals.shape != self.target_points.shape:
            raise ValueError("target_normals must match target_points")
        # Scaling before the norm avoids overflow for large finite normals.
        factors = np.max(np.abs(self.target_normals), axis=1)
        if np.any(factors == 0):
            raise ValueError("target_normals must be nonzero")
        normals = self.target_normals / factors[:, None]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        object.__setattr__(self, "target_normals", _readonly(normals))
        groups = (np.zeros(len(self.source_points), dtype=np.int64)
                  if spatial_groups is None else np.asarray(spatial_groups))
        if groups.shape != (len(self.source_points),) or groups.dtype.kind not in "iu" or np.any(groups < 0):
            raise ValueError("spatial_groups must be nonnegative integer labels per source")
        ids, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
        for name, value in (("max_distance", distance), ("loss_scale", scale),
                            ("min_correspondences", minimum), ("robust_loss", robust_loss),
                            ("spatial_groups", _readonly(groups)), ("_group_ids", _readonly(ids)),
                            ("_group_inverse", _readonly(inverse)), ("_group_sizes", _readonly(sizes))):
            object.__setattr__(self, name, value)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            maximum = float(self._loss(np.array([distance]))[0])
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("max_distance/loss_scale yield an unrepresentable truncation cost")
        object.__setattr__(self, "_max_cost", maximum)
        object.__setattr__(self, "_tree", cKDTree(self.target_points, copy_data=False))

    def _loss(self, residual):
        """Per-point version of fsm_robust_registration._robust_cost."""
        absolute = np.abs(residual)
        if self.robust_loss == "none":
            return (0.5 * absolute) * absolute
        if self.robust_loss == "huber":
            clipped = np.minimum(absolute, self.loss_scale)
            return (0.5 * clipped) * clipped + self.loss_scale * (absolute - clipped)
        # logaddexp avoids overflow of (residual / scale)**2.
        with np.errstate(divide="ignore"):
            log_term = np.logaddexp(0.0, 2 * (np.log(absolute) - np.log(self.loss_scale)))
        return (0.5 * self.loss_scale) * (self.loss_scale * log_term)

    def evaluate(self, pose):
        """Return valid/reason/cost/euclidean_cost/coverage/count/group costs.

        group_costs, group_euclidean_costs and group_counts are dictionaries
        keyed by integer labels. Group counts include unmatched source points.
        All numeric outputs are finite, including failure outputs.
        """
        n = len(self.source_points)
        costs = np.full(n, self._max_cost)
        euclidean = costs.copy()
        matched = np.zeros(n, dtype=bool)
        transform = _pose(pose)
        reason = "invalid_pose" if transform is None else "insufficient_correspondences"
        if transform is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                transformed = self.source_points @ transform[:3, :3].T + transform[:3, 3]
            finite = np.all(np.isfinite(transformed), axis=1)
            indices = np.flatnonzero(finite)
            if indices.size:
                distances, nearest = self._tree.query(transformed[indices], k=1)
                usable = (np.isfinite(distances) & (distances <= self.max_distance)
                          & (nearest < len(self.target_points)))
                indices, nearest, distances = indices[usable], nearest[usable], distances[usable]
                with np.errstate(over="ignore", invalid="ignore"):
                    residuals = np.sum((transformed[indices] - self.target_points[nearest])
                                       * self.target_normals[nearest], axis=1)
                good = np.isfinite(residuals)
                indices, residuals, distances = indices[good], residuals[good], distances[good]
                matched[indices] = True
                costs[indices] = np.minimum(self._loss(np.minimum(np.abs(residuals), self.max_distance)), self._max_cost)
                euclidean[indices] = np.minimum(self._loss(distances), self._max_cost)
            if not np.all(finite):
                reason = "nonfinite_transformed_source"
        count = int(np.count_nonzero(matched))
        valid = transform is not None and reason != "nonfinite_transformed_source" and count >= self.min_correspondences
        group_costs, group_euclidean, group_counts = {}, {}, {}
        for index, (label, size) in enumerate(zip(self._group_ids, self._group_sizes)):
            mask = self._group_inverse == index
            group_costs[int(label)] = float(np.sum(costs[mask] / int(size)))
            group_euclidean[int(label)] = float(np.sum(euclidean[mask] / int(size)))
            group_counts[int(label)] = int(size)
        return {"valid": bool(valid), "reason": "valid" if valid else reason,
                "cost": float(np.sum(costs / n)), "euclidean_cost": float(np.sum(euclidean / n)),
                "coverage": float(count / n), "count": count, "group_costs": group_costs,
                "group_euclidean_costs": group_euclidean, "group_counts": group_counts}

    def evaluate_pair(self, primary_pose, candidate_pose, config=None):
        """Return {accepted: bool, reason: str, metrics: fixed scalar dict}.

        config is a mapping with optional keys/defaults from _DEFAULT_CONFIG:
        relative_tolerance=0, absolute_tolerance=1e-12;
        euclidean_relative_tolerance=0, euclidean_absolute_tolerance=1e-12;
        min_coverage_drop=0 (allowed coverage loss, range [0,1]);
        max_group_cost_increase=1e-12 (both plane and Euclidean groups),
        min_group_size=6;
        max_translation=0.1 (SE3 log translation norm, source length units),
        max_rotation_deg=5 (range [0,180]). All tolerances/bounds are finite
        nonnegative reals; min_group_size is a positive integer. Unknown keys
        and invalid values raise ValueError, never yielding an acceptance.
        Relative motion is Log(inv(primary) @ candidate), invariant under a
        common rigid change of TARGET coordinates (not source-origin shifts).
        """
        options = validate_verifier_config(config)
        primary, candidate = self.evaluate(primary_pose), self.evaluate(candidate_pose)
        worst = max((candidate["group_costs"][g] - primary["group_costs"][g]
                     for g, size in primary["group_counts"].items()
                     if size >= options["min_group_size"]), default=0.0)
        worst_euclidean = max((candidate["group_euclidean_costs"][g]
                               - primary["group_euclidean_costs"][g]
                               for g, size in primary["group_counts"].items()
                               if size >= options["min_group_size"]), default=0.0)
        translation = rotation_deg = 0.0
        reason = "accepted"
        if not primary["valid"]:
            reason = "invalid_primary"
        elif not candidate["valid"]:
            reason = "invalid_candidate"
        else:
            p, c = _pose(primary_pose), _pose(candidate_pose)
            with np.errstate(over="ignore", invalid="ignore"):
                rotation = Rotation.from_matrix(p[:3, :3].T @ c[:3, :3]).as_rotvec()
                offset = p[:3, :3].T @ (c[:3, 3] - p[:3, 3])
                theta = float(np.linalg.norm(rotation))
                x, y, z = rotation
                omega = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                coefficient = (1 / 12 + theta * theta / 720 if theta < 1e-5
                               else (1 - 0.5 * theta / np.tan(theta / 2)) / (theta * theta))
                velocity = (np.eye(3) - 0.5 * omega + coefficient * (omega @ omega)) @ offset
                translation = float(np.hypot.reduce(velocity))
                rotation_deg = float(np.degrees(theta))
            if not np.isfinite(translation) or not np.isfinite(rotation_deg):
                translation = rotation_deg = float(np.finfo(float).max)
                reason = "invalid_relative_motion"
            elif translation > options["max_translation"]:
                reason = "translation_bound"
            elif rotation_deg > options["max_rotation_deg"]:
                reason = "rotation_bound"
            # Divide first to avoid overflow in primary*(1+relative)+absolute.
            elif ((candidate["cost"] - options["absolute_tolerance"]) / (1 + options["relative_tolerance"]) > primary["cost"]):
                reason = "cost_increase"
            elif ((candidate["euclidean_cost"] - options["euclidean_absolute_tolerance"]) / (1 + options["euclidean_relative_tolerance"]) > primary["euclidean_cost"]):
                reason = "euclidean_cost_increase"
            elif candidate["coverage"] < primary["coverage"] - options["min_coverage_drop"]:
                reason = "coverage_drop"
            elif worst > options["max_group_cost_increase"]:
                reason = "spatial_group_degradation"
            elif worst_euclidean > options["max_group_cost_increase"]:
                reason = "spatial_group_euclidean_degradation"
        accepted = reason == "accepted"
        values = (primary["cost"], candidate["cost"], primary["euclidean_cost"],
                  candidate["euclidean_cost"], primary["coverage"], candidate["coverage"],
                  primary["count"], candidate["count"], primary["valid"], candidate["valid"],
                  float(worst), float(worst_euclidean), translation, rotation_deg,
                  len(self.source_points), accepted, reason, primary["reason"], candidate["reason"])
        return {"accepted": accepted, "reason": reason,
                "metrics": dict(zip(VERIFIER_METRIC_FIELDS, values))}
