import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalSubmapConfig:
    near: float
    far: float
    margin_px: float = 0.0
    radius_margin: float = 0.0
    max_points: int | None = None
    sampling_policy: str = "legacy"
    tile_rows: int = 12
    tile_cols: int = 16
    depth_bins: int = 8
    frustum_culling: str = "radius"
    block_size_m: float = 0.3
    projection_pre_cap: int | None = None
    projection_sampling_policy: str = "block_coverage"
    block_test_chunk: int = 65536
    oracle_mode: str = "off"
    oracle_overflow_policy: str = "uncapped_chunked"
    oracle_point_chunk: int = 262144

    def __post_init__(self) -> None:
        if not np.isfinite(self.near) or self.near < 0.0:
            raise ValueError("near must be finite and non-negative")
        if not np.isfinite(self.far) or self.far <= self.near:
            raise ValueError("far must be finite and greater than near")
        if not np.isfinite(self.margin_px) or self.margin_px < 0.0:
            raise ValueError("margin_px must be finite and non-negative")
        if not np.isfinite(self.radius_margin) or self.radius_margin < 0.0:
            raise ValueError("radius_margin must be finite and non-negative")
        if self.max_points is not None and (
            isinstance(self.max_points, (bool, np.bool_)) or int(self.max_points) != self.max_points or self.max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer or None")
        if self.sampling_policy not in ("coverage", "legacy"):
            raise ValueError("sampling_policy must be 'coverage' or 'legacy'")
        if self.frustum_culling not in ("radius", "voxel_blocks"):
            raise ValueError("frustum_culling must be 'radius' or 'voxel_blocks'")
        if not np.isfinite(self.block_size_m) or self.block_size_m <= 0.0:
            raise ValueError("block_size_m must be finite and positive")
        if self.projection_pre_cap is not None and (
            isinstance(self.projection_pre_cap, (bool, np.bool_))
            or int(self.projection_pre_cap) != self.projection_pre_cap
            or self.projection_pre_cap <= 0
        ):
            raise ValueError("projection_pre_cap must be a positive integer or None")
        if self.projection_sampling_policy != "block_coverage":
            raise ValueError("projection_sampling_policy must be 'block_coverage'")
        if self.oracle_mode not in ("off", "measure", "enforce"):
            raise ValueError("oracle_mode must be 'off', 'measure', or 'enforce'")
        if self.oracle_overflow_policy not in ("uncapped_chunked", "fail_closed"):
            raise ValueError("oracle_overflow_policy must be 'uncapped_chunked' or 'fail_closed'")
        for value, name in (
            (self.tile_rows, "tile_rows"),
            (self.tile_cols, "tile_cols"),
            (self.depth_bins, "depth_bins"),
            (self.block_test_chunk, "block_test_chunk"),
            (self.oracle_point_chunk, "oracle_point_chunk"),
        ):
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BlockCSR:
    coords: np.ndarray
    offsets: np.ndarray
    point_ids: np.ndarray
    block_best_score: np.ndarray
    scores: np.ndarray | None
    build_ms: float
    score_hash_ms: float
    geometry_sort_ms: float
    memory_bytes: int
    peak_working_set_estimate_bytes: int
    score_fingerprint: str
    block_size_m: float


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not np.issubdtype(points.dtype, np.number) or np.any(~np.isfinite(points)):
        raise ValueError("points must contain only finite numeric values")
    return points


def _validate_camera(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    c2w = np.asarray(c2w, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if c2w.shape != (4, 4) or np.any(~np.isfinite(c2w)):
        raise ValueError(f"Expected finite c2w with shape [4, 4], got {c2w.shape}")
    if not np.allclose(c2w[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("c2w must be an affine homogeneous transform")
    rotation = c2w[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("c2w rotation must be orthonormal with determinant 1")
    if intrinsics.shape != (3, 3) or np.any(~np.isfinite(intrinsics)):
        raise ValueError(f"Expected finite intrinsics with shape [3, 3], got {intrinsics.shape}")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not np.allclose(intrinsics[2], np.array([0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("intrinsics must use the OpenCV pinhole form")
    for value, name in ((image_height, "image_height"), (image_width, "image_width")):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return c2w, intrinsics, int(image_height), int(image_width)


def _camera_points(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    rotation_c2w = c2w[:3, :3]
    camera_center = c2w[:3, 3]
    return (points_world - camera_center) @ rotation_c2w


def points_in_camera_frustum(
    points_world: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    config: LocalSubmapConfig,
) -> np.ndarray:
    """Return a mask for centers inside an OpenCV +z pinhole frustum."""
    points_world = _validate_points(points_world)
    c2w, intrinsics, image_height, image_width = _validate_camera(
        c2w, intrinsics, image_height, image_width
    )
    if not isinstance(config, LocalSubmapConfig):
        raise TypeError("config must be a LocalSubmapConfig")

    points_camera = _camera_points(points_world.astype(np.float64, copy=False), c2w)
    depth = points_camera[:, 2]
    depth_mask = (depth >= config.near) & (depth <= config.far) & (depth > 0.0)
    mask = np.zeros(points_world.shape[0], dtype=bool)
    valid = np.flatnonzero(depth_mask)
    if valid.size == 0:
        return mask

    projected_u = intrinsics[0, 0] * points_camera[valid, 0] / depth[valid] + intrinsics[0, 2]
    projected_v = intrinsics[1, 1] * points_camera[valid, 1] / depth[valid] + intrinsics[1, 2]
    margin = config.margin_px
    mask[valid] = (
        (projected_u >= -margin)
        & (projected_u < image_width + margin)
        & (projected_v >= -margin)
        & (projected_v < image_height + margin)
    )
    return mask


def coverage_sampling_components(
    projected_u: np.ndarray,
    projected_v: np.ndarray,
    depth: np.ndarray,
    image_height: int,
    image_width: int,
    config: LocalSubmapConfig,
    positive_near: float | None = None,
) -> dict[str, np.ndarray]:
    """Return the exact tile/depth bucket assignment used by coverage sampling."""
    projected_u = np.asarray(projected_u, dtype=np.float64)
    projected_v = np.asarray(projected_v, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if projected_u.ndim != 1 or projected_v.shape != projected_u.shape or depth.shape != projected_u.shape:
        raise ValueError("projected_u, projected_v, and depth must have matching one-dimensional shapes")
    if np.any(~np.isfinite(projected_u)) or np.any(~np.isfinite(projected_v)) or np.any(~np.isfinite(depth)):
        raise ValueError("coverage sampling inputs must be finite")
    if np.any(depth <= 0.0):
        raise ValueError("coverage sampling depth must be positive")
    tile_col = np.clip((projected_u * config.tile_cols / image_width).astype(np.int64), 0, config.tile_cols - 1)
    tile_row = np.clip((projected_v * config.tile_rows / image_height).astype(np.int64), 0, config.tile_rows - 1)
    near = config.near
    if near <= 0.0:
        near = float(np.min(depth)) if positive_near is None else float(positive_near)
        if not np.isfinite(near) or near <= 0.0:
            raise ValueError("positive_near must be finite and positive")
    log_near = np.log(near)
    log_span = np.log(config.far) - log_near
    depth_bin = ((np.log(depth) - log_near) * config.depth_bins / max(log_span, 1e-12)).astype(np.int64)
    depth_bin = np.clip(depth_bin, 0, config.depth_bins - 1)
    tile_id = tile_row * config.tile_cols + tile_col
    return {
        "tile_id": tile_id,
        "depth_bin": depth_bin,
        "bucket_id": tile_id * config.depth_bins + depth_bin,
    }


def coverage_sampling_allocation(occupancy: np.ndarray, cap: int) -> np.ndarray:
    """Allocate a coverage cap to ascending non-empty bucket IDs."""
    occupancy = np.asarray(occupancy, dtype=np.int64)
    if occupancy.ndim != 1 or np.any(occupancy <= 0):
        raise ValueError("occupancy must be a one-dimensional array of positive integers")
    if isinstance(cap, (bool, np.bool_)) or int(cap) != cap or cap <= 0:
        raise ValueError("cap must be a positive integer")
    allocation = np.zeros(occupancy.size, dtype=np.int64)
    if occupancy.size <= int(cap):
        allocation[:] = 1
        remaining = int(cap) - occupancy.size
        capacity = occupancy - 1
        if remaining > 0 and np.any(capacity > 0):
            weights = np.sqrt(occupancy.astype(np.float64))
            ideal = remaining * weights / np.sum(weights)
            extra = np.minimum(np.floor(ideal).astype(np.int64), capacity)
            allocation += extra
            remaining -= int(np.sum(extra))
            while remaining > 0:
                available = allocation < occupancy
                if not np.any(available):
                    break
                remainder = ideal - extra
                choices = np.flatnonzero(available)
                chosen = choices[np.lexsort((choices, -remainder[choices]))[0]]
                allocation[chosen] += 1
                extra[chosen] += 1
                remaining -= 1
    else:
        allocation[: int(cap)] = 1
    return allocation


def camera_frustum_planes_world(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    config: LocalSubmapConfig,
) -> np.ndarray:
    """Return inward-facing normalized world-space planes as ``[nx, ny, nz, d]``."""
    c2w, intrinsics, image_height, image_width = _validate_camera(
        c2w, intrinsics, image_height, image_width
    )
    if not isinstance(config, LocalSubmapConfig):
        raise TypeError("config must be a LocalSubmapConfig")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    margin = config.margin_px
    camera_planes = np.array(
        [
            [0.0, 0.0, 1.0, -config.near],
            [0.0, 0.0, -1.0, config.far],
            [fx, 0.0, cx + margin, 0.0],
            [-fx, 0.0, image_width + margin - cx, 0.0],
            [0.0, fy, cy + margin, 0.0],
            [0.0, -fy, image_height + margin - cy, 0.0],
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(camera_planes[:, :3], axis=1)
    camera_planes /= norms[:, None]
    normals_world = camera_planes[:, :3] @ c2w[:3, :3].T
    offsets_world = camera_planes[:, 3] - normals_world @ c2w[:3, 3]
    return np.column_stack((normals_world, offsets_world))


def conservative_aabb_halfspace_test(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    planes: np.ndarray,
    epsilon: float | None = None,
) -> np.ndarray:
    """Conservatively test AABBs against inward halfspaces without false negatives."""
    lower = np.asarray(aabb_min, dtype=np.float64)
    upper = np.asarray(aabb_max, dtype=np.float64)
    plane_array = np.asarray(planes, dtype=np.float64)
    single = lower.ndim == 1
    if single:
        lower = lower[None, :]
        upper = upper[None, :]
    if lower.ndim != 2 or lower.shape[1] != 3 or upper.shape != lower.shape:
        raise ValueError("aabb_min and aabb_max must have matching shape [N, 3] or [3]")
    if plane_array.ndim != 2 or plane_array.shape[1] != 4:
        raise ValueError("planes must have shape [P, 4]")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(~np.isfinite(plane_array)):
        raise ValueError("AABBs and planes must be finite")
    if np.any(lower > upper):
        raise ValueError("aabb_min must not exceed aabb_max")
    normals = plane_array[:, :3]
    support = np.where(normals[None, :, :] >= 0.0, upper[:, None, :], lower[:, None, :])
    distance = np.einsum("npc,pc->np", support, normals) + plane_array[None, :, 3]
    scale = np.maximum(1.0, np.sum(np.abs(support * normals[None, :, :]), axis=2) + np.abs(plane_array[None, :, 3]))
    tolerance = 8.0 * np.finfo(np.float64).eps * scale if epsilon is None else float(epsilon)
    if np.any(~np.isfinite(np.asarray(tolerance))) or np.any(np.asarray(tolerance) < 0.0):
        raise ValueError("epsilon must be finite and non-negative")
    result = np.all(np.nextafter(distance, np.inf) >= -tolerance, axis=1)
    return result[0] if single else result


aabbs_intersect_halfspaces = conservative_aabb_halfspace_test
conservative_aabb_frustum_test = conservative_aabb_halfspace_test


class MapSpatialIndex:
    """Read-only CPU radius index over world-space map centers."""

    def __init__(
        self,
        points: np.ndarray,
        backend: str = "auto",
        voxel_size: float = 1.0,
    ) -> None:
        self.points = _validate_points(points)
        if backend not in ("auto", "scipy_ckdtree", "numpy_grid"):
            raise ValueError("backend must be 'auto', 'scipy_ckdtree', or 'numpy_grid'")
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size must be finite and positive")
        self.voxel_size = float(voxel_size)
        self._tree: Any | None = None
        self._grid: dict[tuple[int, int, int], np.ndarray] | None = None
        self._active_block_csr: BlockCSR | None = None
        self._active_block_score_token: str | None = None
        self._block_score_token_generation = 0

        start = time.perf_counter()
        if backend in ("auto", "scipy_ckdtree"):
            try:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(self.points)
                self.backend = "scipy_ckdtree"
            except (ImportError, ModuleNotFoundError):
                if backend == "scipy_ckdtree":
                    raise ImportError("scipy.spatial.cKDTree is unavailable")
                self._build_numpy_grid()
        else:
            self._build_numpy_grid()
        self.build_latency_ms = float((time.perf_counter() - start) * 1000.0)
        self.num_points = int(self.points.shape[0])

    @property
    def block_index_build_ms(self) -> float:
        return 0.0 if self._active_block_csr is None else self._active_block_csr.build_ms

    @property
    def block_index_memory_bytes(self) -> int:
        return 0 if self._active_block_csr is None else self._active_block_csr.memory_bytes

    @property
    def block_snapshot_count(self) -> int:
        return int(self._active_block_csr is not None)

    @property
    def memory_bytes(self) -> int:
        return self.block_index_memory_bytes

    def _build_numpy_grid(self) -> None:
        voxel_coords = np.floor(self.points.astype(np.float64, copy=False) / self.voxel_size).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, coord in enumerate(voxel_coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            buckets.setdefault(key, []).append(index)
        self._grid = {key: np.asarray(indices, dtype=np.int64) for key, indices in buckets.items()}
        self.backend = "numpy_grid"

    def query_radius(self, center: np.ndarray, radius: float, sort_result: bool = True) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or np.any(~np.isfinite(center)):
            raise ValueError(f"Expected finite center with shape [3], got {center.shape}")
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and non-negative")

        if self._tree is not None:
            indices = np.asarray(self._tree.query_ball_point(center, float(radius)), dtype=np.int64)
        else:
            lower = np.floor((center - radius) / self.voxel_size).astype(np.int64)
            upper = np.floor((center + radius) / self.voxel_size).astype(np.int64)
            buckets = []
            assert self._grid is not None
            for x in range(int(lower[0]), int(upper[0]) + 1):
                for y in range(int(lower[1]), int(upper[1]) + 1):
                    for z in range(int(lower[2]), int(upper[2]) + 1):
                        bucket = self._grid.get((x, y, z))
                        if bucket is not None:
                            buckets.append(bucket)
            indices = np.concatenate(buckets) if buckets else np.empty(0, dtype=np.int64)
            if indices.size:
                delta = self.points[indices].astype(np.float64, copy=False) - center
                indices = indices[np.einsum("ij,ij->i", delta, delta) <= radius * radius]
        if sort_result:
            indices = np.sort(indices, kind="stable")
        return indices.astype(np.int64, copy=False)

    def _score_array_and_fingerprint(
        self, scores: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str, float]:
        start = time.perf_counter()
        if scores is None:
            return None, "none", float((time.perf_counter() - start) * 1000.0)
        score_array = np.ascontiguousarray(np.asarray(scores, dtype=np.float64))
        if score_array.shape != (self.num_points,) or np.any(~np.isfinite(score_array)):
            raise ValueError(f"Expected finite scores with shape [{self.num_points}]")
        digest = hashlib.sha256()
        digest.update(score_array.dtype.str.encode("ascii"))
        digest.update(memoryview(np.asarray(score_array.shape, dtype=np.int64)))
        digest.update(memoryview(score_array).cast("B"))
        return score_array, digest.hexdigest(), float((time.perf_counter() - start) * 1000.0)

    def _block_metrics(self, csr: BlockCSR, cached: bool) -> dict[str, Any]:
        return {
            "block_index_build_ms": csr.build_ms,
            "block_score_hash_ms": csr.score_hash_ms,
            "block_geometry_sort_ms": csr.geometry_sort_ms,
            "block_index_memory_bytes": csr.memory_bytes,
            "peak_working_set_estimate_bytes": csr.peak_working_set_estimate_bytes,
            "block_count": int(csr.coords.shape[0]),
            "cached": cached,
            "score_fingerprint": csr.score_fingerprint,
            "block_score_token": self._active_block_score_token,
        }

    def prebuild_blocks(self, block_size: float, scores: np.ndarray | None = None) -> dict[str, Any]:
        """Build the sole active block snapshot and return its immutable score token.

        Callers using ``block_score_token`` promise to use the captured score snapshot;
        they must prebuild again after changing scores.
        """
        if not np.isfinite(block_size) or block_size <= 0.0:
            raise ValueError("block_size must be finite and positive")
        start = time.perf_counter()
        score_array, fingerprint, score_hash_ms = self._score_array_and_fingerprint(scores)
        block_size = float(block_size)
        active = self._active_block_csr
        if (
            active is not None
            and active.block_size_m == block_size
            and active.score_fingerprint == fingerprint
        ):
            return self._block_metrics(active, cached=True)

        geometry_start = time.perf_counter()
        coords_for_point = np.floor(self.points.astype(np.float64, copy=False) / block_size).astype(np.int64)
        point_ids = np.arange(self.num_points, dtype=np.int64)
        if score_array is None:
            order = np.lexsort((point_ids, coords_for_point[:, 2], coords_for_point[:, 1], coords_for_point[:, 0]))
        else:
            order = np.lexsort(
                (point_ids, -score_array, coords_for_point[:, 2], coords_for_point[:, 1], coords_for_point[:, 0])
            )
        sorted_coords = coords_for_point[order]
        starts = np.flatnonzero(np.r_[True, np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)])
        coords = sorted_coords[starts]
        offsets = np.r_[starts, self.num_points].astype(np.int64)
        ranked_point_ids = point_ids[order]
        block_best_score = (
            score_array[ranked_point_ids[starts]].astype(np.float64, copy=False)
            if score_array is not None else np.zeros(coords.shape[0], dtype=np.float64)
        )
        geometry_sort_ms = float((time.perf_counter() - geometry_start) * 1000.0)
        score_snapshot = None if score_array is None else score_array.copy()
        if score_snapshot is not None:
            score_snapshot.flags.writeable = False
        build_ms = float((time.perf_counter() - start) * 1000.0)
        memory_bytes = int(
            coords.nbytes + offsets.nbytes + ranked_point_ids.nbytes + block_best_score.nbytes
            + (0 if score_snapshot is None else score_snapshot.nbytes)
        )
        peak_working_set_estimate_bytes = int(
            memory_bytes + coords_for_point.nbytes + point_ids.nbytes + order.nbytes
            + sorted_coords.nbytes + starts.nbytes
            + (0 if score_array is None else score_array.nbytes)
        )
        csr = BlockCSR(
            coords=coords,
            offsets=offsets,
            point_ids=ranked_point_ids,
            block_best_score=block_best_score,
            scores=score_snapshot,
            build_ms=build_ms,
            score_hash_ms=score_hash_ms,
            geometry_sort_ms=geometry_sort_ms,
            memory_bytes=memory_bytes,
            peak_working_set_estimate_bytes=peak_working_set_estimate_bytes,
            score_fingerprint=fingerprint,
            block_size_m=block_size,
        )
        self._active_block_csr = csr
        self._block_score_token_generation += 1
        self._active_block_score_token = f"block-scores-{self._block_score_token_generation}"
        return self._block_metrics(csr, cached=False)

    def _block_csr(
        self, block_size: float, scores: np.ndarray | None = None,
        block_score_token: str | None = None,
    ) -> BlockCSR:
        active = self._active_block_csr
        if block_score_token is not None:
            if scores is not None:
                raise ValueError("scores and block_score_token are mutually exclusive")
            if (
                active is None
                or active.block_size_m != float(block_size)
                or block_score_token != self._active_block_score_token
            ):
                raise ValueError("block_score_token does not match the active block snapshot")
            return active
        _, fingerprint, _ = self._score_array_and_fingerprint(scores)
        if (
            active is None
            or active.block_size_m != float(block_size)
            or active.score_fingerprint != fingerprint
        ):
            self.prebuild_blocks(block_size, scores)
        assert self._active_block_csr is not None
        return self._active_block_csr

    def score_snapshot(self, block_score_token: str) -> np.ndarray | None:
        """Return the immutable score snapshot selected by a valid block token."""
        active = self._active_block_csr
        if active is None or block_score_token != self._active_block_score_token:
            raise ValueError("block_score_token does not match the active block snapshot")
        return active.scores

    def _query_frustum_blocks(
        self, planes: np.ndarray, block_size: float, chunk_size: int,
        scores: np.ndarray | None = None, block_score_token: str | None = None,
    ) -> tuple[np.ndarray, BlockCSR]:
        csr = self._block_csr(block_size, scores, block_score_token)
        coords = csr.coords
        selected_parts = []
        for begin in range(0, coords.shape[0], chunk_size):
            chunk = coords[begin : begin + chunk_size]
            lower = chunk.astype(np.float64) * block_size
            upper = np.nextafter(lower + block_size, np.inf)
            mask = conservative_aabb_halfspace_test(lower, upper, planes)
            selected_parts.append(np.flatnonzero(mask) + begin)
        block_ids = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
        return block_ids.astype(np.int64, copy=False), csr

    @staticmethod
    def expand_blocks(block_ids: np.ndarray, csr: BlockCSR, sort_result: bool = True) -> np.ndarray:
        block_ids = np.asarray(block_ids, dtype=np.int64)
        parts = [csr.point_ids[csr.offsets[i] : csr.offsets[i + 1]] for i in block_ids]
        result = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        return np.sort(result, kind="stable") if sort_result else result

    @staticmethod
    def _block_candidate_count(block_ids: np.ndarray, csr: BlockCSR) -> int:
        return int(np.sum(csr.offsets[block_ids + 1] - csr.offsets[block_ids], dtype=np.int64))

    def _block_representative_pre_cap(
        self, block_ids: np.ndarray, csr: BlockCSR, c2w: np.ndarray, intrinsics: np.ndarray,
        image_height: int, image_width: int, config: LocalSubmapConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Match the point reference ordering in O(G + B log B + cap + pruning)."""
        cap = int(config.projection_pre_cap)
        coords = csr.coords[block_ids]
        centers = (coords.astype(np.float64) + 0.5) * config.block_size_m
        camera = _camera_points(centers, c2w)
        depth = camera[:, 2]
        safe_depth = np.maximum(depth, np.finfo(np.float64).tiny)
        u = intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2]
        tile_col = np.clip(np.floor(u * config.tile_cols / image_width).astype(np.int64), 0, config.tile_cols - 1)
        tile_row = np.clip(np.floor(v * config.tile_rows / image_height).astype(np.int64), 0, config.tile_rows - 1)
        depth_edges = np.geomspace(max(config.near, np.finfo(float).tiny), config.far, config.depth_bins + 1)
        depth_bin = np.clip(np.searchsorted(depth_edges, safe_depth, side="right") - 1, 0, config.depth_bins - 1)
        lower = coords.astype(np.float64) * config.block_size_m
        upper = np.nextafter(lower + config.block_size_m, np.inf)
        planes = camera_frustum_planes_world(c2w, intrinsics, image_height, image_width, config)
        normals = planes[:, :3]
        nearest = np.where(normals[None, :, :] >= 0.0, lower[:, None, :], upper[:, None, :])
        boundary = np.any(np.einsum("bpc,pc->bp", nearest, normals) + planes[None, :, 3] <= 0.0, axis=1)
        bucket = (((~boundary).astype(np.int64) * config.tile_rows + tile_row) * config.tile_cols + tile_col) * config.depth_bins + depth_bin
        best = csr.block_best_score[block_ids]
        block_order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -best, bucket))
        sorted_buckets = bucket[block_order]
        starts = np.flatnonzero(np.r_[True, sorted_buckets[1:] != sorted_buckets[:-1]])
        ends = np.r_[starts[1:], block_order.size]
        ordered_buckets = sorted_buckets[starts]
        blocks_by_bucket = [block_order[begin:end] for begin, end in zip(starts, ends)]

        levels = np.zeros(len(blocks_by_bucket), dtype=np.int64)
        positions = np.zeros(len(blocks_by_bucket), dtype=np.int64)
        active = [blocks.copy() for blocks in blocks_by_bucket]
        active_buckets = deque(range(len(blocks_by_bucket)))
        selected: list[int] = []
        selected_blocks: set[int] = set()
        selected_buckets: set[int] = set()
        pruning_probes = 0
        bucket_turns = 0
        while len(selected) < cap and active_buckets:
            bucket_pos = active_buckets.popleft()
            bucket_turns += 1
            blocks = active[bucket_pos]
            while blocks.size:
                position = int(positions[bucket_pos])
                if position >= blocks.size:
                    levels[bucket_pos] += 1
                    level = int(levels[bucket_pos])
                    lengths = csr.offsets[block_ids[blocks] + 1] - csr.offsets[block_ids[blocks]]
                    pruning_probes += int(blocks.size)
                    blocks = blocks[lengths > level]
                    active[bucket_pos] = blocks
                    positions[bucket_pos] = 0
                    continue
                local_block = int(blocks[position])
                global_block = int(block_ids[local_block])
                point_offset = int(csr.offsets[global_block] + levels[bucket_pos])
                selected.append(int(csr.point_ids[point_offset]))
                selected_blocks.add(local_block)
                selected_buckets.add(bucket_pos)
                positions[bucket_pos] += 1
                active_buckets.append(bucket_pos)
                break
        selected_block_count = len(selected_blocks)
        selected_bucket_count = len(selected_buckets)
        boundary_block_count = int(np.count_nonzero(boundary))
        selected_boundary_block_count = int(sum(boundary[i] for i in selected_blocks))
        return np.asarray(selected, dtype=np.int64), {
            "precap_strategy": "block_representative",
            "representative_points_considered": len(selected) + pruning_probes,
            "representative_pruning_probes": pruning_probes,
            "representative_bucket_turns": bucket_turns,
            "precap_block_count": int(block_ids.size),
            "precap_selected_block_count": selected_block_count,
            "precap_block_coverage": selected_block_count / block_ids.size if block_ids.size else 1.0,
            "precap_bucket_count": len(ordered_buckets),
            "precap_selected_bucket_count": selected_bucket_count,
            "precap_bucket_coverage": selected_bucket_count / len(ordered_buckets) if len(ordered_buckets) else 1.0,
            "precap_boundary_block_count": boundary_block_count,
            "precap_selected_boundary_block_count": selected_boundary_block_count,
            "precap_boundary_coverage": (
                selected_boundary_block_count / boundary_block_count if boundary_block_count else 1.0
            ),
        }

    def _exact_frustum_indices_chunked(
        self, c2w: np.ndarray, intrinsics: np.ndarray, image_height: int,
        image_width: int, config: LocalSubmapConfig,
    ) -> np.ndarray:
        parts = []
        for begin in range(0, self.num_points, config.oracle_point_chunk):
            end = min(begin + config.oracle_point_chunk, self.num_points)
            mask = points_in_camera_frustum(
                self.points[begin:end], c2w, intrinsics, image_height, image_width, config
            )
            parts.append(np.flatnonzero(mask).astype(np.int64) + begin)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def _block_coverage_pre_cap(
        self, candidates: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray,
        image_height: int, image_width: int, config: LocalSubmapConfig,
        scores: np.ndarray | None,
    ) -> np.ndarray:
        cap = int(config.projection_pre_cap)
        if candidates.size <= cap:
            return np.sort(candidates, kind="stable")
        coords = np.floor(self.points[candidates].astype(np.float64, copy=False) / config.block_size_m).astype(np.int64)
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        centers = (unique_coords.astype(np.float64) + 0.5) * config.block_size_m
        camera = _camera_points(centers, c2w)
        depth = camera[:, 2]
        safe_depth = np.maximum(depth, np.finfo(np.float64).tiny)
        u = intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2]
        tile_col = np.clip(np.floor(u * config.tile_cols / image_width).astype(np.int64), 0, config.tile_cols - 1)
        tile_row = np.clip(np.floor(v * config.tile_rows / image_height).astype(np.int64), 0, config.tile_rows - 1)
        depth_edges = np.geomspace(max(config.near, np.finfo(float).tiny), config.far, config.depth_bins + 1)
        depth_bin = np.clip(np.searchsorted(depth_edges, safe_depth, side="right") - 1, 0, config.depth_bins - 1)
        lower = unique_coords.astype(np.float64) * config.block_size_m
        upper = np.nextafter(lower + config.block_size_m, np.inf)
        planes = camera_frustum_planes_world(c2w, intrinsics, image_height, image_width, config)
        normals = planes[:, :3]
        nearest = np.where(normals[None, :, :] >= 0.0, lower[:, None, :], upper[:, None, :])
        boundary = np.any(np.einsum("bpc,pc->bp", nearest, normals) + planes[None, :, 3] <= 0.0, axis=1)
        bucket = (((~boundary).astype(np.int64) * config.tile_rows + tile_row) * config.tile_cols + tile_col) * config.depth_bins + depth_bin

        block_members = []
        block_best_score = np.zeros(unique_coords.shape[0], dtype=np.float64)
        for block_id in range(unique_coords.shape[0]):
            members = np.flatnonzero(inverse == block_id)
            if scores is not None:
                member_order = np.lexsort((candidates[members], -scores[candidates[members]]))
                block_best_score[block_id] = np.max(scores[candidates[members]])
            else:
                member_order = np.argsort(candidates[members], kind="stable")
            block_members.append(candidates[members[member_order]])
        block_order = np.lexsort((unique_coords[:, 2], unique_coords[:, 1], unique_coords[:, 0], -block_best_score, bucket))
        ordered_buckets = np.unique(bucket[block_order])
        blocks_by_bucket = [block_order[bucket[block_order] == value] for value in ordered_buckets]
        bucket_points = []
        for blocks in blocks_by_bucket:
            ordered = []
            level = 0
            while True:
                progressed = False
                for block_id in blocks:
                    members = block_members[int(block_id)]
                    if level < members.size:
                        ordered.append(int(members[level]))
                        progressed = True
                if not progressed:
                    break
                level += 1
            bucket_points.append(ordered)
        selected = []
        bucket_position = np.zeros(len(bucket_points), dtype=np.int64)
        while len(selected) < cap:
            progressed = False
            for bucket_id, ordered in enumerate(bucket_points):
                position = int(bucket_position[bucket_id])
                if position < len(ordered):
                    selected.append(ordered[position])
                    bucket_position[bucket_id] += 1
                    progressed = True
                    if len(selected) == cap:
                        break
            if not progressed:
                break
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _oracle_block_recall(oracle_indices: np.ndarray, block_ids: np.ndarray, csr: BlockCSR, points: np.ndarray) -> float:
        if oracle_indices.size == 0:
            return 1.0
        oracle_coords = np.floor(
            points[oracle_indices].astype(np.float64, copy=False) / csr.block_size_m
        ).astype(np.int64)
        coord_dtype = np.dtype((np.void, csr.coords.dtype.itemsize * csr.coords.shape[1]))
        all_keys = np.ascontiguousarray(csr.coords).view(coord_dtype).ravel()
        selected_keys = all_keys[block_ids]
        oracle_keys = np.ascontiguousarray(oracle_coords).view(coord_dtype).ravel()
        return float(np.count_nonzero(np.isin(oracle_keys, selected_keys)) / oracle_indices.size)

    @staticmethod
    def _coverage_sample(
        indices: np.ndarray,
        projected_u: np.ndarray,
        projected_v: np.ndarray,
        depth: np.ndarray,
        image_height: int,
        image_width: int,
        config: LocalSubmapConfig,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        components = coverage_sampling_components(
            projected_u, projected_v, depth, image_height, image_width, config
        )
        tile_id = components["tile_id"]
        depth_bin = components["depth_bin"]
        bucket_id = components["bucket_id"]

        bucket_order = np.argsort(bucket_id, kind="stable")
        sorted_bucket = bucket_id[bucket_order]
        starts = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
        ends = np.r_[starts[1:], bucket_order.size]
        occupancy = ends - starts
        bucket_count = int(starts.size)
        cap = int(config.max_points)

        allocation = coverage_sampling_allocation(occupancy, cap)

        selected_parts = []
        for bucket_index, (begin, end) in enumerate(zip(starts, ends)):
            take = int(allocation[bucket_index])
            if take == 0:
                continue
            members = bucket_order[begin:end]
            if scores is not None:
                order = np.lexsort((indices[members], depth[members], -scores[indices[members]]))
            else:
                order = np.lexsort((indices[members], depth[members]))
            selected_parts.append(members[order[:take]])
        selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
        return indices[selected], {
            "nonempty_tile_count": int(np.unique(tile_id).size),
            "selected_tile_count": int(np.unique(tile_id[selected]).size),
            "nonempty_depth_bin_count": int(np.unique(depth_bin).size),
            "selected_depth_bin_count": int(np.unique(depth_bin[selected]).size),
            "nonempty_tile_depth_count": bucket_count,
            "selected_tile_depth_count": int(np.unique(bucket_id[selected]).size),
        }

    def stream_exact_final_target(
        self,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        image_height: int,
        image_width: int,
        config: LocalSubmapConfig,
        scores: np.ndarray | None = None,
        block_score_token: str | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        """Compute the exact center-frustum final target with bounded two-pass retention."""
        c2w, intrinsics, image_height, image_width = _validate_camera(c2w, intrinsics, image_height, image_width)
        if not isinstance(config, LocalSubmapConfig):
            raise TypeError("config must be a LocalSubmapConfig")
        if config.max_points is None or config.sampling_policy != "coverage":
            raise ValueError("stream_exact_final_target requires coverage sampling with max_points")
        if config.near <= 0.0:
            raise ValueError("stream_exact_final_target requires near > 0 for bounded two-pass depth binning")
        if block_score_token is not None and scores is not None:
            raise ValueError("scores and block_score_token are mutually exclusive")
        if block_score_token is not None:
            active = self._active_block_csr
            if active is None or block_score_token != self._active_block_score_token:
                raise ValueError("block_score_token does not match the active block snapshot")
            score_array = active.scores
        elif scores is None:
            score_array = None
        else:
            score_array = np.asarray(scores, dtype=np.float64)
            if score_array.shape != (self.num_points,) or np.any(~np.isfinite(score_array)):
                raise ValueError(f"Expected finite scores with shape [{self.num_points}]")
        chunk = config.oracle_point_chunk if chunk_size is None else int(chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        bucket_total = config.tile_rows * config.tile_cols * config.depth_bins
        occupancy = np.zeros(bucket_total, dtype=np.int64)
        visible_count = 0
        for begin in range(0, self.num_points, chunk):
            points = self.points[begin : begin + chunk].astype(np.float64, copy=False)
            camera = _camera_points(points, c2w)
            mask = points_in_camera_frustum(points, c2w, intrinsics, image_height, image_width, config)
            if not np.any(mask):
                continue
            depth = camera[mask, 2]
            u = intrinsics[0, 0] * camera[mask, 0] / depth + intrinsics[0, 2]
            v = intrinsics[1, 1] * camera[mask, 1] / depth + intrinsics[1, 2]
            components = coverage_sampling_components(
                u, v, depth, image_height, image_width, config,
            )
            occupancy += np.bincount(components["bucket_id"], minlength=bucket_total)
            visible_count += int(depth.size)
        nonempty = np.flatnonzero(occupancy)
        allocation = coverage_sampling_allocation(occupancy[nonempty], int(config.max_points))
        take_by_bucket = np.zeros(bucket_total, dtype=np.int64)
        take_by_bucket[nonempty] = allocation
        retained: list[list[tuple[float, float, int]]] = [[] for _ in range(bucket_total)]
        for begin in range(0, self.num_points, chunk):
            points = self.points[begin : begin + chunk].astype(np.float64, copy=False)
            camera = _camera_points(points, c2w)
            mask = points_in_camera_frustum(points, c2w, intrinsics, image_height, image_width, config)
            if not np.any(mask):
                continue
            local = np.flatnonzero(mask)
            ids = local.astype(np.int64) + begin
            depth = camera[local, 2]
            u = intrinsics[0, 0] * camera[local, 0] / depth + intrinsics[0, 2]
            v = intrinsics[1, 1] * camera[local, 1] / depth + intrinsics[1, 2]
            buckets = coverage_sampling_components(
                u, v, depth, image_height, image_width, config,
            )["bucket_id"]
            for point_id, point_depth, bucket_id in zip(ids, depth, buckets):
                take = int(take_by_bucket[bucket_id])
                if take == 0:
                    continue
                primary = -float(score_array[point_id]) if score_array is not None else 0.0
                values = retained[int(bucket_id)]
                values.append((primary, float(point_depth), int(point_id)))
                if len(values) > take:
                    values.sort()
                    del values[take:]
        selected = []
        for bucket_id in nonempty:
            values = retained[int(bucket_id)]
            values.sort()
            selected.extend(value[2] for value in values)
        indices = np.asarray(selected, dtype=np.int64)
        components = {"tile_id": np.empty(0, dtype=np.int64), "depth_bin": np.empty(0, dtype=np.int64), "bucket_id": np.empty(0, dtype=np.int64)}
        if indices.size:
            camera = _camera_points(self.points[indices].astype(np.float64, copy=False), c2w)
            depth = camera[:, 2]
            u = intrinsics[0, 0] * camera[:, 0] / depth + intrinsics[0, 2]
            v = intrinsics[1, 1] * camera[:, 1] / depth + intrinsics[1, 2]
            components = coverage_sampling_components(
                u, v, depth, image_height, image_width, config,
            )
        return {
            "indices": indices,
            "visible_count": visible_count,
            "count": int(indices.size),
            "occupancy": occupancy,
            "allocation": take_by_bucket,
            "tile_id": components["tile_id"],
            "depth_bin": components["depth_bin"],
            "bucket_id": components["bucket_id"],
            "retained_count": int(sum(len(values) for values in retained)),
            "peak_retained_bound": int(np.sum(allocation)),
            "chunk_size": chunk,
        }

    def query_local_submap(
        self,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        image_height: int,
        image_width: int,
        config: LocalSubmapConfig,
        scores: np.ndarray | None = None,
        return_stage_indices: bool = False,
        block_score_token: str | None = None,
    ) -> dict[str, Any]:
        """Query a local submap.

        ``block_score_token`` selects the immutable score snapshot captured by
        ``prebuild_blocks`` and avoids per-query score validation and hashing.
        Without a token, supplied scores are fingerprinted so in-place mutation
        rebuilds the active block snapshot.
        """
        start = time.perf_counter()
        c2w, intrinsics, image_height, image_width = _validate_camera(
            c2w, intrinsics, image_height, image_width
        )
        if not isinstance(config, LocalSubmapConfig):
            raise TypeError("config must be a LocalSubmapConfig")
        if not isinstance(return_stage_indices, (bool, np.bool_)):
            raise TypeError("return_stage_indices must be a boolean")

        if block_score_token is not None and scores is not None:
            raise ValueError("scores and block_score_token are mutually exclusive")
        score_array = None
        if block_score_token is not None:
            active = self._active_block_csr
            if active is None or block_score_token != self._active_block_score_token:
                raise ValueError("block_score_token does not match the active block snapshot")
            score_array = active.scores
        elif scores is not None:
            score_array = np.asarray(scores, dtype=np.float64)
            if score_array.shape != (self.num_points,) or np.any(~np.isfinite(score_array)):
                raise ValueError(f"Expected finite scores with shape [{self.num_points}]")

        block_culling_start = time.perf_counter()
        block_ids = np.empty(0, dtype=np.int64)
        block_csr = None
        block_candidates = None
        if config.frustum_culling == "voxel_blocks":
            planes = camera_frustum_planes_world(c2w, intrinsics, image_height, image_width, config)
            block_ids, block_csr = self._query_frustum_blocks(
                planes, config.block_size_m, config.block_test_chunk,
                None if block_score_token is not None else score_array,
                block_score_token,
            )
            tested_block_count = int(block_csr.coords.shape[0])
            intersecting_block_count = int(block_ids.size)
            candidate_count = self._block_candidate_count(block_ids, block_csr)
            radius_latency_ms = 0.0
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            horizontal = max(abs((-config.margin_px - cx) / fx), abs((image_width + config.margin_px - cx) / fx))
            vertical = max(abs((-config.margin_px - cy) / fy), abs((image_height + config.margin_px - cy) / fy))
            candidate_radius = config.far * np.sqrt(1.0 + horizontal * horizontal + vertical * vertical)
            candidate_radius += config.radius_margin
            radius_start = time.perf_counter()
            block_candidates = self.query_radius(c2w[:3, 3], float(candidate_radius), sort_result=False)
            block_candidates = np.sort(block_candidates.astype(np.int64, copy=False), kind="stable")
            candidate_count = int(block_candidates.size)
            tested_block_count = 0
            intersecting_block_count = 0
            radius_latency_ms = float((time.perf_counter() - radius_start) * 1000.0)
        block_culling_latency_ms = float((time.perf_counter() - block_culling_start) * 1000.0)

        oracle_start = time.perf_counter()
        oracle_indices = np.empty(0, dtype=np.int64)
        oracle_count = 0
        oracle_overflow = False
        oracle_reason = "off"
        status = "ok"
        if config.oracle_mode != "off":
            oracle_indices = self._exact_frustum_indices_chunked(
                c2w, intrinsics, image_height, image_width, config
            )
            oracle_count = int(oracle_indices.size)
            oracle_overflow = config.projection_pre_cap is not None and oracle_count > config.projection_pre_cap
            oracle_reason = "measured" if config.oracle_mode == "measure" else "within_cap"
        oracle_latency_ms = float((time.perf_counter() - oracle_start) * 1000.0)

        precap_start = time.perf_counter()
        projection_input = block_candidates
        bypass_projection_cap = False
        precap_metrics = {
            "precap_strategy": "point_reference",
            "representative_points_considered": 0,
            "representative_pruning_probes": 0,
            "representative_bucket_turns": 0,
            "precap_block_count": intersecting_block_count,
            "precap_selected_block_count": 0,
            "precap_block_coverage": 0.0,
            "precap_bucket_count": 0,
            "precap_selected_bucket_count": 0,
            "precap_bucket_coverage": 0.0,
            "precap_boundary_block_count": 0,
            "precap_selected_boundary_block_count": 0,
            "precap_boundary_coverage": 0.0,
        }
        if config.projection_pre_cap is not None and candidate_count > config.projection_pre_cap:
            if block_csr is not None:
                projection_input, precap_metrics = self._block_representative_pre_cap(
                    block_ids, block_csr, c2w, intrinsics, image_height, image_width, config
                )
            else:
                assert block_candidates is not None
                projection_input = self._block_coverage_pre_cap(
                    block_candidates, c2w, intrinsics, image_height, image_width, config, score_array
                )
                precap_metrics["representative_points_considered"] = candidate_count
        elif block_csr is not None:
            projection_input = self.expand_blocks(block_ids, block_csr)
        assert projection_input is not None
        if config.oracle_mode == "enforce":
            if oracle_overflow and config.oracle_overflow_policy == "fail_closed":
                status = "exact_recall_infeasible"
                oracle_reason = "oracle_exceeds_projection_pre_cap"
                projection_input = np.empty(0, dtype=np.int64)
            elif oracle_overflow:
                if block_csr is not None:
                    projection_input = self.expand_blocks(block_ids, block_csr)
                else:
                    assert block_candidates is not None
                    projection_input = block_candidates
                bypass_projection_cap = True
                oracle_reason = "oracle_overflow_uncapped_chunked"
            elif config.projection_pre_cap is not None:
                oracle_set = set(int(i) for i in oracle_indices)
                remainder = [int(i) for i in projection_input if int(i) not in oracle_set]
                room = int(config.projection_pre_cap) - oracle_count
                projection_input = np.asarray(list(oracle_indices) + remainder[:room], dtype=np.int64)
                oracle_reason = "oracle_injected"
        candidates = projection_input.astype(np.int64, copy=False)
        projection_pre_cap_latency_ms = float((time.perf_counter() - precap_start) * 1000.0)

        projection_start = time.perf_counter()
        positive_depth_count = 0
        depth_range_count = 0
        positive_depth_indices = np.empty(0, dtype=np.int64)
        depth_range_indices = np.empty(0, dtype=np.int64)
        visible_u = np.empty(0, dtype=np.float64)
        visible_v = np.empty(0, dtype=np.float64)
        visible_depth = np.empty(0, dtype=np.float64)
        if candidates.size:
            camera_points = _camera_points(self.points[candidates].astype(np.float64, copy=False), c2w)
            depth = camera_points[:, 2]
            positive_mask = depth > 0.0
            depth_mask = positive_mask & (depth >= config.near) & (depth <= config.far)
            positive_depth_count = int(np.count_nonzero(positive_mask))
            depth_range_count = int(np.count_nonzero(depth_mask))
            if return_stage_indices:
                positive_depth_indices = candidates[positive_mask].astype(np.int64, copy=False)
                depth_range_indices = candidates[depth_mask].astype(np.int64, copy=False)
            projected_u = np.full(depth.shape, np.nan, dtype=np.float64)
            projected_v = np.full(depth.shape, np.nan, dtype=np.float64)
            projected_u[depth_mask] = intrinsics[0, 0] * camera_points[depth_mask, 0] / depth[depth_mask] + intrinsics[0, 2]
            projected_v[depth_mask] = intrinsics[1, 1] * camera_points[depth_mask, 1] / depth[depth_mask] + intrinsics[1, 2]
            margin = config.margin_px
            mask = depth_mask & (
                (projected_u >= -margin)
                & (projected_u < image_width + margin)
                & (projected_v >= -margin)
                & (projected_v < image_height + margin)
            )
            indices = candidates[mask]
            visible_u = projected_u[mask]
            visible_v = projected_v[mask]
            visible_depth = depth[mask]
        else:
            indices = candidates
        projection_latency_ms = float((time.perf_counter() - projection_start) * 1000.0)

        frustum_count = int(indices.size)
        exact_frustum_indices = indices.astype(np.int64, copy=False)
        frustum_indices = indices.astype(np.int64, copy=False) if return_stage_indices else None
        sampling_counts = {
            "nonempty_tile_count": 0,
            "selected_tile_count": 0,
            "nonempty_depth_bin_count": 0,
            "selected_depth_bin_count": 0,
            "nonempty_tile_depth_count": 0,
            "selected_tile_depth_count": 0,
        }
        sampling_start = time.perf_counter()
        if config.max_points is not None and indices.size > config.max_points:
            if config.sampling_policy == "coverage":
                indices, sampling_counts = self._coverage_sample(
                    indices, visible_u, visible_v, visible_depth,
                    image_height, image_width, config, score_array,
                )
            else:
                if score_array is not None:
                    order = np.lexsort((indices, -score_array[indices]))
                else:
                    order = np.lexsort((indices, visible_depth))
                indices = indices[order[: int(config.max_points)]]
        elif config.sampling_policy == "coverage" and indices.size:
            tile_col = np.clip((visible_u * config.tile_cols / image_width).astype(np.int64), 0, config.tile_cols - 1)
            tile_row = np.clip((visible_v * config.tile_rows / image_height).astype(np.int64), 0, config.tile_rows - 1)
            tile_id = tile_row * config.tile_cols + tile_col
            depth_edges = np.geomspace(max(config.near, np.finfo(float).tiny), config.far, config.depth_bins + 1)
            depth_bin = np.clip(np.searchsorted(depth_edges, visible_depth, side="right") - 1, 0, config.depth_bins - 1)
            bucket_id = tile_id * config.depth_bins + depth_bin
            sampling_counts = {
                "nonempty_tile_count": int(np.unique(tile_id).size),
                "selected_tile_count": int(np.unique(tile_id).size),
                "nonempty_depth_bin_count": int(np.unique(depth_bin).size),
                "selected_depth_bin_count": int(np.unique(depth_bin).size),
                "nonempty_tile_depth_count": int(np.unique(bucket_id).size),
                "selected_tile_depth_count": int(np.unique(bucket_id).size),
            }
        sampling_latency_ms = float((time.perf_counter() - sampling_start) * 1000.0)
        final_start = time.perf_counter()
        indices = indices.astype(np.int64, copy=False)
        final_latency_ms = float((time.perf_counter() - final_start) * 1000.0)

        if oracle_count:
            if block_csr is not None:
                block_recall = self._oracle_block_recall(oracle_indices, block_ids, block_csr, self.points)
            else:
                assert block_candidates is not None
                block_recall = float(np.intersect1d(oracle_indices, block_candidates, assume_unique=True).size / oracle_count)
            projection_recall = float(np.intersect1d(oracle_indices, candidates, assume_unique=True).size / oracle_count)
            frustum_recall = float(np.intersect1d(oracle_indices, exact_frustum_indices, assume_unique=True).size / oracle_count)
        else:
            block_recall = projection_recall = frustum_recall = 1.0

        result = {
            "indices": indices,
            "status": status,
            "query_latency_ms": float((time.perf_counter() - start) * 1000.0),
            "radius_latency_ms": radius_latency_ms,
            "block_culling_latency_ms": block_culling_latency_ms,
            "projection_pre_cap_latency_ms": projection_pre_cap_latency_ms,
            "precap_latency_ms": projection_pre_cap_latency_ms,
            "oracle_latency_ms": oracle_latency_ms,
            "projection_latency_ms": projection_latency_ms,
            "sampling_latency_ms": sampling_latency_ms,
            "final_latency_ms": final_latency_ms,
            "candidate_count": candidate_count,
            "culling_mode": config.frustum_culling,
            "tested_block_count": tested_block_count,
            "intersecting_block_count": intersecting_block_count,
            "block_candidate_count": candidate_count,
            "projection_input_count": int(candidates.size),
            "positive_depth_count": positive_depth_count,
            "depth_range_count": depth_range_count,
            # Kept for compatibility; this is the final-sampling input, not the projection pre-cap input.
            "precap_count": frustum_count,
            "final_sampling_input_count": frustum_count,
            "frustum_count": frustum_count,
            "count": int(indices.size),
            "backend": self.backend,
            "oracle_count": oracle_count,
            "oracle_block_recall": block_recall,
            "oracle_projection_recall": projection_recall,
            "oracle_frustum_recall": frustum_recall,
            "block_recall": block_recall,
            "projection_recall": projection_recall,
            "frustum_recall": frustum_recall,
            "oracle_overflow": bool(oracle_overflow),
            "oracle_reason": oracle_reason,
            "projection_cap_bypassed": bypass_projection_cap,
            **precap_metrics,
            **sampling_counts,
        }
        if return_stage_indices:
            if block_csr is not None:
                block_candidates = self.expand_blocks(block_ids, block_csr)
            assert block_candidates is not None
            result.update(
                {
                    "block_candidate_indices": block_candidates,
                    "projection_input_indices": candidates,
                    "candidate_indices": block_candidates,
                    "positive_depth_indices": positive_depth_indices,
                    "depth_range_indices": depth_range_indices,
                    "frustum_indices": frustum_indices,
                    "selected_indices": indices,
                }
            )
        return result


def query_local_submap(
    index: MapSpatialIndex,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    config: LocalSubmapConfig,
    scores: np.ndarray | None = None,
    return_stage_indices: bool = False,
    block_score_token: str | None = None,
) -> dict[str, Any]:
    if not isinstance(index, MapSpatialIndex):
        raise TypeError("index must be a MapSpatialIndex")
    return index.query_local_submap(
        c2w,
        intrinsics,
        image_height,
        image_width,
        config,
        scores=scores,
        return_stage_indices=return_stage_indices,
        block_score_token=block_score_token,
    )


def stream_exact_final_target(
    index: MapSpatialIndex,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    config: LocalSubmapConfig,
    scores: np.ndarray | None = None,
    block_score_token: str | None = None,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    if not isinstance(index, MapSpatialIndex):
        raise TypeError("index must be a MapSpatialIndex")
    return index.stream_exact_final_target(
        c2w,
        intrinsics,
        image_height,
        image_width,
        config,
        scores=scores,
        block_score_token=block_score_token,
        chunk_size=chunk_size,
    )
