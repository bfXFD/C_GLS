import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.fsm_local_map import LocalSubmapConfig, MapSpatialIndex
from utils.fsm_map import load_activation_geometry
from utils.fsm_registration import build_target_index
from utils.fsm_robust_registration import estimate_normals, evaluate_point_to_plane_diagnostics


_STAGES = ("candidate", "projection_input", "positive_depth", "depth_range", "frustum", "selected")
_TRIGGER_BITS = {
    "condition_number": 1,
    "normalized_lambda_min": 2,
    "min_fitness": 4,
    "max_iterations": 8,
}


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _normal_config(config: dict[str, Any], registration: dict[str, Any]) -> tuple[int, float]:
    return (
        int(registration.get("normal_k", registration.get("normal_max_nn", 30))),
        float(registration.get("rank_tolerance", 1e-9)),
    )


def _empty_fixed(source_count: int) -> dict[str, Any]:
    return {
        "fitness": 0.0,
        "num_correspondences": 0,
        "hessian": np.zeros((6, 6), dtype=np.float64),
        "eigenvalues": np.zeros(6, dtype=np.float64),
        "condition_number": float("inf"),
        "effective_rank": 0,
        "target_ids": np.empty(0, dtype=np.int64),
        "source_ids": np.empty(0, dtype=np.int64),
        "jacobians": np.empty((0, 6), dtype=np.float64),
        "weights": np.empty(0, dtype=np.float64),
        "source_count": int(source_count),
    }


class RuntimeActivationDiagnostics:
    """Read-only sidecar diagnostics. Results never feed back into tracking."""

    def __init__(
        self,
        config: dict[str, Any],
        fsm_map: dict[str, Any],
        map_path: str,
        local_config: LocalSubmapConfig | None,
        local_spatial_index: MapSpatialIndex | None,
        target_points: np.ndarray,
        target_normals: np.ndarray | None,
        target_scores: np.ndarray | None,
        evaluation_protocol: str,
        backend: str,
    ) -> None:
        cfg = config.get("activation_diagnostics", {}) or {}
        if not isinstance(cfg, dict):
            raise ValueError("activation_diagnostics must be a dictionary")
        self.enabled = bool(cfg.get("enabled", False))
        self.records: list[dict[str, Any]] = []
        self.config = cfg
        self.output_dir: Path | None = None
        self.role_unavailable_reason = "disabled"
        if not self.enabled:
            return
        if evaluation_protocol != "mvp_strict" or local_config is None or local_spatial_index is None:
            raise ValueError("activation_diagnostics requires mvp_strict with local_submap.enabled")
        if backend != "numpy_robust_point_to_plane":
            raise ValueError("activation_diagnostics requires numpy_robust_point_to_plane")

        self.local_config = local_config
        self.actual_index = local_spatial_index
        self.actual_points = np.asarray(target_points, dtype=np.float64)
        self.actual_normals = None if target_normals is None else np.asarray(target_normals, dtype=np.float64)
        self.actual_scores = target_scores
        self.actual_global_ids = np.asarray(fsm_map["indices"], dtype=np.int64)
        self.registration = config.get("registration", {})
        self.normal_source = str(self.registration.get("normal_source", "map"))
        self.normal_k, self.rank_tolerance = _normal_config(config, self.registration)
        self.max_distance = float(self.registration.get("max_correspondence_distance", 0.05))
        self.min_correspondences = int(self.registration.get("min_correspondences", 6))
        self.loss = str(self.registration.get("robust_loss", "huber"))
        self.loss_scale = float(self.registration.get("loss_scale", 0.02))
        self.include_stage_ids = bool(cfg.get("include_stage_ids", False))
        self.save_correspondences = bool(cfg.get("save_correspondences", False))
        self.max_detail_frames = int(cfg.get("max_detail_frames", 64))
        if self.max_detail_frames < 1:
            raise ValueError("activation_diagnostics.max_detail_frames must be positive")
        self.counterfactual_enabled = bool(cfg.get("counterfactual", True))
        self.weak_subspace_dim = int(cfg.get("weak_subspace_dim", 1))
        if self.weak_subspace_dim < 1 or self.weak_subspace_dim > 6:
            raise ValueError("activation_diagnostics.weak_subspace_dim must be between 1 and 6")
        budgets = np.asarray(cfg.get("reserve_budget", [1000, 2000, 4000]), dtype=np.int64)
        if budgets.ndim != 1 or budgets.size == 0 or np.any(budgets <= 0):
            raise ValueError("activation_diagnostics.reserve_budget must contain positive integers")
        self.reserve_budgets = budgets
        trigger = cfg.get("trigger", {}) or {}
        if not isinstance(trigger, dict) or trigger.get("mode", "plan_only") != "plan_only":
            raise ValueError("activation_diagnostics.trigger only supports mode='plan_only'")
        self.trigger = trigger

        geometry_path = cfg.get("geometry_path")
        automatic = os.path.join(os.path.dirname(os.path.abspath(map_path)), "activation_geometry.npz")
        explicit_geometry = geometry_path not in (None, "", "auto")
        self.geometry_path = str(geometry_path) if explicit_geometry else (automatic if os.path.isfile(automatic) else None)
        expected_params = str(np.asarray(fsm_map.get("source_params_sha256", "")))
        expected_controls = str(np.asarray(fsm_map.get("selection_controls_sha256", "")))
        if self.geometry_path and (not expected_params or not expected_controls):
            if explicit_geometry:
                raise ValueError("activation geometry requires map provenance hashes")
            self.geometry = None
            self.role_unavailable_reason = "map_provenance_unavailable"
        else:
            self.geometry = load_activation_geometry(
                self.geometry_path, fsm_map,
                source_params_sha256=expected_params or None,
                selection_controls_sha256=expected_controls or None,
            ) if self.geometry_path else None
            self.role_unavailable_reason = None if self.geometry is not None else "activation_geometry_unavailable"
        self.role_available = self.geometry is not None
        self.replacement_mask = np.zeros(self.actual_points.shape[0], dtype=bool)
        self.counter_index = None
        self.counter_points = None
        self.counter_normals = None
        self.counter_scores = None
        self.counter_global_ids = None
        self.removed_mask = None
        if self.geometry is not None:
            local_by_global = {int(value): index for index, value in enumerate(self.actual_global_ids)}
            replacement_local = np.asarray(
                [local_by_global[int(value)] for value in np.asarray(self.geometry["replacement_global_ids"], dtype=np.int64)],
                dtype=np.int64,
            )
            self.replacement_mask[replacement_local] = True
            if self.counterfactual_enabled:
                preserved_global = np.asarray(self.geometry["preserved_global_ids"], dtype=np.int64)
                preserved_local = np.asarray([local_by_global[int(value)] for value in preserved_global], dtype=np.int64)
                removed_global = np.asarray(self.geometry["removed_global_ids"], dtype=np.int64)
                self.counter_points = np.concatenate(
                    [self.actual_points[preserved_local], np.asarray(self.geometry["removed_means"], dtype=np.float64)], axis=0
                )
                self.counter_global_ids = np.concatenate([preserved_global, removed_global])
                self.removed_mask = np.isin(self.counter_global_ids, removed_global, assume_unique=True)
                if self.actual_normals is not None:
                    self.counter_normals = np.concatenate(
                        [self.actual_normals[preserved_local], np.asarray(self.geometry["removed_normals"], dtype=np.float64)], axis=0
                    )
                if self.actual_scores is not None:
                    self.counter_scores = np.concatenate(
                        [self.actual_scores[preserved_local], np.asarray(self.geometry["removed_scores"], dtype=np.float64)]
                    )
                spatial = config.get("local_submap", {}).get("spatial", {})
                self.counter_index = MapSpatialIndex(
                    self.counter_points,
                    backend=str(config.get("local_submap", {}).get("spatial_backend", spatial.get("backend", "auto"))),
                    voxel_size=float(config.get("local_submap", {}).get("grid_voxel", spatial.get("voxel_size", 1.0))),
                )

    def _query(
        self,
        index: MapSpatialIndex,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        height: int,
        width: int,
        scores: np.ndarray | None,
    ) -> dict[str, Any]:
        return index.query_local_submap(
            pose, intrinsics, height, width, self.local_config, scores=scores, return_stage_indices=True
        )

    def _fixed(
        self,
        source_points: np.ndarray,
        pose: np.ndarray,
        points: np.ndarray,
        normals: np.ndarray | None,
        global_ids: np.ndarray,
        indices: np.ndarray,
        cached: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        if indices.size == 0:
            return _empty_fixed(source_points.shape[0]), 0.0, 0.0
        local_points = points[indices]
        normal_start = time.perf_counter()
        local_normals = None
        local_tree = None
        if cached is not None:
            local_tree = cached.get("target_tree")
            local_normals = cached.get("target_normals")
        if local_tree is None:
            local_tree = build_target_index(local_points)["target_tree"]
        if local_normals is None:
            if self.normal_source == "map" and normals is not None:
                local_normals = normals[indices]
            else:
                local_normals = estimate_normals(local_points, k=self.normal_k, target_tree=local_tree)
        normal_ms = float((time.perf_counter() - normal_start) * 1000.0)
        eval_start = time.perf_counter()
        fixed = evaluate_point_to_plane_diagnostics(
            source_points=source_points,
            target_points=local_points,
            target_normals=local_normals,
            transform=pose,
            max_correspondence_distance=self.max_distance,
            target_tree=local_tree,
            loss=self.loss,
            loss_scale=self.loss_scale,
            min_correspondences=self.min_correspondences,
            rank_tolerance=self.rank_tolerance,
            source_ids=np.arange(source_points.shape[0], dtype=np.int64),
            target_ids=global_ids[indices],
            return_correspondences=True,
        )
        fixed["source_count"] = int(source_points.shape[0])
        return fixed, normal_ms, float((time.perf_counter() - eval_start) * 1000.0)

    @staticmethod
    def _role_metrics(
        query: dict[str, Any], fixed: dict[str, Any], role_mask: np.ndarray, global_ids: np.ndarray
    ) -> dict[str, Any]:
        role_global = global_ids[role_mask]
        stage_counts = {
            stage: int(np.count_nonzero(role_mask[np.asarray(query[f"{stage}_indices"], dtype=np.int64)]))
            for stage in _STAGES
        }
        corr_role = np.isin(np.asarray(fixed["target_ids"]), role_global, assume_unique=False)
        jacobians = np.asarray(fixed["jacobians"], dtype=np.float64)[corr_role]
        weights = np.asarray(fixed["weights"], dtype=np.float64)[corr_role]
        hessian = jacobians.T @ (jacobians * weights[:, None]) if jacobians.size else np.zeros((6, 6))
        hessian = 0.5 * (hessian + hessian.T)
        return {
            "total": int(np.count_nonzero(role_mask)),
            **stage_counts,
            "correspondence": int(np.count_nonzero(corr_role)),
            "unique_target": int(np.unique(np.asarray(fixed["target_ids"])[corr_role]).size),
            "hessian": hessian,
            "corr_role_mask": corr_role,
        }

    def _hessian_metrics(self, fixed: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
        total_h = np.asarray(fixed["hessian"], dtype=np.float64)
        role_h = np.asarray(role["hessian"], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(total_h)
        trace = float(np.trace(total_h))
        role_trace = float(np.trace(role_h))
        weak_vectors = eigenvectors[:, : self.weak_subspace_dim]
        weak_total = float(np.sum(eigenvalues[: self.weak_subspace_dim]))
        weak_role = float(np.trace(weak_vectors.T @ role_h @ weak_vectors))
        role_eigenvalues = np.linalg.eigvalsh(role_h)
        corr_mask = role["corr_role_mask"]
        jacobians = np.asarray(fixed["jacobians"], dtype=np.float64)[corr_mask]
        weights = np.asarray(fixed["weights"], dtype=np.float64)[corr_mask]
        if jacobians.size:
            projected = jacobians @ weak_vectors
            contributions = weights * np.sum(projected * projected, axis=1)
            contributions = np.sort(contributions)[::-1]
        else:
            contributions = np.empty(0, dtype=np.float64)
        reserve_gain = np.asarray(
            [float(np.sum(contributions[: min(int(budget), contributions.size)])) for budget in self.reserve_budgets]
        )
        return {
            "trace": trace,
            "lambda_min": float(eigenvalues[0]),
            "normalized_lambda_min": _safe_fraction(float(eigenvalues[0]), trace),
            "role_trace": role_trace,
            "role_lambda_min": float(role_eigenvalues[0]),
            "trace_fraction": _safe_fraction(role_trace, trace),
            "weak_fraction": _safe_fraction(weak_role, weak_total),
            "reserve_gain": reserve_gain,
            "reserve_gain_fraction": reserve_gain / weak_total if weak_total > 0.0 else np.zeros_like(reserve_gain),
        }

    def _compact_query(self, query: dict[str, Any] | None, global_ids: np.ndarray | None) -> dict[str, Any] | None:
        if query is None:
            return None
        compact = {
            key: query[key]
            for key in (
                "candidate_count", "projection_input_count", "positive_depth_count", "depth_range_count",
                "frustum_count", "precap_count", "tested_block_count", "intersecting_block_count",
                "block_candidate_count", "oracle_count", "oracle_block_recall", "oracle_projection_recall",
                "oracle_frustum_recall", "oracle_overflow", "block_culling_latency_ms", "precap_latency_ms",
                "oracle_latency_ms", "culling_mode", "status",
            )
            if key in query
        }
        if self.include_stage_ids and global_ids is not None:
            compact["stage_global_ids"] = {
                stage: np.asarray(global_ids[np.asarray(query[f"{stage}_indices"], dtype=np.int64)], dtype=np.int64)
                for stage in _STAGES
            }
        return compact

    def _compact_fixed(self, fixed: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "fitness": float(fixed["fitness"]),
            "num_correspondences": int(fixed["num_correspondences"]),
            "hessian": np.asarray(fixed["hessian"], dtype=np.float64),
            "condition_number": float(fixed["condition_number"]),
        }
        if self.save_correspondences:
            compact["source_ids"] = np.asarray(fixed["source_ids"], dtype=np.int64)
            compact["target_ids"] = np.asarray(fixed["target_ids"], dtype=np.int64)
        return compact

    @staticmethod
    def _compact_role(role: dict[str, Any]) -> dict[str, Any]:
        return {
            "total": int(role["total"]),
            **{stage: int(role[stage]) for stage in _STAGES},
            "correspondence": int(role["correspondence"]),
            "unique_target": int(role["unique_target"]),
            "hessian": np.asarray(role["hessian"], dtype=np.float64),
        }

    @staticmethod
    def _compact_h(hessian: dict[str, Any]) -> dict[str, Any]:
        return {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else float(value)
            for key, value in hessian.items()
        }

    @property
    def retained_ndarray_bytes(self) -> int:
        def retained(value: Any) -> int:
            if isinstance(value, np.ndarray):
                return int(value.nbytes)
            if isinstance(value, dict):
                return sum(retained(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(retained(item) for item in value)
            return 0

        return retained(self.records)

    def _trigger(self, fixed: dict[str, Any], hessian: dict[str, Any], registration_result: dict[str, Any]) -> tuple[bool, int]:
        mask = 0
        condition_limit = self.trigger.get("condition_number", 120.0)
        if condition_limit is not None and float(fixed["condition_number"]) > float(condition_limit):
            mask |= _TRIGGER_BITS["condition_number"]
        normalized_limit = self.trigger.get("normalized_lambda_min")
        if normalized_limit is not None and hessian["normalized_lambda_min"] < float(normalized_limit):
            mask |= _TRIGGER_BITS["normalized_lambda_min"]
        fitness_limit = self.trigger.get("min_fitness")
        if fitness_limit is not None and float(fixed["fitness"]) < float(fitness_limit):
            mask |= _TRIGGER_BITS["min_fitness"]
        iteration_limit = self.trigger.get("max_iterations")
        if iteration_limit is not None and int(registration_result.get("iterations", 0)) >= int(iteration_limit):
            mask |= _TRIGGER_BITS["max_iterations"]
        return mask != 0, mask

    def evaluate_frame(
        self,
        frame_id: int,
        timestamp: float,
        source_points: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        image_height: int,
        image_width: int,
        registration_result: dict[str, Any],
        final_attempt: dict[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        if (self.include_stage_ids or self.save_correspondences) and len(self.records) >= self.max_detail_frames:
            raise RuntimeError(
                "activation diagnostics detail capture is diagnostic-only and limited to "
                f"{self.max_detail_frames} frames; disable include_stage_ids/save_correspondences or explicitly "
                "raise max_detail_frames for a small diagnostic run"
            )
        start = time.perf_counter()
        cached_query = None
        cached_target = None
        if final_attempt is not None and np.array_equal(np.asarray(final_attempt.get("query_pose")), np.asarray(pose)):
            candidate = final_attempt.get("query")
            if candidate is not None and all(f"{stage}_indices" in candidate for stage in _STAGES):
                cached_query = candidate
                cached_target = final_attempt
        query_start = time.perf_counter()
        actual_query = cached_query or self._query(
            self.actual_index, pose, intrinsics, image_height, image_width, self.actual_scores
        )
        actual_query_ms = 0.0 if cached_query is not None else float((time.perf_counter() - query_start) * 1000.0)
        if cached_target is None and final_attempt is not None and np.array_equal(
            np.asarray(final_attempt.get("query_pose")), np.asarray(pose)
        ):
            final_indices = final_attempt.get("local_indices")
            if final_indices is not None and np.array_equal(
                np.asarray(final_indices), np.asarray(actual_query["indices"])
            ):
                cached_target = final_attempt
        actual_fixed, actual_normal_ms, actual_eval_ms = self._fixed(
            source_points, pose, self.actual_points, self.actual_normals, self.actual_global_ids,
            np.asarray(actual_query["indices"], dtype=np.int64), cached=cached_target,
        )
        actual_role = self._role_metrics(actual_query, actual_fixed, self.replacement_mask, self.actual_global_ids)
        actual_h = self._hessian_metrics(actual_fixed, actual_role)
        triggered, trigger_mask = self._trigger(actual_fixed, actual_h, registration_result)

        counter_query = None
        counter_fixed = _empty_fixed(source_points.shape[0])
        removed_role = {
            "total": 0, **{stage: 0 for stage in _STAGES}, "correspondence": 0, "unique_target": 0,
            "hessian": np.zeros((6, 6)), "corr_role_mask": np.empty(0, dtype=bool),
        }
        counter_h = self._hessian_metrics(counter_fixed, removed_role)
        counter_query_ms = counter_normal_ms = counter_eval_ms = 0.0
        if self.counter_index is not None:
            counter_start = time.perf_counter()
            counter_query = self._query(
                self.counter_index, pose, intrinsics, image_height, image_width, self.counter_scores
            )
            counter_query_ms = float((time.perf_counter() - counter_start) * 1000.0)
            counter_fixed, counter_normal_ms, counter_eval_ms = self._fixed(
                source_points, pose, self.counter_points, self.counter_normals, self.counter_global_ids,
                np.asarray(counter_query["indices"], dtype=np.int64),
            )
            removed_role = self._role_metrics(counter_query, counter_fixed, self.removed_mask, self.counter_global_ids)
            counter_h = self._hessian_metrics(counter_fixed, removed_role)

        record = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "role_available": self.role_available,
            "counterfactual_available": counter_query is not None,
            "actual_query": self._compact_query(actual_query, self.actual_global_ids),
            "actual_fixed": self._compact_fixed(actual_fixed),
            "actual_role": self._compact_role(actual_role),
            "actual_h": self._compact_h(actual_h),
            "counter_query": self._compact_query(counter_query, self.counter_global_ids),
            "counter_fixed": self._compact_fixed(counter_fixed),
            "removed_role": self._compact_role(removed_role),
            "counter_h": self._compact_h(counter_h),
            "trigger": triggered,
            "trigger_reason_mask": trigger_mask,
            "actual_query_latency_ms": actual_query_ms,
            "actual_normal_latency_ms": actual_normal_ms,
            "actual_eval_latency_ms": actual_eval_ms,
            "counterfactual_query_latency_ms": counter_query_ms,
            "counterfactual_normal_latency_ms": counter_normal_ms,
            "counterfactual_eval_latency_ms": counter_eval_ms,
            "diagnostic_latency_ms": float((time.perf_counter() - start) * 1000.0),
        }
        self.records.append(record)

    def _arrays(self) -> dict[str, np.ndarray]:
        records = self.records
        trigger_reasons = []
        for record in records:
            trigger_reasons.append("|".join(
                name for name, bit in _TRIGGER_BITS.items() if record["trigger_reason_mask"] & bit
            ))
        arrays: dict[str, np.ndarray] = {
            "frame_id": np.asarray([r["frame_id"] for r in records], dtype=np.int64),
            "timestamp": np.asarray([r["timestamp"] for r in records], dtype=np.float64),
            "role_available": np.asarray([r["role_available"] for r in records], dtype=bool),
            "counterfactual_available": np.asarray([r["counterfactual_available"] for r in records], dtype=bool),
            "trigger": np.asarray([r["trigger"] for r in records], dtype=bool),
            "trigger_reason_mask": np.asarray([r["trigger_reason_mask"] for r in records], dtype=np.uint8),
            "trigger_reasons": np.asarray(trigger_reasons),
            "reserve_budget": self.reserve_budgets,
        }
        for prefix, query_key, role_key, fixed_key, h_key in (
            ("actual_replacement", "actual_query", "actual_role", "actual_fixed", "actual_h"),
            ("counterfactual_removed", "counter_query", "removed_role", "counter_fixed", "counter_h"),
        ):
            arrays[f"{prefix}_total_count"] = np.asarray([r[role_key]["total"] for r in records], dtype=np.int64)
            for stage in _STAGES:
                arrays[f"{prefix}_{stage}_count"] = np.asarray([r[role_key][stage] for r in records], dtype=np.int64)
            arrays[f"{prefix}_correspondence_count"] = np.asarray([r[role_key]["correspondence"] for r in records], dtype=np.int64)
            arrays[f"{prefix}_unique_target_count"] = np.asarray([r[role_key]["unique_target"] for r in records], dtype=np.int64)
            arrays[f"{prefix}_hessian"] = np.stack([r[role_key]["hessian"] for r in records])
            arrays[f"{prefix}_hessian_trace_fraction"] = np.asarray([r[h_key]["trace_fraction"] for r in records])
            arrays[f"{prefix}_weak_fraction"] = np.asarray([r[h_key]["weak_fraction"] for r in records])
            arrays[f"{prefix}_hessian_lambda_min"] = np.asarray([r[h_key]["role_lambda_min"] for r in records])
            arrays[f"{prefix}_hessian_trace"] = np.asarray([r[h_key]["role_trace"] for r in records])
        for prefix, fixed_key, h_key in (
            ("actual", "actual_fixed", "actual_h"),
            ("counterfactual", "counter_fixed", "counter_h"),
        ):
            arrays[f"{prefix}_total_hessian"] = np.stack([r[fixed_key]["hessian"] for r in records])
            arrays[f"{prefix}_correspondence_count"] = np.asarray([r[fixed_key]["num_correspondences"] for r in records], dtype=np.int64)
            arrays[f"{prefix}_fitness"] = np.asarray([r[fixed_key]["fitness"] for r in records])
            arrays[f"{prefix}_condition_number"] = np.asarray([r[fixed_key]["condition_number"] for r in records])
            arrays[f"{prefix}_lambda_min"] = np.asarray([r[h_key]["lambda_min"] for r in records])
            arrays[f"{prefix}_normalized_lambda_min"] = np.asarray([r[h_key]["normalized_lambda_min"] for r in records])
        arrays["reserve_predicted_weak_gain"] = np.stack([r["actual_h"]["reserve_gain"] for r in records])
        arrays["reserve_predicted_weak_gain_fraction"] = np.stack([r["actual_h"]["reserve_gain_fraction"] for r in records])
        for field in (
            "actual_query_latency_ms", "actual_normal_latency_ms", "actual_eval_latency_ms",
            "counterfactual_query_latency_ms", "counterfactual_normal_latency_ms",
            "counterfactual_eval_latency_ms", "diagnostic_latency_ms",
        ):
            arrays[field] = np.asarray([r[field] for r in records])
        arrays["local_projection_input_count"] = np.asarray(
            [r["actual_query"].get("projection_input_count", r["actual_query"]["candidate_count"]) for r in records],
            dtype=np.int64,
        )
        arrays["local_frustum_to_projection_input_ratio"] = np.asarray([
            _safe_fraction(
                r["actual_query"]["frustum_count"],
                r["actual_query"].get("projection_input_count", r["actual_query"]["candidate_count"]),
            )
            for r in records
        ])
        arrays["local_precap_to_projection_input_ratio"] = np.asarray([
            _safe_fraction(
                r["actual_query"]["precap_count"],
                r["actual_query"].get("projection_input_count", r["actual_query"]["candidate_count"]),
            )
            for r in records
        ])
        for field in (
            "tested_block_count", "intersecting_block_count", "block_candidate_count",
            "oracle_count", "oracle_block_recall", "oracle_projection_recall", "oracle_frustum_recall",
            "oracle_overflow", "block_culling_latency_ms", "precap_latency_ms", "oracle_latency_ms",
        ):
            arrays[f"local_{field}"] = np.asarray([
                r["actual_query"].get(field, 0 if "recall" not in field else 1.0) for r in records
            ])
        arrays["local_culling_mode"] = np.asarray([
            r["actual_query"].get("culling_mode", self.local_config.frustum_culling) for r in records
        ])
        arrays["local_oracle_status"] = np.asarray([
            r["actual_query"].get("status", "ok") for r in records
        ])
        self._append_optional_details(arrays)
        return arrays

    def _append_optional_details(self, arrays: dict[str, np.ndarray]) -> None:
        if self.save_correspondences:
            for prefix, key in (("actual", "actual_fixed"), ("counterfactual", "counter_fixed")):
                lengths = [len(r[key]["target_ids"]) for r in self.records]
                arrays[f"{prefix}_correspondence_offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
                arrays[f"{prefix}_correspondence_source_ids"] = np.concatenate(
                    [np.asarray(r[key]["source_ids"], dtype=np.int64) for r in self.records]
                ) if sum(lengths) else np.empty(0, dtype=np.int64)
                arrays[f"{prefix}_correspondence_target_global_ids"] = np.concatenate(
                    [np.asarray(r[key]["target_ids"], dtype=np.int64) for r in self.records]
                ) if sum(lengths) else np.empty(0, dtype=np.int64)
        if self.include_stage_ids:
            for prefix, query_key in (
                ("actual", "actual_query"),
                ("counterfactual", "counter_query"),
            ):
                for stage in _STAGES:
                    parts = [
                        np.asarray(r[query_key]["stage_global_ids"][stage], dtype=np.int64)
                        if r[query_key] is not None else np.empty(0, dtype=np.int64)
                        for r in self.records
                    ]
                    arrays[f"{prefix}_{stage}_global_id_offsets"] = np.r_[0, np.cumsum([p.size for p in parts])]
                    arrays[f"{prefix}_{stage}_global_ids"] = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def save(self, output_dir: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        arrays = self._arrays()
        output = Path(output_dir)
        np.savez(output / "runtime_activation_diagnostics.npz", **arrays)
        replacement_local = arrays["actual_replacement_selected_count"]
        replacement_corr = arrays["actual_replacement_correspondence_count"]
        removed_local = arrays["counterfactual_removed_selected_count"]
        removed_corr = arrays["counterfactual_removed_correspondence_count"]
        candidates = arrays["local_projection_input_count"]
        frustum_counts = np.asarray(
            [record["actual_query"]["frustum_count"] for record in self.records], dtype=np.int64
        )
        target_cap = self.local_config.max_points
        pre_cap_sweep = (
            [int(target_cap * multiplier) for multiplier in (4, 8, 16)]
            if target_cap is not None else []
        )
        summary = {
            "enabled": True,
            "geometry_path": self.geometry_path,
            "role_available": self.role_available,
            "role_unavailable_reason": self.role_unavailable_reason,
            "counterfactual_enabled": self.counterfactual_enabled,
            "counterfactual_available": bool(np.any(arrays["counterfactual_available"])),
            "counterfactual_scope": "preserved_selected_rows_plus_removed_geometry" if self.counter_index is not None else "unavailable",
            "reserve_source": "replacement_set" if self.role_available else "unavailable",
            "reserve_budget": self.reserve_budgets.tolist(),
            "num_frames": len(self.records),
            "replacement_local_activation_rate": _finite_mean(replacement_local > 0) if self.role_available else None,
            "replacement_frames_active": int(np.count_nonzero(replacement_local)) if self.role_available else None,
            "replacement_correspondence_activation_rate": _finite_mean(replacement_corr > 0) if self.role_available else None,
            "replacement_correspondence_rate": _finite_mean(
                replacement_corr / np.maximum(arrays["actual_correspondence_count"], 1)
            ) if self.role_available else None,
            "replacement_hessian_trace_fraction_mean": _finite_mean(arrays["actual_replacement_hessian_trace_fraction"]) if self.role_available else None,
            "replacement_weak_fraction_mean": _finite_mean(arrays["actual_replacement_weak_fraction"]) if self.role_available else None,
            "removed_local_activation_rate": _finite_mean(removed_local > 0) if self.counter_index is not None else None,
            "removed_frames_active": int(np.count_nonzero(removed_local)) if self.counter_index is not None else None,
            "removed_correspondence_activation_rate": _finite_mean(removed_corr > 0) if self.counter_index is not None else None,
            "removed_correspondence_rate": _finite_mean(
                removed_corr / np.maximum(arrays["counterfactual_correspondence_count"], 1)
            ) if self.counter_index is not None else None,
            "removed_hessian_trace_fraction_mean": _finite_mean(arrays["counterfactual_removed_hessian_trace_fraction"]) if self.counter_index is not None else None,
            "removed_weak_fraction_mean": _finite_mean(arrays["counterfactual_removed_weak_fraction"]) if self.counter_index is not None else None,
            "replacement_minus_removed_local_activation_rate": None,
            "replacement_minus_removed_correspondence_rate": None,
            "replacement_minus_removed_hessian_trace_fraction": None,
            "replacement_minus_removed_weak_fraction": None,
            "trigger_mode": "plan_only",
            "trigger_rate": _finite_mean(arrays["trigger"]),
            "trigger_reason_counts": {
                name: int(np.count_nonzero(arrays["trigger_reason_mask"] & bit)) for name, bit in _TRIGGER_BITS.items()
            },
            "diagnostic_latency_mean_ms": _finite_mean(arrays["diagnostic_latency_ms"]),
            "diagnostic_latency_p95_ms": float(np.percentile(arrays["diagnostic_latency_ms"], 95)),
            "actual_query_latency_mean_ms": _finite_mean(arrays["actual_query_latency_ms"]),
            "counterfactual_query_latency_mean_ms": _finite_mean(arrays["counterfactual_query_latency_ms"]),
            "local_projection_input_count_mean": _finite_mean(candidates),
            "oracle_recall_semantics": "exact center pipeline; projection and frustum recall are identical",
            "local_candidate_fraction_mean": _finite_mean(
                candidates / max(self.actual_points.shape[0], 1)
            ),
            "local_frustum_to_projection_input_ratio_mean": _finite_mean(arrays["local_frustum_to_projection_input_ratio"]),
            "local_precap_to_projection_input_ratio_mean": _finite_mean(arrays["local_precap_to_projection_input_ratio"]),
            "observed_projection_input_count_p95": (
                int(np.ceil(np.percentile(candidates, 95))) if candidates.size else None
            ),
            "observed_frustum_count_p95": (
                int(np.ceil(np.percentile(frustum_counts, 95))) if frustum_counts.size else None
            ),
            "projection_pre_cap_sweep": pre_cap_sweep,
            "projection_pre_cap_sweep_semantics": (
                "diagnostic candidates equal to 4x/8x/16x the final local target cap; "
                "validate against exact frustum tile/depth recall before enabling"
            ),
        }
        if self.counter_index is not None:
            for key, replacement_key, removed_key in (
                ("replacement_minus_removed_local_activation_rate", "replacement_local_activation_rate", "removed_local_activation_rate"),
                ("replacement_minus_removed_correspondence_rate", "replacement_correspondence_rate", "removed_correspondence_rate"),
                ("replacement_minus_removed_hessian_trace_fraction", "replacement_hessian_trace_fraction_mean", "removed_hessian_trace_fraction_mean"),
                ("replacement_minus_removed_weak_fraction", "replacement_weak_fraction_mean", "removed_weak_fraction_mean"),
            ):
                summary[key] = float(summary[replacement_key] - summary[removed_key])
        with open(output / "runtime_activation_summary.json", "w") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return summary
