"""Diagnostic-only comparison of production and exact final local targets."""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.fsm_local_map import (
    LocalSubmapConfig,
    MapSpatialIndex,
    _camera_points,
    coverage_sampling_components,
)
from utils.fsm_registration import build_target_index
from utils.fsm_robust_registration import estimate_normals, evaluate_point_to_plane_diagnostics


@dataclass(frozen=True)
class FinalTargetOracleConfig:
    diagnostic_only: bool = True
    chunk_size: int = 262144
    normal_k: int = 20
    max_correspondence_distance: float = 0.05
    min_correspondences: int = 6
    robust_loss: str = "huber"
    loss_scale: float = 0.02
    rank_tolerance: float = 1e-9
    weak_subspace_dim: int = 1
    top_score_fractions: tuple[float, ...] = (0.01, 0.05, 0.1)
    include_target_ids: bool = False

    def __post_init__(self) -> None:
        if self.diagnostic_only is not True:
            raise ValueError("FinalTargetOracleConfig requires diagnostic_only=True")
        for value, name, minimum in (
            (self.chunk_size, "chunk_size", 1),
            (self.normal_k, "normal_k", 3),
            (self.min_correspondences, "min_correspondences", 1),
            (self.weak_subspace_dim, "weak_subspace_dim", 1),
        ):
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.weak_subspace_dim > 6:
            raise ValueError("weak_subspace_dim must not exceed 6")
        for value, name in (
            (self.max_correspondence_distance, "max_correspondence_distance"),
            (self.loss_scale, "loss_scale"),
            (self.rank_tolerance, "rank_tolerance"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.robust_loss not in ("none", "huber", "cauchy"):
            raise ValueError("robust_loss must be 'none', 'huber', or 'cauchy'")
        fractions = np.asarray(self.top_score_fractions, dtype=np.float64)
        if fractions.ndim != 1 or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0) or np.any(fractions > 1.0):
            raise ValueError("top_score_fractions must contain values in (0, 1]")


def _ratio(numerator: float, denominator: float, empty: float = 1.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(empty)


def _set_metrics(exact: np.ndarray, production: np.ndarray) -> dict[str, Any]:
    exact = np.unique(np.asarray(exact, dtype=np.int64))
    production = np.unique(np.asarray(production, dtype=np.int64))
    intersection = np.intersect1d(exact, production, assume_unique=True)
    union_count = exact.size + production.size - intersection.size
    return {
        "exact_count": int(exact.size),
        "production_count": int(production.size),
        "intersection_count": int(intersection.size),
        "recall": _ratio(intersection.size, exact.size),
        "precision": _ratio(intersection.size, production.size),
        "jaccard": _ratio(intersection.size, union_count),
        "symmetric_difference_count": int(exact.size + production.size - 2 * intersection.size),
    }


def _histogram_metrics(exact: np.ndarray, production: np.ndarray) -> dict[str, float]:
    exact = np.asarray(exact, dtype=np.float64)
    production = np.asarray(production, dtype=np.float64)
    overlap = float(np.sum(np.minimum(exact, production)))
    union = float(np.sum(np.maximum(exact, production)))
    exact_sum = float(np.sum(exact))
    production_sum = float(np.sum(production))
    exact_prob = exact / exact_sum if exact_sum else np.zeros_like(exact)
    production_prob = production / production_sum if production_sum else np.zeros_like(production)
    l1 = float(np.sum(np.abs(exact_prob - production_prob)))
    return {
        "recall": _ratio(overlap, exact_sum),
        "jaccard": _ratio(overlap, union),
        "tv": 0.5 * l1,
        "l1": l1,
    }


def _target_components(
    points: np.ndarray,
    indices: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    local_config: LocalSubmapConfig,
    positive_near: float | None = None,
) -> dict[str, np.ndarray]:
    if indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return {"tile_id": empty, "depth_bin": empty, "bucket_id": empty}
    camera = _camera_points(points[indices].astype(np.float64, copy=False), pose)
    depth = camera[:, 2]
    u = intrinsics[0, 0] * camera[:, 0] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / depth + intrinsics[1, 2]
    if local_config.near <= 0.0 and positive_near is None:
        positive_near = float(np.min(depth))
    return coverage_sampling_components(u, v, depth, height, width, local_config, positive_near=positive_near)


def _distribution_comparison(
    exact: dict[str, np.ndarray], production: dict[str, np.ndarray], local_config: LocalSubmapConfig
) -> dict[str, float]:
    sizes = {
        "tile": local_config.tile_rows * local_config.tile_cols,
        "depth": local_config.depth_bins,
        "joint": local_config.tile_rows * local_config.tile_cols * local_config.depth_bins,
    }
    keys = {"tile": "tile_id", "depth": "depth_bin", "joint": "bucket_id"}
    result: dict[str, float] = {}
    for name, key in keys.items():
        exact_hist = np.bincount(exact[key], minlength=sizes[name])
        production_hist = np.bincount(production[key], minlength=sizes[name])
        for metric, value in _histogram_metrics(exact_hist, production_hist).items():
            result[f"{name}_{metric}"] = value
    return result


def _score_metrics(
    scores: np.ndarray | None,
    exact: np.ndarray,
    production: np.ndarray,
    fractions: tuple[float, ...],
) -> dict[str, Any]:
    if scores is None:
        return {"available": False}
    result: dict[str, Any] = {"available": True}
    for name, ids in (("exact", exact), ("production", production)):
        values = scores[ids]
        result[f"{name}_mean"] = float(np.mean(values)) if values.size else None
        result[f"{name}_std"] = float(np.std(values)) if values.size else None
        result[f"{name}_min"] = float(np.min(values)) if values.size else None
        result[f"{name}_max"] = float(np.max(values)) if values.size else None
    for fraction in fractions:
        take = max(1, int(np.ceil(exact.size * fraction))) if exact.size else 0
        top = exact[np.lexsort((exact, -scores[exact]))[:take]] if take else np.empty(0, dtype=np.int64)
        overlap = np.intersect1d(top, production, assume_unique=False).size
        result[f"top_{fraction:g}_recall"] = _ratio(overlap, top.size)
    return result


def _empty_hessian() -> dict[str, Any]:
    return {
        "hessian": np.zeros((6, 6), dtype=np.float64),
        "eigenvalues": np.zeros(6, dtype=np.float64),
        "eigenvectors": np.eye(6, dtype=np.float64),
        "trace": 0.0,
        "lambda_min": 0.0,
        "normalized_lambda_min": 0.0,
        "condition_number": None,
        "effective_rank": 0,
        "num_correspondences": 0,
    }


def _fixed_hessian(
    source_points: np.ndarray,
    pose: np.ndarray,
    target_points: np.ndarray,
    config: FinalTargetOracleConfig,
) -> dict[str, Any]:
    if target_points.shape[0] < 3:
        return _empty_hessian()
    target_tree = build_target_index(target_points)["target_tree"]
    normals = estimate_normals(target_points, k=min(config.normal_k, target_points.shape[0]), target_tree=target_tree)
    result = evaluate_point_to_plane_diagnostics(
        source_points=source_points,
        target_points=target_points,
        target_normals=normals,
        transform=pose,
        max_correspondence_distance=config.max_correspondence_distance,
        target_tree=target_tree,
        loss=config.robust_loss,
        loss_scale=config.loss_scale,
        min_correspondences=config.min_correspondences,
        rank_tolerance=config.rank_tolerance,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(result["hessian"], dtype=np.float64))
    trace = float(np.trace(result["hessian"]))
    condition = float(result["condition_number"])
    return {
        "hessian": np.asarray(result["hessian"], dtype=np.float64),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace": trace,
        "lambda_min": float(eigenvalues[0]),
        "normalized_lambda_min": _ratio(float(eigenvalues[0]), trace, empty=0.0),
        "condition_number": condition if np.isfinite(condition) else None,
        "effective_rank": int(result["effective_rank"]),
        "num_correspondences": int(result["num_correspondences"]),
    }


def _hessian_comparison(exact: dict[str, Any], production: dict[str, Any], weak_dim: int) -> dict[str, Any]:
    exact_h = exact["hessian"]
    production_h = production["hessian"]
    weak_vectors = exact["eigenvectors"][:, :weak_dim]
    exact_weak = float(np.trace(weak_vectors.T @ exact_h @ weak_vectors))
    production_weak = float(np.trace(weak_vectors.T @ production_h @ weak_vectors))
    production_vectors = production["eigenvectors"][:, :weak_dim]
    singular = np.linalg.svd(weak_vectors.T @ production_vectors, compute_uv=False)
    exact_condition = exact["condition_number"]
    production_condition = production["condition_number"]
    result = {
        "trace_recall": _ratio(production["trace"], exact["trace"], empty=0.0),
        "lambda_min_recall": _ratio(production["lambda_min"], exact["lambda_min"], empty=0.0),
        "normalized_lambda_min_recall": _ratio(
            production["normalized_lambda_min"], exact["normalized_lambda_min"], empty=0.0
        ),
        "weak_energy_recall": _ratio(production_weak, exact_weak, empty=0.0),
        "weak_energy_loss": exact_weak - production_weak,
        "weak_subspace_min_cosine": float(np.min(singular)),
        "weak_subspace_mean_cosine": float(np.mean(singular)),
        "hessian_frobenius_l1": float(np.linalg.norm(exact_h - production_h, ord="fro")),
        "rank_loss": int(exact["effective_rank"] - production["effective_rank"]),
        "condition_number_ratio": (
            float(production_condition / exact_condition)
            if exact_condition not in (None, 0.0) and production_condition is not None else None
        ),
    }
    for prefix, values in (("exact", exact), ("production", production)):
        for key in (
            "trace", "lambda_min", "normalized_lambda_min", "condition_number",
            "effective_rank", "num_correspondences",
        ):
            result[f"{prefix}_{key}"] = values[key]
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), item, output)
    elif np.isscalar(value) or value is None:
        output[prefix] = value


class FinalTargetOracleRuntime:
    """Read-only sidecar; frame evaluation records diagnostics and always returns None."""

    def __init__(
        self,
        config: FinalTargetOracleConfig,
        index: MapSpatialIndex,
        local_config: LocalSubmapConfig,
        scores: np.ndarray | None = None,
        block_score_token: str | None = None,
    ) -> None:
        if not isinstance(config, FinalTargetOracleConfig) or config.diagnostic_only is not True:
            raise ValueError("a strict diagnostic-only FinalTargetOracleConfig is required")
        if not isinstance(index, MapSpatialIndex):
            raise TypeError("index must be a MapSpatialIndex")
        if local_config.max_points is None or local_config.sampling_policy != "coverage":
            raise ValueError("local_config must use coverage sampling with max_points")
        if scores is not None and block_score_token is not None:
            raise ValueError("scores and block_score_token are mutually exclusive")
        self.config = config
        self.index = index
        self.local_config = local_config
        if block_score_token is not None:
            self.scores = index.score_snapshot(block_score_token)
        else:
            self.scores = None if scores is None else np.asarray(scores, dtype=np.float64)
        self.block_score_token = block_score_token
        self.records: list[dict[str, Any]] = []
        self.hessians: list[dict[str, np.ndarray]] = []

    def evaluate_frame(
        self,
        frame_id: int,
        timestamp: float,
        source_points: np.ndarray,
        final_pose: np.ndarray,
        intrinsics: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> None:
        start = time.perf_counter()
        record: dict[str, Any] = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "status": "ok",
            "error_type": "",
            "error_message": "",
        }
        try:
            production = self.index.query_local_submap(
                final_pose, intrinsics, image_height, image_width, self.local_config,
                scores=None if self.block_score_token is not None else self.scores,
                block_score_token=self.block_score_token,
            )
            exact = self.index.stream_exact_final_target(
                final_pose, intrinsics, image_height, image_width, self.local_config,
                scores=None if self.block_score_token is not None else self.scores,
                block_score_token=self.block_score_token,
                chunk_size=self.config.chunk_size,
            )
            exact_ids = np.asarray(exact["indices"], dtype=np.int64)
            production_ids = np.asarray(production["indices"], dtype=np.int64)
            exact_components = {
                key: np.asarray(exact[key], dtype=np.int64) for key in ("tile_id", "depth_bin", "bucket_id")
            }
            production_components = _target_components(
                self.index.points, production_ids, final_pose, intrinsics,
                image_height, image_width, self.local_config,
            )
            exact_h = _fixed_hessian(
                np.asarray(source_points, dtype=np.float64), final_pose,
                self.index.points[exact_ids].astype(np.float64, copy=False), self.config,
            )
            production_h = _fixed_hessian(
                np.asarray(source_points, dtype=np.float64), final_pose,
                self.index.points[production_ids].astype(np.float64, copy=False), self.config,
            )
            record.update({
                "sets": _set_metrics(exact_ids, production_ids),
                "histograms": _distribution_comparison(exact_components, production_components, self.local_config),
                "scores": _score_metrics(self.scores, exact_ids, production_ids, self.config.top_score_fractions),
                "hessian": _hessian_comparison(exact_h, production_h, self.config.weak_subspace_dim),
                "exact_visible_count": int(exact["visible_count"]),
                "exact_retained_count": int(exact["retained_count"]),
                "exact_peak_retained_bound": int(exact["peak_retained_bound"]),
                "production_projection_input_count": int(production["projection_input_count"]),
                "production_frustum_count": int(production["frustum_count"]),
            })
            if self.config.include_target_ids:
                record["exact_target_ids"] = exact_ids.tolist()
                record["production_target_ids"] = production_ids.tolist()
            self.hessians.append({
                "exact": exact_h["hessian"],
                "production": production_h["hessian"],
                "exact_eigenvalues": exact_h["eigenvalues"],
                "production_eigenvalues": production_h["eigenvalues"],
            })
        except Exception as error:
            record.update({
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            self.hessians.append({
                "exact": np.full((6, 6), np.nan),
                "production": np.full((6, 6), np.nan),
                "exact_eigenvalues": np.full(6, np.nan),
                "production_eigenvalues": np.full(6, np.nan),
            })
        record["diagnostic_latency_ms"] = float((time.perf_counter() - start) * 1000.0)
        self.records.append(record)
        return None

    def save(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for record in self.records:
            row: dict[str, Any] = {}
            _flatten("", record, row)
            rows.append(_json_value(row))
        fieldnames = sorted({key for row in rows for key in row})
        with open(output / "frames.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        np.savez(
            output / "hessians.npz",
            frame_id=np.asarray([record["frame_id"] for record in self.records], dtype=np.int64),
            exact_hessian=np.stack([item["exact"] for item in self.hessians]) if self.hessians else np.empty((0, 6, 6)),
            production_hessian=np.stack([item["production"] for item in self.hessians]) if self.hessians else np.empty((0, 6, 6)),
            exact_eigenvalues=np.stack([item["exact_eigenvalues"] for item in self.hessians]) if self.hessians else np.empty((0, 6)),
            production_eigenvalues=np.stack([item["production_eigenvalues"] for item in self.hessians]) if self.hessians else np.empty((0, 6)),
        )
        ok = [record for record in self.records if record["status"] == "ok"]
        summary = {
            "diagnostic_only": True,
            "num_frames": len(self.records),
            "num_ok": len(ok),
            "num_errors": len(self.records) - len(ok),
            "set_recall_mean": float(np.mean([record["sets"]["recall"] for record in ok])) if ok else None,
            "set_jaccard_mean": float(np.mean([record["sets"]["jaccard"] for record in ok])) if ok else None,
            "joint_histogram_recall_mean": float(np.mean([record["histograms"]["joint_recall"] for record in ok])) if ok else None,
            "tile_histogram_recall_mean": float(np.mean([record["histograms"]["tile_recall"] for record in ok])) if ok else None,
            "depth_histogram_recall_mean": float(np.mean([record["histograms"]["depth_recall"] for record in ok])) if ok else None,
            "score_exact_mean": float(np.mean([
                record["scores"]["exact_mean"] for record in ok
                if record["scores"].get("available") and record["scores"].get("exact_mean") is not None
            ])) if any(record["scores"].get("available") and record["scores"].get("exact_mean") is not None for record in ok) else None,
            "score_production_mean": float(np.mean([
                record["scores"]["production_mean"] for record in ok
                if record["scores"].get("available") and record["scores"].get("production_mean") is not None
            ])) if any(record["scores"].get("available") and record["scores"].get("production_mean") is not None for record in ok) else None,
            "hessian_trace_recall_mean": float(np.mean([record["hessian"]["trace_recall"] for record in ok])) if ok else None,
            "hessian_lambda_min_recall_mean": float(np.mean([record["hessian"]["lambda_min_recall"] for record in ok])) if ok else None,
            "hessian_condition_number_ratio_mean": float(np.mean([
                record["hessian"]["condition_number_ratio"] for record in ok
                if record["hessian"].get("condition_number_ratio") is not None
            ])) if any(record["hessian"].get("condition_number_ratio") is not None for record in ok) else None,
            "diagnostic_latency_total_ms": float(sum(record["diagnostic_latency_ms"] for record in self.records)),
            "diagnostic_latency_mean_ms": float(np.mean([
                record["diagnostic_latency_ms"] for record in self.records
            ])) if self.records else 0.0,
            "errors": [
                {key: record[key] for key in ("frame_id", "error_type", "error_message")}
                for record in self.records if record["status"] == "error"
            ],
        }
        with open(output / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(_json_value(summary), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return summary
