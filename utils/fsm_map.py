import os
from typing import Any

import numpy as np


REQUIRED_SPLATAM_FIELDS = ("means3D", "log_scales", "unnorm_rotations")
REQUIRED_FSM_FIELDS = (
    "means",
    "covariances",
    "scales",
    "rotations",
    "indices",
    "scores",
    "strategy",
    "compression_ratio",
    "lambda_min",
    "delta",
    "source_params_path",
)


def normalize_quaternions(quats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape [N, 4], got {quats.shape}")

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Quaternion norms must be positive")
    return quats / norms


def quaternions_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    quats = normalize_quaternions(quats)
    w, x, y, z = quats.T

    rotations = np.empty((quats.shape[0], 3, 3), dtype=np.float64)
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


def expand_log_scales(log_scales: np.ndarray) -> np.ndarray:
    log_scales = np.asarray(log_scales, dtype=np.float64)
    if log_scales.ndim != 2 or log_scales.shape[1] not in (1, 3):
        raise ValueError(f"Expected log scales with shape [N, 1] or [N, 3], got {log_scales.shape}")
    if log_scales.shape[1] == 1:
        log_scales = np.repeat(log_scales, 3, axis=1)
    return log_scales


def gaussians_to_covariances(
    log_scales: np.ndarray,
    unnorm_rotations: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    if eps < 0:
        raise ValueError("eps must be non-negative")

    expanded_log_scales = expand_log_scales(log_scales)
    rotations = quaternions_to_rotation_matrices(unnorm_rotations)
    if expanded_log_scales.shape[0] != rotations.shape[0]:
        raise ValueError(
            "log_scales and unnorm_rotations must have the same number of Gaussians: "
            f"{expanded_log_scales.shape[0]} != {rotations.shape[0]}"
        )

    variances = np.exp(expanded_log_scales) ** 2
    covariances = rotations @ np.array([np.diag(v) for v in variances]) @ np.transpose(rotations, (0, 2, 1))
    if eps > 0:
        covariances = covariances + eps * np.eye(3, dtype=np.float64)[None, :, :]
    return 0.5 * (covariances + np.transpose(covariances, (0, 2, 1)))


def load_splatam_params(params_path: str) -> dict[str, Any]:
    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)
    params = dict(np.load(params_path, allow_pickle=True))
    missing = [field for field in REQUIRED_SPLATAM_FIELDS if field not in params]
    if missing:
        raise KeyError(f"Missing required SplaTAM fields: {missing}")
    return params


def _as_string_array(value: str | None) -> np.ndarray:
    return np.asarray("" if value is None else value)


def _as_float_or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def build_fsm_map(
    params: dict[str, Any],
    indices: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    strategy: str = "full",
    compression_ratio: float = 1.0,
    lambda_min: float | None = None,
    delta: float | None = None,
    source_params_path: str | None = None,
    covariance_eps: float = 1e-6,
) -> dict[str, Any]:
    missing = [field for field in REQUIRED_SPLATAM_FIELDS if field not in params]
    if missing:
        raise KeyError(f"Missing required SplaTAM fields: {missing}")

    means = np.asarray(params["means3D"], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means3D with shape [N, 3], got {means.shape}")
    num_gaussians = means.shape[0]

    log_scales = np.asarray(params["log_scales"], dtype=np.float64)
    rotations = normalize_quaternions(np.asarray(params["unnorm_rotations"], dtype=np.float64))
    if log_scales.shape[0] != num_gaussians or rotations.shape[0] != num_gaussians:
        raise ValueError("means3D, log_scales, and unnorm_rotations must have matching first dimension")

    if indices is None:
        indices = np.arange(num_gaussians, dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"Expected indices with shape [N*], got {indices.shape}")
        if indices.size == 0:
            raise ValueError("indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= num_gaussians):
            raise IndexError("indices contain values outside the Gaussian range")
        if np.unique(indices).size != indices.size:
            raise ValueError("indices must be unique")

    if scores is None:
        score_array = np.asarray([], dtype=np.float64)
    else:
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.ndim != 1 or score_array.shape[0] != num_gaussians:
            raise ValueError(f"Expected scores with shape [{num_gaussians}], got {score_array.shape}")

    expanded_log_scales = expand_log_scales(log_scales)
    covariances = gaussians_to_covariances(log_scales, rotations, eps=covariance_eps)
    fsm_map = {
        "means": means[indices].astype(np.float32),
        "covariances": covariances[indices].astype(np.float32),
        "scales": np.exp(expanded_log_scales[indices]).astype(np.float32),
        "rotations": rotations[indices].astype(np.float32),
        "indices": indices.astype(np.int64),
        "scores": score_array.astype(np.float32),
        "strategy": _as_string_array(strategy),
        "compression_ratio": np.asarray(float(compression_ratio), dtype=np.float32),
        "lambda_min": np.asarray(_as_float_or_nan(lambda_min), dtype=np.float32),
        "delta": np.asarray(_as_float_or_nan(delta), dtype=np.float32),
        "source_params_path": _as_string_array(source_params_path),
    }
    validate_fsm_map(fsm_map)
    return fsm_map


def save_fsm_map(fsm_map: dict[str, Any], output_path: str) -> None:
    validate_fsm_map(fsm_map)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **fsm_map)


def load_fsm_map(map_path: str) -> dict[str, Any]:
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    fsm_map = dict(np.load(map_path, allow_pickle=True))
    validate_fsm_map(fsm_map)
    return fsm_map


def validate_fsm_map(fsm_map: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FSM_FIELDS if field not in fsm_map]
    if missing:
        raise KeyError(f"Missing required FSM map fields: {missing}")

    means = np.asarray(fsm_map["means"])
    covariances = np.asarray(fsm_map["covariances"])
    scales = np.asarray(fsm_map["scales"])
    rotations = np.asarray(fsm_map["rotations"])
    indices = np.asarray(fsm_map["indices"])
    scores = np.asarray(fsm_map["scores"])

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means with shape [N*, 3], got {means.shape}")
    num_retained = means.shape[0]
    if num_retained == 0:
        raise ValueError("FSM map must retain at least one Gaussian")
    if covariances.shape != (num_retained, 3, 3):
        raise ValueError(f"Expected covariances with shape [{num_retained}, 3, 3], got {covariances.shape}")
    if scales.shape != (num_retained, 3):
        raise ValueError(f"Expected scales with shape [{num_retained}, 3], got {scales.shape}")
    if rotations.shape != (num_retained, 4):
        raise ValueError(f"Expected rotations with shape [{num_retained}, 4], got {rotations.shape}")
    if indices.shape != (num_retained,):
        raise ValueError(f"Expected indices with shape [{num_retained}], got {indices.shape}")
    if scores.ndim != 1:
        raise ValueError(f"Expected scores with shape [N] or empty [0], got {scores.shape}")
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(covariances)):
        raise ValueError("means and covariances must be finite")
    if np.any(scales <= 0) or np.any(~np.isfinite(scales)):
        raise ValueError("scales must be finite and positive")
    if np.unique(indices).size != indices.size:
        raise ValueError("indices must be unique")

    if not np.allclose(covariances, np.transpose(covariances, (0, 2, 1)), atol=1e-5):
        raise ValueError("covariances must be symmetric")
    eigvals = np.linalg.eigvalsh(covariances.astype(np.float64))
    if np.any(eigvals <= 0):
        raise ValueError("covariances must be positive definite")

    rotation_norms = np.linalg.norm(rotations.astype(np.float64), axis=1)
    if not np.allclose(rotation_norms, 1.0, atol=1e-5):
        raise ValueError("rotations must be normalized quaternions")
