import hashlib
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
ACTIVATION_GEOMETRY_SCHEMA_VERSION = 2
RUNTIME_RESERVE_SCHEMA_VERSION = 3
ACTIVATION_GEOMETRY_COMPATIBLE_SCHEMA_VERSIONS = (1, 2)
ACTIVATION_GEOMETRY_ID_SPACE = "source_params_global"
REQUIRED_ACTIVATION_GEOMETRY_FIELDS = (
    "schema_version",
    "id_space",
    "selected_global_ids",
    "baseline_global_ids",
    "preserved_global_ids",
    "replacement_global_ids",
    "removed_global_ids",
    "removed_means",
    "removed_log_scales",
    "removed_rotations",
    "removed_scales",
    "removed_covariances",
    "removed_scores",
    "removed_normals",
    "normal_source",
    "source_params_sha256",
    "selection_controls_sha256",
    "map_selected_ids_sha256",
)
REQUIRED_ACTIVATION_GEOMETRY_RESERVE_FIELDS = (
    "reserve_global_ids",
    "reserve_means",
    "reserve_normals",
    "reserve_scores",
    "reserve_covariances",
    "reserve_scales",
    "reserve_rotations",
    "reserve_count",
    "reserve_policy",
)


def normalize_quaternions(quats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")
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
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")

    expanded_log_scales = expand_log_scales(log_scales)
    rotations = quaternions_to_rotation_matrices(unnorm_rotations)
    if expanded_log_scales.shape[0] != rotations.shape[0]:
        raise ValueError(
            "log_scales and unnorm_rotations must have the same number of Gaussians: "
            f"{expanded_log_scales.shape[0]} != {rotations.shape[0]}"
        )

    variances = np.exp(expanded_log_scales) ** 2
    covariances = (rotations * variances[:, None, :]) @ np.transpose(rotations, (0, 2, 1))
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
    source_params_sha256: str | None = None,
    selection_controls_sha256: str | None = None,
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

    retained_means = means[indices]
    retained_log_scales = log_scales[indices]
    retained_rotations = rotations[indices]
    expanded_log_scales = expand_log_scales(retained_log_scales)
    covariances = gaussians_to_covariances(retained_log_scales, retained_rotations, eps=covariance_eps)
    fsm_map = {
        "means": retained_means.astype(np.float32),
        "covariances": covariances.astype(np.float32),
        "scales": np.exp(expanded_log_scales).astype(np.float32),
        "rotations": retained_rotations.astype(np.float32),
        "indices": indices.astype(np.int64),
        "scores": score_array[indices].astype(np.float32) if score_array.size else score_array.astype(np.float32),
        "strategy": _as_string_array(strategy),
        "compression_ratio": np.asarray(float(compression_ratio), dtype=np.float32),
        "lambda_min": np.asarray(_as_float_or_nan(lambda_min), dtype=np.float32),
        "delta": np.asarray(_as_float_or_nan(delta), dtype=np.float32),
        "source_params_path": _as_string_array(source_params_path),
        "source_params_sha256": _as_string_array(source_params_sha256),
        "selection_controls_sha256": _as_string_array(selection_controls_sha256),
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
    scores = np.asarray(fsm_map.get("scores", []))
    indices = np.asarray(fsm_map.get("indices", []), dtype=np.int64)
    if scores.ndim == 1 and indices.ndim == 1 and scores.size > indices.size and indices.size > 0:
        if np.max(indices) >= scores.size:
            raise ValueError("Legacy full-length scores do not cover all retained indices")
        fsm_map["scores"] = scores[indices]
    validate_fsm_map(fsm_map)
    return fsm_map


def selected_ids_sha256(indices: np.ndarray) -> str:
    ids = np.asarray(indices, dtype="<i8")
    if ids.ndim != 1:
        raise ValueError(f"Expected selected IDs with shape [N], got {ids.shape}")
    digest = hashlib.sha256()
    digest.update(np.asarray([ids.size], dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(ids).tobytes())
    return digest.hexdigest()


def load_activation_geometry(
    geometry_path: str,
    fsm_map: dict[str, Any] | str,
    *,
    source_params_sha256: str | None = None,
    selection_controls_sha256: str | None = None,
) -> dict[str, Any]:
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(geometry_path)
    geometry = dict(np.load(geometry_path, allow_pickle=False))
    map_value = load_fsm_map(fsm_map) if isinstance(fsm_map, (str, os.PathLike)) else fsm_map
    validate_activation_geometry(
        geometry,
        map_value,
        source_params_sha256=source_params_sha256,
        selection_controls_sha256=selection_controls_sha256,
    )
    return geometry


_RUNTIME_RESERVE_ARRAYS = {
    "reserve_means": (3,), "reserve_normals": (3,), "reserve_scores": (),
    "reserve_covariances": (3, 3),
}
_RUNTIME_RESERVE_METADATA = (
    "id_space", "map_selected_ids_sha256", "source_params_sha256", "selection_controls_sha256",
    "reserve_count", "reserve_normal_source",
)


def _validate_runtime_reserve(geometry: dict[str, Any], fsm_map: dict[str, Any]) -> None:
    version = np.asarray(geometry["schema_version"])
    if version.shape != () or version.dtype.kind not in "iu" or int(version) != RUNTIME_RESERVE_SCHEMA_VERSION:
        raise ValueError("Invalid runtime reserve schema version")
    if _scalar_string(geometry["id_space"], "id_space") != ACTIVATION_GEOMETRY_ID_SPACE:
        raise ValueError("Invalid runtime reserve ID space")
    indices = np.asarray(fsm_map["indices"])
    if selected_ids_sha256(indices) != _scalar_string(geometry["map_selected_ids_sha256"], "map_selected_ids_sha256"):
        raise ValueError("Runtime reserve does not match main map indices and order")
    for key in ("source_params_sha256", "selection_controls_sha256"):
        digest = _scalar_string(geometry[key], key)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())
                or digest != _scalar_string(fsm_map[key], key)):
            raise ValueError(f"Runtime reserve {key} does not match main map")
    ids = _global_ids(geometry, "reserve_global_ids")
    if np.isin(ids, indices).any():
        raise ValueError("Runtime reserve overlaps main map")
    count = np.asarray(geometry["reserve_count"])
    if count.shape != () or count.dtype.kind not in "iu" or int(count) != len(ids):
        raise ValueError("Runtime reserve count does not match its IDs")
    _scalar_string(geometry["reserve_normal_source"], "reserve_normal_source")
    for key, shape in _RUNTIME_RESERVE_ARRAYS.items():
        values = np.asarray(geometry[key])
        if values.shape != (len(ids), *shape) or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"Invalid runtime reserve field: {key}")
    covariances = np.asarray(geometry["reserve_covariances"], dtype=np.float64)
    if (not np.allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-5)
            or np.any(np.linalg.eigvalsh(covariances) <= 0)):
        raise ValueError("Runtime reserve covariances must be symmetric positive definite")
    if "reserve_supported" in geometry:
        supported = np.asarray(geometry["reserve_supported"])
        if supported.dtype.kind != "b" or supported.shape != (len(ids),):
            raise ValueError("Invalid runtime reserve support mask")


def load_runtime_reserve_geometry(path: str, fsm_map: dict[str, Any]) -> dict[str, Any] | None:
    """Load a runtime-only sidecar or the historical complete activation archive."""
    with np.load(path, allow_pickle=False) as archive:
        version = int(np.asarray(archive["schema_version"])) if "schema_version" in archive else 0
        if version == RUNTIME_RESERVE_SCHEMA_VERSION:
            geometry = dict(archive)
            _validate_runtime_reserve(geometry, fsm_map)
            return geometry
    if version < 2:
        return None
    return load_activation_geometry(
        path, fsm_map, source_params_sha256=str(np.asarray(fsm_map["source_params_sha256"])),
        selection_controls_sha256=str(np.asarray(fsm_map["selection_controls_sha256"])),
    )


def export_runtime_reserve_geometry(source: str, fsm_map: dict[str, Any], output: str) -> None:
    """Derive an exact-geometry deployment sidecar without rewriting the audit archive."""
    archive = load_activation_geometry(source, fsm_map)
    if int(archive["schema_version"]) < 2:
        raise ValueError("The audit archive does not contain a reserve")
    geometry = {key: archive[key] for key in (*_RUNTIME_RESERVE_ARRAYS, *_RUNTIME_RESERVE_METADATA, "reserve_global_ids")
                if key != "reserve_normal_source"}
    geometry["schema_version"] = np.asarray(RUNTIME_RESERVE_SCHEMA_VERSION, dtype=np.int64)
    geometry["reserve_normal_source"] = archive.get("reserve_normal_source", np.asarray("sidecar_reserve_normals"))
    if "reserve_supported" in archive:
        geometry["reserve_supported"] = np.asarray(archive["reserve_supported"], dtype=bool)
    _validate_runtime_reserve(geometry, fsm_map)
    with open(output, "xb") as stream:
        np.savez_compressed(stream, **geometry)


def fsm_map_field_sizes(fsm_map: dict[str, Any]) -> dict[str, int]:
    validate_fsm_map(fsm_map)
    return {key: int(np.asarray(value).nbytes) for key, value in fsm_map.items()}


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
    if scores.ndim != 1 or scores.size not in (0, num_retained):
        raise ValueError(f"Expected scores with shape [{num_retained}] or empty [0], got {scores.shape}")
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
    for field in ("source_params_sha256", "selection_controls_sha256"):
        if field in fsm_map:
            value = _scalar_string(fsm_map[field], field)
            if value and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower())):
                raise ValueError(f"{field} must be empty or a SHA-256 hex digest")


def _scalar_string(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{field} must be a scalar string")
    return str(array)


def _global_ids(geometry: dict[str, Any], field: str) -> np.ndarray:
    values = np.asarray(geometry[field])
    if values.dtype.kind not in "iu" or values.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.unique(values).size != values.size:
        raise ValueError(f"{field} must contain unique non-negative global IDs")
    return values


def validate_activation_geometry(
    geometry: dict[str, Any],
    fsm_map: dict[str, Any],
    *,
    source_params_sha256: str | None = None,
    selection_controls_sha256: str | None = None,
) -> None:
    validate_fsm_map(fsm_map)
    missing = [field for field in REQUIRED_ACTIVATION_GEOMETRY_FIELDS if field not in geometry]
    if missing:
        raise KeyError(f"Missing required activation geometry fields: {missing}")

    version = np.asarray(geometry["schema_version"])
    if (
        version.shape != ()
        or version.dtype.kind not in "iu"
        or int(version) not in ACTIVATION_GEOMETRY_COMPATIBLE_SCHEMA_VERSIONS
    ):
        raise ValueError(f"Unsupported activation geometry schema_version: {version}")
    schema_version = int(version)
    if schema_version >= 2:
        missing_reserve = [field for field in REQUIRED_ACTIVATION_GEOMETRY_RESERVE_FIELDS if field not in geometry]
        if missing_reserve:
            raise KeyError(f"Missing required activation geometry reserve fields: {missing_reserve}")
    if _scalar_string(geometry["id_space"], "id_space") != ACTIVATION_GEOMETRY_ID_SPACE:
        raise ValueError(f"id_space must be {ACTIVATION_GEOMETRY_ID_SPACE!r}")

    selected = _global_ids(geometry, "selected_global_ids")
    baseline = _global_ids(geometry, "baseline_global_ids")
    preserved = _global_ids(geometry, "preserved_global_ids")
    replacement = _global_ids(geometry, "replacement_global_ids")
    removed = _global_ids(geometry, "removed_global_ids")
    map_indices = np.asarray(fsm_map["indices"], dtype=np.int64)
    if not np.array_equal(selected, map_indices):
        raise ValueError("selected_global_ids must exactly match FSM map indices and order")
    if selected_ids_sha256(map_indices) != _scalar_string(geometry["map_selected_ids_sha256"], "map_selected_ids_sha256"):
        raise ValueError("map_selected_ids_sha256 does not match FSM map indices")

    selected_set = set(selected.tolist())
    baseline_set = set(baseline.tolist())
    preserved_set = set(preserved.tolist())
    replacement_set = set(replacement.tolist())
    removed_set = set(removed.tolist())
    if not replacement_set <= selected_set:
        raise ValueError("replacement_global_ids must belong to selected_global_ids")
    if removed_set & selected_set:
        raise ValueError("removed_global_ids must not belong to selected_global_ids")
    if preserved_set & removed_set or baseline_set != preserved_set | removed_set:
        raise ValueError("baseline_global_ids must equal disjoint preserved_global_ids plus removed_global_ids")
    if baseline.size != preserved.size + removed.size:
        raise ValueError("baseline/preserved/removed counts are inconsistent")
    if preserved_set & replacement_set or selected_set != preserved_set | replacement_set:
        raise ValueError("selected_global_ids must equal disjoint preserved_global_ids plus replacement_global_ids")
    if selected.size != preserved.size + replacement.size:
        raise ValueError("selected/preserved/replacement counts are inconsistent")

    removed_count = removed.size
    expected_shapes = {
        "removed_means": (removed_count, 3),
        "removed_rotations": (removed_count, 4),
        "removed_scales": (removed_count, 3),
        "removed_covariances": (removed_count, 3, 3),
        "removed_scores": (removed_count,),
        "removed_normals": (removed_count, 3),
    }
    for field, shape in expected_shapes.items():
        values = np.asarray(geometry[field])
        if values.shape != shape:
            raise ValueError(f"Expected {field} with shape {shape}, got {values.shape}")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{field} must be finite")
    removed_log_scales = np.asarray(geometry["removed_log_scales"])
    if removed_log_scales.ndim != 2 or removed_log_scales.shape[0] != removed_count or removed_log_scales.shape[1] not in (1, 3):
        raise ValueError(
            f"Expected removed_log_scales with shape [{removed_count}, 1] or [{removed_count}, 3], "
            f"got {removed_log_scales.shape}"
        )
    if np.any(~np.isfinite(removed_log_scales)):
        raise ValueError("removed_log_scales must be finite")
    scales = np.asarray(geometry["removed_scales"], dtype=np.float64)
    if np.any(scales <= 0) or not np.allclose(scales, np.exp(expand_log_scales(removed_log_scales)), rtol=1e-5, atol=1e-7):
        raise ValueError("removed_scales do not match removed_log_scales")
    rotations = np.asarray(geometry["removed_rotations"], dtype=np.float64)
    if not np.allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-5):
        raise ValueError("removed_rotations must be normalized quaternions")
    covariances = np.asarray(geometry["removed_covariances"], dtype=np.float64)
    if not np.allclose(covariances, np.transpose(covariances, (0, 2, 1)), atol=1e-5):
        raise ValueError("removed_covariances must be symmetric")
    if removed_count and np.any(np.linalg.eigvalsh(covariances) <= 0):
        raise ValueError("removed_covariances must be positive definite")
    _scalar_string(geometry["normal_source"], "normal_source")

    params_hash = _scalar_string(geometry["source_params_sha256"], "source_params_sha256")
    controls_hash = _scalar_string(geometry["selection_controls_sha256"], "selection_controls_sha256")
    if (
        len(params_hash) != 64
        or len(controls_hash) != 64
        or any(character not in "0123456789abcdef" for character in params_hash.lower())
        or any(character not in "0123456789abcdef" for character in controls_hash.lower())
    ):
        raise ValueError("source params and selection controls hashes must be SHA-256 hex digests")
    if source_params_sha256 is not None and params_hash != source_params_sha256:
        raise ValueError("source_params_sha256 does not match expected provenance")
    if selection_controls_sha256 is not None and controls_hash != selection_controls_sha256:
        raise ValueError("selection_controls_sha256 does not match expected provenance")

    if "replacement_selected_local_ids" in geometry:
        local_ids = np.asarray(geometry["replacement_selected_local_ids"])
        if local_ids.dtype.kind not in "iu" or local_ids.shape != (replacement.size,):
            raise ValueError("replacement_selected_local_ids must be an integer array matching replacement count")
        local_ids = local_ids.astype(np.int64, copy=False)
        if np.any(local_ids < 0) or np.any(local_ids >= selected.size) or np.unique(local_ids).size != local_ids.size:
            raise ValueError("replacement_selected_local_ids contain invalid or duplicate map-local IDs")
        if not np.array_equal(selected[local_ids], replacement):
            raise ValueError("replacement_selected_local_ids do not map to replacement_global_ids")

    if schema_version >= 2:
        reserve = _global_ids(geometry, "reserve_global_ids")
        if set(reserve.tolist()) & selected_set:
            raise ValueError("reserve_global_ids must be strictly outside selected_global_ids")
        reserve_count = reserve.size
        encoded_count = np.asarray(geometry["reserve_count"])
        if encoded_count.shape != () or encoded_count.dtype.kind not in "iu" or int(encoded_count) != reserve_count:
            raise ValueError("reserve_count must exactly match reserve_global_ids length")
        _scalar_string(geometry["reserve_policy"], "reserve_policy")
        reserve_shapes = {
            "reserve_means": (reserve_count, 3),
            "reserve_normals": (reserve_count, 3),
            "reserve_scores": (reserve_count,),
            "reserve_covariances": (reserve_count, 3, 3),
            "reserve_scales": (reserve_count, 3),
            "reserve_rotations": (reserve_count, 4),
        }
        for field, shape in reserve_shapes.items():
            values = np.asarray(geometry[field])
            if values.shape != shape:
                raise ValueError(f"Expected {field} with shape {shape}, got {values.shape}")
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{field} must be finite")
        reserve_scales = np.asarray(geometry["reserve_scales"], dtype=np.float64)
        if np.any(reserve_scales <= 0):
            raise ValueError("reserve_scales must be positive")
        reserve_rotations = np.asarray(geometry["reserve_rotations"], dtype=np.float64)
        if not np.allclose(np.linalg.norm(reserve_rotations, axis=1), 1.0, atol=1e-5):
            raise ValueError("reserve_rotations must be normalized quaternions")
        reserve_covariances = np.asarray(geometry["reserve_covariances"], dtype=np.float64)
        if not np.allclose(
            reserve_covariances, np.transpose(reserve_covariances, (0, 2, 1)), atol=1e-5
        ):
            raise ValueError("reserve_covariances must be symmetric")
        if reserve_count and np.any(np.linalg.eigvalsh(reserve_covariances) <= 0):
            raise ValueError("reserve_covariances must be positive definite")
        if "reserve_source_codes" in geometry:
            source_codes = np.asarray(geometry["reserve_source_codes"])
            if source_codes.dtype.kind not in "iu" or source_codes.shape != (reserve_count,):
                raise ValueError("reserve_source_codes must be an integer array matching reserve count")
        if "reserve_trace_scores" in geometry:
            trace_scores = np.asarray(geometry["reserve_trace_scores"])
            if trace_scores.shape != (reserve_count,) or np.any(~np.isfinite(trace_scores)):
                raise ValueError("reserve_trace_scores must be a finite array matching reserve count")
