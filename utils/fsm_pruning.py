from typing import Any

import numpy as np

from utils.fsm_map import expand_log_scales


SUPPORTED_STRATEGIES = (
    "full",
    "random",
    "opacity",
    "scale_small",
    "scale_large",
    "uniform_stride",
    "external_scores",
    "fisher_score",
    "fisher_x_scale",
    "fisher_x_opacity",
    "fisher_x_scale_opacity",
)


def resolve_budget(num_items: int, compression_ratio: float | None = None, count: int | None = None) -> int:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if compression_ratio is not None and count is not None:
        raise ValueError("Specify either compression_ratio or count, not both")

    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        return min(int(count), int(num_items))

    if compression_ratio is None:
        return int(num_items)
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")

    budget = int(np.ceil(float(compression_ratio) * int(num_items)))
    return min(max(budget, 1), int(num_items))


def select_topk(scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"Expected scores with shape [N], got {scores.shape}")
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), scores.size)
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores must be finite")

    order = np.argsort(-scores if largest else scores, kind="mergesort")
    return order[:k].astype(np.int64)


def score_opacity(params: dict[str, Any]) -> np.ndarray:
    if "logit_opacities" not in params:
        raise KeyError("Missing required field 'logit_opacities' for opacity scoring")
    logits = np.asarray(params["logit_opacities"], dtype=np.float64)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(f"Expected logit_opacities with shape [N] or [N, 1], got {logits.shape}")
    return 1.0 / (1.0 + np.exp(-logits))


def _scale_magnitude(params: dict[str, Any]) -> np.ndarray:
    if "log_scales" not in params:
        raise KeyError("Missing required field 'log_scales' for scale scoring")
    scales = np.exp(expand_log_scales(np.asarray(params["log_scales"], dtype=np.float64)))
    return np.linalg.norm(scales, axis=1)


def score_scale_small(params: dict[str, Any]) -> np.ndarray:
    return -_scale_magnitude(params)


def score_scale_large(params: dict[str, Any]) -> np.ndarray:
    return _scale_magnitude(params)


def _normalize_score(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"Expected scores with shape [N], got {scores.shape}")
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores must be finite")
    min_value = float(np.min(scores))
    max_value = float(np.max(scores))
    if max_value - min_value <= eps:
        return np.ones_like(scores, dtype=np.float64)
    return (scores - min_value) / (max_value - min_value + eps)


def score_fisher_hybrid(
    params: dict[str, Any],
    fisher_scores: np.ndarray,
    use_scale: bool = False,
    use_opacity: bool = False,
) -> np.ndarray:
    num_items = _num_gaussians_from_params(params)
    fisher_scores = np.asarray(fisher_scores, dtype=np.float64)
    if fisher_scores.ndim != 1 or fisher_scores.shape[0] != num_items:
        raise ValueError(f"Expected Fisher scores with shape [{num_items}], got {fisher_scores.shape}")

    score = _normalize_score(fisher_scores) + 1e-12
    if use_scale:
        score *= 1.0 + _normalize_score(score_scale_large(params))
    if use_opacity:
        score *= 1.0 + _normalize_score(score_opacity(params))
    return score.astype(np.float64)


def select_random(num_items: int, k: int, seed: int = 0) -> np.ndarray:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), int(num_items))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=k, replace=False)).astype(np.int64)


def select_uniform_stride(num_items: int, k: int) -> np.ndarray:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), int(num_items))
    if k == num_items:
        return np.arange(num_items, dtype=np.int64)
    indices = np.linspace(0, num_items - 1, num=k, dtype=np.int64)
    return np.unique(indices).astype(np.int64)


def _num_gaussians_from_params(params: dict[str, Any]) -> int:
    if "means3D" in params:
        means = np.asarray(params["means3D"])
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"Expected means3D with shape [N, 3], got {means.shape}")
        return int(means.shape[0])
    if "log_scales" in params:
        return int(np.asarray(params["log_scales"]).shape[0])
    if "logit_opacities" in params:
        return int(np.asarray(params["logit_opacities"]).shape[0])
    raise KeyError("Unable to infer Gaussian count from params")


def _validate_indices(indices: np.ndarray, num_items: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"Expected indices with shape [N*], got {indices.shape}")
    if indices.size == 0:
        raise ValueError("selected indices must not be empty")
    if np.any(indices < 0) or np.any(indices >= num_items):
        raise IndexError("selected indices contain values outside the Gaussian range")
    if np.unique(indices).size != indices.size:
        raise ValueError("selected indices must be unique")
    return indices


def select_by_strategy(
    params: dict[str, Any],
    strategy: str,
    compression_ratio: float | None = None,
    count: int | None = None,
    scores: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown pruning strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}")

    num_items = _num_gaussians_from_params(params)
    k = resolve_budget(num_items, compression_ratio=compression_ratio, count=count)

    if strategy == "full":
        indices = np.arange(num_items, dtype=np.int64)
        score_array = np.ones(num_items, dtype=np.float64)
    elif strategy == "random":
        indices = select_random(num_items, k, seed=seed)
        score_array = np.zeros(num_items, dtype=np.float64)
        score_array[indices] = 1.0
    elif strategy == "uniform_stride":
        indices = select_uniform_stride(num_items, k)
        score_array = np.zeros(num_items, dtype=np.float64)
        score_array[indices] = 1.0
    elif strategy == "opacity":
        score_array = score_opacity(params)
        indices = select_topk(score_array, k, largest=True)
    elif strategy == "scale_small":
        score_array = score_scale_small(params)
        indices = select_topk(score_array, k, largest=True)
    elif strategy == "scale_large":
        score_array = score_scale_large(params)
        indices = select_topk(score_array, k, largest=True)
    elif strategy == "external_scores":
        if scores is None:
            raise ValueError("scores must be provided when strategy='external_scores'")
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.ndim != 1 or score_array.shape[0] != num_items:
            raise ValueError(f"Expected external scores with shape [{num_items}], got {score_array.shape}")
        indices = select_topk(score_array, k, largest=True)
    else:
        if scores is None:
            raise ValueError(f"scores must be provided when strategy='{strategy}'")
        score_array = score_fisher_hybrid(
            params,
            fisher_scores=scores,
            use_scale=strategy in ("fisher_x_scale", "fisher_x_scale_opacity"),
            use_opacity=strategy in ("fisher_x_opacity", "fisher_x_scale_opacity"),
        )
        indices = select_topk(score_array, k, largest=True)

    return _validate_indices(indices, num_items), np.asarray(score_array, dtype=np.float64)
