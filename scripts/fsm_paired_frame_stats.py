"""Frame-clustered paired statistics for paired-reserve GT results.

This module intentionally depends only on the Python standard library and NumPy.
Random-policy seeds are aggregated within each frame before frames receive equal
weight in point estimates and pair-cluster bootstrap inference.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
TIMESTAMP_ATOL = 5e-9
DEFAULT_COMPARATORS = ("random", "saliency", "runtime_weak")
DEFAULT_METRICS = ("translation", "rotation")
KEY_FIELDS = ("pair_id", "frame_id", "policy", "seed", "budget")
PAIRED_REQUIRED_FIELDS = {*KEY_FIELDS, "timestamp", "status"}
GT_REQUIRED_FIELDS = {
    *KEY_FIELDS, "timestamp", "candidate_valid",
    "translation_delta_m", "rotation_delta_deg",
}

SUMMARY_FIELDS = (
    "metric", "unit", "budget", "corrweak_policy", "comparator", "estimand",
    "n_pairs", "n_candidate_pairs", "n_dropped_pairs",
    "random_seed_count_min", "random_seed_count_median", "random_seed_count_max",
    "expected_random_seed_count", "complete_seed_panel_pair_count",
    "mean_effect", "mean_effect_ci_low", "mean_effect_ci_high",
    "median_effect", "median_effect_ci_low", "median_effect_ci_high",
    "std_effect", "paired_effect_dz", "paired_effect_dz_ci_low", "paired_effect_dz_ci_high",
    "win_rate", "win_rate_ci_low", "win_rate_ci_high", "tie_rate", "loss_rate",
    "probability_superiority", "rank_biserial", "rank_biserial_ci_low",
    "rank_biserial_ci_high", "frame_majority_win_rate", "frame_any_win_rate",
    "frame_all_win_rate", "seed_percentile_mean", "seed_percentile_median",
    "seed_percentile_p10", "seed_percentile_p25", "seed_percentile_p75",
    "seed_percentile_p90", "seed_percentile_ci_low", "seed_percentile_ci_high",
    "valid_bootstrap_replicates", "valid_dz_bootstrap_replicates",
)

FRAME_FIELDS = (
    "pair_id", "frame_id", "timestamp", "budget", "metric", "comparator",
    "corrweak_gain", "comparator_gain", "effect", "win", "tie", "loss",
    "random_seed_count", "random_effect_mean", "random_effect_median",
    "random_win_fraction", "random_tie_fraction", "random_loss_fraction",
    "seed_percentile", "complete_random_panel",
)

RANDOM_SEED_FIELDS = (
    "pair_id", "frame_id", "timestamp", "budget", "metric", "seed",
    "corrweak_gain", "random_gain", "effect", "win", "tie", "loss",
)

BUDGET_FIELDS = (
    "row_type", "metric", "comparator", "budget", "budget_low", "budget_high",
    "n_pairs", "mean_effect", "ci_low", "ci_high", "win_rate",
    "seed_percentile_mean", "valid_bootstrap_replicates",
)


@dataclass(frozen=True)
class PairedStatsConfig:
    corrweak_policy: str = "runtime_corr_weak"
    comparators: tuple[str, ...] = DEFAULT_COMPARATORS
    budgets: tuple[int, ...] | None = None
    metrics: tuple[str, ...] = DEFAULT_METRICS
    bootstrap_replicates: int = 10_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 0
    min_random_seeds: int = 1
    expected_random_seeds: tuple[int, ...] | None = None
    require_complete_random_panel: bool = False
    complete_across_budgets: bool = False
    translation_tolerance_m: float = 0.0
    rotation_tolerance_deg: float = 0.0
    strict: bool = True

    def __post_init__(self) -> None:
        requested_comparators = set(str(value) for value in self.comparators)
        requested_metrics = set(str(value) for value in self.metrics)
        comparators = tuple(value for value in DEFAULT_COMPARATORS if value in requested_comparators)
        metrics = tuple(value for value in DEFAULT_METRICS if value in requested_metrics)
        budgets = None if self.budgets is None else tuple(sorted(set(int(value) for value in self.budgets)))
        expected = (
            None if self.expected_random_seeds is None
            else tuple(sorted(set(int(value) for value in self.expected_random_seeds)))
        )
        if not comparators or requested_comparators != set(comparators):
            raise ValueError(f"comparators must be selected from {DEFAULT_COMPARATORS}")
        if not metrics or requested_metrics != set(metrics):
            raise ValueError(f"metrics must be selected from {DEFAULT_METRICS}")
        if budgets is not None and (not budgets or any(value <= 0 for value in budgets)):
            raise ValueError("budgets must contain positive integers")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.min_random_seeds < 1:
            raise ValueError("min_random_seeds must be positive")
        if expected is not None and (not expected or any(value < 0 for value in expected)):
            raise ValueError("expected_random_seeds must contain non-negative integers")
        if self.translation_tolerance_m < 0.0 or self.rotation_tolerance_deg < 0.0:
            raise ValueError("metric tolerances must be non-negative")
        object.__setattr__(self, "comparators", comparators)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "expected_random_seeds", expected)


@dataclass(frozen=True)
class PairedStatsResult:
    summary_rows: tuple[dict[str, Any], ...]
    frame_rows: tuple[dict[str, Any], ...]
    random_seed_rows: tuple[dict[str, Any], ...]
    budget_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


def _frame_value(row: Mapping[str, Any]) -> Any:
    has_id = "frame_id" in row and str(row.get("frame_id", "")).strip() != ""
    has_frame = "frame" in row and str(row.get("frame", "")).strip() != ""
    if has_id and has_frame and int(row["frame_id"]) != int(row["frame"]):
        raise ValueError("frame and frame_id disagree")
    if has_id:
        return row["frame_id"]
    if has_frame:
        return row["frame"]
    raise KeyError("missing required field frame_id (or frame)")


def _required(row: Mapping[str, Any], field: str) -> Any:
    if field not in row:
        raise KeyError(f"missing required field {field}")
    return row[field]


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} must be an integer")
    try:
        parsed_float = float(text)
        parsed = int(parsed_float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(parsed_float) or parsed_float != parsed:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise ValueError(f"{name} must be boolean")


def _normalize_seed(value: Any, policy: str) -> int | None:
    text = "" if value is None else str(value).strip()
    if policy == "random":
        if text == "":
            raise ValueError("random policy requires an integer seed")
        return _integer(text, "seed", minimum=0)
    if text != "":
        raise ValueError(f"deterministic policy {policy} must have an empty seed")
    return None


def _normalize_base(row: Mapping[str, Any], allowed_policies: set[str]) -> dict[str, Any]:
    pair_id = str(_required(row, "pair_id")).strip()
    if not pair_id:
        raise ValueError("pair_id must be non-empty")
    frame_id = _integer(_frame_value(row), "frame_id", minimum=0)
    timestamp = _finite_float(_required(row, "timestamp"), "timestamp")
    policy = str(_required(row, "policy")).strip()
    if policy not in allowed_policies:
        raise ValueError(f"unknown policy {policy!r}")
    seed = _normalize_seed(row.get("seed", ""), policy)
    budget = _integer(_required(row, "budget"), "budget", minimum=1)
    return {
        "pair_id": pair_id, "frame_id": frame_id, "timestamp": timestamp,
        "policy": policy, "seed": seed, "budget": budget,
    }


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def _key_text(key: tuple[Any, ...]) -> str:
    return "/".join("" if value is None else str(value) for value in key)


def _normalize_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    config: PairedStatsConfig,
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]], dict[str, Any]]:
    allowed = {config.corrweak_policy, *DEFAULT_COMPARATORS}
    paired: dict[tuple[Any, ...], dict[str, Any]] = {}
    gt: dict[tuple[Any, ...], dict[str, Any]] = {}
    pair_meta: dict[str, tuple[int, float]] = {}

    def check_meta(normalized: Mapping[str, Any]) -> None:
        value = (int(normalized["frame_id"]), float(normalized["timestamp"]))
        previous = pair_meta.get(str(normalized["pair_id"]))
        if previous is not None and (
            previous[0] != value[0] or not np.isclose(previous[1], value[1], rtol=0.0, atol=TIMESTAMP_ATOL)
        ):
            raise ValueError(f"pair_id {normalized['pair_id']} maps to inconsistent frame/timestamp")
        pair_meta[str(normalized["pair_id"])] = value

    for raw in paired_rows:
        normalized = _normalize_base(raw, allowed)
        normalized["status"] = str(_required(raw, "status")).strip()
        check_meta(normalized)
        key = _key(normalized)
        if key in paired:
            raise ValueError(f"duplicate paired join key: {_key_text(key)}")
        paired[key] = normalized

    metric_fields = {
        "translation": "translation_delta_m",
        "rotation": "rotation_delta_deg",
    }
    for raw in gt_rows:
        normalized = _normalize_base(raw, allowed)
        normalized["candidate_valid"] = _boolean(_required(raw, "candidate_valid"), "candidate_valid")
        normalized["gains"] = {}
        normalized["metric_errors"] = {}
        for metric in config.metrics:
            field = metric_fields[metric]
            try:
                normalized["gains"][metric] = _finite_float(_required(raw, field), field)
            except (KeyError, ValueError) as exc:
                normalized["gains"][metric] = None
                normalized["metric_errors"][metric] = str(exc)
        check_meta(normalized)
        key = _key(normalized)
        if key in gt:
            raise ValueError(f"duplicate GT join key: {_key_text(key)}")
        gt[key] = normalized

    paired_keys = set(paired)
    gt_keys = set(gt)
    unmatched_paired = sorted(paired_keys - gt_keys, key=_key_text)
    orphan_gt = sorted(gt_keys - paired_keys, key=_key_text)
    invalid_status = sorted((key for key, row in paired.items() if row["status"] != "ok"), key=_key_text)
    invalid_candidate = sorted((key for key, row in gt.items() if not row["candidate_valid"]), key=_key_text)
    invalid_metric = {
        metric: sorted(
            (key for key, row in gt.items() if row["gains"].get(metric) is None), key=_key_text
        ) for metric in config.metrics
    }
    if config.strict:
        problems = []
        if unmatched_paired:
            problems.append(f"{len(unmatched_paired)} paired rows lack GT matches")
        if orphan_gt:
            problems.append(f"{len(orphan_gt)} GT rows lack paired matches")
        if invalid_status:
            problems.append(f"{len(invalid_status)} paired rows have non-ok status")
        if invalid_candidate:
            problems.append(f"{len(invalid_candidate)} GT rows have invalid candidates")
        invalid_count = sum(len(value) for value in invalid_metric.values())
        if invalid_count:
            problems.append(f"{invalid_count} metric values are missing or non-finite")
        if problems:
            raise ValueError("strict validation failed: " + "; ".join(problems))

    diagnostics = {
        "input_paired_row_count": len(paired_rows),
        "input_gt_row_count": len(gt_rows),
        "joined_key_count": len(paired_keys & gt_keys),
        "unmatched_paired_count": len(unmatched_paired),
        "orphan_gt_count": len(orphan_gt),
        "invalid_status_count": len(invalid_status),
        "invalid_candidate_count": len(invalid_candidate),
        "invalid_metric_count": {metric: len(keys) for metric, keys in invalid_metric.items()},
        "examples": {
            "unmatched_paired": [_key_text(value) for value in unmatched_paired[:10]],
            "orphan_gt": [_key_text(value) for value in orphan_gt[:10]],
            "invalid_status": [_key_text(value) for value in invalid_status[:10]],
            "invalid_candidate": [_key_text(value) for value in invalid_candidate[:10]],
        },
    }
    return paired, gt, diagnostics


def _tolerance(config: PairedStatsConfig, metric: str) -> float:
    return config.translation_tolerance_m if metric == "translation" else config.rotation_tolerance_deg


def _classify(effect: np.ndarray | float, tolerance: float) -> tuple[Any, Any, Any]:
    values = np.asarray(effect, dtype=np.float64)
    return values > tolerance, np.abs(values) <= tolerance, values < -tolerance


def _safe_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _dz(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / std) if std > 0.0 and np.isfinite(std) else float("nan")


def _ci(values: np.ndarray, confidence: float, requested: int) -> tuple[float | None, float | None, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count < int(np.ceil(0.9 * requested)):
        return None, None, count
    alpha = (1.0 - confidence) * 0.5
    bounds = np.quantile(finite, [alpha, 1.0 - alpha], method="linear")
    return float(bounds[0]), float(bounds[1]), count


def _bootstrap_stat(
    values: np.ndarray,
    bootstrap_draws: np.ndarray,
    statistic: str,
) -> np.ndarray:
    result = np.full(bootstrap_draws.shape[0], np.nan, dtype=np.float64)
    if values.size == 0:
        return result
    # The same PCG64 draw matrix is reused for every cell. Each cell maps its
    # first n columns to its sorted complete-case panel and draws n clusters.
    indices = bootstrap_draws[:, : values.size] % np.uint64(values.size)
    for index, sampled in enumerate(indices):
        sampled_values = values[sampled]
        if statistic == "mean":
            result[index] = float(np.mean(sampled_values))
        elif statistic == "median":
            result[index] = float(np.median(sampled_values))
        elif statistic == "dz":
            result[index] = _dz(sampled_values)
        else:
            raise ValueError(f"unknown bootstrap statistic {statistic}")
    return result


def _summary(
    rows: list[dict[str, Any]],
    all_pair_ids: tuple[str, ...],
    bootstrap_draws: np.ndarray,
    config: PairedStatsConfig,
    metric: str,
    budget: int,
    comparator: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    n = len(rows)
    effects = np.asarray([row["effect"] for row in rows], dtype=np.float64)
    wins = np.asarray([row["win"] for row in rows], dtype=np.float64)
    ties = np.asarray([row["tie"] for row in rows], dtype=np.float64)
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    percentiles = np.asarray([row["seed_percentile"] for row in rows], dtype=np.float64) if comparator == "random" else None
    mean_boot = _bootstrap_stat(effects, bootstrap_draws, "mean")
    median_boot = _bootstrap_stat(effects, bootstrap_draws, "median")
    dz_boot = _bootstrap_stat(effects, bootstrap_draws, "dz")
    win_boot = _bootstrap_stat(wins, bootstrap_draws, "mean")
    rank_boot = _bootstrap_stat(wins - losses, bootstrap_draws, "mean")
    mean_ci = _ci(mean_boot, config.confidence_level, config.bootstrap_replicates)
    median_ci = _ci(median_boot, config.confidence_level, config.bootstrap_replicates)
    dz_ci = _ci(dz_boot, config.confidence_level, config.bootstrap_replicates)
    win_ci = _ci(win_boot, config.confidence_level, config.bootstrap_replicates)
    rank_ci = _ci(rank_boot, config.confidence_level, config.bootstrap_replicates)
    percentile_ci = (None, None, 0)
    if percentiles is not None:
        percentile_boot = _bootstrap_stat(percentiles, bootstrap_draws, "mean")
        percentile_ci = _ci(percentile_boot, config.confidence_level, config.bootstrap_replicates)

    seed_counts = np.asarray([row["random_seed_count"] for row in rows], dtype=np.int64) if comparator == "random" else np.empty(0)
    complete_count = int(sum(bool(row["complete_random_panel"]) for row in rows)) if comparator == "random" else None
    mean_win = float(np.mean(wins)) if n else float("nan")
    mean_tie = float(np.mean(ties)) if n else float("nan")
    mean_loss = float(np.mean(losses)) if n else float("nan")
    output = {
        "metric": metric,
        "unit": "m" if metric == "translation" else "deg",
        "budget": budget,
        "corrweak_policy": config.corrweak_policy,
        "comparator": comparator,
        "estimand": "corrweak_improvement_minus_comparator_improvement",
        "n_pairs": n,
        "n_candidate_pairs": len(all_pair_ids),
        "n_dropped_pairs": len(all_pair_ids) - n,
        "random_seed_count_min": int(np.min(seed_counts)) if seed_counts.size else None,
        "random_seed_count_median": _safe_float(np.median(seed_counts)) if seed_counts.size else None,
        "random_seed_count_max": int(np.max(seed_counts)) if seed_counts.size else None,
        "expected_random_seed_count": len(expected_seeds) if comparator == "random" else None,
        "complete_seed_panel_pair_count": complete_count,
        "mean_effect": _safe_float(np.mean(effects)) if n else None,
        "mean_effect_ci_low": mean_ci[0], "mean_effect_ci_high": mean_ci[1],
        "median_effect": _safe_float(np.median(effects)) if n else None,
        "median_effect_ci_low": median_ci[0], "median_effect_ci_high": median_ci[1],
        "std_effect": _safe_float(np.std(effects, ddof=1)) if n >= 2 else None,
        "paired_effect_dz": _safe_float(_dz(effects)) if n else None,
        "paired_effect_dz_ci_low": dz_ci[0], "paired_effect_dz_ci_high": dz_ci[1],
        "win_rate": _safe_float(mean_win) if n else None,
        "win_rate_ci_low": win_ci[0], "win_rate_ci_high": win_ci[1],
        "tie_rate": _safe_float(mean_tie) if n else None,
        "loss_rate": _safe_float(mean_loss) if n else None,
        "probability_superiority": _safe_float(mean_win + 0.5 * mean_tie) if n else None,
        "rank_biserial": _safe_float(mean_win - mean_loss) if n else None,
        "rank_biserial_ci_low": rank_ci[0], "rank_biserial_ci_high": rank_ci[1],
        "frame_majority_win_rate": _safe_float(np.mean(wins > 0.5)) if n else None,
        "frame_any_win_rate": _safe_float(np.mean(wins > 0.0)) if n else None,
        "frame_all_win_rate": _safe_float(np.mean(wins == 1.0)) if n else None,
        "seed_percentile_mean": _safe_float(np.mean(percentiles)) if percentiles is not None and n else None,
        "seed_percentile_median": _safe_float(np.median(percentiles)) if percentiles is not None and n else None,
        "seed_percentile_p10": _safe_float(np.quantile(percentiles, 0.10, method="linear")) if percentiles is not None and n else None,
        "seed_percentile_p25": _safe_float(np.quantile(percentiles, 0.25, method="linear")) if percentiles is not None and n else None,
        "seed_percentile_p75": _safe_float(np.quantile(percentiles, 0.75, method="linear")) if percentiles is not None and n else None,
        "seed_percentile_p90": _safe_float(np.quantile(percentiles, 0.90, method="linear")) if percentiles is not None and n else None,
        "seed_percentile_ci_low": percentile_ci[0], "seed_percentile_ci_high": percentile_ci[1],
        "valid_bootstrap_replicates": mean_ci[2],
        "valid_dz_bootstrap_replicates": dz_ci[2],
    }
    return output


def analyze_paired_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    config: PairedStatsConfig = PairedStatsConfig(),
) -> PairedStatsResult:
    if bool(paired_rows) != bool(gt_rows):
        raise ValueError("paired rows and GT rows must either both be empty or both be non-empty")
    if not paired_rows:
        if config.budgets is None:
            raise ValueError("header-only inputs require explicitly requested budgets")
        diagnostics = {
            "status": "no_trigger_pairs", "strict": config.strict,
            "input_paired_row_count": 0, "input_gt_row_count": 0,
            "joined_key_count": 0, "unmatched_paired_count": 0, "orphan_gt_count": 0,
            "invalid_status_count": 0, "invalid_candidate_count": 0,
            "invalid_metric_count": {metric: 0 for metric in config.metrics},
            "examples": {"unmatched_paired": [], "orphan_gt": [], "invalid_status": [], "invalid_candidate": []},
            "selected_budgets": list(config.budgets), "missing_requested_budgets": [],
            "pair_count": 0, "drop_reasons": {}, "summary_row_count": 0,
            "frame_row_count": 0, "random_seed_row_count": 0,
            "expected_random_seeds_by_budget": {
                str(budget): list(config.expected_random_seeds or ()) for budget in config.budgets
            },
            "paired_no_trigger": True,
        }
        provenance = _provenance(config)
        return PairedStatsResult((), (), (), (), diagnostics, provenance)
    paired, gt, diagnostics = _normalize_rows(paired_rows, gt_rows, config)
    joined = set(paired) & set(gt)
    valid: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key in joined:
        if paired[key]["status"] == "ok" and gt[key]["candidate_valid"]:
            valid[key] = gt[key]

    discovered_budgets = sorted({int(key[4]) for key in joined})
    budgets = config.budgets if config.budgets is not None else tuple(discovered_budgets)
    if not budgets:
        raise ValueError("no budgets were discovered")
    missing_budgets = sorted(set(budgets) - set(discovered_budgets))
    if missing_budgets and config.strict:
        raise ValueError(f"requested budgets not present in inputs: {missing_budgets}")

    all_pair_ids = tuple(sorted({str(key[0]) for key in joined if int(key[4]) in budgets}))
    if not all_pair_ids:
        raise ValueError("no joined pairs are available for selected budgets")
    bit_generator = np.random.PCG64(config.bootstrap_seed)
    bootstrap_draws = bit_generator.random_raw((config.bootstrap_replicates, len(all_pair_ids)))

    random_seed_sets: dict[int, set[int]] = defaultdict(set)
    for key, row in valid.items():
        if key[2] == "random" and row["candidate_valid"] and int(key[4]) in budgets:
            random_seed_sets[int(key[4])].add(int(key[3]))
    expected_by_budget = {
        budget: (
            config.expected_random_seeds
            if config.expected_random_seeds is not None
            else tuple(sorted(random_seed_sets.get(budget, set())))
        ) for budget in budgets
    }

    frame_rows: list[dict[str, Any]] = []
    random_rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    drop_reasons: Counter[str] = Counter()
    rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    def gain(pair_id: str, budget: int, policy: str, seed: int | None, metric: str) -> float | None:
        row = valid.get((pair_id, next_frame[pair_id], policy, seed, budget))
        if row is None:
            return None
        value = row["gains"].get(metric)
        return None if value is None else float(value)

    next_frame = {pair_id: int(next(key[1] for key in joined if key[0] == pair_id)) for pair_id in all_pair_ids}
    pair_timestamp = {pair_id: float(next(row["timestamp"] for key, row in gt.items() if key[0] == pair_id)) for pair_id in all_pair_ids}

    candidate_cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for pair_id in all_pair_ids:
        for budget in budgets:
            for metric in config.metrics:
                corr = gain(pair_id, budget, config.corrweak_policy, None, metric)
                for comparator in config.comparators:
                    cell_key = (pair_id, budget, metric, comparator)
                    if corr is None:
                        drop_reasons[f"{metric}/{budget}/{comparator}:missing_corrweak"] += 1
                        continue
                    tolerance = _tolerance(config, metric)
                    if comparator != "random":
                        other = gain(pair_id, budget, comparator, None, metric)
                        if other is None:
                            drop_reasons[f"{metric}/{budget}/{comparator}:missing_comparator"] += 1
                            continue
                        effect = corr - other
                        win, tie, loss = _classify(effect, tolerance)
                        candidate_cells[cell_key] = {
                            "pair_id": pair_id, "frame_id": next_frame[pair_id],
                            "timestamp": pair_timestamp[pair_id], "budget": budget, "metric": metric,
                            "comparator": comparator, "corrweak_gain": corr, "comparator_gain": other,
                            "effect": effect, "win": float(win), "tie": float(tie), "loss": float(loss),
                            "random_seed_count": None, "random_effect_mean": None,
                            "random_effect_median": None, "random_win_fraction": None,
                            "random_tie_fraction": None, "random_loss_fraction": None,
                            "seed_percentile": None, "complete_random_panel": None,
                        }
                        continue
                    seeds = expected_by_budget[budget]
                    available = []
                    actual_seed_keys = sorted(
                        int(key[3]) for key in valid
                        if key[0] == pair_id and key[2] == "random" and int(key[4]) == budget
                        and valid[key]["gains"].get(metric) is not None
                    )
                    for seed in actual_seed_keys:
                        random_gain = gain(pair_id, budget, "random", seed, metric)
                        if random_gain is not None:
                            available.append((seed, random_gain))
                    complete = bool(seeds) and set(seed for seed, _ in available) == set(seeds)
                    if len(available) < config.min_random_seeds:
                        drop_reasons[f"{metric}/{budget}/random:insufficient_seeds"] += 1
                        continue
                    if config.require_complete_random_panel and not complete:
                        drop_reasons[f"{metric}/{budget}/random:incomplete_seed_panel"] += 1
                        continue
                    effects = np.asarray([corr - value for _, value in available], dtype=np.float64)
                    wins, ties, losses = _classify(effects, tolerance)
                    for (seed, random_gain), effect, win, tie, loss in zip(available, effects, wins, ties, losses):
                        random_rows_by_cell[(pair_id, budget, metric)].append({
                            "pair_id": pair_id, "frame_id": next_frame[pair_id],
                            "timestamp": pair_timestamp[pair_id], "budget": budget, "metric": metric,
                            "seed": seed, "corrweak_gain": corr, "random_gain": random_gain,
                            "effect": float(effect), "win": bool(win), "tie": bool(tie), "loss": bool(loss),
                        })
                    mean_effect = float(np.mean(effects))
                    candidate_cells[cell_key] = {
                        "pair_id": pair_id, "frame_id": next_frame[pair_id],
                        "timestamp": pair_timestamp[pair_id], "budget": budget, "metric": metric,
                        "comparator": comparator, "corrweak_gain": corr,
                        "comparator_gain": float(np.mean([value for _, value in available])),
                        "effect": mean_effect, "win": float(np.mean(wins)),
                        "tie": float(np.mean(ties)), "loss": float(np.mean(losses)),
                        "random_seed_count": len(available), "random_effect_mean": mean_effect,
                        "random_effect_median": float(np.median(effects)),
                        "random_win_fraction": float(np.mean(wins)),
                        "random_tie_fraction": float(np.mean(ties)),
                        "random_loss_fraction": float(np.mean(losses)),
                        "seed_percentile": float(100.0 * (np.sum(wins) + 0.5 * np.sum(ties)) / len(effects)),
                        "complete_random_panel": complete,
                    }

    if config.complete_across_budgets:
        for metric in config.metrics:
            for comparator in config.comparators:
                complete_pairs = {
                    pair_id for pair_id in all_pair_ids
                    if all((pair_id, budget, metric, comparator) in candidate_cells for budget in budgets)
                }
                for pair_id in all_pair_ids:
                    if pair_id not in complete_pairs:
                        for budget in budgets:
                            candidate_cells.pop((pair_id, budget, metric, comparator), None)

    for key in sorted(candidate_cells, key=lambda value: (value[0], value[1], value[2], value[3])):
        row = candidate_cells[key]
        frame_rows.append(row)
        rows_by_cell[(row["metric"], row["budget"], row["comparator"])].append(row)
    random_rows = [
        seed_row
        for key in sorted(random_rows_by_cell, key=lambda value: (value[0], value[1], value[2]))
        if (key[0], key[1], key[2], "random") in candidate_cells
        for seed_row in random_rows_by_cell[key]
    ]

    summary_rows = []
    for metric in config.metrics:
        for budget in budgets:
            for comparator in config.comparators:
                rows = rows_by_cell[(metric, budget, comparator)]
                summary_rows.append(_summary(
                    rows, all_pair_ids, bootstrap_draws, config, metric, budget,
                    comparator, expected_by_budget[budget],
                ))

    budget_rows: list[dict[str, Any]] = []
    summary_lookup = {(row["metric"], row["comparator"], row["budget"]): row for row in summary_rows}
    for metric in config.metrics:
        for comparator in config.comparators:
            for budget in budgets:
                row = summary_lookup[(metric, comparator, budget)]
                budget_rows.append({
                    "row_type": "point", "metric": metric, "comparator": comparator,
                    "budget": budget, "budget_low": None, "budget_high": None,
                    "n_pairs": row["n_pairs"], "mean_effect": row["mean_effect"],
                    "ci_low": row["mean_effect_ci_low"], "ci_high": row["mean_effect_ci_high"],
                    "win_rate": row["win_rate"], "seed_percentile_mean": row["seed_percentile_mean"],
                    "valid_bootstrap_replicates": row["valid_bootstrap_replicates"],
                })
            for low, high in zip(budgets, budgets[1:]):
                low_rows = {row["pair_id"]: row for row in rows_by_cell[(metric, low, comparator)]}
                high_rows = {row["pair_id"]: row for row in rows_by_cell[(metric, high, comparator)]}
                common = sorted(set(low_rows) & set(high_rows))
                values = np.asarray([high_rows[pair]["effect"] - low_rows[pair]["effect"] for pair in common], dtype=np.float64)
                boot = _bootstrap_stat(values, bootstrap_draws, "mean")
                ci = _ci(boot, config.confidence_level, config.bootstrap_replicates)
                budget_rows.append({
                    "row_type": "adjacent_change", "metric": metric, "comparator": comparator,
                    "budget": None, "budget_low": low, "budget_high": high,
                    "n_pairs": len(common), "mean_effect": _safe_float(np.mean(values)) if values.size else None,
                    "ci_low": ci[0], "ci_high": ci[1], "win_rate": None,
                    "seed_percentile_mean": None, "valid_bootstrap_replicates": ci[2],
                })

    diagnostics.update({
        "status": "ok",
        "strict": config.strict,
        "selected_budgets": list(budgets),
        "missing_requested_budgets": missing_budgets,
        "pair_count": len(all_pair_ids),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "summary_row_count": len(summary_rows),
        "frame_row_count": len(frame_rows),
        "random_seed_row_count": len(random_rows),
        "expected_random_seeds_by_budget": {str(key): list(value) for key, value in expected_by_budget.items()},
        "paired_no_trigger": False,
    })
    provenance = _provenance(config)
    return PairedStatsResult(
        tuple(summary_rows), tuple(frame_rows), tuple(random_rows), tuple(budget_rows),
        diagnostics, provenance,
    )


def _provenance(config: PairedStatsConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": dataclasses.asdict(config),
        "bootstrap": {
            "cluster": "pair_id", "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed, "generator": "numpy.PCG64",
            "ci_method": "percentile", "quantile_method": "linear",
            "confidence_level": config.confidence_level,
            "index_matrix_reused_across_all_budgets_metrics_contrasts": True,
        },
        "effect_direction": "positive_means_corrweak_improvement_exceeds_comparator_improvement",
        "random_aggregation": "within_pair_seed_distribution_then_equal_pair_weight",
        "timestamp_absolute_tolerance": TIMESTAMP_ATOL,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], set[str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return list(reader), set(reader.fieldnames)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_paired_files(
    paired_rows_csv: str | Path,
    gt_rows_csv: str | Path,
    config: PairedStatsConfig = PairedStatsConfig(),
) -> PairedStatsResult:
    paired_path = Path(paired_rows_csv).resolve()
    gt_path = Path(gt_rows_csv).resolve()
    if not paired_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError("paired rows CSV and GT rows CSV are required")
    paired_rows, paired_fields = _read_csv(paired_path)
    gt_rows, gt_fields = _read_csv(gt_path)
    missing_paired = sorted(PAIRED_REQUIRED_FIELDS - paired_fields)
    missing_gt = sorted(GT_REQUIRED_FIELDS - gt_fields)
    if missing_paired or missing_gt:
        raise ValueError(
            f"invalid CSV schema: paired missing {missing_paired}; GT missing {missing_gt}"
        )
    result = analyze_paired_rows(paired_rows, gt_rows, config)
    provenance = dict(result.provenance)
    provenance.update({
        "paired_rows_path": str(paired_path), "paired_rows_sha256": _sha256(paired_path),
        "gt_rows_path": str(gt_path), "gt_rows_sha256": _sha256(gt_path),
    })
    return dataclasses.replace(result, provenance=provenance)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def write_paired_stats(result: PairedStatsResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output / "summary.csv",
        "summary_json": output / "summary.json",
        "frames_csv": output / "frames.csv",
        "random_seeds_csv": output / "random_seeds.csv",
        "budget_curve_csv": output / "budget_curve.csv",
        "diagnostics_json": output / "diagnostics.json",
        "provenance_json": output / "provenance.json",
    }
    _write_csv(paths["summary_csv"], result.summary_rows, SUMMARY_FIELDS)
    _write_json(paths["summary_json"], result.summary_rows)
    _write_csv(paths["frames_csv"], result.frame_rows, FRAME_FIELDS)
    _write_csv(paths["random_seeds_csv"], result.random_seed_rows, RANDOM_SEED_FIELDS)
    _write_csv(paths["budget_curve_csv"], result.budget_rows, BUDGET_FIELDS)
    _write_json(paths["diagnostics_json"], result.diagnostics)
    _write_json(paths["provenance_json"], result.provenance)
    return paths


def run(
    paired_rows_csv: str | Path,
    gt_rows_csv: str | Path,
    output_dir: str | Path,
    config: PairedStatsConfig = PairedStatsConfig(),
) -> dict[str, Any]:
    result = analyze_paired_files(paired_rows_csv, gt_rows_csv, config)
    paths = write_paired_stats(result, output_dir)
    return {
        "status": result.diagnostics.get("status", "ok"),
        "paired_no_trigger": bool(result.diagnostics.get("paired_no_trigger", False)),
        "output_dir": str(Path(output_dir).resolve()),
        "budgets": result.diagnostics["selected_budgets"],
        "pair_count": result.diagnostics["pair_count"],
        "comparison_count": len(result.summary_rows),
        "outputs": {key: str(value) for key, value in sorted(paths.items())},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frame-clustered paired statistics for paired-reserve GT rows.")
    parser.add_argument("--paired-rows", required=True)
    parser.add_argument("--gt-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrweak-policy", default="runtime_corr_weak")
    parser.add_argument("--comparators", nargs="+", choices=DEFAULT_COMPARATORS, default=list(DEFAULT_COMPARATORS))
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=list(DEFAULT_METRICS))
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--min-random-seeds", type=int, default=1)
    parser.add_argument("--expected-random-seeds", nargs="+", type=int)
    parser.add_argument("--require-complete-random-panel", action="store_true")
    parser.add_argument("--complete-across-budgets", action="store_true")
    parser.add_argument("--translation-tolerance-m", type=float, default=0.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.0)
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict", dest="strict", action="store_true", default=True)
    strict.add_argument("--non-strict", dest="strict", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PairedStatsConfig(
        corrweak_policy=args.corrweak_policy, comparators=tuple(args.comparators),
        budgets=None if args.budgets is None else tuple(args.budgets), metrics=tuple(args.metrics),
        bootstrap_replicates=args.bootstrap_replicates, confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed, min_random_seeds=args.min_random_seeds,
        expected_random_seeds=(
            None if args.expected_random_seeds is None else tuple(args.expected_random_seeds)
        ),
        require_complete_random_panel=args.require_complete_random_panel,
        complete_across_budgets=args.complete_across_budgets,
        translation_tolerance_m=args.translation_tolerance_m,
        rotation_tolerance_deg=args.rotation_tolerance_deg, strict=args.strict,
    )
    print(json.dumps(run(args.paired_rows, args.gt_rows, args.output_dir, config), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
