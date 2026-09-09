import csv
import json
import sys

import numpy as np
import pytest

from scripts.fsm_paired_frame_stats import (
    BUDGET_FIELDS,
    FRAME_FIELDS,
    RANDOM_SEED_FIELDS,
    SUMMARY_FIELDS,
    PairedStatsConfig,
    analyze_paired_rows,
    run,
    write_paired_stats,
)
from scripts.fsm_score_paired_reserve_gt import _GT_CSV_FIELDS
from utils.fsm_paired_reserve import _CSV_BASE_FIELDS, _CSV_V2_FIELDS


POLICIES = ("runtime_corr_weak", "random", "saliency", "runtime_weak")


def _arm(pair, frame, policy, budget, gain_t, gain_r=None, seed=None, **overrides):
    paired = {
        "pair_id": pair, "frame_id": frame, "timestamp": f"{frame + 0.25:.9f}",
        "policy": policy, "seed": "" if seed is None else seed, "budget": budget,
        "status": "ok",
    }
    gt = {
        "pair_id": pair, "frame_id": frame, "timestamp": f"{frame + 0.25:.9f}",
        "policy": policy, "seed": "" if seed is None else seed, "budget": budget,
        "candidate_valid": True, "translation_delta_m": gain_t,
        "rotation_delta_deg": gain_t if gain_r is None else gain_r,
    }
    for key, value in overrides.items():
        target, field = key.split("__", 1)
        (paired if target == "paired" else gt)[field] = value
    return paired, gt


def _dataset(spec, budgets=(64,)):
    paired, gt = [], []
    for frame, values in enumerate(spec):
        pair = f"pair-{frame}"
        for budget in budgets:
            for policy in ("runtime_corr_weak", "saliency", "runtime_weak"):
                arm = _arm(pair, frame, policy, budget, values.get(policy, 0.0))
                paired.append(arm[0]); gt.append(arm[1])
            for seed, gain in enumerate(values.get("random", [])):
                arm = _arm(pair, frame, "random", budget, gain, seed=seed)
                paired.append(arm[0]); gt.append(arm[1])
    return paired, gt


def _summary(result, metric="translation", budget=64, comparator="saliency"):
    return next(
        row for row in result.summary_rows
        if row["metric"] == metric and row["budget"] == budget and row["comparator"] == comparator
    )


def test_deterministic_math_direction_effect_sizes_and_ties():
    paired, gt = _dataset([
        {"runtime_corr_weak": 2, "saliency": 1, "runtime_weak": 2, "random": [2]},
        {"runtime_corr_weak": 4, "saliency": 2, "runtime_weak": 5, "random": [4]},
        {"runtime_corr_weak": 6, "saliency": 3, "runtime_weak": 4, "random": [6]},
    ])
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=200))
    row = _summary(result)
    assert row["mean_effect"] == pytest.approx(2.0)
    assert row["median_effect"] == pytest.approx(2.0)
    assert row["std_effect"] == pytest.approx(1.0)
    assert row["paired_effect_dz"] == pytest.approx(2.0)
    assert row["win_rate"] == 1.0
    assert row["probability_superiority"] == 1.0
    assert row["rank_biserial"] == 1.0
    weak = _summary(result, comparator="runtime_weak")
    assert (weak["win_rate"], weak["tie_rate"], weak["loss_rate"]) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_random_uses_frame_equal_weight_not_row_weight_and_midrank_percentile():
    paired, gt = _dataset([
        {"runtime_corr_weak": 10, "random": [0, 10]},
        {"runtime_corr_weak": 0, "random": [1]},
    ])
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=100))
    row = _summary(result, comparator="random")
    assert row["mean_effect"] == pytest.approx(2.0)
    assert row["win_rate"] == pytest.approx(0.25)
    assert row["tie_rate"] == pytest.approx(0.25)
    assert row["loss_rate"] == pytest.approx(0.5)
    assert row["seed_percentile_mean"] == pytest.approx(37.5)

    paired, gt = _dataset([{"runtime_corr_weak": 2, "random": [1, 2, 2, 4]}])
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=50))
    assert _summary(result, comparator="random")["seed_percentile_mean"] == 50.0


def test_bootstrap_is_clustered_deterministic_order_invariant_and_budget_aware():
    paired, gt = _dataset([
        {"runtime_corr_weak": 4, "saliency": 1, "random": [0, 2]},
        {"runtime_corr_weak": 2, "saliency": 3, "random": [1]},
        {"runtime_corr_weak": 8, "saliency": 4, "random": [3, 6, 9]},
    ], budgets=(64, 128))
    # Give budget 128 a known additional corrweak effect.
    for row in gt:
        if row["policy"] == "runtime_corr_weak" and row["budget"] == 128:
            row["translation_delta_m"] += 2
            row["rotation_delta_deg"] += 2
    config = PairedStatsConfig(bootstrap_replicates=400, bootstrap_seed=17)
    first = analyze_paired_rows(paired, gt, config)
    second = analyze_paired_rows(list(reversed(paired)), list(reversed(gt)), config)
    assert first.summary_rows == second.summary_rows
    assert first.budget_rows == second.budget_rows
    change = next(
        row for row in first.budget_rows
        if row["row_type"] == "adjacent_change" and row["metric"] == "translation"
        and row["comparator"] == "saliency"
    )
    assert change["mean_effect"] == pytest.approx(2.0)
    assert change["ci_low"] == pytest.approx(2.0)
    assert change["ci_high"] == pytest.approx(2.0)
    changed_seed = analyze_paired_rows(paired, gt, dataclasses_replace(config, bootstrap_seed=18))
    assert _summary(first)["mean_effect"] == _summary(changed_seed)["mean_effect"]
    reordered = analyze_paired_rows(paired, gt, dataclasses_replace(
        config, comparators=("runtime_weak", "random", "saliency"),
        metrics=("rotation", "translation"),
    ))
    assert first.summary_rows == reordered.summary_rows
    assert first.budget_rows == reordered.budget_rows


def dataclasses_replace(config, **values):
    data = config.__dict__.copy()
    data.update(values)
    return PairedStatsConfig(**data)


def test_missing_contrast_and_complete_random_panels_are_cell_specific():
    paired, gt = _dataset([
        {"runtime_corr_weak": 3, "saliency": 1, "runtime_weak": 2, "random": [0, 1, 2]},
        {"runtime_corr_weak": 4, "saliency": 2, "runtime_weak": 3, "random": [0, 1]},
    ])
    # Remove pair-1 saliency only.
    paired = [row for row in paired if not (row["pair_id"] == "pair-1" and row["policy"] == "saliency")]
    gt = [row for row in gt if not (row["pair_id"] == "pair-1" and row["policy"] == "saliency")]
    partial = analyze_paired_rows(paired, gt, PairedStatsConfig(
        bootstrap_replicates=100, expected_random_seeds=(0, 1, 2),
    ))
    assert _summary(partial, comparator="saliency")["n_pairs"] == 1
    assert _summary(partial, comparator="runtime_weak")["n_pairs"] == 2
    assert _summary(partial, comparator="random")["n_pairs"] == 2
    complete = analyze_paired_rows(paired, gt, PairedStatsConfig(
        bootstrap_replicates=100, expected_random_seeds=(0, 1, 2),
        require_complete_random_panel=True,
    ))
    assert _summary(complete, comparator="random")["n_pairs"] == 1


def test_complete_across_budgets_and_requested_budget_handling():
    paired, gt = _dataset([
        {"runtime_corr_weak": 3, "saliency": 1, "random": [0]},
        {"runtime_corr_weak": 4, "saliency": 2, "random": [1]},
    ], budgets=(64, 128))
    paired = [row for row in paired if not (row["pair_id"] == "pair-1" and row["budget"] == 128 and row["policy"] == "saliency")]
    gt = [row for row in gt if not (row["pair_id"] == "pair-1" and row["budget"] == 128 and row["policy"] == "saliency")]
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(
        bootstrap_replicates=50, complete_across_budgets=True,
    ))
    assert _summary(result, budget=64, comparator="saliency")["n_pairs"] == 1
    assert _summary(result, budget=128, comparator="saliency")["n_pairs"] == 1
    assert all(row["pair_id"] == "pair-0" for row in result.frame_rows if row["comparator"] == "saliency")
    selected = analyze_paired_rows(paired, gt, PairedStatsConfig(
        budgets=(128,), bootstrap_replicates=50,
    ))
    assert {row["budget"] for row in selected.summary_rows} == {128}
    with pytest.raises(ValueError, match="requested budgets"):
        analyze_paired_rows(paired, gt, PairedStatsConfig(budgets=(256,), bootstrap_replicates=50))


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate paired join key"),
    ("timestamp", "inconsistent frame/timestamp"),
    ("deterministic_seed", "must have an empty seed"),
    ("random_seed", "requires an integer seed"),
    ("budget", "at least 1"),
    ("unknown_policy", "unknown policy"),
])
def test_structural_join_validation(mutation, match):
    paired, gt = _dataset([{"runtime_corr_weak": 2, "random": [1]}])
    if mutation == "duplicate":
        paired.append(dict(paired[0]))
    elif mutation == "timestamp":
        gt[0]["timestamp"] = "99"
    elif mutation == "deterministic_seed":
        paired[0]["seed"] = 3
    elif mutation == "random_seed":
        next(row for row in paired if row["policy"] == "random")["seed"] = ""
    elif mutation == "budget":
        paired[0]["budget"] = 0
    elif mutation == "unknown_policy":
        paired[0]["policy"] = "mystery"
    with pytest.raises((ValueError, KeyError), match=match):
        analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20, strict=False))


def test_strict_and_non_strict_invalid_rows_and_small_samples():
    paired, gt = _dataset([{"runtime_corr_weak": 2, "saliency": 1, "random": [0]}])
    bad = next(row for row in gt if row["policy"] == "runtime_weak")
    bad["candidate_valid"] = False
    with pytest.raises(ValueError, match="strict validation failed"):
        analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20))
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20, strict=False))
    assert _summary(result, comparator="runtime_weak")["n_pairs"] == 0
    saliency = _summary(result)
    assert saliency["n_pairs"] == 1
    assert saliency["paired_effect_dz"] is None
    assert saliency["valid_dz_bootstrap_replicates"] == 0
    assert result.diagnostics["invalid_candidate_count"] == 1


def test_timestamp_tolerance_and_unmatched_join_behavior():
    paired, gt = _dataset([{"runtime_corr_weak": 2, "random": [1]}])
    gt[0]["timestamp"] = f"{float(gt[0]['timestamp']) + 4e-9:.9f}"
    analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20))
    orphan = dict(gt[0]); orphan["pair_id"] = "orphan"
    orphan["frame_id"] = 9; orphan["timestamp"] = "9.25"
    gt.append(orphan)
    with pytest.raises(ValueError, match="GT rows lack paired matches"):
        analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20))
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20, strict=False))
    assert result.diagnostics["orphan_gt_count"] == 1


def test_outputs_cli_contract_provenance_sha_and_standard_json(tmp_path):
    paired, gt = _dataset([{"runtime_corr_weak": 2, "saliency": 1, "random": [0, 1]}])
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(bootstrap_replicates=20))
    paths = write_paired_stats(result, tmp_path / "memory")
    assert {path.name for path in paths.values()} == {
        "summary.csv", "summary.json", "frames.csv", "random_seeds.csv",
        "budget_curve.csv", "diagnostics.json", "provenance.json",
    }
    json.loads(paths["summary_json"].read_text())

    paired_path = tmp_path / "paired.csv"; gt_path = tmp_path / "gt.csv"
    for path, rows in ((paired_path, paired), (gt_path, gt)):
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    payload = run(paired_path, gt_path, tmp_path / "files", PairedStatsConfig(bootstrap_replicates=20))
    provenance = json.loads((tmp_path / "files/provenance.json").read_text())
    assert payload["status"] == "ok"
    assert len(provenance["paired_rows_sha256"]) == 64
    assert provenance["bootstrap"]["generator"] == "numpy.PCG64"
    assert provenance["numpy_version"] == np.__version__
    assert provenance["python_version"]


def test_header_only_office0_outputs_no_trigger_contract(tmp_path):
    paired_path = tmp_path / "paired_reserve_rows.csv"
    gt_path = tmp_path / "paired_reserve_gt_rows.csv"
    for path, fields in (
        (paired_path, _CSV_BASE_FIELDS + _CSV_V2_FIELDS),
        (gt_path, _GT_CSV_FIELDS),
    ):
        with open(path, "w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    payload = run(
        paired_path,
        gt_path,
        tmp_path / "stats",
        PairedStatsConfig(budgets=(64, 128, 256), bootstrap_replicates=20),
    )
    assert payload["status"] == "no_trigger_pairs"
    assert payload["paired_no_trigger"] is True
    assert payload["budgets"] == [64, 128, 256]
    assert payload["pair_count"] == 0
    assert payload["comparison_count"] == 0
    diagnostics = json.loads((tmp_path / "stats/diagnostics.json").read_text())
    assert diagnostics["status"] == "no_trigger_pairs"
    assert diagnostics["selected_budgets"] == [64, 128, 256]
    assert json.loads((tmp_path / "stats/summary.json").read_text()) == []
    for name, fields in (
        ("summary.csv", SUMMARY_FIELDS),
        ("frames.csv", FRAME_FIELDS),
        ("random_seeds.csv", RANDOM_SEED_FIELDS),
        ("budget_curve.csv", BUDGET_FIELDS),
    ):
        with open(tmp_path / "stats" / name, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == list(fields)
            assert list(reader) == []


def test_header_only_requires_both_empty_valid_schemas_and_requested_budgets(tmp_path):
    paired_path = tmp_path / "paired.csv"
    gt_path = tmp_path / "gt.csv"
    with open(paired_path, "w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=_CSV_BASE_FIELDS + _CSV_V2_FIELDS).writeheader()
    with open(gt_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_GT_CSV_FIELDS)
        writer.writeheader()
        writer.writerow({
            "row_index": 0, "frame_id": 0, "timestamp": "0.000000000",
            "pair_id": "office0-pair", "candidate_valid": True,
            "policy": "runtime_corr_weak", "seed": "", "budget": 64,
            "translation_delta_m": 0.0, "rotation_delta_deg": 0.0,
        })
    with pytest.raises(ValueError, match="both be empty or both be non-empty"):
        run(paired_path, gt_path, tmp_path / "one-sided", PairedStatsConfig(budgets=(64,), bootstrap_replicates=20))

    gt_path.write_text(",".join(_GT_CSV_FIELDS) + "\n")
    with pytest.raises(ValueError, match="explicitly requested budgets"):
        run(paired_path, gt_path, tmp_path / "no-budgets", PairedStatsConfig(bootstrap_replicates=20))

    bad_gt = tmp_path / "bad_gt.csv"
    bad_gt.write_text("frame_id,timestamp,pair_id\n")
    with pytest.raises(ValueError, match="invalid CSV schema"):
        run(paired_path, bad_gt, tmp_path / "bad-schema", PairedStatsConfig(budgets=(64,), bootstrap_replicates=20))


def test_module_has_no_scipy_dependency_and_config_rejects_bad_values():
    module = sys.modules["scripts.fsm_paired_frame_stats"]
    source = open(module.__file__, encoding="utf-8").read()
    assert "import scipy" not in source
    with pytest.raises(ValueError):
        PairedStatsConfig(bootstrap_replicates=0)
    with pytest.raises(ValueError):
        PairedStatsConfig(confidence_level=1.0)
    with pytest.raises(ValueError):
        PairedStatsConfig(min_random_seeds=0)


def test_selected_comparator_accepts_full_input_policy_schema():
    paired, gt = _dataset([{"runtime_corr_weak": 2, "saliency": 1, "random": [0]}])
    result = analyze_paired_rows(paired, gt, PairedStatsConfig(
        comparators=("saliency",), metrics=("translation",), bootstrap_replicates=20,
    ))
    assert len(result.summary_rows) == 1
    assert result.summary_rows[0]["comparator"] == "saliency"
