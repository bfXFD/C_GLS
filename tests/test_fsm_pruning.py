import numpy as np
import pytest

from utils.fsm_pruning import (
    resolve_budget,
    score_opacity,
    score_fisher_hybrid,
    score_scale_large,
    score_scale_small,
    select_by_strategy,
    select_random,
    select_topk,
    select_uniform_stride,
)


def make_params(num_gaussians=5):
    return {
        "means3D": np.zeros((num_gaussians, 3), dtype=np.float32),
        "log_scales": np.log(np.array([[0.5], [0.1], [0.4], [0.2], [0.3]], dtype=np.float32)[:num_gaussians]),
        "logit_opacities": np.array([[-2.0], [3.0], [0.0], [1.0], [-1.0]], dtype=np.float32)[:num_gaussians],
    }


def test_resolve_budget_from_ratio_uses_ceil():
    assert resolve_budget(100, compression_ratio=0.05) == 5
    assert resolve_budget(101, compression_ratio=0.05) == 6


def test_resolve_budget_has_minimum_one_and_caps_to_num_items():
    assert resolve_budget(100, compression_ratio=0.0001) == 1
    assert resolve_budget(10, count=20) == 10


def test_resolve_budget_rejects_ambiguous_inputs():
    with pytest.raises(ValueError):
        resolve_budget(10, compression_ratio=0.5, count=5)


def test_select_topk_largest_stable_for_ties():
    scores = np.array([1.0, 3.0, 3.0, 2.0])
    indices = select_topk(scores, 3, largest=True)
    np.testing.assert_array_equal(indices, np.array([1, 2, 3]))


def test_select_topk_smallest():
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    indices = select_topk(scores, 2, largest=False)
    np.testing.assert_array_equal(indices, np.array([1, 3]))


def test_score_opacity_applies_sigmoid():
    params = make_params()
    scores = score_opacity(params)
    assert scores.shape == (5,)
    assert np.argmax(scores) == 1
    np.testing.assert_allclose(scores[2], 0.5)


def test_scale_small_and_large_have_opposite_order():
    params = make_params()
    small_scores = score_scale_small(params)
    large_scores = score_scale_large(params)
    assert np.argmax(small_scores) == 1
    assert np.argmax(large_scores) == 0


def test_select_random_is_deterministic_and_unique():
    first = select_random(20, 5, seed=7)
    second = select_random(20, 5, seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 5


def test_select_uniform_stride_includes_endpoints():
    indices = select_uniform_stride(10, 4)
    np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9]))


@pytest.mark.parametrize("strategy", ["full", "random", "opacity", "scale_small", "scale_large", "uniform_stride"])
def test_select_by_strategy_outputs_valid_indices_and_scores(strategy):
    params = make_params()
    indices, scores = select_by_strategy(params, strategy, compression_ratio=0.4, seed=0)
    expected_count = 5 if strategy == "full" else 2
    assert indices.shape == (expected_count,)
    assert scores.shape == (5,)
    assert np.unique(indices).size == indices.size
    assert np.all(indices >= 0)
    assert np.all(indices < 5)


def test_select_by_opacity_prefers_highest_opacity():
    params = make_params()
    indices, _ = select_by_strategy(params, "opacity", count=2)
    np.testing.assert_array_equal(indices, np.array([1, 3]))


def test_select_by_scale_small_prefers_small_scales():
    params = make_params()
    indices, _ = select_by_strategy(params, "scale_small", count=2)
    np.testing.assert_array_equal(indices, np.array([1, 3]))


def test_select_by_scale_large_prefers_large_scales():
    params = make_params()
    indices, _ = select_by_strategy(params, "scale_large", count=2)
    np.testing.assert_array_equal(indices, np.array([0, 2]))


def test_select_by_external_scores():
    params = make_params()
    external = np.array([0.0, 10.0, 2.0, 5.0, 1.0])
    indices, scores = select_by_strategy(params, "external_scores", count=3, scores=external)
    np.testing.assert_array_equal(indices, np.array([1, 3, 2]))
    np.testing.assert_allclose(scores, external)


def test_score_fisher_hybrid_combines_scale_and_opacity():
    params = make_params()
    fisher = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    scores = score_fisher_hybrid(params, fisher, use_scale=True, use_opacity=True)
    assert scores.shape == (5,)
    assert np.argmax(scores) == 2


def test_select_by_fisher_score_requires_scores_and_prefers_highest():
    params = make_params()
    fisher = np.array([0.0, 2.0, 5.0, 1.0, 4.0])
    indices, scores = select_by_strategy(params, "fisher_score", count=2, scores=fisher)
    np.testing.assert_array_equal(indices, np.array([2, 4]))
    assert scores.shape == (5,)


def test_select_by_fisher_x_scale_uses_hybrid_order():
    params = make_params()
    fisher = np.array([0.1, 1.0, 0.1, 0.1, 0.1])
    indices, _ = select_by_strategy(params, "fisher_x_scale", count=1, scores=fisher)
    np.testing.assert_array_equal(indices, np.array([1]))


def test_select_by_external_scores_requires_scores():
    params = make_params()
    with pytest.raises(ValueError):
        select_by_strategy(params, "external_scores", count=2)


def test_select_by_strategy_rejects_unknown_strategy():
    params = make_params()
    with pytest.raises(ValueError):
        select_by_strategy(params, "missing", count=2)


def test_select_topk_rejects_nonfinite_scores():
    with pytest.raises(ValueError):
        select_topk(np.array([1.0, np.nan]), 1)
