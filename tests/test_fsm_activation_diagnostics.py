import numpy as np
import pytest

from utils.fsm_activation_diagnostics import RuntimeActivationDiagnostics, _STAGES


def make_runtime(*, include_stage_ids=False, save_correspondences=False, max_detail_frames=64):
    runtime = RuntimeActivationDiagnostics.__new__(RuntimeActivationDiagnostics)
    runtime.enabled = True
    runtime.records = []
    runtime.include_stage_ids = include_stage_ids
    runtime.save_correspondences = save_correspondences
    runtime.max_detail_frames = max_detail_frames
    runtime.actual_global_ids = np.arange(100_000, dtype=np.int64)
    runtime.counter_global_ids = None
    runtime.actual_scores = None
    runtime.counter_scores = None
    runtime.actual_index = object()
    runtime.counter_index = None
    runtime.actual_points = np.zeros((100_000, 3), dtype=np.float64)
    runtime.actual_normals = None
    runtime.replacement_mask = np.zeros(100_000, dtype=bool)
    runtime.reserve_budgets = np.asarray([1, 2, 4], dtype=np.int64)
    runtime.role_available = False
    runtime.weak_subspace_dim = 1
    runtime.trigger = {}
    runtime.local_config = type("Local", (), {"frustum_culling": "radius", "max_points": None})()
    return runtime


def huge_query():
    indices = np.arange(100_000, dtype=np.int64)
    return {
        "indices": indices,
        **{f"{stage}_indices": indices for stage in _STAGES},
        **{f"{stage}_count": indices.size for stage in _STAGES},
        "precap_count": indices.size,
        "culling_mode": "radius",
        "status": "ok",
    }


def huge_fixed():
    count = 100_000
    return {
        "fitness": 0.5,
        "num_correspondences": count,
        "hessian": np.eye(6),
        "condition_number": 2.0,
        "source_ids": np.arange(count, dtype=np.int64),
        "target_ids": np.arange(count, dtype=np.int64),
        "jacobians": np.ones((count, 6), dtype=np.float64),
        "weights": np.ones(count, dtype=np.float64),
    }


def aggregate_role():
    return {
        "total": 0, **{stage: 0 for stage in _STAGES}, "correspondence": 0, "unique_target": 0,
        "hessian": np.zeros((6, 6)), "corr_role_mask": np.zeros(100_000, dtype=bool),
    }


def aggregate_hessian():
    return {
        "trace": 6.0, "lambda_min": 1.0, "normalized_lambda_min": 1.0 / 6.0,
        "role_trace": 0.0, "role_lambda_min": 0.0, "trace_fraction": 0.0, "weak_fraction": 0.0,
        "reserve_gain": np.zeros(3), "reserve_gain_fraction": np.zeros(3),
    }


def configure_fake_evaluation(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "_query", lambda *args, **kwargs: huge_query())
    monkeypatch.setattr(runtime, "_fixed", lambda *args, **kwargs: (huge_fixed(), 0.0, 0.0))
    monkeypatch.setattr(runtime, "_role_metrics", lambda *args, **kwargs: aggregate_role())
    monkeypatch.setattr(runtime, "_hessian_metrics", lambda *args, **kwargs: aggregate_hessian())


def evaluate(runtime, frame_id):
    runtime.evaluate_frame(
        frame_id, float(frame_id), np.zeros((1, 3)), np.eye(4), np.eye(3), 1, 1,
        {"iterations": 1}, None,
    )


def test_compact_records_bound_retained_arrays_for_huge_frames(monkeypatch):
    runtime = make_runtime()
    configure_fake_evaluation(runtime, monkeypatch)
    for frame_id in range(200):
        before = runtime.retained_ndarray_bytes
        evaluate(runtime, frame_id)
        assert runtime.retained_ndarray_bytes - before < 10 * 1024
    assert runtime.retained_ndarray_bytes < 2 * 1024 * 1024
    record = runtime.records[0]
    assert "indices" not in record["actual_query"]
    assert "stage_global_ids" not in record["actual_query"]
    assert not {"source_ids", "target_ids", "jacobians", "weights"} & record["actual_fixed"].keys()
    arrays = runtime._arrays()
    assert arrays["actual_total_hessian"].shape == (200, 6, 6)
    assert arrays["actual_correspondence_count"][0] == 100_000
    assert arrays["actual_fitness"][0] == pytest.approx(0.5)


@pytest.mark.parametrize("option", ["include_stage_ids", "save_correspondences"])
def test_optional_detail_capture_fails_before_exceeding_limit(monkeypatch, option):
    runtime = make_runtime(**{option: True}, max_detail_frames=1)
    configure_fake_evaluation(runtime, monkeypatch)
    evaluate(runtime, 0)
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        evaluate(runtime, 1)
    assert len(runtime.records) == 1
