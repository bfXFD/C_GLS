import csv
import json

import numpy as np
import pytest

from scripts.fsm_critical_adoption import parse_args
from utils.fsm_critical_adoption import (
    CriticalAdoptionConfig, analyze_directories, load_run, run_analysis, write_outputs,
)


FRAME_FIELDS = [
    "frame_id", "timestamp", "tracking_valid", "reserve_attempted", "reserve_used",
    "reserve_information_v3_branch", "reserve_refinement_translation_m", "reserve_corr_selected_p_mean",
]
ERROR_FIELDS = [
    "frame_id", "timestamp", "tracking_valid", "raw_translation_error_m", "raw_rotation_error_deg",
    "aligned_translation_error_m", "aligned_rotation_error_deg",
]


def _tum(timestamp, x):
    return f"{timestamp:.9f} {x:.9f} 0 0 0 0 0 1\n"


def _make_run(root, name, errors, attempts=(), adoptions=(), valid=None, est=None, gt=None):
    path = root / name
    path.mkdir()
    n = len(errors)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    est = np.asarray(errors if est is None else est, dtype=float)
    gt = np.zeros(n) if gt is None else np.asarray(gt, dtype=float)
    frame_rows, error_rows = [], []
    for index in range(n):
        frame_rows.append({
            "frame_id": index, "timestamp": f"{index:.9f}", "tracking_valid": bool(valid[index]),
            "reserve_attempted": index in attempts, "reserve_used": index in adoptions,
            "reserve_information_v3_branch": "both" if index in attempts else "",
            "reserve_refinement_translation_m": 0.01 * index if index in attempts else "",
            "reserve_corr_selected_p_mean": 0.8 if index in attempts else "",
        })
        error_rows.append({
            "frame_id": index, "timestamp": f"{index:.9f}", "tracking_valid": bool(valid[index]),
            "raw_translation_error_m": errors[index], "raw_rotation_error_deg": 0,
            "aligned_translation_error_m": errors[index], "aligned_rotation_error_deg": 0,
        })
    for filename, fields, rows in (
        ("frame_metrics.csv", FRAME_FIELDS, frame_rows), ("per_frame_errors.csv", ERROR_FIELDS, error_rows),
    ):
        with open(path / filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (path / "trajectory_est.txt").write_text("".join(_tum(i, est[i]) for i in range(n)))
    (path / "trajectory_gt.txt").write_text("".join(_tum(i, gt[i]) for i in range(n)))
    return path


def _result(tmp_path, n=8):
    baseline = _make_run(tmp_path, "base", np.arange(n, dtype=float))
    run = _make_run(tmp_path, "corr", np.arange(n, dtype=float) - 1, attempts=(1, 3, 4), adoptions=(1, 4))
    random = _make_run(tmp_path, "random_seed2", np.arange(n, dtype=float) - 0.5,
                       attempts=(2, 4), adoptions=(2, 4))
    config = CriticalAdoptionConfig(horizons=(2, 5), rpe_deltas=(1, 2), burst_gaps=(1, 3), temporal_windows=(1, 2))
    return analyze_directories("base", baseline, [("corr", run), ("random_seed2", random)], config, ("random_seed2",))


def test_strict_schema_join_adoption_consistency_and_gt(tmp_path):
    baseline = _make_run(tmp_path, "base", [0, 1, 2])
    bad = _make_run(tmp_path, "bad", [0, 1, 2], adoptions=(1,))
    with pytest.raises(ValueError, match="requires attempt"):
        load_run("bad", bad, CriticalAdoptionConfig())
    other = _make_run(tmp_path, "other", [0, 1, 2], gt=[0, 0, 1])
    with pytest.raises(ValueError, match="GT trajectory differs"):
        analyze_directories("base", baseline, [("other", other)])
    rows = list(csv.DictReader(open(baseline / "per_frame_errors.csv")))
    rows[1]["frame_id"] = 9
    with open(baseline / "per_frame_errors.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ERROR_FIELDS); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="frame IDs differ|strictly increasing"):
        load_run("base", baseline, CriticalAdoptionConfig())


def test_future_right_censor_auc_counts_and_passthrough(tmp_path):
    result = _result(tmp_path)
    first = next(row for row in result.adoption_rows if row["run"] == "corr" and row["frame_id"] == 1)
    assert first["current_point_error_delta_m"] == -1
    assert first["future_h2_mean_delta_m"] == -1
    assert first["future_h2_auc_delta_m_frames"] == -2
    assert first["future_h2_other_attempt_count"] == 1
    assert first["next_adoption_gap_frames"] == 3
    assert first["reserve_information_v3_branch"] == "both"
    last = next(row for row in result.adoption_rows if row["run"] == "corr" and row["frame_id"] == 4)
    assert last["future_h5_right_censored"] is True
    assert last["future_h5_observed_frames"] == 4


def test_rpe_excludes_pairs_crossing_invalid_gap(tmp_path):
    base = _make_run(tmp_path, "base", [0] * 6, valid=[1, 1, 0, 1, 1, 1], est=[0, 1, 2, 3, 4, 5])
    run = _make_run(tmp_path, "run", [0] * 6, attempts=(0,), adoptions=(0,),
                    valid=[1, 1, 0, 1, 1, 1], est=[0, 2, 4, 6, 8, 10])
    config = CriticalAdoptionConfig(horizons=(5,), rpe_deltas=(1, 2), burst_gaps=(1,), temporal_windows=(1,))
    result = analyze_directories("base", base, [("run", run)], config)
    row = result.adoption_rows[0]
    assert row["future_h5_rpe_d1_pair_count"] == 3
    assert row["future_h5_rpe_d1_coverage"] == pytest.approx(3 / 5)
    assert row["future_h5_rpe_d1_delta_m"] == pytest.approx(1.0)
    assert row["future_h5_rpe_d2_pair_count"] == 1


def test_lost_frames_allow_blank_errors_and_use_common_valid_pairs(tmp_path):
    base = _make_run(tmp_path, "base", [0, 1, "", 3, 4], valid=[1, 1, 0, 1, 1], est=[0, 1, 2, 3, 4])
    run = _make_run(tmp_path, "run", [0, 0.5, 1, "", 2], attempts=(1,), adoptions=(1,),
                    valid=[1, 1, 1, 0, 1], est=[0, 1.5, 3, 4, 5])
    result = analyze_directories(
        "base", base, [("run", run)],
        CriticalAdoptionConfig(horizons=(4,), rpe_deltas=(1,), burst_gaps=(1,), temporal_windows=(1,)),
    )
    row = result.adoption_rows[0]
    assert row["future_h4_observed_frames"] == 2
    assert row["future_h4_point_error_delta_m"] is not None
    assert row["future_h4_rpe_d1_pair_count"] == 0


def test_blank_error_on_valid_frame_is_rejected(tmp_path):
    run = _make_run(tmp_path, "run", [0, "", 2], valid=[1, 1, 1], est=[0, 1, 2])
    with pytest.raises(ValueError, match="missing raw error on valid frames"):
        load_run("run", run, CriticalAdoptionConfig())


def test_bursts_and_temporal_lag_are_frame_overlap_not_point_overlap(tmp_path):
    result = _result(tmp_path)
    gap1 = [row for row in result.burst_rows if row["run"] == "corr" and row["gap_threshold"] == 1]
    gap3 = [row for row in result.burst_rows if row["run"] == "corr" and row["gap_threshold"] == 3]
    assert [row["size"] for row in gap1] == [1, 1]
    assert gap3[0]["size"] == 2 and gap3[0]["duration_frames"] == 4
    temporal = next(row for row in result.temporal_rows if row["run"] == "corr" and row["event_type"] == "adoption")
    assert temporal["frame_intersection_count"] == 1
    assert temporal["frame_jaccard"] == pytest.approx(1 / 3)
    assert temporal["nearest_signed_lag_mean"] == pytest.approx(0.5)
    assert "not_selected_point_set_overlap" in temporal["overlap_definition"]


def test_unavailable_and_no_lost_are_not_estimable(tmp_path):
    result = _result(tmp_path)
    assert result.availability["selected_ids_available"] is False
    assert result.availability["point_p_corr_available"] is False
    lost = result.summary["runs"]["corr"]["lost"]
    assert lost["status"] == "not_estimable"
    assert lost["avoidance_rate"] is None and lost["avoided_count"] is None


def test_provenance_outputs_standard_json_atomic_npz_and_cli(tmp_path):
    result = _result(tmp_path)
    paths = write_outputs(result, tmp_path / "out")
    assert {path.name for path in paths.values()} == {
        "run_inventory.csv", "attempt_events.csv", "adoption_events.csv", "bursts.csv",
        "temporal_comparison.csv", "summary.json", "arrays.npz", "availability.json", "provenance.json",
    }
    for name in ("summary", "availability", "provenance"):
        json.loads(paths[name].read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    provenance = json.loads(paths["provenance"].read_text())
    assert len(provenance["inputs"]["base"]["sha256"]["frame_metrics.csv"]) == 64
    assert provenance["allow_pickle"] is False
    with np.load(paths["arrays"], allow_pickle=False) as archive:
        assert archive["corr__adopted"].dtype == np.bool_
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    assert not list((tmp_path / "out").glob("*.tmp"))

    args = parse_args([
        "--baseline", "base=/x", "--run", "corr=/y", "--run", "random_seed2=/z",
        "--output-dir", "/o", "--compare-run", "random_seed2",
    ])
    assert args.baseline == ("base", "/x") and len(args.run) == 2


def test_run_analysis_writes_all_outputs(tmp_path):
    base = _make_run(tmp_path, "base", [0, 1, 2])
    run = _make_run(tmp_path, "run", [0, 0, 1], attempts=(1,), adoptions=(1,))
    payload = run_analysis("base", base, [("run", run)], tmp_path / "analysis",
                           CriticalAdoptionConfig(horizons=(1,), rpe_deltas=(1,), burst_gaps=(1,), temporal_windows=(1,)))
    assert payload["status"] == "ok" and payload["adoption_event_count"] == 1
    assert set(path.name for path in (tmp_path / "analysis").iterdir()) == {
        "run_inventory.csv", "attempt_events.csv", "adoption_events.csv", "bursts.csv",
        "temporal_comparison.csv", "summary.json", "arrays.npz", "availability.json", "provenance.json",
    }
