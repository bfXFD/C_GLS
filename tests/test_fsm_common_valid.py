import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from utils.fsm_common_valid import (
    CommonValidConfig,
    analyze_common_valid,
    load_run,
    parse_method_specs,
    run_common_valid,
)
from utils.fsm_metrics import save_tum_trajectory


def pose(x=0.0, y=0.0, yaw=0.0):
    angle = np.deg2rad(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    result[:3, 3] = [x, y, 0.0]
    return result


def make_run(tmp_path, name, frame_ids, est, gt=None, valid=None, statuses=None, timestamps=None):
    root = tmp_path / name
    root.mkdir()
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    timestamps = frame_ids.astype(np.float64) if timestamps is None else np.asarray(timestamps, dtype=np.float64)
    gt = np.asarray(est if gt is None else gt, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    valid = [True] * len(frame_ids) if valid is None else valid
    statuses = ["tracked"] * len(frame_ids) if statuses is None else statuses
    save_tum_trajectory(str(root / "trajectory_est.txt"), timestamps, est)
    save_tum_trajectory(str(root / "trajectory_gt.txt"), timestamps, gt)
    with open(root / "frame_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_id", "timestamp", "tracking_valid", "tracking_status"])
        writer.writeheader()
        for frame_id, timestamp, is_valid, status in zip(frame_ids, timestamps, valid, statuses):
            writer.writerow({
                "frame_id": frame_id, "timestamp": f"{timestamp:.9f}",
                "tracking_valid": is_valid, "tracking_status": status,
            })
    (root / "summary.json").write_text(json.dumps({"num_valid_frames": sum(bool(x) for x in valid)}))
    return root


def line_trajectory(count=6):
    return np.stack([pose(x=float(index), yaw=2.0 * index) for index in range(count)])


def summary(result, scope, method, group, delta=None):
    return next(
        row for row in result.summary_rows
        if row["scope"] == scope and row["method"] == method
        and row["metric_group"] == group and row["delta_frames"] == delta
    )


def comparison(result, scope, group, metric="translation_rmse", delta=None):
    return next(
        row for row in result.comparison_rows
        if row["scope"] == scope and row["metric_group"] == group
        and row["metric"] == metric and row["delta_frames"] == delta
    )


def test_masks_union_intersection_pairwise_and_all(tmp_path):
    gt = line_trajectory(4)
    a = make_run(tmp_path, "a", [0, 1, 2, 3], gt, gt, [1, 1, 0, 1])
    b = make_run(tmp_path, "b", [1, 2, 3], gt[1:], gt[1:], [1, 1, 1])
    c = make_run(tmp_path, "c", [0, 1, 2, 3], gt, gt, [1, 0, 1, 1])
    result = analyze_common_valid({"a": a, "b": b, "c": c}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))
    assert result.frame_ids.tolist() == [0, 1, 2, 3]
    assert result.individual_masks.tolist() == [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]
    assert result.pair_masks.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]]
    assert result.all_mask.tolist() == [0, 0, 0, 1]
    assert summary(result, "pair:b|c", "b", "raw_absolute")["sample_count"] == 2
    assert summary(result, "all", "a", "raw_absolute")["sample_count"] == 1


def test_tracking_valid_fail_closed_and_held_pose_invalid(tmp_path):
    trajectory = line_trajectory(3)
    run = make_run(tmp_path, "held", range(3), trajectory, trajectory, [True, "maybe", True], ["tracked", "tracked", "lost_hold_last_valid"])
    loaded = load_run("held", run)
    assert loaded.valid_mask.tolist() == [True, False, False]
    assert loaded.diagnostics["held_pose_count"] == 1
    assert loaded.diagnostics["tracking_valid_parse_failure_count"] == 1


def test_alignment_fits_common_mask_only_and_needs_three_frames(tmp_path):
    gt = np.stack([pose(0, 0), pose(1, 0), pose(0, 1), pose(1, 1)])
    transform = pose(5, -3, 90)
    est_a = np.stack([transform @ item for item in gt])
    est_a[3, :3, 3] = [100, 100, 0]
    a = make_run(tmp_path, "a", range(4), est_a, gt, [1, 1, 1, 1])
    b = make_run(tmp_path, "b", range(4), gt, gt, [1, 1, 1, 0])
    result = analyze_common_valid({"a": a, "b": b}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))
    aligned = summary(result, "all", "a", "aligned_absolute")
    assert aligned["sample_count"] == 3
    assert aligned["translation_rmse_m"] == pytest.approx(0.0, abs=1e-8)
    b2 = make_run(tmp_path, "b2", range(4), gt, gt, [1, 1, 0, 0])
    short = analyze_common_valid({"a": a, "b": b2}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))
    assert summary(short, "all", "a", "aligned_absolute")["alignment_status"] == "insufficient_frames"
    assert summary(short, "all", "a", "aligned_absolute")["translation_rmse_m"] is None


def test_absolute_rotation_and_comparison_direction(tmp_path):
    gt = line_trajectory(4)
    a_est = gt.copy()
    b_est = np.stack([pose(x=float(index), yaw=10 + 2 * index) for index in range(4)])
    a = make_run(tmp_path, "a", range(4), a_est, gt)
    b = make_run(tmp_path, "b", range(4), b_est, gt)
    result = analyze_common_valid({"a": a, "b": b}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=50))
    assert summary(result, "all", "b", "raw_absolute")["rotation_rmse_deg"] == pytest.approx(10.0)
    row = comparison(result, "all", "raw_absolute", "rotation_rmse")
    assert row["delta_b_minus_a"] == pytest.approx(10.0)
    assert result.provenance["comparison_direction"].endswith("method B is lower")


def test_gap_rpe_requires_complete_integer_common_interval(tmp_path):
    gt = line_trajectory(6)
    est = gt.copy(); est[:, 0, 3] += np.arange(6) * 0.1
    a = make_run(tmp_path, "a", range(6), est, gt, [1, 1, 0, 1, 1, 1])
    b = make_run(tmp_path, "b", range(6), gt, gt, [1] * 6)
    result = analyze_common_valid({"a": a, "b": b}, config=CommonValidConfig(frame_deltas=(1, 2, 3), bootstrap_replicates=0))
    assert summary(result, "all", "a", "frame_rpe", 1)["sample_count"] == 3
    assert summary(result, "all", "a", "frame_rpe", 2)["sample_count"] == 1
    assert summary(result, "all", "a", "frame_rpe", 3)["sample_count"] == 0
    assert all(not (row["start_frame_id"] < 2 < row["end_frame_id"]) for row in result.rpe_pair_rows)


def test_nonconsecutive_frame_ids_do_not_form_rpe_pair(tmp_path):
    gt = np.stack([pose(0), pose(1), pose(3)])
    a = make_run(tmp_path, "a", [0, 1, 3], gt, gt)
    b = make_run(tmp_path, "b", [0, 1, 3], gt, gt)
    result = analyze_common_valid({"a": a, "b": b}, config=CommonValidConfig(frame_deltas=(2,), bootstrap_replicates=0))
    assert summary(result, "all", "a", "frame_rpe", 2)["sample_count"] == 0


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_frame", "duplicate frame_id"),
    ("duplicate_timestamp", "duplicate timestamp"),
    ("bad_schema", "invalid frame_metrics.csv schema"),
    ("join", "matched 0 rows"),
])
def test_frame_join_duplicate_timestamp_and_schema_validation(tmp_path, mutation, match):
    trajectory = line_trajectory(3)
    run = make_run(tmp_path, "run", range(3), trajectory, trajectory)
    path = run / "frame_metrics.csv"
    rows = list(csv.DictReader(path.open()))
    fields = list(rows[0])
    if mutation == "duplicate_frame":
        rows[1]["frame_id"] = rows[0]["frame_id"]
    elif mutation == "duplicate_timestamp":
        rows[1]["timestamp"] = rows[0]["timestamp"]
    elif mutation == "bad_schema":
        fields.remove("tracking_valid")
    else:
        rows[1]["timestamp"] = "9.0"
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match=match):
        load_run("run", run)


def test_gt_consistency_timestamp_tolerance_and_external_gt(tmp_path):
    gt = line_trajectory(3)
    a = make_run(tmp_path, "a", range(3), gt, gt, timestamps=[0, 1, 2])
    b = make_run(tmp_path, "b", range(3), gt, gt, timestamps=[4e-9, 1 + 4e-9, 2 + 4e-9])
    external = tmp_path / "external.txt"
    save_tum_trajectory(str(external), np.arange(3, dtype=float), gt)
    analyze_common_valid({"a": a, "b": b}, gt=external, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))
    bad_gt = gt.copy(); bad_gt[1, 0, 3] += 0.1
    c = make_run(tmp_path, "c", range(3), gt, bad_gt)
    with pytest.raises(ValueError, match="inconsistent GT poses"):
        analyze_common_valid({"a": a, "c": c}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))


def test_empty_common_and_pairwise_can_differ_from_all(tmp_path):
    gt = line_trajectory(3)
    a = make_run(tmp_path, "a", range(3), gt, gt, [1, 1, 0])
    b = make_run(tmp_path, "b", range(3), gt, gt, [1, 1, 0])
    c = make_run(tmp_path, "c", range(3), gt, gt, [0, 0, 1])
    result = analyze_common_valid({"a": a, "b": b, "c": c}, config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=0))
    assert summary(result, "pair:a|b", "a", "raw_absolute")["sample_count"] == 2
    assert summary(result, "all", "a", "raw_absolute")["sample_count"] == 0
    assert summary(result, "all", "a", "aligned_absolute")["alignment_status"] == "insufficient_frames"


def test_bootstrap_deterministic_shared_and_paired_units(tmp_path):
    gt = line_trajectory(8)
    a_est = gt.copy(); b_est = gt.copy()
    a_est[:, 0, 3] += np.linspace(0, 0.4, 8)
    b_est[:, 0, 3] += np.linspace(0, 0.8, 8)
    a = make_run(tmp_path, "a", range(8), a_est, gt)
    b = make_run(tmp_path, "b", range(8), b_est, gt)
    config = CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=200, bootstrap_seed=17)
    first = analyze_common_valid({"a": a, "b": b}, config=config)
    second = analyze_common_valid({"a": a, "b": b}, config=config)
    assert first.comparison_rows == second.comparison_rows
    absolute = comparison(first, "all", "raw_absolute")
    rpe = comparison(first, "all", "frame_rpe", delta=1)
    assert absolute["bootstrap_method"] == "IID paired percentile over frames"
    assert rpe["bootstrap_method"] == "IID paired percentile over RPE pairs"
    assert absolute["ci_low"] <= absolute["delta_b_minus_a"] <= absolute["ci_high"]


def test_outputs_provenance_standard_json_npz_and_cli(tmp_path):
    gt = line_trajectory(4)
    a = make_run(tmp_path, "a", range(4), gt, gt)
    b = make_run(tmp_path, "b", range(4), gt, gt)
    output = tmp_path / "output"
    payload = run_common_valid(
        {"a": a, "b": b}, output,
        config=CommonValidConfig(frame_deltas=(1,), bootstrap_replicates=10),
    )
    assert payload["status"] == "ok"
    assert {path.name for path in output.iterdir()} == {
        "summary.csv", "comparisons.csv", "frames.csv", "rpe_pairs.csv",
        "result.json", "arrays.npz", "provenance.json",
    }
    result_json = json.loads((output / "result.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert result_json["diagnostics"]["input_hashes"]["a"]["trajectory_est"]
    assert len(provenance["runs"][0]["input_sha256"]["trajectory_est"]) == 64
    assert provenance["bootstrap"]["method"] == "IID paired percentile"
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["method_names"].tolist() == ["a", "b"]
        assert arrays["all_mask"].dtype == np.bool_
    completed = subprocess.run([
        sys.executable, "scripts/fsm_common_valid.py", "--method", f"a={a}",
        "--method", f"b={b}", "--output-dir", str(tmp_path / "cli"),
        "--frame-deltas", "1", "--bootstrap-reps", "5",
    ], cwd=str(tmp_path.parents[1] / "C_GLS_code") if (tmp_path.parents[1] / "C_GLS_code").exists() else "/home/ras/xbf/C_GLS_code", text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["method_count"] == 2


def test_method_spec_and_minimum_method_validation():
    assert list(parse_method_specs(["a=/tmp/a", "b=/tmp/b"])) == ["a", "b"]
    with pytest.raises(ValueError, match="duplicate method"):
        parse_method_specs(["a=/tmp/a", "a=/tmp/b"])
    with pytest.raises(ValueError, match="at least two"):
        analyze_common_valid({"a": "/tmp/a"})
