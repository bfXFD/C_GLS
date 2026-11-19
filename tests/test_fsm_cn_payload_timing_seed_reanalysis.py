import importlib.util
import math
import statistics
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fsm_cn_payload_timing_seed_reanalysis.py"
SPEC = importlib.util.spec_from_file_location("cn_reanalysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_sample_sd_and_range():
    result = analysis.stats([1, 2, 6])
    assert result == {"n": 3, "mean": 3, "sample_sd": math.sqrt(7),
                      "median": 2, "min": 1, "max": 6, "range": 5}
    assert analysis.stats([4])["sample_sd"] is None


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_stats_reject_bad_observations(values):
    with pytest.raises(ValueError):
        analysis.stats(values)


def test_jts_strict_thresholds_invalid_and_nan():
    def row(valid, translation, rotation):
        return {"tracking_valid": valid, "raw_translation_error_m": str(translation),
                "raw_rotation_error_deg": str(rotation)}
    rows = [row("True", 0, 0), row("False", 0, 0), row("True", .25, 5),
            row("True", "nan", 0)]
    assert analysis.jts(rows) == 37.5


def test_zip_member_and_container_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    path = tmp_path / "map.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("means.npy", b"x" * 4096)
        archive.writestr("source_params_sha256.npy", b"provenance")
    fields, result = analysis.zip_account(path, "test", "compact")
    assert result["member_file_size_sum"] == 4096 + len(b"provenance")
    assert result["serialized_bytes"] == path.stat().st_size
    assert result["member_compress_size_sum"] + result["zip_container_overhead_bytes"] == path.stat().st_size
    assert fields[0]["field_class"] == "runtime_field_candidate"
    assert fields[1]["field_class"] == "audit_provenance_or_validation_overhead"


def test_timing_equal_weight_runs_not_frames(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(analysis, "MVP", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    for prefix in ("office0", "tum_fr2", "tum_fr3"):
        for method, leaf in (("baseline", "baseline"), ("corrweak", "corrweak128")):
            if prefix.startswith("tum_"):
                leaf = "fps10" if method == "baseline" else "fps_corrweak128"
            for rep, mean in enumerate([1, 3, 8], 1):
                path = tmp_path / f"timing_v1/{prefix}_repeat{rep}_{method}/localize/{leaf}/summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"total_frame_latency_mean_ms": mean,
                                            "total_frame_latency_p95_ms": 10 * mean,
                                            "num_frames": rep * 100}))
    runs, aggregates = analysis.timing_tables(set())
    result = next(row for row in aggregates if row["scene"] == "Office0" and row["method"] == "baseline")
    assert result["mean"] == 4
    assert result["sample_sd"] == statistics.stdev([1, 3, 8])
    assert [row["per_run_p95"] for row in runs if row["scene"] == "Office0" and row["method"] == "baseline"
            and row["metric"] == "total_frame_latency_mean_ms"] == [10, 30, 80]


def test_empty_jts_rejected():
    with pytest.raises(ValueError):
        analysis.jts([])


def test_saved_seed_counts_and_evidence_manifest():
    import json
    import os
    output = analysis.OUTPUT
    if not (output / "payload_timing_seed_evidence_manifest.json").exists():
        pytest.skip("Saved evidence tables have not been generated")
    runs = analysis.read_csv(output / "random_seed_jts_runs.csv")
    for row in analysis.read_csv(output / "random_seed_jts_statistics.csv"):
        selected = [run for run in runs if run["scene"] == row["scene"]]
        random = [float(run["jts_percent"]) for run in selected if run["method"].startswith("random")]
        corr = next(float(run["jts_percent"]) for run in selected if run["method"] == "corrweak128")
        assert len(random) == 5
        assert float(row["mean"]) == pytest.approx(statistics.mean(random), rel=1e-14)
        assert float(row["sample_sd"]) == pytest.approx(statistics.stdev(random), rel=1e-14)
        assert int(row["random_seeds_above_corrweak"]) == sum(value > corr for value in random)
    manifest = json.loads((output / "payload_timing_seed_evidence_manifest.json").read_text())
    # Frozen inputs include historical sources, not necessarily this checkout.
    records = list(manifest["outputs"])
    if os.getenv("FSM_VERIFY_FROZEN_SEED_INPUTS") == "1":
        records.extend(manifest["inputs"])
    for record in records:
        path = analysis.ROOT / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert analysis.sha256(path) == record["sha256"]
