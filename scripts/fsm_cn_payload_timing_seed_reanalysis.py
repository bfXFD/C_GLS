#!/usr/bin/env python3
"""Read-only, CPU-only accounting of saved payloads, timings and seed controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
OUTPUT = MVP / "cn_revision_analysis_v1"
MAP_CANDIDATES = {"means", "covariances", "normals", "scores", "indices"}
RESERVE_CANDIDATES = {
    "reserve_global_ids", "reserve_means", "reserve_normals", "reserve_scores",
    "reserve_covariances", "reserve_supported", "reserve_normal_source",
}
TIMING_METRICS = (
    "total_frame_latency_mean_ms", "production_equivalent_total_frame_latency_mean_ms",
    "local_query_latency_mean_ms", "latency_mean_ms",
    "reserve_latency_all_tracking_frames_mean_ms", "reserve_latency_triggered_mean_ms",
    "reserve_corr_latency_mean_ms", "cold_start_latency_ms", "peak_rss_mb",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = list(values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected nonempty finite observations")
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "range": max(values) - min(values)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jts(rows):
    """Percent joint success averaged at 100 strict raw-error thresholds."""
    if not rows:
        raise ValueError("Empty frame table")
    successes = 0
    for row in rows:
        if row["tracking_valid"].lower() not in ("true", "false"):
            raise ValueError("Invalid tracking_valid value")
        if row["tracking_valid"].lower() != "true":
            continue
        translation = float(row["raw_translation_error_m"] or "nan")
        rotation = float(row["raw_rotation_error_deg"] or "nan")
        successes += sum(translation < .005 * k and rotation < .1 * k
                         for k in range(1, 101))
    return successes / len(rows)


def zip_account(path, scene, role):
    candidates = RESERVE_CANDIDATES if role == "sidecar" else MAP_CANDIDATES
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        rows = [{"scene": scene, "role": role, "path": str(path.relative_to(ROOT)),
                 "member": item.filename,
                 "field_class": ("runtime_field_candidate" if Path(item.filename).stem in candidates
                                 else "audit_provenance_or_validation_overhead"),
                 "file_size": item.file_size, "compress_size": item.compress_size,
                 "compression_method": item.compress_type}
                for item in members]
    serialized = path.stat().st_size
    summary = {"scene": scene, "role": role, "path": str(path.relative_to(ROOT)),
               "serialized_bytes": serialized,
               "member_file_size_sum": sum(row["file_size"] for row in rows),
               "member_compress_size_sum": sum(row["compress_size"] for row in rows),
               "zip_container_overhead_bytes": serialized - sum(row["compress_size"] for row in rows),
               "sha256": sha256(path)}
    for category in ("runtime_field_candidate", "audit_provenance_or_validation_overhead"):
        for key in ("file_size", "compress_size"):
            summary[f"{category}_{key}"] = sum(row[key] for row in rows if row["field_class"] == category)
    return rows, summary


def payload_tables(inputs):
    fields, files, totals = [], [], []
    for scene, source, compact_dir in (
        ("Room2", "replica_room2_0", "runtime_reserve_v1/maps/replica_room2_0/logdet"),
        ("Office0", "replica_office0_0", "runtime_reserve_v1/maps/replica_office0_0/logdet"),
        ("fr3/office", "tum_freiburg3_long_office_household",
         "tum_fps_hybrid_v1/maps/freiburg3_long_office_household"),
    ):
        assets = {}
        for role, path in (
            ("compact", MVP / compact_dir / "fsm_map.npz"),
            ("sidecar", MVP / compact_dir / "activation_geometry.npz"),
            ("full", ROOT / f"experiments/FSM/audit_v2_full_rerun/main/{source}/full/maps/full_ratio_1/fsm_map.npz"),
        ):
            rows, summary = zip_account(path, scene, role)
            inputs.add(path)
            fields.extend(rows)
            files.append(summary)
            assets[role] = summary["serialized_bytes"]
        totals.append({"scene": scene, "compact_bytes": assets["compact"],
                       "sidecar_bytes": assets["sidecar"],
                       "compact_plus_reserve_bytes": assets["compact"] + assets["sidecar"],
                       "full_bytes": assets["full"],
                       "actual_serialized_ratio_to_full": (assets["compact"] + assets["sidecar"]) / assets["full"]})
    return fields, files, totals


def timing_tables(inputs):
    runs, aggregates = [], []
    for scene, prefix in (("Office0", "office0"), ("fr2/xyz", "tum_fr2"), ("fr3/office", "tum_fr3")):
        for method, leaf in (("baseline", "baseline"), ("corrweak", "corrweak128")):
            if prefix.startswith("tum_"):
                leaf = "fps10" if method == "baseline" else "fps_corrweak128"
            observations = {metric: [] for metric in TIMING_METRICS}
            for rep in range(1, 4):
                run = MVP / f"timing_v1/{prefix}_repeat{rep}_{method}"
                path = run / f"localize/{leaf}/summary.json"
                inputs.add(path)
                inputs.update((run / "config_snapshot").glob("*"))
                summary = json.loads(path.read_text())
                for metric in TIMING_METRICS:
                    value = summary.get(metric)
                    p95_key = metric.replace("_mean_ms", "_p95_ms") if metric.endswith("_mean_ms") else None
                    p95 = summary.get(p95_key) if p95_key else None
                    if value is not None:
                        observations[metric].append(float(value))
                    runs.append({"scene": scene, "method": method, "rep": rep, "metric": metric,
                                 "value": value, "per_run_p95": p95,
                                 "p95_key": p95_key or "", "source": str(path.relative_to(ROOT)),
                                 "source_sha256": sha256(path)})
            for metric, values in observations.items():
                if values:
                    aggregates.append({"scene": scene, "method": method, "metric": metric,
                                       "aggregation_unit": "saved_run_mean" if "_mean_" in metric else "saved_run_value",
                                       **stats(values)})
    return runs, aggregates


def seed_tables(inputs):
    runs, aggregates = [], []
    for scene, suite, prefix in (
        ("Room2", "reviewer_ablation_suite_v2", "replica_room2"),
        ("Office0", "reviewer_ablation_suite_v2", "replica_office0"),
        ("Office4", "reviewer_ablation_suite_v2", "replica_office4"),
        ("fr3/office", "reviewer_ablation_suite_v2_tum", "tum_freiburg3_long_office_household"),
    ):
        values, frame_ids = [], None
        deterministic = None
        for label in [f"random128_s{seed}" for seed in range(5)] + ["corrweak128"]:
            case = f"{prefix}_{label}"
            run = MVP / suite / "runs" / case
            path = run / "localize" / case / "per_frame_errors.csv"
            inputs.add(path)
            inputs.add(path.with_name("summary.json"))
            inputs.update((run / "config_snapshot").glob("*"))
            rows = read_csv(path)
            ids = [(row["frame_id"], row["timestamp"]) for row in rows]
            if len(set(ids)) != len(ids) or (frame_ids is not None and ids != frame_ids):
                raise ValueError(f"Mismatched or duplicate evaluation frames: {path}")
            frame_ids = ids
            score = jts(rows)
            if label == "corrweak128":
                deterministic = score
            else:
                values.append(score)
            runs.append({"scene": scene, "suite": suite, "method": label, "frames": len(rows),
                         "jts_percent": score, "source": str(path.relative_to(ROOT)),
                         "source_sha256": sha256(path)})
        aggregates.append({"scene": scene, "suite": suite, **stats(values),
                           "deterministic_corrweak_jts_percent": deterministic,
                           "random_seeds_above_corrweak": sum(value > deterministic for value in values),
                           "random_seeds_equal_corrweak": sum(value == deterministic for value in values)})
    return runs, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {Path(__file__), ROOT / "tests/test_fsm_cn_payload_timing_seed_reanalysis.py",
              ROOT / "utils/fsm_map.py", ROOT / "utils/fsm_runtime_reserve.py", ROOT / "scripts/fsm_localize.py"}
    fields, files, totals = payload_tables(inputs)
    timing_runs, timing_stats = timing_tables(inputs)
    seed_runs, seed_stats = seed_tables(inputs)
    outputs = []
    for name, rows in (
        ("payload_zip_fields.csv", fields), ("payload_zip_files.csv", files),
        ("payload_serialized_totals.csv", totals), ("timing_saved_runs.csv", timing_runs),
        ("timing_saved_run_statistics.csv", timing_stats), ("random_seed_jts_runs.csv", seed_runs),
        ("random_seed_jts_statistics.csv", seed_stats),
    ):
        path = args.output_dir / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs.append(path)
    notes = {
        "storage": "ZIP file_size includes NPY headers (and pickled object serialization where present), not ndarray nbytes; compress_size excludes ZIP headers and directory. Actual serialized totals use stat().st_size. No lean map exported; no memory savings estimated.",
        "runtime_candidates": {"map": sorted(MAP_CANDIDATES), "sidecar": sorted(RESERVE_CANDIDATES)},
        "compatibility": "Candidates reflect direct localization/runtime-reserve field reads only. Current loaders eagerly load and validate additional fields, geometry, schema and provenance hashes; overhead is not safely removable. Full and compact exports are historical artifacts, not new matched runtime measurements. No production or source_snapshot changes.",
        "timing": "Three equally weighted saved repetitions per scene/method; sample SD uses n-1 across run means, not frames. p95 values are per-run saved summary quantiles; no pooled or averaged p95. Metrics have distinct scopes and are not summed. RSS and cold-start values are run values, not run means. Missing p95 is blank. Legacy ICP latency is not total pipeline latency.",
        "random_seed": "JTS percent = 100 times joint success averaged over k=1..100 at raw translation < .005*k m and raw rotation < .1*k deg. All saved frames, including initialized first pose, form denominator; invalid/NaN errors fail. Five random-selection seeds, one deterministic Corr-Weak; SD is seed variation, not repeated deterministic-run uncertainty. Exact frame/timestamp equality required within scene. Count above uses strict greater-than. No inferential p-value.",
        "suite_separation": "Random-seed controls use reviewer_ablation_suite_v2(_tum); timings use timing_v1. Do not pool them or treat timings as current production benchmarks. No spectral calculation performed.",
        "fr3_sidecar_2392136_bytes_verified": next(row["sidecar_bytes"] == 2392136 for row in totals if row["scene"] == "fr3/office"),
    }
    notes_path = args.output_dir / "payload_timing_seed_caveats.json"
    notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True) + "\n")
    outputs.append(notes_path)
    def record(path):
        return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema_version": 1, "inputs": [record(path) for path in sorted(inputs)],
                "outputs": [record(path) for path in outputs]}
    (args.output_dir / "payload_timing_seed_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "inputs": len(inputs), "tables": 7,
                      "fr3_verified": notes["fr3_sidecar_2392136_bytes_verified"]}))


if __name__ == "__main__":
    main()
