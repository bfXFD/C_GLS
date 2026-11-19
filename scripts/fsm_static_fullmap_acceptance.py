#!/usr/bin/env python3
"""Audit and summarize the frozen static-reserve/full-map panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "experiments/FSM/mvp_observable_v1/static_reserve_fullmap_v1"
OUTPUT = PANEL / "acceptance_v1"
SCENES = (
    "replica_room2_0",
    "replica_office0_0",
    "tum_freiburg3_long_office_household",
)
METHODS = ("compact", "corrweak128", "static4096", "full")


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_metric(rows, method, group, delta=""):
    for row in rows:
        if row["scope"] == "all" and row["method"] == method and row["metric_group"] == group and row["delta_frames"] == delta:
            return row
    raise KeyError((method, group, delta))


def joint_success_auc(run: Path) -> float:
    rows = read_csv(run / "per_frame_errors.csv")
    valid = np.asarray([row["tracking_valid"].lower() == "true" for row in rows])
    translation = np.asarray([float(row["raw_translation_error_m"] or "nan") for row in rows])
    rotation = np.asarray([float(row["raw_rotation_error_deg"] or "nan") for row in rows])
    translation_thresholds = np.linspace(0.005, 0.5, 100)
    rotation_thresholds = np.linspace(0.1, 10.0, 100)
    return float(np.mean([
        np.mean(valid & (translation < translation_threshold) & (rotation < rotation_threshold))
        for translation_threshold, rotation_threshold in zip(translation_thresholds, rotation_thresholds)
    ]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failures, results, materialized, protocol_rows, output_paths = [], [], [], [], []

    source_manifest = read_json(PANEL / "frozen_source_manifest.json")
    for item in source_manifest:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"frozen source mismatch: {path}")

    for scene in SCENES:
        map_root = PANEL / f"maps/{scene}/static_all_reserve"
        done_path = map_root / "fsm_map.npz.done.json"
        static_path = map_root / "fsm_map.npz"
        done = read_json(done_path)
        provenance = done["provenance"]
        static_valid = (
            static_path.is_file()
            and static_path.stat().st_size == done["output_size_bytes"]
            and sha256(static_path) == done["output_sha256"]
            and provenance["reserve_count"] == 4096
            and provenance["merged_count"] == provenance["base_count"] + 4096
        )
        if not static_valid:
            failures.append(f"{scene}: invalid static materialization")
        materialized.append({
            "scene": scene,
            "base_count": provenance["base_count"],
            "reserve_count": provenance["reserve_count"],
            "merged_count": provenance["merged_count"],
            "static_map_mib": f"{static_path.stat().st_size / 2**20:.2f}",
            "output_sha256": done["output_sha256"],
            "valid": static_valid,
        })

        eval_root = PANEL / f"eval/{scene}"
        manifest = read_json(eval_root / "manifest.json")
        cases = {case["name"]: case for case in manifest["cases"]}
        if set(cases) != set(METHODS):
            failures.append(f"{scene}: unexpected cases {sorted(cases)}")
        baseline_hashes = {cases[name]["resolved_config_sha256"] for name in ("compact", "static4096", "full")}
        if len(baseline_hashes) != 1:
            failures.append(f"{scene}: compact/static/full resolved configs differ")
        for method, case in cases.items():
            valid_protocol = (
                case["evaluation_protocol"] == "mvp_strict"
                and case["first_pose_protocol"] == "gt_first_frame"
                and case["motion_model"] == "constant_velocity"
                and not case["uses_gt_current_frame_initialization"]
                and not case["uses_gt_fallback"]
                and not case["final_target_oracle"]["enabled"]
                and case["local_submap"]["max_points"] == 10000
            )
            protocol_rows.append({
                "scene": scene,
                "method": method,
                "resolved_config_sha256": case["resolved_config_sha256"],
                "map_sha256": case["map_input"]["sha256"],
                "map_size_bytes": case["map_input"]["size_bytes"],
                "strict_protocol_valid": valid_protocol,
            })
            if not valid_protocol:
                failures.append(f"{scene}/{method}: protocol violation")
            for field in ("config_input", "common_config_input", "map_input"):
                artifact = case[field]
                path = Path(artifact["path"])
                if not path.is_file() or path.stat().st_size != artifact["size_bytes"] or sha256(path) != artifact["sha256"]:
                    failures.append(f"{scene}/{method}: invalid {field}")
            if method == "corrweak128":
                artifact = case["activation_geometry_input"]
                path = Path(artifact["path"])
                if not path.is_file() or path.stat().st_size != artifact["size_bytes"] or sha256(path) != artifact["sha256"]:
                    failures.append(f"{scene}/{method}: invalid sidecar")

        summaries = {row["case"]: row for row in read_csv(eval_root / "summary_all.csv")}
        common = read_csv(PANEL / f"analysis/{scene}/common_valid/summary.csv")
        for method in METHODS:
            raw = common_metric(common, method, "raw_absolute")
            aligned = common_metric(common, method, "aligned_absolute")
            rpe = common_metric(common, method, "frame_rpe", "1")
            summary = summaries[method]
            run = eval_root / f"localize/{method}"
            run_summary = read_json(run / "summary.json")
            map_path = Path(summary["map"])
            results.append({
                "scene": scene,
                "method": method,
                "map_gaussians": int(summary["N"]),
                "map_mib": f"{map_path.stat().st_size / 2**20:.2f}",
                "common_frames_all_four": int(raw["common_frame_count"]),
                "valid_frames": int(summary["valid_frames"]),
                "lost_frames": int(summary["lost_frames"]),
                "common_raw_ate_m": f"{float(raw['translation_rmse_m']):.9f}",
                "common_raw_rotation_deg": f"{float(raw['rotation_rmse_deg']):.6f}",
                "common_aligned_ate_m": f"{float(aligned['translation_rmse_m']):.9f}",
                "common_rpe_d1_translation_m": f"{float(rpe['translation_rmse_m']):.9f}",
                "common_rpe_d1_rotation_deg": f"{float(rpe['rotation_rmse_deg']):.6f}",
                "joint_success_auc_percent": f"{100.0 * joint_success_auc(run):.3f}",
                "production_latency_mean_ms": f"{float(summary['production_equivalent_total_frame_latency_mean_ms']):.3f}",
                "production_latency_p95_ms": f"{float(summary['production_equivalent_total_frame_latency_p95_ms']):.3f}",
                "peak_rss_mb": f"{float(run_summary['peak_rss_mb']):.2f}",
                "attempts": int(summary["num_reserve_attempt_frames"]),
                "adoptions": int(summary["num_reserve_adopted_frames"]),
            })
            output_paths.extend((run / "summary.json", run / "frame_metrics.csv", run / "trajectory_est.txt", run / "per_frame_errors.csv"))
        output_paths.extend((eval_root / "manifest.json", eval_root / "summary_all.csv", static_path, done_path))

    write_csv(OUTPUT / "results.csv", results)
    write_csv(OUTPUT / "materialization.csv", materialized)
    write_csv(OUTPUT / "protocol_verification.csv", protocol_rows)
    output_paths.extend((PANEL / "frozen_protocol.json", PANEL / "frozen_source_manifest.json"))
    unique = sorted(set(path.resolve() for path in output_paths))
    output_manifest = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique]
    (OUTPUT / "output_manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "scene_count": len(SCENES),
        "run_count": len(SCENES) * len(METHODS),
        "source_hash_checks": len(source_manifest),
        "static_materializations_valid": all(row["valid"] for row in materialized),
        "protocol_checks": len(protocol_rows),
        "output_manifest_count": len(output_manifest),
        "failures": failures,
        "accepted": not failures,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
