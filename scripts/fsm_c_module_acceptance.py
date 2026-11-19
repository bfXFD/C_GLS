#!/usr/bin/env python3
"""Audit the completed Corr-Weak versus runtime-weak all-seven panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "experiments/FSM/mvp_observable_v1/c_module_runtimeweak_v4_all7"
OUTPUT = PANEL / "acceptance_v1"
DATASETS = (
    "replica_room0_0",
    "replica_room1_0",
    "replica_room2_0",
    "replica_office0_0",
    "tum_freiburg1_desk",
    "tum_freiburg2_xyz",
    "tum_freiburg3_long_office_household",
)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left, right, prefix: str = "") -> list[dict]:
    output = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                output.append({"path": path, "corrweak": left.get(key), "weak": right.get(key)})
            else:
                output.extend(differences(left[key], right[key], path))
    elif isinstance(left, list) and isinstance(right, list):
        if left != right:
            output.append({"path": prefix, "corrweak": left, "weak": right})
    elif left != right:
        output.append({"path": prefix, "corrweak": left, "weak": right})
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_rows = []
    config_rows = []
    manifest_rows = []
    failures = []
    common_rows = []
    expected_config_differences = {
        "observable_reserve.selection",
    }
    for dataset in DATASETS:
        root = PANEL / dataset
        manifest = read_json(root / "manifest.json")
        if len(manifest.get("cases", [])) != 2:
            failures.append(f"{dataset}: manifest does not contain two cases")
        case_by_name = {case["name"]: case for case in manifest["cases"]}
        if set(case_by_name) != {"corrweak128", "weak128"}:
            failures.append(f"{dataset}: unexpected manifest cases {sorted(case_by_name)}")
        corr_case, weak_case = case_by_name["corrweak128"], case_by_name["weak128"]
        for field in ("map_input", "activation_geometry_input", "common_config_input"):
            if corr_case[field]["sha256"] != weak_case[field]["sha256"]:
                failures.append(f"{dataset}: mismatched {field} hash")
        for case in (corr_case, weak_case):
            if case["evaluation_protocol"] != "mvp_strict":
                failures.append(f"{dataset}/{case['name']}: non-strict protocol")
            if case["uses_gt_current_frame_initialization"] or case["uses_gt_fallback"]:
                failures.append(f"{dataset}/{case['name']}: prohibited GT use")
            if case["final_target_oracle"]["enabled"]:
                failures.append(f"{dataset}/{case['name']}: final target oracle enabled")
            for artifact_field in ("config_input", "common_config_input", "map_input", "activation_geometry_input"):
                artifact = case[artifact_field]
                path = Path(artifact["path"])
                valid = path.is_file() and path.stat().st_size == artifact["size_bytes"] and sha256(path) == artifact["sha256"]
                manifest_rows.append({
                    "dataset": dataset,
                    "case": case["name"],
                    "artifact": artifact_field,
                    "path": str(path),
                    "sha256": artifact["sha256"],
                    "valid": valid,
                })
                if not valid:
                    failures.append(f"{dataset}/{case['name']}: invalid {artifact_field}")

        corr_config = read_json(root / "config_snapshot/corrweak128.json")
        weak_config = read_json(root / "config_snapshot/weak128.json")
        diffs = differences(corr_config, weak_config)
        diff_paths = {row["path"] for row in diffs}
        config_rows.append({
            "dataset": dataset,
            "difference_count": len(diffs),
            "difference_paths": "|".join(sorted(diff_paths)),
            "selection_only": diff_paths == expected_config_differences,
        })
        if diff_paths != expected_config_differences:
            failures.append(f"{dataset}: resolved configs differ at {sorted(diff_paths)}")

        summaries = {row["case"]: row for row in read_csv(root / "summary_all.csv")}
        corr, weak = summaries["corrweak128"], summaries["weak128"]
        corr_ate, weak_ate = float(corr["raw_ate_rmse_m"]), float(weak["raw_ate_rmse_m"])
        corr_rot, weak_rot = float(corr["raw_rotation_abs_rmse_deg"]), float(weak["raw_rotation_abs_rmse_deg"])
        result_rows.append({
            "dataset": dataset,
            "frames": int(corr["frames"]),
            "corrweak_valid": int(corr["valid_frames"]),
            "weak_valid": int(weak["valid_frames"]),
            "corrweak_lost": int(corr["lost_frames"]),
            "weak_lost": int(weak["lost_frames"]),
            "corrweak_raw_ate_m": f"{corr_ate:.9f}",
            "weak_raw_ate_m": f"{weak_ate:.9f}",
            "corrweak_vs_weak_ate_percent": f"{100.0 * (1.0 - corr_ate / weak_ate):.3f}",
            "corrweak_raw_rot_deg": f"{corr_rot:.6f}",
            "weak_raw_rot_deg": f"{weak_rot:.6f}",
            "corrweak_vs_weak_rot_percent": f"{100.0 * (1.0 - corr_rot / weak_rot):.3f}",
            "corrweak_attempts": int(corr["num_reserve_attempt_frames"]),
            "weak_attempts": int(weak["num_reserve_attempt_frames"]),
            "corrweak_adoptions": int(corr["num_reserve_adopted_frames"]),
            "weak_adoptions": int(weak["num_reserve_adopted_frames"]),
            "corrweak_prod_latency_ms": f"{float(corr['production_equivalent_total_frame_latency_mean_ms']):.3f}",
            "weak_prod_latency_ms": f"{float(weak['production_equivalent_total_frame_latency_mean_ms']):.3f}",
        })
        for case in ("corrweak128", "weak128"):
            required = (
                root / f"localize/{case}/summary.json",
                root / f"localize/{case}/frame_metrics.csv",
                root / f"localize/{case}/trajectory_est.txt",
                root / f"localize/{case}/trajectory_gt.txt",
                root / f"metrics/{case}.json",
                root / f"logs/{case}.log",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                failures.extend(f"missing {path}" for path in missing)

        common_summary = read_csv(OUTPUT / f"common_valid/{dataset}/summary.csv")
        common = {
            (row["method"], row["metric_group"], row["delta_frames"]): row
            for row in common_summary if row["scope"] == "all"
        }
        corr_raw = common[("corrweak128", "raw_absolute", "")]
        weak_raw = common[("weak128", "raw_absolute", "")]
        corr_aligned = common[("corrweak128", "aligned_absolute", "")]
        weak_aligned = common[("weak128", "aligned_absolute", "")]
        corr_rpe = common[("corrweak128", "frame_rpe", "1")]
        weak_rpe = common[("weak128", "frame_rpe", "1")]
        corr_raw_ate = float(corr_raw["translation_rmse_m"])
        weak_raw_ate = float(weak_raw["translation_rmse_m"])
        corr_raw_rot = float(corr_raw["rotation_rmse_deg"])
        weak_raw_rot = float(weak_raw["rotation_rmse_deg"])
        common_rows.append({
            "dataset": dataset,
            "common_frames": int(corr_raw["common_frame_count"]),
            "corrweak_raw_ate_m": f"{corr_raw_ate:.9f}",
            "weak_raw_ate_m": f"{weak_raw_ate:.9f}",
            "corrweak_vs_weak_raw_ate_percent": f"{100.0 * (1.0 - corr_raw_ate / weak_raw_ate):.3f}",
            "corrweak_aligned_ate_m": f"{float(corr_aligned['translation_rmse_m']):.9f}",
            "weak_aligned_ate_m": f"{float(weak_aligned['translation_rmse_m']):.9f}",
            "corrweak_raw_rotation_deg": f"{corr_raw_rot:.6f}",
            "weak_raw_rotation_deg": f"{weak_raw_rot:.6f}",
            "corrweak_vs_weak_raw_rotation_percent": f"{100.0 * (1.0 - corr_raw_rot / weak_raw_rot):.3f}",
            "corrweak_rpe_d1_translation_m": f"{float(corr_rpe['translation_rmse_m']):.9f}",
            "weak_rpe_d1_translation_m": f"{float(weak_rpe['translation_rmse_m']):.9f}",
            "corrweak_rpe_d1_rotation_deg": f"{float(corr_rpe['rotation_rmse_deg']):.6f}",
            "weak_rpe_d1_rotation_deg": f"{float(weak_rpe['rotation_rmse_deg']):.6f}",
        })

    write_csv(OUTPUT / "results.csv", result_rows)
    write_csv(OUTPUT / "config_equivalence.csv", config_rows)
    write_csv(OUTPUT / "manifest_verification.csv", manifest_rows)
    write_csv(OUTPUT / "common_frame_results.csv", common_rows)
    summary = {
        "schema_version": 1,
        "panel_root": str(PANEL),
        "dataset_count": len(DATASETS),
        "run_count": len(DATASETS) * 2,
        "selection_only_config_equivalence": all(row["selection_only"] for row in config_rows),
        "artifact_checks": len(manifest_rows),
        "artifact_failures": sum(not row["valid"] for row in manifest_rows),
        "failures": failures,
        "accepted": not failures,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
