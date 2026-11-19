#!/usr/bin/env python3
"""Accept and summarize the frozen Replica Office1-4 held-out panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "experiments/FSM/mvp_observable_v1/replica_office1_4_untouched_v1"
OUTPUT = PANEL / "acceptance_v1"
SCENES = ("office1", "office2", "office3", "office4")


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


def differences(left, right, prefix="") -> list[str]:
    output = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                output.append(path)
            else:
                output.extend(differences(left[key], right[key], path))
    elif left != right:
        output.append(prefix)
    return output


def metric(summary_rows, method, group, delta=""):
    for row in summary_rows:
        if row["scope"] == "all" and row["method"] == method and row["metric_group"] == group and row["delta_frames"] == delta:
            return row
    raise KeyError((method, group, delta))


def per_frame_success(run: Path) -> tuple[float, float, float, float]:
    rows = read_csv(run / "per_frame_errors.csv")
    valid = np.asarray([row["tracking_valid"].lower() == "true" for row in rows])
    t = np.asarray([float(row["raw_translation_error_m"] or "nan") for row in rows])
    r = np.asarray([float(row["raw_rotation_error_deg"] or "nan") for row in rows])
    success_01_5 = np.mean(valid & (t < .1) & (r < 5.0))
    success_02_5 = np.mean(valid & (t < .2) & (r < 5.0))
    success_05_10 = np.mean(valid & (t < .5) & (r < 10.0))
    t_grid = np.linspace(.005, .5, 100)
    r_grid = np.linspace(.1, 10.0, 100)
    auc = np.mean([np.mean(valid & (t < ti) & (r < ri)) for ti, ri in zip(t_grid, r_grid)])
    return success_01_5, success_02_5, success_05_10, auc


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failures = []
    results, config_rows, artifact_rows = [], [], []
    output_paths = []
    for scene in SCENES:
        root = PANEL / f"eval/replica_{scene}_0"
        manifest = read_json(root / "manifest.json")
        cases = {case["name"]: case for case in manifest["cases"]}
        if set(cases) != {"baseline", "corrweak128", "weak128"}:
            failures.append(f"{scene}: unexpected cases {sorted(cases)}")
        hashes = {name: case["map_input"]["sha256"] for name, case in cases.items()}
        if len(set(hashes.values())) != 1:
            failures.append(f"{scene}: map hash differs across arms")
        for name, case in cases.items():
            if case["evaluation_protocol"] != "mvp_strict" or case["uses_gt_current_frame_initialization"] or case["uses_gt_fallback"] or case["final_target_oracle"]["enabled"]:
                failures.append(f"{scene}/{name}: protocol violation")
            for field in ("config_input", "common_config_input", "map_input"):
                artifact = case[field]
                path = Path(artifact["path"])
                valid = path.is_file() and path.stat().st_size == artifact["size_bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
                artifact_rows.append({"scene": scene, "case": name, "artifact": field, "path": str(path), "valid": valid})
                if not valid:
                    failures.append(f"{scene}/{name}: invalid {field}")
            if name != "baseline":
                artifact = case["activation_geometry_input"]
                path = Path(artifact["path"])
                valid = path.is_file() and path.stat().st_size == artifact["size_bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
                artifact_rows.append({"scene": scene, "case": name, "artifact": "activation_geometry_input", "path": str(path), "valid": valid})
                if not valid:
                    failures.append(f"{scene}/{name}: invalid activation geometry")

        corr_config = read_json(root / "config_snapshot/corrweak128.json")
        weak_config = read_json(root / "config_snapshot/weak128.json")
        diff_paths = differences(corr_config, weak_config)
        config_rows.append({"scene": scene, "difference_count": len(diff_paths), "difference_paths": "|".join(diff_paths), "selection_only": diff_paths == ["observable_reserve.selection"]})
        if diff_paths != ["observable_reserve.selection"]:
            failures.append(f"{scene}: active configs differ at {diff_paths}")

        common_rows = read_csv(OUTPUT / f"common_valid/{scene}/summary.csv")
        summaries = {row["case"]: row for row in read_csv(root / "summary_all.csv")}
        row = {"scene": scene}
        for method in ("baseline", "corrweak128", "weak128"):
            raw = metric(common_rows, method, "raw_absolute")
            aligned = metric(common_rows, method, "aligned_absolute")
            rpe = metric(common_rows, method, "frame_rpe", "1")
            run = root / f"localize/{method}"
            s01, s02, s05, auc = per_frame_success(run)
            row[f"{method}_valid"] = int(summaries[method]["valid_frames"])
            row[f"{method}_lost"] = int(summaries[method]["lost_frames"])
            row[f"{method}_raw_ate_m"] = f"{float(raw['translation_rmse_m']):.9f}"
            row[f"{method}_raw_rotation_deg"] = f"{float(raw['rotation_rmse_deg']):.6f}"
            row[f"{method}_aligned_ate_m"] = f"{float(aligned['translation_rmse_m']):.9f}"
            row[f"{method}_rpe_d1_translation_m"] = f"{float(rpe['translation_rmse_m']):.9f}"
            row[f"{method}_success_01m_5deg_percent"] = f"{100*s01:.3f}"
            row[f"{method}_success_02m_5deg_percent"] = f"{100*s02:.3f}"
            row[f"{method}_success_05m_10deg_percent"] = f"{100*s05:.3f}"
            row[f"{method}_joint_auc_percent"] = f"{100*auc:.3f}"
            output_paths.extend((run / "summary.json", run / "frame_metrics.csv", run / "trajectory_est.txt", run / "per_frame_errors.csv"))
        row["common_frames_all_three"] = int(metric(common_rows, "baseline", "raw_absolute")["common_frame_count"])
        base_ate = float(row["baseline_raw_ate_m"])
        corr_ate = float(row["corrweak128_raw_ate_m"])
        weak_ate = float(row["weak128_raw_ate_m"])
        row["corrweak_vs_baseline_ate_percent"] = f"{100*(1-corr_ate/base_ate):.3f}"
        row["corrweak_vs_weak_ate_percent"] = f"{100*(1-corr_ate/weak_ate):.3f}"
        results.append(row)
        output_paths.extend((root / "manifest.json", root / "summary_all.csv"))
        map_root = PANEL / f"maps/replica_{scene}_0/logdet"
        output_paths.extend((map_root / "selection.npz.done.json", map_root / "fsm_map.npz", map_root / "activation_geometry.npz"))

    write_csv(OUTPUT / "heldout_results.csv", results)
    write_csv(OUTPUT / "config_equivalence.csv", config_rows)
    write_csv(OUTPUT / "artifact_verification.csv", artifact_rows)
    output_paths.extend((PANEL / "frozen_protocol.json", PANEL / "frozen_source_manifest.json"))
    unique = sorted(set(path.resolve() for path in output_paths))
    manifest_rows = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in unique]
    (OUTPUT / "output_manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "scenes": list(SCENES),
        "run_count": 12,
        "mapping_outputs_present": all((Path(f"/home/ras/xbf/SplaTAM/experiments/Replica/{scene}_0/params.npz")).is_file() for scene in SCENES),
        "selection_only_config_equivalence": all(row["selection_only"] for row in config_rows),
        "artifact_checks": len(artifact_rows),
        "artifact_failures": sum(not row["valid"] for row in artifact_rows),
        "output_manifest_count": len(manifest_rows),
        "failures": failures,
        "accepted": not failures,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
