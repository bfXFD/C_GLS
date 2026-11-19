import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
OUT = MVP / "reviewer_paper_tables_v2"

SUITES = {
    "replica_essential": MVP / "reviewer_ablation_suite_v2",
    "tum_essential": MVP / "reviewer_ablation_suite_v2_tum",
    "primary_v4": MVP / "reviewer_primary_only_v4_v2",
    "budgets": MVP / "reviewer_budget_sensitivity_v2",
    "perturb4": MVP / "reviewer_first_pose_perturb4_v2",
    "fixed_normals": MVP / "reviewer_fixed_normals_v2",
}


def load_runs(suite):
    runs = {}
    for summary_path in sorted((suite / "runs").glob("*/localize/*/summary.json")):
        case_dir = summary_path.parent
        case = case_dir.name
        errors = np.genfromtxt(
            case_dir / "per_frame_errors.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        summary = json.loads(summary_path.read_text())
        config = json.loads(next((suite / "runs" / case).glob("config_snapshot/*.json")).read_text())
        runs[case] = {"summary": summary, "errors": errors, "config": config}
    return runs


def jts(run):
    rows = run["errors"]
    valid = rows["tracking_valid"].astype(bool)
    translation = rows["raw_translation_error_m"].astype(float)
    rotation = rows["raw_rotation_error_deg"].astype(float)
    thresholds = np.arange(1, 101)
    success = valid[:, None] & (translation[:, None] < 0.005 * thresholds) & (rotation[:, None] < 0.1 * thresholds)
    return 100.0 * float(success.mean())


def pair_metrics(first, second):
    a, b = first["errors"], second["errors"]
    valid = a["tracking_valid"].astype(bool) & b["tracking_valid"].astype(bool)
    return {
        "common": int(valid.sum()),
        "first_translation_rmse_m": float(np.sqrt(np.mean(a["raw_translation_error_m"][valid] ** 2))),
        "second_translation_rmse_m": float(np.sqrt(np.mean(b["raw_translation_error_m"][valid] ** 2))),
        "first_rotation_rmse_deg": float(np.sqrt(np.mean(a["raw_rotation_error_deg"][valid] ** 2))),
        "second_rotation_rmse_deg": float(np.sqrt(np.mean(b["raw_rotation_error_deg"][valid] ** 2))),
    }


def write_csv(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_runs = {name: load_runs(path) for name, path in SUITES.items()}
    essential_rows = []
    for suite_name in ("replica_essential", "tum_essential"):
        for case, run in sorted(all_runs[suite_name].items()):
            summary = run["summary"]
            review = run["config"].get("reviewer_ablation", {})
            essential_rows.append({
                "suite": suite_name,
                "scene": review.get("scene"),
                "case": case,
                "method": review.get("method"),
                "seed": review.get("seed"),
                "target_mode": review.get("target_mode"),
                "normal_mode": review.get("normal_mode"),
                "use_v4": review.get("use_v4"),
                "valid": summary["num_valid_frames"],
                "lost": summary["num_lost_frames"],
                "jts_percent": jts(run),
                "raw_translation_rmse_m": summary["raw_ate_rmse_m"],
                "raw_rotation_rmse_deg": summary["raw_rotation_abs_rmse_deg"],
                "attempts": summary.get("num_reserve_attempt_frames", 0),
                "adoptions": summary.get("num_reserve_adopted_frames", 0),
                "latency_mean_ms": summary["production_equivalent_total_frame_latency_mean_ms"],
            })
    write_csv("table_closed_loop_interventions.csv", essential_rows)

    budget_rows = []
    baseline_runs = all_runs["replica_essential"]
    budget_runs = all_runs["budgets"]
    for scene in ("room2", "office0", "office4"):
        baseline = baseline_runs[f"replica_{scene}_baseline"]
        methods = [(0, baseline)]
        for budget in (32, 64):
            methods.append((budget, budget_runs[f"replica_{scene}_corrweak{budget}"]))
        methods.append((128, baseline_runs[f"replica_{scene}_corrweak128"]))
        methods.append((256, budget_runs[f"replica_{scene}_corrweak256"]))
        valid = np.logical_and.reduce([run["errors"]["tracking_valid"].astype(bool) for _, run in methods])
        for budget, run in methods:
            summary = run["summary"]
            budget_rows.append({
                "scene": scene,
                "budget": "baseline" if budget == 0 else budget,
                "common": int(valid.sum()),
                "valid": summary["num_valid_frames"],
                "lost": summary["num_lost_frames"],
                "jts_percent": jts(run),
                "common_translation_rmse_m": float(np.sqrt(np.mean(run["errors"]["raw_translation_error_m"][valid] ** 2))),
                "common_rotation_rmse_deg": float(np.sqrt(np.mean(run["errors"]["raw_rotation_error_deg"][valid] ** 2))),
                "attempts": summary.get("num_reserve_attempt_frames", 0),
                "adoptions": summary.get("num_reserve_adopted_frames", 0),
            })
    write_csv("table_budget_sensitivity.csv", budget_rows)

    normal_rows = []
    for scene, standard_suite, standard_case, fixed_case in (
        ("room2", "replica_essential", "replica_room2_corrweak128", "replica_room2_corrweak128_frozen_normals"),
        ("office0", "replica_essential", "replica_office0_corrweak128", "replica_office0_corrweak128_frozen_normals"),
        ("office4", "replica_essential", "replica_office4_corrweak128", "replica_office4_corrweak128_frozen_normals"),
        ("fr3", "tum_essential", "tum_freiburg3_long_office_household_corrweak128", "tum_freiburg3_long_office_household_corrweak128_frozen_normals"),
    ):
        standard = all_runs[standard_suite][standard_case]
        fixed = all_runs["fixed_normals"][fixed_case]
        pair = pair_metrics(standard, fixed)
        for label, run, prefix in (("recomputed", standard, "first"), ("frozen", fixed, "second")):
            summary = run["summary"]
            normal_rows.append({
                "scene": scene,
                "normals": label,
                "common": pair["common"],
                "valid": summary["num_valid_frames"],
                "lost": summary["num_lost_frames"],
                "jts_percent": jts(run),
                "common_translation_rmse_m": pair[f"{prefix}_translation_rmse_m"],
                "common_rotation_rmse_deg": pair[f"{prefix}_rotation_rmse_deg"],
                "attempts": summary["num_reserve_attempt_frames"],
                "adoptions": summary["num_reserve_adopted_frames"],
            })
    write_csv("table_normal_recomputation.csv", normal_rows)

    perturb_rows = []
    perturb_runs = all_runs["perturb4"]
    for scene in ("room2", "office4"):
        for perturbation in ("txn02", "txp02", "yawn1", "yawp1"):
            baseline = perturb_runs[f"replica_{scene}_perturb_{perturbation}_baseline"]
            corrweak = perturb_runs[f"replica_{scene}_perturb_{perturbation}_corrweak128"]
            pair = pair_metrics(baseline, corrweak)
            perturb_rows.append({
                "scene": scene,
                "perturbation": perturbation,
                "common": pair["common"],
                "baseline_valid": baseline["summary"]["num_valid_frames"],
                "corrweak_valid": corrweak["summary"]["num_valid_frames"],
                "baseline_jts_percent": jts(baseline),
                "corrweak_jts_percent": jts(corrweak),
                "delta_jts_pp": jts(corrweak) - jts(baseline),
                **pair,
            })
    write_csv("table_first_pose_perturbations.csv", perturb_rows)

    primary_rows = []
    for scene in ("room2", "office0", "office4"):
        baseline = all_runs["replica_essential"][f"replica_{scene}_baseline"]
        primary = all_runs["primary_v4"][f"replica_{scene}_primary_only_v4"]
        pair = pair_metrics(baseline, primary)
        primary_rows.append({
            "scene": scene,
            "attempts": primary["summary"]["num_reserve_attempt_frames"],
            "adoptions": primary["summary"]["num_reserve_adopted_frames"],
            "baseline_jts_percent": jts(baseline),
            "primary_only_jts_percent": jts(primary),
            **pair,
        })
    write_csv("table_primary_only_v4.csv", primary_rows)

    manifest = []
    for path in sorted(OUT.glob("*.csv")):
        import hashlib
        manifest.append({
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "tables": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
