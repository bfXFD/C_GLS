#!/usr/bin/env python3
"""Offline evidence strengthening for the C-GLS RA-L submission."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
OUTPUT = MVP / "ral_evidence_strengthening_v1"
C_PANEL = MVP / "c_module_runtimeweak_v4_all7"

RUNS = {
    "Room0": (MVP / "core_route_v11_oomfix/room0_baseline/localize/baseline", MVP / "core_route_v11_oomfix/room0_corrweak/localize/corrweak128"),
    "Room1": (MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/baseline", MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/corrweak128"),
    "Room2": (MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/baseline", MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/corrweak128"),
    "Office0": (MVP / "core_route_v11_oomfix/office0_baseline/localize/baseline", MVP / "core_route_v11_oomfix/office0_corrweak/localize/corrweak128"),
    "fr1/desk": (MVP / "tum_fps_hybrid_v1/freiburg1_desk/closed_loop/localize/fps10", MVP / "tum_fps_hybrid_v1/freiburg1_desk/closed_loop/localize/fps_corrweak128"),
    "fr2/xyz": (MVP / "tum_fps_hybrid_v1/freiburg2_xyz/closed_loop/localize/fps10", MVP / "tum_fps_hybrid_v1/freiburg2_xyz/closed_loop/localize/fps_corrweak128"),
    "fr3/office": (MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps10", MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps_corrweak128"),
}

PAIRED = {
    "Room0": MVP / "core_route_v2/room0_0_500_paired/localize/paired_core/paired_reserve/paired_reserve_ids.npz",
    "Room1": MVP / "room1_paired_late_v1/paired/room1/localize/paired/paired_reserve/paired_reserve_ids.npz",
    "Room2": MVP / "room12_generalization_v1/paired/room2/localize/paired/paired_reserve/paired_reserve_ids.npz",
    "Office0": MVP / "core_route_v5/office0_0_500_paired/localize/paired_core/paired_reserve/paired_reserve_ids.npz",
    "fr1/desk": MVP / "tum_core_v1/freiburg1_desk/paired/localize/paired/paired_reserve/paired_reserve_ids.npz",
    "fr2/xyz": MVP / "tum_core_v1/freiburg2_xyz/paired/localize/paired/paired_reserve/paired_reserve_ids.npz",
    "fr3/office": MVP / "tum_core_v1/freiburg3_long_office_household/paired/localize/paired/paired_reserve/paired_reserve_ids.npz",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def errors(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path / "per_frame_errors.csv")
    frame = np.asarray([int(row["frame_id"]) for row in rows])
    valid = np.asarray([row["tracking_valid"].lower() == "true" for row in rows])
    translation = np.asarray([float(row["raw_translation_error_m"] or "nan") for row in rows])
    rotation = np.asarray([float(row["raw_rotation_error_deg"] or "nan") for row in rows])
    return frame, valid, translation, rotation


def success_rows() -> list[dict]:
    thresholds = ((.10, 5.0), (.20, 5.0), (.50, 10.0))
    output = []
    for scene, (baseline, method) in RUNS.items():
        _, base_valid, base_t, base_r = errors(baseline)
        _, ours_valid, ours_t, ours_r = errors(method)
        row = {"scene": scene, "frames": base_valid.size}
        for threshold_t, threshold_r in thresholds:
            key = f"success_{threshold_t:g}m_{threshold_r:g}deg"
            base_success = np.mean(base_valid & (base_t < threshold_t) & (base_r < threshold_r))
            ours_success = np.mean(ours_valid & (ours_t < threshold_t) & (ours_r < threshold_r))
            row[f"base_{key}_percent"] = f"{100 * base_success:.2f}"
            row[f"ours_{key}_percent"] = f"{100 * ours_success:.2f}"
            row[f"delta_{key}_pp"] = f"{100 * (ours_success - base_success):.2f}"
        translation_grid = np.linspace(0.0, .5, 101)[1:]
        rotation_grid = np.linspace(0.0, 10.0, 101)[1:]
        base_joint = np.mean([
            np.mean(base_valid & (base_t < threshold_t) & (base_r < threshold_r))
            for threshold_t, threshold_r in zip(translation_grid, rotation_grid)
        ])
        ours_joint = np.mean([
            np.mean(ours_valid & (ours_t < threshold_t) & (ours_r < threshold_r))
            for threshold_t, threshold_r in zip(translation_grid, rotation_grid)
        ])
        row["base_joint_auc_percent"] = f"{100 * base_joint:.2f}"
        row["ours_joint_auc_percent"] = f"{100 * ours_joint:.2f}"
        row["joint_auc_delta_pp"] = f"{100 * (ours_joint - base_joint):.2f}"
        output.append(row)
    return output


def office0_figure() -> None:
    corr = C_PANEL / "replica_office0_0/localize/corrweak128"
    weak = C_PANEL / "replica_office0_0/localize/weak128"
    frame, corr_valid, corr_t, corr_r = errors(corr)
    _, weak_valid, weak_t, weak_r = errors(weak)
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 4.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, .38]})
    axes[0].plot(frame, corr_t, color="#0072B2", linewidth=.8, label="Corr-Weak")
    axes[0].plot(frame, weak_t, color="#D55E00", linewidth=.8, label="Weak-only", alpha=.85)
    axes[0].set_ylabel("translation (m)")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].plot(frame, corr_r, color="#0072B2", linewidth=.8)
    axes[1].plot(frame, weak_r, color="#D55E00", linewidth=.8, alpha=.85)
    axes[1].axhline(5.0, color="#555555", linestyle="--", linewidth=.7)
    axes[1].set_ylabel("rotation (deg)")
    axes[2].scatter(frame[~corr_valid], np.ones((~corr_valid).sum()), color="#0072B2", s=5, label="Corr-Weak lost")
    axes[2].scatter(frame[~weak_valid], np.zeros((~weak_valid).sum()), color="#D55E00", s=5, label="Weak-only lost")
    axes[2].set_yticks([0, 1], ["weak", "corr"])
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("lost")
    for ax in axes:
        ax.grid(alpha=.16, linewidth=.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Office0 Closed-Loop Translation, Rotation, and Availability", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96), pad=.45)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"fig_office0_multiobjective.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calibration_and_hessian_rows() -> tuple[list[dict], list[dict]]:
    calibration = []
    hessian = []
    reliability = []
    bins = np.linspace(0.0, 1.0, 11)
    for scene, path in PAIRED.items():
        with np.load(path, allow_pickle=False) as data:
            policy = np.asarray(data["policy"]).astype(str)
            budget = np.asarray(data["budget"], dtype=int)
            selected = np.where((policy == "runtime_corr_weak") & (budget == 128))[0]
            point_p, point_y = [], []
            offsets = np.asarray(data["offsets"], dtype=int)
            all_p = np.asarray(data["selected_p_corr"], dtype=float)
            all_y = np.asarray(data["selected_actual_correspondence"], dtype=bool)
            for index in selected:
                start, end = offsets[index], offsets[index + 1]
                valid = np.isfinite(all_p[start:end])
                point_p.append(all_p[start:end][valid])
                point_y.append(all_y[start:end][valid].astype(float))
            p = np.concatenate(point_p) if point_p else np.empty(0)
            y = np.concatenate(point_y) if point_y else np.empty(0)
            calibration.append({
                "scene": scene,
                "selected_points": p.size,
                "actual_correspondence_rate": f"{np.mean(y):.6f}" if p.size else "",
                "mean_compatibility": f"{np.mean(p):.6f}" if p.size else "",
                "brier_score": f"{np.mean((p-y)**2):.6f}" if p.size else "",
                "mean_log_loss": f"{np.mean(-(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))):.6f}" if p.size else "",
            })
            if p.size:
                ids = np.clip(np.digitize(p, bins) - 1, 0, 9)
                for bin_index in range(10):
                    mask = ids == bin_index
                    if np.any(mask):
                        reliability.append((scene, .5*(bins[bin_index]+bins[bin_index+1]), np.mean(p[mask]), np.mean(y[mask]), mask.sum()))

            primary = np.asarray(data["primary_hessian_normalized"], dtype=float)[selected]
            augmented = np.asarray(data["augmented_hessian_normalized"], dtype=float)[selected]
            corr_count = np.asarray(data["actual_correspondence_count"], dtype=float)[selected]
            primary_min = np.linalg.eigvalsh(primary)[:, 0]
            augmented_min = np.linalg.eigvalsh(augmented)[:, 0]
            raw_ratio = augmented_min / np.maximum(primary_min, 1e-12)
            per_corr_ratio = (augmented_min / np.maximum(corr_count, 1.0)) / np.maximum(primary_min / np.maximum(corr_count - np.asarray(data["actual_correspondence_unique_count"], dtype=float)[selected], 1.0), 1e-12)
            trace_ratio = (augmented_min / np.maximum(np.trace(augmented, axis1=1, axis2=2), 1e-12)) / np.maximum(primary_min / np.maximum(np.trace(primary, axis1=1, axis2=2), 1e-12), 1e-12)
            hessian.append({
                "scene": scene,
                "rows": selected.size,
                "median_raw_lambda_ratio": f"{np.median(raw_ratio):.6f}",
                "median_per_correspondence_ratio": f"{np.median(per_corr_ratio[np.isfinite(per_corr_ratio)]):.6f}",
                "median_trace_normalized_ratio": f"{np.median(trace_ratio):.6f}",
                "fraction_raw_ratio_gt_1": f"{np.mean(raw_ratio > 1):.6f}",
                "fraction_per_corr_ratio_gt_1": f"{np.mean(per_corr_ratio > 1):.6f}",
                "fraction_trace_ratio_gt_1": f"{np.mean(trace_ratio > 1):.6f}",
            })
    if reliability:
        fig, ax = plt.subplots(figsize=(3.4, 2.65))
        for scene in PAIRED:
            rows = [row for row in reliability if row[0] == scene]
            if rows:
                ax.plot([row[2] for row in rows], [row[3] for row in rows], marker="o", markersize=3, linewidth=.8, label=scene)
        ax.plot([0,1], [0,1], color="#555555", linestyle="--", linewidth=.8)
        ax.set_xlabel("mean compatibility")
        ax.set_ylabel("actual correspondence rate")
        ax.set_title("Offline Corr-Weak compatibility reliability", fontsize=9)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=.4)
        for suffix in ("pdf", "png"):
            fig.savefig(OUTPUT / f"fig_compatibility_reliability.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return calibration, hessian


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    success = success_rows()
    calibration, hessian = calibration_and_hessian_rows()
    write_csv(OUTPUT / "table_failure_aware_success.csv", success)
    write_csv(OUTPUT / "table_compatibility_calibration.csv", calibration)
    write_csv(OUTPUT / "table_normalized_hessian.csv", hessian)
    office0_figure()
    inputs = sorted({path.resolve() for pair in RUNS.values() for run in pair for path in (run / "per_frame_errors.csv",)} | {path.resolve() for path in PAIRED.values()})
    manifest = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs]
    (OUTPUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "joint_success": "valid and translation/rotation below both thresholds; lost frames count as failures",
        "joint_auc": "mean joint success along thresholds 0--0.5 m and 0--10 deg",
        "calibration_scope": "offline paired-shadow selected Corr-Weak points at K=128",
        "hessian_scope": "paired-shadow camera/scale-normalized Hessians at K=128",
        "input_count": len(manifest),
    }, indent=2) + "\n")
    print(json.dumps({"output_dir": str(OUTPUT), "input_count": len(manifest)}))


if __name__ == "__main__":
    main()
