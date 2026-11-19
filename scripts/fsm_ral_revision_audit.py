#!/usr/bin/env python3
"""Generate reviewer-driven RA-L audit artifacts from frozen localization outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ROOT = Path(__file__).resolve().parent.parent
MVP = ROOT / "experiments/FSM/mvp_observable_v1"

RUNS = {
    "Room0": (
        MVP / "core_route_v11_oomfix/room0_baseline/localize/baseline",
        MVP / "core_route_v11_oomfix/room0_corrweak/localize/corrweak128",
        "observable20",
    ),
    "Room1": (
        MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/baseline",
        MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/corrweak128",
        "observable20",
    ),
    "Room2": (
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/baseline",
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/corrweak128",
        "observable20",
    ),
    "Office0": (
        MVP / "core_route_v11_oomfix/office0_baseline/localize/baseline",
        MVP / "core_route_v11_oomfix/office0_corrweak/localize/corrweak128",
        "observable20",
    ),
    "fr1/desk": (
        MVP / "tum_fps_hybrid_v1/freiburg1_desk/closed_loop/localize/fps10",
        MVP / "tum_fps_hybrid_v1/freiburg1_desk/closed_loop/localize/fps_corrweak128",
        "fps10",
    ),
    "fr2/xyz": (
        MVP / "tum_fps_hybrid_v1/freiburg2_xyz/closed_loop/localize/fps10",
        MVP / "tum_fps_hybrid_v1/freiburg2_xyz/closed_loop/localize/fps_corrweak128",
        "fps10",
    ),
    "fr3/office": (
        MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps10",
        MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps_corrweak128",
        "fps10",
    ),
}

PAIRED = {
    "Room0": MVP / "core_route_v2/room0_0_500_paired/localize/paired_core/paired_frame_stats/frames.csv",
    "Room1": MVP / "room1_paired_late_v1/paired_stats/room1/frames.csv",
    "Room2": MVP / "room12_generalization_v1/paired_stats/room2/frames.csv",
    "Office0": MVP / "core_route_v5/office0_0_500_paired/localize/paired_core/paired_frame_stats/frames.csv",
}

CRITICAL = {
    "Room2": MVP / "paper_revision_audit_v1/critical_adoption/room2/adoption_events.csv",
    "Office0": MVP / "paper_revision_audit_v1/critical_adoption/office0/adoption_events.csv",
}

ASSETS = {
    "Room0": MVP / "runtime_reserve_v1/maps/replica_room0_0/logdet",
    "Room1": MVP / "runtime_reserve_v1/maps/replica_room1_0/logdet",
    "Room2": MVP / "runtime_reserve_v1/maps/replica_room2_0/logdet",
    "Office0": MVP / "runtime_reserve_v1/maps/replica_office0_0/logdet",
    "fr1/desk": MVP / "tum_fps_hybrid_v1/maps/freiburg1_desk",
    "fr2/xyz": MVP / "tum_fps_hybrid_v1/maps/freiburg2_xyz",
    "fr3/office": MVP / "tum_fps_hybrid_v1/maps/freiburg3_long_office_household",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(MVP / "paper_revision_audit_v1/summary"))
    parser.add_argument("--block-length", type=int, default=30)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_rows(inputs: list[Path]) -> list[dict]:
    output = []
    aliases = {
        "Room0": "room0", "Room1": "room1", "Room2": "room2", "Office0": "office0",
        "fr1/desk": "tum_fr1", "fr2/xyz": "tum_fr2", "fr3/office": "tum_fr3",
    }
    for scene, (baseline_dir, method_dir, base) in RUNS.items():
        common_path = MVP / f"paper_revision_audit_v1/common_valid/{aliases[scene]}/summary.csv"
        rows = read_csv(common_path)
        inputs.append(common_path)
        method_names = ("baseline", "corrweak128") if base == "observable20" else ("fps10", "corrweak128")
        selected = {}
        for row in rows:
            if row["scope"] != "all" or row["method"] not in method_names:
                continue
            key = (row["method"], row["metric_group"], row["delta_frames"])
            selected.setdefault(key, row)
        baseline_name, method_name = method_names
        base_raw = selected[(baseline_name, "raw_absolute", "")]
        ours_raw = selected[(method_name, "raw_absolute", "")]
        base_aligned = selected[(baseline_name, "aligned_absolute", "")]
        ours_aligned = selected[(method_name, "aligned_absolute", "")]
        base_rpe = selected[(baseline_name, "frame_rpe", "1")]
        ours_rpe = selected[(method_name, "frame_rpe", "1")]
        base_summary, ours_summary = read_json(baseline_dir / "summary.json"), read_json(method_dir / "summary.json")
        inputs.extend((baseline_dir / "summary.json", method_dir / "summary.json"))
        common_base = float(base_raw["translation_rmse_m"])
        common_ours = float(ours_raw["translation_rmse_m"])
        output.append({
            "scene": scene,
            "base_protocol": base,
            "common_frames": int(base_raw["common_frame_count"]),
            "base_valid": int(base_summary["num_valid_frames"]),
            "ours_valid": int(ours_summary["num_valid_frames"]),
            "base_lost": int(base_summary["num_lost_frames"]),
            "ours_lost": int(ours_summary["num_lost_frames"]),
            "common_raw_ate_base_m": f"{common_base:.6f}",
            "common_raw_ate_ours_m": f"{common_ours:.6f}",
            "common_raw_ate_gain_percent": f"{100.0 * (1.0 - common_ours / common_base):.2f}",
            "common_aligned_ate_base_m": f"{float(base_aligned['translation_rmse_m']):.6f}",
            "common_aligned_ate_ours_m": f"{float(ours_aligned['translation_rmse_m']):.6f}",
            "common_raw_rotation_base_deg": f"{float(base_raw['rotation_rmse_deg']):.4f}",
            "common_raw_rotation_ours_deg": f"{float(ours_raw['rotation_rmse_deg']):.4f}",
            "rpe_d1_base_m": f"{float(base_rpe['translation_rmse_m']):.6f}",
            "rpe_d1_ours_m": f"{float(ours_rpe['translation_rmse_m']):.6f}",
            "trigger_percent": f"{100.0 * float(ours_summary.get('observable_reserve_trigger_rate', 0.0)):.2f}",
            "attempts": int(ours_summary.get("num_reserve_attempt_frames", 0)),
            "adoptions": int(ours_summary.get("num_reserve_adopted_frames", 0)),
        })
    return output


def temporal_block_interval(frame_ids: np.ndarray, effects: np.ndarray, block_length: int,
                            reps: int, rng: np.random.Generator) -> tuple[float, float]:
    block_ids = frame_ids // block_length
    groups = [effects[block_ids == value] for value in np.unique(block_ids)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[value] for value in chosen])
        estimates[index] = np.mean(sample)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_rows(inputs: list[Path], block_length: int, reps: int, seed: int) -> list[dict]:
    output = []
    comparators = ("random", "saliency", "runtime_weak")
    for scene_index, (scene, path) in enumerate(PAIRED.items()):
        rows = read_csv(path)
        inputs.append(path)
        for comparator_index, comparator in enumerate(comparators):
            selected = [row for row in rows if int(row["budget"]) == 128
                        and row["metric"] == "translation" and row["comparator"] == comparator]
            frame_ids = np.asarray([int(row["frame_id"]) for row in selected], dtype=np.int64)
            effects = np.asarray([float(row["effect"]) for row in selected], dtype=np.float64)
            rng = np.random.default_rng(seed + 31 * scene_index + comparator_index)
            low, high = temporal_block_interval(frame_ids, effects, block_length, reps, rng)
            output.append({
                "scene": scene,
                "comparator": comparator,
                "budget": 128,
                "pair_count": effects.size,
                "occupied_time_blocks": np.unique(frame_ids // block_length).size,
                "block_length_frames": block_length,
                "mean_effect_mm": f"{1000.0 * np.mean(effects):.3f}",
                "block_ci_low_mm": f"{1000.0 * low:.3f}",
                "block_ci_high_mm": f"{1000.0 * high:.3f}",
                "win_rate_percent": f"{100.0 * np.mean(effects > 0.0):.1f}",
            })
    return output


def resource_rows(inputs: list[Path]) -> list[dict]:
    output = []
    for scene, (baseline_dir, method_dir, base_protocol) in RUNS.items():
        base, ours = read_json(baseline_dir / "summary.json"), read_json(method_dir / "summary.json")
        map_path = ASSETS[scene] / "fsm_map.npz"
        sidecar_path = ASSETS[scene] / "activation_geometry.npz"
        inputs.extend((baseline_dir / "summary.json", method_dir / "summary.json", map_path, sidecar_path))
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            reserve_count = int(np.asarray(sidecar["reserve_global_ids"]).size)
        output.append({
            "scene": scene,
            "base_protocol": base_protocol,
            "base_map_mib": f"{map_path.stat().st_size / 2**20:.2f}",
            "full_sidecar_mib": f"{sidecar_path.stat().st_size / 2**20:.2f}",
            "reserve_count": reserve_count,
            "base_peak_rss_mb": f"{float(base.get('peak_rss_mb', 0.0)):.2f}",
            "ours_peak_rss_mb": f"{float(ours.get('peak_rss_mb', 0.0)):.2f}",
            "rss_delta_mb": f"{float(ours.get('peak_rss_mb', 0.0)) - float(base.get('peak_rss_mb', 0.0)):.2f}",
            "reserve_attempts": int(ours.get("num_reserve_attempt_frames", 0)),
            "reserve_adoptions": int(ours.get("num_reserve_adopted_frames", 0)),
            "attempt_to_adopt_percent": f"{100.0 * float(ours.get('num_reserve_adopted_frames', 0)) / max(1, float(ours.get('num_reserve_attempt_frames', 0))):.2f}",
            "reserve_extra_latency_ms": f"{float(ours.get('observable_reserve_extra_latency_mean_ms', 0.0)):.2f}",
            "corrweak_scoring_latency_ms": f"{float(ours.get('reserve_corr_latency_mean_ms', 0.0)):.2f}",
        })
    return output


def plot_critical_events(output: Path, inputs: list[Path]) -> None:
    windows = {"Room2": (1900, 2000), "Office0": (200, 280)}
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.35))
    for ax, scene in zip(axes, ("Room2", "Office0")):
        baseline_dir, method_dir, _ = RUNS[scene]
        base_errors = read_csv(baseline_dir / "per_frame_errors.csv")
        ours_errors = read_csv(method_dir / "per_frame_errors.csv")
        events = read_csv(CRITICAL[scene])
        inputs.extend((baseline_dir / "per_frame_errors.csv", method_dir / "per_frame_errors.csv", CRITICAL[scene]))
        start, end = windows[scene]
        base_map = {int(row["frame_id"]): row for row in base_errors}
        ours_map = {int(row["frame_id"]): row for row in ours_errors}
        frames = np.arange(start, end)
        base_values = np.asarray([float(base_map[int(frame)]["raw_translation_error_m"] or "nan") for frame in frames])
        ours_values = np.asarray([float(ours_map[int(frame)]["raw_translation_error_m"] or "nan") for frame in frames])
        ax.plot(frames, base_values, color="#666666", lw=1.1, label="compact baseline")
        ax.plot(frames, ours_values, color="#0072B2", lw=1.1, label="Corr-Weak128")
        for row in events:
            frame = int(row["frame_id"])
            if start <= frame < end:
                ax.axvline(frame, color="#D55E00", lw=0.65, alpha=0.75)
        ax.set_title(scene, fontsize=10, fontweight="normal", fontfamily="serif", pad=3)
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Raw translation error (m)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(pad=0.45, h_pad=0.55)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"fig_critical_adoptions.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs: list[Path] = []
    common = common_rows(inputs)
    paired = paired_rows(inputs, args.block_length, args.bootstrap_reps, args.seed)
    resources = resource_rows(inputs)
    write_csv(output / "table_common_frame_metrics.csv", common)
    write_csv(output / "table_paired_block_bootstrap.csv", paired)
    write_csv(output / "table_resource_attempts.csv", resources)
    plot_critical_events(output, inputs)
    unique_inputs = sorted(set(path.resolve() for path in inputs))
    manifest = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique_inputs]
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "common_frame_policy": "pairwise intersection of fail-closed tracking_valid masks",
        "absolute_metrics": "raw and rigid SE(3)-aligned without scale, each on the same common-frame mask",
        "rpe_policy": "raw SE(3) relative error at one-frame delta; intervals crossing invalid frames excluded",
        "paired_estimand": "Corr-Weak candidate improvement minus comparator candidate improvement; positive is better",
        "paired_bootstrap": f"non-overlapping {args.block_length}-frame temporal-block percentile bootstrap",
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "correspondence_calibration_status": "not estimable: frozen runs do not export per-point selected IDs and p_corr",
        "input_count": len(manifest),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_dir": str(output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
