import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MVP = ROOT / "experiments/FSM/mvp_observable_v1"

REPLICA = {
    "Room0": (
        MVP / "core_route_v11_oomfix/room0_baseline/localize/baseline/summary.json",
        MVP / "core_route_v11_oomfix/room0_corrweak/localize/corrweak128/summary.json",
    ),
    "Room1": (
        MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/baseline/summary.json",
        MVP / "room12_generalization_2000_v1/closed_loop/room1/localize/corrweak128/summary.json",
    ),
    "Room2": (
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/baseline/summary.json",
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/corrweak128/summary.json",
    ),
    "Office0": (
        MVP / "core_route_v11_oomfix/office0_baseline/localize/baseline/summary.json",
        MVP / "core_route_v11_oomfix/office0_corrweak/localize/corrweak128/summary.json",
    ),
}

TUM = {
    "fr1/desk": MVP / "tum_fps_hybrid_v1/freiburg1_desk/closed_loop",
    "fr2/xyz": MVP / "tum_fps_hybrid_v1/freiburg2_xyz/closed_loop",
    "fr3/office": MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop",
}

TIMING = {
    "Office0": ("office0", "baseline", "corrweak", "baseline", "corrweak128"),
    "TUM fr2": ("tum_fr2", "baseline", "corrweak", "fps10", "fps_corrweak128"),
    "TUM fr3": ("tum_fr3", "baseline", "corrweak", "fps10", "fps_corrweak128"),
}

TRAJECTORIES = {
    "Replica Room2": (
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/baseline",
        MVP / "room12_generalization_2000_v1/closed_loop/room2/localize/corrweak128",
    ),
    "Replica Office0": (
        MVP / "core_route_v11_oomfix/office0_baseline/localize/baseline",
        MVP / "core_route_v11_oomfix/office0_corrweak/localize/corrweak128",
    ),
    "TUM fr3/office": (
        MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps10",
        MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/localize/fps_corrweak128",
    ),
}

PAIRED = {
    "Room0": MVP / "core_route_v2/room0_0_500_paired/localize/paired_core/paired_frame_stats/summary.json",
    "Room1": MVP / "room1_paired_late_v1/paired_stats/room1/summary.json",
    "Room2": MVP / "room12_generalization_v1/paired_stats/room2/summary.json",
    "Office0": MVP / "core_route_v5/office0_0_500_paired/localize/paired_core/paired_frame_stats/summary.json",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate final MVP paper tables, figures, and provenance.")
    parser.add_argument("--output-dir", default=str(MVP / "paper_summary_v1"))
    return parser.parse_args(argv)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=0.0):
    value = row.get(key, default)
    return float(default if value in (None, "") else value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value):
    return str(value).replace("%", r"\%").replace("_", r"\_")


def write_latex(path, rows, columns):
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(latex_escape(label) for _, label in columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(row[key]) for key, _ in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def replica_rows(inputs):
    rows = []
    for scene, (baseline_path, method_path) in REPLICA.items():
        baseline, method = read_json(baseline_path), read_json(method_path)
        inputs.extend((baseline_path, method_path))
        ate_base, ate_method = baseline["raw_ate_rmse_m"], method["raw_ate_rmse_m"]
        rows.append({
            "scene": scene,
            "retention_percent": "20",
            "baseline_valid": baseline["num_valid_frames"],
            "method_valid": method["num_valid_frames"],
            "baseline_lost": baseline["num_lost_frames"],
            "method_lost": method["num_lost_frames"],
            "baseline_ate_m": f"{ate_base:.6f}",
            "method_ate_m": f"{ate_method:.6f}",
            "ate_improvement_percent": f"{100.0 * (1.0 - ate_method / ate_base):.2f}",
            "baseline_rotation_deg": f"{baseline['raw_rotation_abs_rmse_deg']:.4f}",
            "method_rotation_deg": f"{method['raw_rotation_abs_rmse_deg']:.4f}",
            "trigger_rate_percent": f"{100.0 * method.get('observable_reserve_trigger_rate', 0.0):.2f}",
            "adoptions": method.get("num_reserve_adopted_frames", 0),
        })
    return rows


def tum_rows(inputs):
    rows = []
    for scene, root in TUM.items():
        summary_path = root / "summary_all.csv"
        values = {row["case"]: row for row in read_csv(summary_path)}
        inputs.append(summary_path)
        baseline, method = values["fps10"], values["fps_corrweak128"]
        ate_base, ate_method = number(baseline, "raw_ate_rmse_m"), number(method, "raw_ate_rmse_m")
        rows.append({
            "scene": scene,
            "retention_percent": "10",
            "baseline_valid": int(number(baseline, "valid_frames")),
            "method_valid": int(number(method, "valid_frames")),
            "baseline_ate_m": f"{ate_base:.6f}",
            "method_ate_m": f"{ate_method:.6f}",
            "ate_improvement_percent": f"{100.0 * (1.0 - ate_method / ate_base):.2f}",
            "baseline_rotation_deg": f"{number(baseline, 'raw_rotation_abs_rmse_deg'):.4f}",
            "method_rotation_deg": f"{number(method, 'raw_rotation_abs_rmse_deg'):.4f}",
            "trigger_rate_percent": f"{100.0 * number(method, 'observable_reserve_trigger_rate'):.2f}",
            "adoptions": int(number(method, "num_reserve_adopted_frames")),
        })
    return rows


def timing_rows(inputs):
    rows = []
    timing_root = MVP / "timing_v1"
    for scene, (prefix, baseline_token, method_token, baseline_case, method_case) in TIMING.items():
        aggregates = {}
        for label, token, case in (("baseline", baseline_token, baseline_case), ("method", method_token, method_case)):
            means, p95s, p50s = [], [], []
            for repeat in (1, 2, 3):
                root = timing_root / f"{prefix}_repeat{repeat}_{token}"
                summary_path = root / "summary_all.csv"
                frame_path = root / "localize" / case / "frame_metrics.csv"
                summary = read_csv(summary_path)[0]
                frames = read_csv(frame_path)
                values = [number(row, "production_equivalent_total_frame_latency_ms") for row in frames]
                means.append(number(summary, "production_equivalent_total_frame_latency_mean_ms"))
                p95s.append(number(summary, "production_equivalent_total_frame_latency_p95_ms"))
                p50s.append(float(np.median(values)))
                inputs.extend((summary_path, frame_path))
            aggregates[label] = {
                "mean": np.mean(means), "mean_sd": np.std(means, ddof=1),
                "p50": np.mean(p50s), "p50_sd": np.std(p50s, ddof=1),
                "p95": np.mean(p95s), "p95_sd": np.std(p95s, ddof=1),
            }
        base, method = aggregates["baseline"], aggregates["method"]
        rows.append({
            "scene": scene,
            "baseline_mean_ms": f"{base['mean']:.2f}",
            "baseline_mean_sd_ms": f"{base['mean_sd']:.2f}",
            "method_mean_ms": f"{method['mean']:.2f}",
            "method_mean_sd_ms": f"{method['mean_sd']:.2f}",
            "mean_change_percent": f"{100.0 * (method['mean'] / base['mean'] - 1.0):.2f}",
            "baseline_p50_ms": f"{base['p50']:.2f}",
            "method_p50_ms": f"{method['p50']:.2f}",
            "baseline_p95_ms": f"{base['p95']:.2f}",
            "method_p95_ms": f"{method['p95']:.2f}",
        })
    return rows


def covariance_rows(inputs):
    rows = []
    root = MVP / "tum_covariance_map_order_v1"
    for scene in ("freiburg2_xyz", "freiburg3_long_office_household"):
        path = root / scene / "summary_all.csv"
        inputs.append(path)
        for row in read_csv(path):
            rows.append({
                "scene": scene,
                "method": row["case"],
                "valid_frames": int(number(row, "valid_frames")),
                "lost_frames": int(number(row, "lost_frames")),
                "raw_ate_m": f"{number(row, 'raw_ate_rmse_m'):.6f}",
                "rotation_deg": f"{number(row, 'raw_rotation_abs_rmse_deg'):.4f}",
            })
    return rows


def paired_rows(inputs):
    rows = []
    for scene, path in PAIRED.items():
        values = read_json(path)
        inputs.append(path)
        selected = {
            row["metric"]: row for row in values
            if row["budget"] == 128 and row["comparator"] == "random"
        }
        rows.append({
            "scene": scene,
            "pair_count": selected["translation"]["n_candidate_pairs"],
            "translation_effect_mm": f"{1000.0 * selected['translation']['mean_effect']:.3f}",
            "translation_ci_low_mm": f"{1000.0 * selected['translation']['mean_effect_ci_low']:.3f}",
            "translation_ci_high_mm": f"{1000.0 * selected['translation']['mean_effect_ci_high']:.3f}",
            "rotation_effect_deg": f"{selected['rotation']['mean_effect']:.5f}",
            "rotation_ci_low_deg": f"{selected['rotation']['mean_effect_ci_low']:.5f}",
            "rotation_ci_high_deg": f"{selected['rotation']['mean_effect_ci_high']:.5f}",
        })
    return rows


def trigger_rows(inputs):
    path = MVP / "tum_fps_trigger_ablation_v1/freiburg3_long_office_household/summary_all.csv"
    inputs.append(path)
    labels = {"fps10": "FPS baseline", "gate_mid": "Gate mid", "gate_strict": "Gate strict"}
    rows = []
    for row in read_csv(path):
        rows.append({
            "method": labels[row["case"]],
            "trigger_rate_percent": f"{100.0 * number(row, 'observable_reserve_trigger_rate'):.2f}",
            "adoptions": int(number(row, "num_reserve_adopted_frames")),
            "raw_ate_m": f"{number(row, 'raw_ate_rmse_m'):.6f}",
            "rotation_deg": f"{number(row, 'raw_rotation_abs_rmse_deg'):.4f}",
            "latency_ms": f"{number(row, 'production_equivalent_total_frame_latency_mean_ms'):.2f}",
        })
    frozen_path = MVP / "tum_fps_hybrid_v1/freiburg3_long_office_household/closed_loop/summary_all.csv"
    inputs.append(frozen_path)
    frozen = next(row for row in read_csv(frozen_path) if row["case"] == "fps_corrweak128")
    rows.insert(1, {
        "method": "Frozen V4",
        "trigger_rate_percent": f"{100.0 * number(frozen, 'observable_reserve_trigger_rate'):.2f}",
        "adoptions": int(number(frozen, "num_reserve_adopted_frames")),
        "raw_ate_m": f"{number(frozen, 'raw_ate_rmse_m'):.6f}",
        "rotation_deg": f"{number(frozen, 'raw_rotation_abs_rmse_deg'):.4f}",
        "latency_ms": f"{number(frozen, 'production_equivalent_total_frame_latency_mean_ms'):.2f}",
    })
    return rows


def load_positions(path):
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    return values[:, 1:4]


def plot_main(replica, tum, output):
    rows = replica + tum
    labels = [row["scene"] for row in rows]
    improvements = [float(row["ate_improvement_percent"]) for row in rows]
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in improvements]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(labels, improvements, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Raw ATE improvement (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Closed-loop improvement over matched compact baseline")
    for index, value in enumerate(improvements):
        ax.text(index, value + (1.5 if value >= 0 else -1.5), f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    save_figure(fig, output, "fig_main_ate_improvement")


def plot_timing(rows, output):
    labels = [row["scene"] for row in rows]
    baseline = [float(row["baseline_mean_ms"]) for row in rows]
    method = [float(row["method_mean_ms"]) for row in rows]
    x = np.arange(len(rows)); width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - width / 2, baseline, width, label="Compact baseline", color="#999999")
    ax.bar(x + width / 2, method, width, label="Corr-Weak128", color="#0072B2")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Production-equivalent latency (ms)")
    ax.set_title("CPU latency over three repeated runs")
    ax.legend(frameon=False)
    save_figure(fig, output, "fig_timing_repeats")


def plot_covariance(rows, output):
    scenes = ["freiburg2_xyz", "freiburg3_long_office_household"]
    methods = ["full", "fps10", "observable10"]
    labels = ["Full", "FPS 10%", "Observable 10%"]
    colors = ["#666666", "#009E73", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, scene in zip(axes, scenes):
        lookup = {(row["scene"], row["method"]): float(row["raw_ate_m"]) for row in rows}
        ax.bar(labels, [lookup[(scene, method)] for method in methods], color=colors)
        ax.set_yscale("log")
        ax.set_title(scene.replace("freiburg", "fr"))
        ax.tick_params(axis="x", rotation=22)
    axes[0].set_ylabel("Raw ATE (m, log scale)")
    fig.suptitle("Map ordering under covariance-weighted registration")
    save_figure(fig, output, "fig_second_backend_map_order")


def plot_paired(rows, output):
    labels = [row["scene"] for row in rows]
    translation = [float(row["translation_effect_mm"]) for row in rows]
    rotation = [float(row["rotation_effect_deg"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, values, ylabel in zip(axes, (translation, rotation), ("Translation effect vs random (mm)", "Rotation effect vs random (deg)")):
        ax.bar(labels, values, color=["#0072B2" if value >= 0 else "#D55E00" for value in values])
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Paired Corr-Weak128 effect; positive is better")
    save_figure(fig, output, "fig_replica_paired_effect")


def plot_trigger(rows, output):
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    x = [float(row["latency_ms"]) for row in rows]
    y = [float(row["raw_ate_m"]) for row in rows]
    rates = [float(row["trigger_rate_percent"]) for row in rows]
    scatter = ax.scatter(x, y, c=rates, cmap="viridis", s=70, edgecolor="#333333")
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(row["method"], (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Production-equivalent latency (ms)")
    ax.set_ylabel("Raw ATE (m)")
    ax.set_title("fr3 trigger efficiency ablation")
    fig.colorbar(scatter, ax=ax, label="Trigger rate (%)")
    save_figure(fig, output, "fig_trigger_accuracy_latency")


def plot_trajectories(inputs, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    for ax, (scene, (baseline, method)) in zip(axes, TRAJECTORIES.items()):
        gt_path = baseline / "trajectory_gt.txt"
        baseline_path = baseline / "trajectory_est.txt"
        method_path = method / "trajectory_est.txt"
        inputs.extend((gt_path, baseline_path, method_path))
        gt, first, second = map(load_positions, (gt_path, baseline_path, method_path))
        ax.plot(gt[:, 0], gt[:, 1], color="#222222", linewidth=1.2, label="GT")
        ax.plot(first[:, 0], first[:, 1], color="#999999", linewidth=0.9, label="Baseline")
        ax.plot(second[:, 0], second[:, 1], color="#0072B2", linewidth=0.9, label="Corr-Weak128")
        ax.set_title(scene)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, output, "fig_representative_trajectories")


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    replica = replica_rows(inputs)
    tum = tum_rows(inputs)
    timing = timing_rows(inputs)
    covariance = covariance_rows(inputs)
    paired = paired_rows(inputs)
    trigger = trigger_rows(inputs)
    write_csv(output / "table_replica_main.csv", replica)
    write_csv(output / "table_tum_hybrid.csv", tum)
    write_csv(output / "table_timing.csv", timing)
    write_csv(output / "table_second_backend.csv", covariance)
    write_csv(output / "table_replica_paired.csv", paired)
    write_csv(output / "table_trigger_ablation.csv", trigger)
    write_latex(output / "table_replica_main.tex", replica, (("scene", "Scene"), ("baseline_ate_m", "Base ATE"), ("method_ate_m", "Ours ATE"), ("ate_improvement_percent", "Gain (percent)"), ("baseline_lost", "Base lost"), ("method_lost", "Ours lost")))
    write_latex(output / "table_tum_hybrid.tex", tum, (("scene", "Scene"), ("baseline_ate_m", "FPS ATE"), ("method_ate_m", "Hybrid ATE"), ("ate_improvement_percent", "Gain (percent)"), ("trigger_rate_percent", "Trigger (percent)")))
    write_latex(output / "table_timing.tex", timing, (("scene", "Scene"), ("baseline_mean_ms", "Base mean"), ("method_mean_ms", "Ours mean"), ("mean_change_percent", "Change (percent)"), ("baseline_p95_ms", "Base p95"), ("method_p95_ms", "Ours p95")))
    write_latex(output / "table_replica_paired.tex", paired, (("scene", "Scene"), ("pair_count", "Frames"), ("translation_effect_mm", "Trans. effect (mm)"), ("rotation_effect_deg", "Rot. effect (deg)")))
    write_latex(output / "table_trigger_ablation.tex", trigger, (("method", "Method"), ("trigger_rate_percent", "Trigger (percent)"), ("adoptions", "Adopt"), ("raw_ate_m", "ATE"), ("latency_ms", "Latency")))
    plot_main(replica, tum, output)
    plot_timing(timing, output)
    plot_covariance(covariance, output)
    plot_paired(paired, output)
    plot_trigger(trigger, output)
    plot_trajectories(inputs, output)
    unique_inputs = sorted(set(path.resolve() for path in inputs))
    manifest = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique_inputs]
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "replica_scene_count": len(replica),
        "tum_scene_count": len(tum),
        "timing_scene_count": len(timing),
        "second_backend_row_count": len(covariance),
        "paired_scene_count": len(paired),
        "trigger_ablation_row_count": len(trigger),
        "input_file_count": len(manifest),
        "output_dir": str(output.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
