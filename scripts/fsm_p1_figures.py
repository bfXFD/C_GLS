import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/ras/xbf/C_GLS_code")
DEFAULT_OUTPUT = Path("/home/ras/xbf/C_GLS/ralsubmission/figures/supplementary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P1 ablation/cost figures for C-GLS reviewer experiments.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if name != "fig_reviewer_analyses_combined":
        fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_table(output_dir: Path, name: str, columns: list[str], rows: list[list[str]], figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.18)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    _save(fig, output_dir, name)


def _gb_label(gamma: float, beta: float) -> str:
    if np.isclose(gamma, 1.0 / 3.0):
        gamma_text = "1/3"
    else:
        gamma_text = f"{gamma:g}"
    return f"({gamma_text}, {beta:g})"


def table_saliency(output_dir: Path) -> None:
    path = ROOT / "experiments/FSM/saliency_exponent_ablation_v1_0/summary_all.csv"
    df = pd.read_csv(path)
    df["ratio"] = df["compression_ratio"].round(2)
    df["setting"] = [_gb_label(g, b) for g, b in zip(df["gamma"], df["beta"])]
    summary = (
        df.groupby(["ratio", "setting"], as_index=False)
        .agg(ate=("ate_rmse", "mean"), rpe=("rpe_rmse", "mean"), occ=("occupied_voxel_ratio", "mean"))
        .sort_values(["ratio", "ate"])
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append([f"{row['ratio']:.2f}", row["setting"], f"{row['ate']:.5f}", f"{row['rpe']:.5f}", f"{row['occ']:.3f}"])
    _save_table(
        output_dir,
        "table_saliency_exponent_ablation",
        ["Ratio", "(gamma, beta)", "Mean ATE", "Mean RPE", "Occ. vox."],
        rows,
        figsize=(5.6, 2.35),
    )


def fig_voxel(output_dir: Path) -> None:
    path = ROOT / "experiments/FSM/voxel_resolution_ablation_v1_0/summary_all.csv"
    df = pd.read_csv(path)
    summary = (
        df.groupby("voxel_multiplier", as_index=False)
        .agg(ate=("ate_rmse", "mean"), rpe=("rpe_rmse", "mean"), occ=("occupied_voxel_ratio", "mean"))
        .sort_values("voxel_multiplier")
    )
    x = np.arange(len(summary))
    labels = [f"{v:g}x" for v in summary["voxel_multiplier"]]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.0))
    axes[0].bar(x, summary["ate"], color="#009E73", edgecolor="#333333", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean ATE")
    axes[0].set_xlabel("Voxel-size multiplier")
    axes[0].set_title("Localization")
    axes[1].bar(x, summary["occ"], color="#56B4E9", edgecolor="#333333", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Occupied-voxel ratio")
    axes[1].set_xlabel("Voxel-size multiplier")
    axes[1].set_title("Coverage")
    for ax, values in zip(axes, [summary["ate"], summary["occ"]]):
        ymax = float(max(values))
        for i, value in enumerate(values):
            ax.text(i, float(value) + ymax * 0.025, f"{float(value):.3f}", ha="center", va="bottom", fontsize=6)
    _save(fig, output_dir, "fig_voxel_resolution_sensitivity")


def table_compression_cost(output_dir: Path) -> None:
    path = ROOT / "experiments/FSM/compression_cost_v1_0/summary_all.csv"
    df = pd.read_csv(path)
    df["ratio"] = df["compression_ratio"].round(2)
    labels = {
        "voxel_scale_opacity": "C-GLS-Voxel",
        "fps_scale_opacity": "FPS-C-GLS",
        "lightgaussian_proxy": "LG proxy",
    }
    summary = (
        df.groupby(["strategy", "ratio"], as_index=False)
        .agg(time=("compression_time_s", "mean"), size=("map_size_mb", "mean"))
        .sort_values(["ratio", "strategy"])
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append([labels.get(row["strategy"], row["strategy"]), f"{row['ratio']:.2f}", f"{row['time']:.2f} s", f"{row['size']:.2f} MB"])
    _save_table(
        output_dir,
        "table_compression_cost",
        ["Method", "Ratio", "Offline cost", "Map size"],
        rows,
        figsize=(4.7, 1.8),
    )


def table_replica_per_scene(output_dir: Path) -> None:
    scene_paths = {
        "room0_0": ROOT / "experiments/FSM/replica_room0_0/batch_500_pareto_v1_1/summary_all.csv",
        "room1_0": ROOT / "experiments/FSM/replica_room1_0/batch_500_pareto_v1_1/summary_all.csv",
        "room2_0": ROOT / "experiments/FSM/replica_room2_0/batch_500_pareto_v1_1/summary_all.csv",
        "office0_0": ROOT / "experiments/FSM/replica_office0_0/batch_500_pareto_v1_1/summary_all.csv",
    }
    methods = [
        ("full", 1.0, "Full"),
        ("random", 0.2, "Random 20%"),
        ("voxel_scale_opacity", 0.2, "C-GLS-Voxel 20%"),
        ("fps_scale_opacity", 0.2, "FPS-C-GLS 20%"),
    ]
    rows = []
    for scene, path in scene_paths.items():
        df = pd.read_csv(path)
        row = [scene]
        for strategy, ratio, _label in methods:
            sub = df[(df["strategy"] == strategy) & np.isclose(df["compression_ratio"], ratio)]
            if sub.empty:
                row.extend(["--", "--"])
                continue
            item = sub.iloc[0]
            row.extend([f"{item['ate_rmse']:.5f}", f"{item['rpe_rmse']:.5f}"])
        rows.append(row)
    columns = ["Scene"]
    for _strategy, _ratio, label in methods:
        columns.extend([f"{label} ATE", f"{label} RPE"])
    _save_table(output_dir, "table_replica_per_scene_results", columns, rows, figsize=(7.2, 1.65))


def fig_reviewer_analyses_combined(output_dir: Path) -> None:
    sal = pd.read_csv(ROOT / "experiments/FSM/saliency_exponent_ablation_v1_0/summary_all.csv")
    vox = pd.read_csv(ROOT / "experiments/FSM/voxel_resolution_ablation_v1_0/summary_all.csv")
    cost = pd.read_csv(ROOT / "experiments/FSM/compression_cost_v1_0/summary_all.csv")
    root = ROOT / "experiments/FSM"
    scenes = ["room0_0", "room1_0", "room2_0", "office0_0"]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.72))
    ax_sal, ax_vox, ax_cost, ax_long = axes.ravel()

    sal["ratio"] = sal["compression_ratio"].round(2)
    sal["setting"] = [_gb_label(g, b) for g, b in zip(sal["gamma"], sal["beta"])]
    order = ["(1/3, 1)", "(1, 0)", "(0, 1)", "(1, 1)", "(1/3, 0.5)"]
    sal_summary = sal.groupby(["ratio", "setting"], as_index=False).agg(ate=("ate_rmse", "mean"))
    x = np.arange(len(order))
    width = 0.36
    for offset, ratio, color in [(-width / 2, 0.10, "#56B4E9"), (width / 2, 0.20, "#009E73")]:
        vals = [float(sal_summary[(sal_summary["ratio"] == ratio) & (sal_summary["setting"] == setting)]["ate"].iloc[0]) for setting in order]
        ax_sal.bar(x + offset, vals, width=width, label=f"{ratio:.1f}", color=color, edgecolor="#333333", linewidth=0.6)
    ax_sal.set_title("A. Saliency exponents")
    ax_sal.set_ylabel("Mean ATE", labelpad=2)
    ax_sal.set_xticks(x)
    ax_sal.set_xticklabels(order, rotation=25, ha="right")
    ax_sal.legend(title="Ratio", frameon=True, ncol=2, fontsize=6, title_fontsize=6)

    vox_summary = vox.groupby("voxel_multiplier", as_index=False).agg(ate=("ate_rmse", "mean"), occ=("occupied_voxel_ratio", "mean"))
    labels = [f"{v:g}x" for v in vox_summary["voxel_multiplier"]]
    x = np.arange(len(labels))
    ax_vox.bar(x, vox_summary["ate"], color="#009E73", edgecolor="#333333", linewidth=0.6, label="ATE")
    ax_vox.set_title("B. Voxel resolution")
    ax_vox.set_ylabel("Mean ATE", labelpad=1)
    ax_vox.set_xticks(x)
    ax_vox.set_xticklabels(labels)
    ax_vox.set_xlabel("Voxel multiplier")
    ax_vox_occ = ax_vox.twinx()
    ax_vox_occ.plot(x, vox_summary["occ"], color="#0072B2", marker="o", linewidth=1.4, label="Coverage")
    ax_vox_occ.set_ylabel("Occupied-voxel ratio", labelpad=2)
    ax_vox_occ.set_ylim(0.93, 1.0)

    cost["ratio"] = cost["compression_ratio"].round(2)
    cost_summary = cost.groupby(["strategy", "ratio"], as_index=False).agg(time=("compression_time_s", "mean"))
    methods = [("lightgaussian_proxy", "LG proxy"), ("voxel_scale_opacity", "C-GLS-V"), ("fps_scale_opacity", "FPS-C")]
    x = np.arange(len(methods))
    for offset, ratio, color in [(-width / 2, 0.10, "#56B4E9"), (width / 2, 0.20, "#009E73")]:
        vals = [float(cost_summary[(cost_summary["strategy"] == strategy) & (cost_summary["ratio"] == ratio)]["time"].iloc[0]) for strategy, _ in methods]
        ax_cost.bar(x + offset, vals, width=width, color=color, edgecolor="#333333", linewidth=0.6, label=f"{ratio:.1f}")
    ax_cost.set_title("C. Offline compression cost")
    ax_cost.set_ylabel("Time (s)", labelpad=2)
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([label for _, label in methods])
    ax_cost.legend(title="Ratio", frameon=True, ncol=2, fontsize=6, title_fontsize=6)

    rel_rows = []
    for scene in scenes:
        d500 = pd.read_csv(root / f"replica_{scene}/batch_500_pareto_v1_1/summary_all.csv")
        d2000 = pd.read_csv(root / f"replica_{scene}/batch_2000_cgls_validation_v1_0/summary_all.csv")
        f500 = d500[(d500["strategy"] == "full") & np.isclose(d500["compression_ratio"], 1.0)].iloc[0]
        f2000 = d2000[(d2000["strategy"] == "full") & np.isclose(d2000["compression_ratio"], 1.0)].iloc[0]
        for strategy, label in [("random", "Random20"), ("voxel_scale_opacity", "C-GLS20")]:
            r500 = d500[(d500["strategy"] == strategy) & np.isclose(d500["compression_ratio"], 0.2)].iloc[0]
            r2000 = d2000[(d2000["strategy"] == strategy) & np.isclose(d2000["compression_ratio"], 0.2)].iloc[0]
            rel_rows.append(
                {
                    "method": label,
                    "ate500": float(r500["ate_rmse"] / f500["ate_rmse"]),
                    "ate2000": float(r2000["ate_rmse"] / f2000["ate_rmse"]),
                    "speed500": float(f500["latency_mean_ms"] / r500["latency_mean_ms"]),
                    "speed2000": float(f2000["latency_mean_ms"] / r2000["latency_mean_ms"]),
                }
            )
    rel = pd.DataFrame(rel_rows).groupby("method", as_index=False).mean(numeric_only=True)
    methods_long = ["Random20", "C-GLS20"]
    x = np.arange(len(methods_long))
    vals500 = [float(rel[rel["method"] == method]["ate500"].iloc[0]) for method in methods_long]
    vals2000 = [float(rel[rel["method"] == method]["ate2000"].iloc[0]) for method in methods_long]
    ax_long.bar(x - width / 2, vals500, width=width, color="#BDBDBD", edgecolor="#333333", linewidth=0.6, label="0-500")
    ax_long.bar(x + width / 2, vals2000, width=width, color="#0072B2", edgecolor="#333333", linewidth=0.6, label="0-2000")
    ax_long.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    ax_long.set_title("D. Long-horizon validation")
    ax_long.set_ylabel("ATE / full", labelpad=2)
    ax_long.set_xticks(x)
    ax_long.set_xticklabels(methods_long)
    ax_long.legend(frameon=True, ncol=2, fontsize=6)
    ax_long.set_ylim(0, 1.75)
    for idx, method in enumerate(methods_long):
        speed = float(rel[rel["method"] == method]["speed2000"].iloc[0])
        y_text = min(max(vals500[idx], vals2000[idx]) + 0.07, 1.62)
        ax_long.text(idx, y_text, f"{speed:.1f}x", ha="center", va="bottom", fontsize=6, clip_on=True)

    for ax in [ax_sal, ax_vox, ax_cost, ax_long]:
        ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.105, top=0.93, hspace=0.52, wspace=0.62)
    _save(fig, output_dir, "fig_reviewer_analyses_combined")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _style()
    table_saliency(output_dir)
    fig_voxel(output_dir)
    table_compression_cost(output_dir)
    table_replica_per_scene(output_dir)
    fig_reviewer_analyses_combined(output_dir)
    print(f"Wrote P1 figures to {output_dir}")


if __name__ == "__main__":
    main()
