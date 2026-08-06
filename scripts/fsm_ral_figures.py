import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/ras/xbf/C_GLS_code")
FSM_ROOT = Path("/home/ras/xbf/C_GLS")


METHOD_LABELS = {
    "full": "Full",
    "random": "Random",
    "scale_large": "Scale-only",
    "voxel_grid_pure": "Voxel-only",
    "voxel_scale_opacity": "C-GLS-Voxel (ours)",
    "fps_scale_opacity": "FPS-C-GLS (ours)",
}

COLORS = {
    "Full": "#4D4D4D",
    "Random": "#999999",
    "Scale-only": "#E69F00",
    "Voxel-only": "#CC79A7",
    "C-GLS-Voxel (ours)": "#009E73",
    "FPS-C-GLS (ours)": "#0072B2",
}

BAR_EDGE = "#333333"
OTHER_COLOR = "#56B4E9"
BEFORE_COLOR = "#BDBDBD"

MARKERS = {
    "Full": "*",
    "Random": "o",
    "Scale-only": "s",
    "Voxel-only": "^",
    "C-GLS-Voxel (ours)": "D",
    "FPS-C-GLS (ours)": "P",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RA-L paper figures for C-GLS experiments.")
    parser.add_argument("--output-dir", default="/home/ras/xbf/C_GLS/ralsubmission/figures/main")
    parser.add_argument("--latex-pack", default="/home/ras/xbf/Latex_pack")
    parser.add_argument("--prepare-template", action="store_true")
    return parser.parse_args()


def _ensure_style() -> None:
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
    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _save_table_figure(output_dir: Path, name: str, columns: list[str], rows: list[list[str]], figsize: tuple[float, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.22)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
        if row > 0 and col == 0 and "C-GLS" in cell.get_text().get_text():
            cell.set_facecolor("#E5F5EF")
        if row > 0 and col == 0 and "FPS" in cell.get_text().get_text():
            cell.set_facecolor("#E6F0F8")
    fig.tight_layout(pad=0.15)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _method_label(strategy: str) -> str:
    return METHOD_LABELS.get(strategy, strategy)


def prepare_ral_template(latex_pack: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for archive_name in ["ieeeconf.zip", "IEEEtranBST.zip"]:
        archive = latex_pack / archive_name
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(output_root)


def plot_replica_pareto(output_dir: Path) -> None:
    df = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    df = df[df["strategy"] != "full"].copy()
    strategies = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45))
    for strategy in strategies:
        group = df[df["strategy"] == strategy].groupby("compression_ratio", as_index=False).mean(numeric_only=True)
        group = group.sort_values("compression_ratio", ascending=False)
        label = _method_label(strategy)
        axes[0].plot(
            group["speedup_over_full"],
            group["ate_over_full"],
            marker=MARKERS[label],
            color=COLORS[label],
            linewidth=1.4,
            markersize=4,
            label=label,
        )
        axes[1].plot(
            group["size_reduction_over_full"],
            group["ate_over_full"],
            marker=MARKERS[label],
            color=COLORS[label],
            linewidth=1.4,
            markersize=4,
            label=label,
        )
    for ax, xlabel in zip(axes, ["CPU speedup over full", "Map-size reduction over full"]):
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ATE / full")
        ax.set_ylim(0.75, 2.2)
    axes[0].set_title("Accuracy vs. speed")
    axes[1].set_title("Accuracy vs. storage")
    axes[1].legend(loc="upper left", frameon=True, ncol=1)
    _save(fig, output_dir, "fig_replica_pareto")


def plot_replica_main_operating_point(output_dir: Path) -> None:
    df = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    methods = [
        ("full", 1.0, "Full"),
        ("random", 0.2, "Random"),
        ("scale_large", 0.2, "Scale"),
        ("voxel_grid_pure", 0.2, "Voxel"),
        ("voxel_scale_opacity", 0.2, "C-GLS\n(ours)"),
        ("fps_scale_opacity", 0.2, "FPS-CGLS\n(ours)"),
    ]
    rows = []
    for strategy, ratio, label in methods:
        sub = df[(df["strategy"] == strategy) & np.isclose(df["compression_ratio"], ratio)]
        rows.append(
            {
                "label": label,
                "ate_over_full": sub["ate_over_full"].mean(),
                "speedup_over_full": sub["speedup_over_full"].mean(),
                "size_reduction_over_full": sub["size_reduction_over_full"].mean(),
                "strategy": strategy,
            }
        )
    data = pd.DataFrame(rows)
    colors = [COLORS[_method_label(strategy)] for strategy in data["strategy"]]
    edgecolors = [BAR_EDGE for _ in data["strategy"]]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=True)
    panels = [
        ("ate_over_full", "ATE / full\n(lower is better)", 1.0, 1.75),
        ("speedup_over_full", "CPU speedup\n(higher is better)", None, 5.5),
        ("size_reduction_over_full", "Map reduction\n(higher is better)", None, 5.5),
    ]
    for ax, (metric, ylabel, ref, ymax) in zip(axes, panels):
        ax.bar(x, data[metric], color=colors, edgecolor=edgecolors, linewidth=1.0)
        if ref is not None:
            ax.axhline(ref, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(data["label"], rotation=25, ha="right")
        ax.set_ylim(0, ymax)
        for idx, value in enumerate(data[metric]):
            ax.text(idx, value + ymax * 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=6)
    axes[0].set_title("Accuracy")
    axes[1].set_title("Runtime")
    axes[2].set_title("Storage")
    fig.suptitle("Replica 4-scene main operating point", y=1.02, fontsize=9)
    fig.subplots_adjust(bottom=0.28, top=0.82)
    _save(fig, output_dir, "fig_replica_main_operating_point")


def _plot_ratio_curve(
    output_dir: Path,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    y_ref: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    df = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    strategies = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for strategy in strategies:
        group = df[(df["strategy"] == strategy) & (df["compression_ratio"] < 1.0)].groupby("compression_ratio", as_index=False).mean(numeric_only=True)
        group = group.sort_values("compression_ratio")
        label = _method_label(strategy)
        ax.plot(
            group["compression_ratio"] * 100.0,
            group[metric],
            marker=MARKERS[label],
            color=COLORS[label],
            linewidth=1.45,
            markersize=4.2,
            label=label,
        )
    if y_ref is not None:
        ax.axhline(y_ref, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Retention ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xlim(4, 26)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=True, fontsize=6, loc="best")
    _save(fig, output_dir, filename)


def plot_replica_ratio_curves(output_dir: Path) -> None:
    _plot_ratio_curve(
        output_dir=output_dir,
        metric="ate_over_full",
        ylabel="ATE / full (lower is better)",
        title="Accuracy under compression",
        filename="fig_replica_ratio_ate",
        y_ref=1.0,
        ylim=(0.8, 2.4),
    )
    _plot_ratio_curve(
        output_dir=output_dir,
        metric="speedup_over_full",
        ylabel="CPU speedup over full",
        title="Runtime under compression",
        filename="fig_replica_ratio_speedup",
        y_ref=1.0,
        ylim=(0.0, 14.0),
    )
    _plot_ratio_curve(
        output_dir=output_dir,
        metric="size_reduction_over_full",
        ylabel="Map-size reduction over full",
        title="Storage under compression",
        filename="fig_replica_ratio_storage",
        y_ref=1.0,
        ylim=(0.0, 12.0),
    )


def plot_replica_ratio_curves_combined(output_dir: Path) -> None:
    df = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    strategies = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
    panels = [
        ("ate_over_full", "ATE / full\n(lower is better)", "Accuracy", (0.8, 2.4), 1.0),
        ("speedup_over_full", "CPU speedup", "Runtime", (0.0, 14.0), 1.0),
        ("size_reduction_over_full", "Map-size reduction", "Storage", (0.0, 12.0), 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharex=True)
    for ax, (metric, ylabel, title, ylim, y_ref) in zip(axes, panels):
        for strategy in strategies:
            group = df[(df["strategy"] == strategy) & (df["compression_ratio"] < 1.0)].groupby("compression_ratio", as_index=False).mean(numeric_only=True)
            group = group.sort_values("compression_ratio")
            label = _method_label(strategy)
            ax.plot(
                group["compression_ratio"] * 100.0,
                group[metric],
                marker=MARKERS[label],
                color=COLORS[label],
                linewidth=1.35,
                markersize=3.8,
                label=label,
            )
        ax.axhline(y_ref, color="#333333", linestyle="--", linewidth=0.75)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(4, 26)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Retention (%)")
    axes[1].legend(frameon=True, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3)
    _save(fig, output_dir, "fig_replica_ratio_curves_combined")


def _plot_grouped_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    y_ref: float | None = None,
    show_legend: bool = False,
) -> None:
    strategies = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
    ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    x = np.arange(len(ratios))
    total_width = 0.78
    width = total_width / len(strategies)
    offsets = (np.arange(len(strategies)) - (len(strategies) - 1) / 2.0) * width
    for offset, strategy in zip(offsets, strategies):
        group = data[data["strategy"] == strategy].groupby("compression_ratio", as_index=False).mean(numeric_only=True)
        values = []
        for ratio in ratios:
            row = group[np.isclose(group["compression_ratio"], ratio)]
            values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        label = _method_label(strategy)
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=COLORS[label],
            edgecolor=BAR_EDGE,
            linewidth=0.45,
            label=label,
        )
        if metric == "ate_over_full":
            for bar, value in zip(bars, values):
                if np.isfinite(value) and value > ylim[1]:
                    ax.text(bar.get_x() + bar.get_width() / 2, ylim[1] * 0.86, f"{value:.2f}", ha="center", va="top", fontsize=6, color=BAR_EDGE)
                    ax.text(bar.get_x() + bar.get_width() / 2, ylim[1] * 0.98, "//", ha="center", va="top", fontsize=7, color=BAR_EDGE)
    if y_ref is not None:
        ax.axhline(y_ref, color="#222222", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["5", "10", "15", "20", "25"])
    ax.set_xlabel("Retention ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(loc="upper right", frameon=True, fontsize=6, ncol=1)


def plot_replica_grouped_bars(output_dir: Path) -> None:
    df = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    df = df[(df["strategy"] != "full") & (df["compression_ratio"] < 1.0)].copy()
    single_specs = [
        ("ate_over_full", "ATE / full (lower is better)", "Accuracy under compression", (0.0, 2.4), 1.0, "fig_replica_grouped_ate"),
        ("speedup_over_full", "CPU speedup over full", "Runtime under compression", (0.0, 14.0), 1.0, "fig_replica_grouped_speedup"),
        ("size_reduction_over_full", "Map-size reduction over full", "Storage under compression", (0.0, 12.0), 1.0, "fig_replica_grouped_storage"),
    ]
    for metric, ylabel, title, ylim, y_ref, filename in single_specs:
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        _plot_grouped_bar(ax, df, metric, ylabel, title, ylim, y_ref=y_ref, show_legend=True)
        _save(fig, output_dir, filename)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.92), sharex=True)
    combined_specs = [
        ("ate_over_full", "ATE / full\n(lower is better)", "Accuracy", (0.0, 2.4), 1.0),
        ("speedup_over_full", "CPU speedup", "Runtime", (0.0, 14.0), 1.0),
        ("size_reduction_over_full", "Map-size reduction", "Storage", (0.0, 12.0), 1.0),
    ]
    handles = None
    labels = None
    for idx, (ax, spec) in enumerate(zip(axes, combined_specs)):
        metric, ylabel, title, ylim, y_ref = spec
        _plot_grouped_bar(ax, df, metric, ylabel, title, ylim, y_ref=y_ref, show_legend=False)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=True, fontsize=6, borderaxespad=0.3)
    fig.subplots_adjust(bottom=0.20, right=0.70, wspace=0.48)
    _save(fig, output_dir, "fig_replica_grouped_bars_combined")


def _read_decoupling() -> pd.DataFrame:
    rows = []
    for scene in ["room0_0", "room1_0", "room2_0", "office0_0"]:
        path = ROOT / f"experiments/FSM/replica_{scene}/render_decoupling_v1_0_e/decoupling_summary.csv"
        if path.exists():
            data = pd.read_csv(path)
            data["scene"] = scene
            rows.append(data)
    return pd.concat(rows, ignore_index=True)


def plot_render_localization(output_dir: Path) -> None:
    df = _read_decoupling()
    psnr_col = "psnr" if "psnr" in df.columns else "psnr_mean"
    fig, ax = plt.subplots(figsize=(3.45, 1.78))
    for _, row in df.iterrows():
        name = str(row.get("method", row.get("name", "")))
        if "Full" in name or str(row.get("strategy", "")) == "full":
            label = "Full"
        elif "C_GLS" in name or "voxel_scale_opacity" in name:
            label = "C-GLS-Voxel (ours)"
        elif "random" in name.lower():
            label = "Random"
        elif "scale" in name.lower():
            label = "Scale-only"
        else:
            label = "Other compressed maps"
        color = COLORS.get(label, OTHER_COLOR)
        ax.scatter(row[psnr_col], row["ate_rmse"], s=22, color=color, alpha=0.85)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("ATE RMSE")
    ax.set_title("Rendering fidelity is not sufficient")
    handles = []
    for label, color in [("Full", COLORS["Full"]), ("C-GLS-Voxel (ours)", COLORS["C-GLS-Voxel (ours)"]), ("Random", COLORS["Random"]), ("Scale-only", COLORS["Scale-only"]), ("Other compressed maps", OTHER_COLOR)]:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=4))
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=6)
    _save(fig, output_dir, "fig_render_localization_mismatch")


def _read_skeleton() -> pd.DataFrame:
    rows = []
    for scene in ["room0_0", "room1_0", "room2_0", "office0_0"]:
        path = ROOT / f"experiments/FSM/replica_{scene}/skeleton_analysis_v1_1.csv"
        if path.exists():
            data = pd.read_csv(path)
            data["scene"] = scene
            rows.append(data)
    return pd.concat(rows, ignore_index=True)


def plot_skeleton_mechanism(output_dir: Path) -> None:
    df = _read_skeleton()
    methods = ["random_0p1", "voxel_grid_pure_0p1", "voxel_scale_opacity_0p1", "scale_large_0p05", "voxel_scale_opacity_0p05"]
    labels = ["Random\n10%", "Voxel\n10%", "C-GLS\n10%", "Scale\n5%", "C-GLS\n5%"]
    group = df[df["method"].isin(methods)].groupby("method", as_index=False).mean(numeric_only=True)
    group = group.set_index("method").loc[methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.15))
    skel_colors = [COLORS["Random"], COLORS["Voxel-only"], COLORS["C-GLS-Voxel (ours)"], COLORS["Scale-only"], COLORS["C-GLS-Voxel (ours)"]]
    axes[0].bar(x, group["occupied_voxels"], color=skel_colors, edgecolor=BAR_EDGE, linewidth=0.45)
    axes[0].set_ylabel("Occupied voxels")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Spatial coverage")
    axes[1].bar(x, group["scale_mean"], color=skel_colors, edgecolor=BAR_EDGE, linewidth=0.45)
    axes[1].set_ylabel("Mean scale")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Structural saliency")
    _save(fig, output_dir, "fig_skeleton_mechanism")


def plot_quantization(output_dir: Path) -> None:
    replica = pd.read_csv(ROOT / "experiments/FSM/quantization_v1_1_summary.csv")
    tum = pd.read_csv(ROOT / "experiments/FSM/tum_quantization_v1_2_summary.csv")
    rows = []
    for ratio, label in [(0.2, "Replica\n20%"), (0.15, "Replica\n15%")]:
        sub = replica[np.isclose(replica["compression_ratio"], ratio)]
        rows.append(
            {
                "label": label,
                "base": sub["base_map_size_mb"].mean(),
                "quantized": sub["quantized_map_size_mb"].mean(),
                "ate_rel": sub["ate_relative"].mean(),
            }
        )
    for _, row in tum.iterrows():
        rows.append(
            {
                "label": row["scene"].replace("freiburg", "fr").replace("_", "\n"),
                "base": row["base_map_size_mb"],
                "quantized": row["quantized_map_size_mb"],
                "ate_rel": row["ate_relative"],
            }
        )
    qdf = pd.DataFrame(rows)
    x = np.arange(len(qdf))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45))
    axes[0].bar(x - width / 2, qdf["base"], width, label="Before", color=BEFORE_COLOR, edgecolor=BAR_EDGE, linewidth=0.45)
    axes[0].bar(x + width / 2, qdf["quantized"], width, label="After", color=COLORS["C-GLS-Voxel (ours)"], edgecolor=BAR_EDGE, linewidth=0.45)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(qdf["label"])
    axes[0].set_ylabel("Map size (MB)")
    axes[0].set_title("Storage after safe quantization")
    axes[0].legend(frameon=True)
    axes[1].bar(x, qdf["ate_rel"], color=COLORS["FPS-C-GLS (ours)"], edgecolor=BAR_EDGE, linewidth=0.45)
    axes[1].axhline(1.0, color="#444444", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(qdf["label"])
    axes[1].set_ylabel("ATE after / before")
    axes[1].set_title("Negligible localization change")
    axes[1].set_ylim(0.995, 1.005)
    _save(fig, output_dir, "fig_quantization")


def plot_tum_sampler(output_dir: Path) -> None:
    tum = pd.read_csv(ROOT / "experiments/FSM/tum_pareto_v1_1_summary.csv")
    extra = pd.read_csv(ROOT / "experiments/FSM/tum_freiburg3_long_office_household/batch_500_tum_extra_v1_2/summary_all.csv")
    extra_full = extra[(extra["strategy"] == "full") & np.isclose(extra["compression_ratio"], 1.0)].iloc[0]
    extra = extra[np.isclose(extra["compression_ratio"], 0.1)].copy()
    extra["scene"] = "freiburg3_long_office_household"
    extra["ate_over_full"] = extra["ate_rmse"] / float(extra_full["ate_rmse"])
    data = pd.concat([tum[np.isclose(tum["compression_ratio"], 0.1)], extra], ignore_index=True)
    strategies = ["random", "scale_large", "voxel_grid_pure", "voxel_scale_opacity", "fps_scale_opacity"]
    pivot = data[data["strategy"].isin(strategies)].pivot_table(index="scene", columns="strategy", values="ate_over_full", aggfunc="mean")
    scenes = list(pivot.index)
    x = np.arange(len(scenes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    for i, strategy in enumerate(strategies):
        label = _method_label(strategy)
        ax.bar(x + (i - 2) * width, pivot[strategy], width, color=COLORS[label], label=label)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("freiburg", "fr").replace("_long_office_household", " office") for s in scenes])
    ax.set_ylabel("ATE / full at 10%")
    ax.set_title("TUM real-data sampler comparison")
    ax.legend(frameon=True, ncol=3, loc="upper right")
    _save(fig, output_dir, "fig_tum_sampler")


def _load_robustness_rows() -> pd.DataFrame:
    rows = []
    for scene in ["room0_0", "room1_0", "room2_0", "office0_0"]:
        for method in ["full_ratio_1", "voxel_scale_opacity_ratio_0p2", "random_ratio_0p2"]:
            for init in ["previous_estimate", "identity"]:
                path = ROOT / f"experiments/FSM/replica_{scene}/init_robustness_v1_2/{method}_{init}/localize/summary.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "init": init,
                        "ate_rmse": data["ate_rmse"],
                        "latency_mean_ms": data["latency_mean_ms"],
                    }
                )
    return pd.DataFrame(rows)


def plot_init_stress(output_dir: Path) -> None:
    df = _load_robustness_rows()
    label_map = {
        "full_ratio_1": "Full",
        "voxel_scale_opacity_ratio_0p2": "C-GLS 20%\n(ours)",
        "random_ratio_0p2": "Random 20%",
    }
    group = df.groupby(["method", "init"], as_index=False).mean(numeric_only=True)
    methods = ["full_ratio_1", "voxel_scale_opacity_ratio_0p2", "random_ratio_0p2"]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    prev = [group[(group["method"] == m) & (group["init"] == "previous_estimate")]["ate_rmse"].iloc[0] for m in methods]
    ident = [group[(group["method"] == m) & (group["init"] == "identity")]["ate_rmse"].iloc[0] for m in methods]
    ax.bar(x - width / 2, prev, width, label="Previous estimate", color="#1b9e77")
    ax.bar(x + width / 2, ident, width, label="Identity", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Mean ATE")
    ax.set_title("Initialization stress test")
    ax.legend(frameon=True)
    _save(fig, output_dir, "fig_init_stress")


def generate_table_figures(output_dir: Path) -> None:
    replica = pd.read_csv(ROOT / "experiments/FSM/pareto_v1_1_summary.csv")
    main_specs = [
        ("Full", "full", 1.0),
        ("Scale-only", "scale_large", 0.20),
        ("Voxel-only", "voxel_grid_pure", 0.20),
        ("Random", "random", 0.20),
        ("C-GLS-Voxel (ours)", "voxel_scale_opacity", 0.20),
        ("FPS-C-GLS (ours)", "fps_scale_opacity", 0.20),
        ("C-GLS-Voxel (ours)", "voxel_scale_opacity", 0.15),
    ]
    rows = []
    for label, strategy, ratio in main_specs:
        sub = replica[(replica["strategy"] == strategy) & np.isclose(replica["compression_ratio"], ratio)]
        rows.append(
            [
                label,
                f"{ratio:.2f}",
                f"{sub['ate_rmse'].mean():.5f}",
                f"{sub['rpe_rmse'].mean():.5f}",
                f"{sub['ate_over_full'].mean():.3f}",
                f"{sub['speedup_over_full'].mean():.2f}x",
                f"{sub['size_reduction_over_full'].mean():.2f}x",
            ]
        )
    _save_table_figure(
        output_dir,
        "table_replica_main_results",
        ["Method", "Ratio", "ATE", "RPE", "ATE/Full", "Speedup", "Size Red."],
        rows,
        figsize=(7.1, 2.25),
    )

    tum = pd.read_csv(ROOT / "experiments/FSM/tum_pareto_v1_1_summary.csv")
    extra = pd.read_csv(ROOT / "experiments/FSM/tum_freiburg3_long_office_household/batch_500_tum_extra_v1_2/summary_all.csv")
    extra_full = extra[(extra["strategy"] == "full") & np.isclose(extra["compression_ratio"], 1.0)].iloc[0]
    tum_specs = [
        ("fr1/desk", tum, "freiburg1_desk", "full", 1.0),
        ("fr1/desk", tum, "freiburg1_desk", "fps_scale_opacity", 0.10),
        ("fr2/xyz", tum, "freiburg2_xyz", "full", 1.0),
        ("fr2/xyz", tum, "freiburg2_xyz", "fps_scale_opacity", 0.10),
    ]
    rows = []
    for scene_label, frame, scene, strategy, ratio in tum_specs:
        sub = frame[(frame["scene"] == scene) & (frame["strategy"] == strategy) & np.isclose(frame["compression_ratio"], ratio)]
        label = "Full" if strategy == "full" else "FPS-C-GLS (ours)"
        rows.append(
            [
                scene_label,
                label,
                f"{ratio:.2f}",
                f"{sub['ate_rmse'].mean():.5f}",
                f"{sub['rpe_rmse'].mean():.5f}",
                f"{sub['ate_over_full'].mean():.3f}",
                f"{sub['size_reduction_over_full'].mean():.2f}x",
            ]
        )
    fr3_fps = extra[(extra["strategy"] == "fps_scale_opacity") & np.isclose(extra["compression_ratio"], 0.10)].iloc[0]
    rows.append(["fr3/office", "Full", "1.00", f"{float(extra_full['ate_rmse']):.5f}", f"{float(extra_full['rpe_rmse']):.5f}", "1.000", "1.00x"])
    rows.append(["fr3/office", "FPS-C-GLS (ours)", "0.10", f"{float(fr3_fps['ate_rmse']):.5f}", f"{float(fr3_fps['rpe_rmse']):.5f}", f"{float(fr3_fps['ate_rmse']) / float(extra_full['ate_rmse']):.3f}", f"{float(extra_full['map_size_mb']) / float(fr3_fps['map_size_mb']):.2f}x"])
    _save_table_figure(
        output_dir,
        "table_tum_realworld_results",
        ["Scene", "Method", "Ratio", "ATE", "RPE", "ATE/Full", "Size Red."],
        rows,
        figsize=(7.1, 2.3),
    )

    replica_q = pd.read_csv(ROOT / "experiments/FSM/quantization_v1_1_summary.csv")
    tum_q = pd.read_csv(ROOT / "experiments/FSM/tum_quantization_v1_2_summary.csv")
    rows = []
    for ratio, label in [(0.20, "Replica C-GLS 20%"), (0.15, "Replica C-GLS 15%")]:
        sub = replica_q[np.isclose(replica_q["compression_ratio"], ratio)]
        rows.append([label, f"{sub['base_ate_rmse'].mean():.5f}", f"{sub['quantized_ate_rmse'].mean():.5f}", f"{sub['base_rpe_rmse'].mean():.5f}", f"{sub['quantized_rpe_rmse'].mean():.5f}", f"{sub['ate_relative'].mean():.3f}", f"{sub['extra_size_reduction'].mean():.2f}x"])
    for _, row in tum_q.iterrows():
        rows.append([row["scene"].replace("freiburg", "fr"), f"{row['base_ate_rmse']:.5f}", f"{row['quantized_ate_rmse']:.5f}", f"{row['base_rpe_rmse']:.5f}", f"{row['quantized_rpe_rmse']:.5f}", f"{row['ate_relative']:.3f}", f"{row['extra_size_reduction']:.2f}x"])
    _save_table_figure(
        output_dir,
        "table_quantization_results",
        ["Setting", "ATE w/o Quant.", "ATE w/ Quant.", "RPE w/o Quant.", "RPE w/ Quant.", "ATE Ratio", "Extra Size Red."],
        rows,
        figsize=(7.1, 1.95),
    )

    robust = _load_robustness_rows()
    robust_specs = [
        ("Full", "full_ratio_1", "previous_estimate"),
        ("C-GLS 20% (ours)", "voxel_scale_opacity_ratio_0p2", "previous_estimate"),
        ("Random 20%", "random_ratio_0p2", "previous_estimate"),
        ("Full", "full_ratio_1", "identity"),
        ("C-GLS 20% (ours)", "voxel_scale_opacity_ratio_0p2", "identity"),
        ("Random 20%", "random_ratio_0p2", "identity"),
    ]
    rows = []
    for label, method, init in robust_specs:
        sub = robust[(robust["method"] == method) & (robust["init"] == init)]
        init_label = "Prev. pose" if init == "previous_estimate" else "Identity"
        rows.append([label, init_label, f"{sub['ate_rmse'].mean():.5f}", f"{sub['latency_mean_ms'].mean():.1f} ms"])
    _save_table_figure(
        output_dir,
        "table_init_stress_results",
        ["Map", "Init mode", "Mean ATE", "Latency"],
        rows,
        figsize=(5.9, 2.0),
    )


def main() -> None:
    args = parse_args()
    _ensure_style()
    output_dir = Path(args.output_dir)
    if args.prepare_template:
        prepare_ral_template(Path(args.latex_pack), output_dir.parent)
    plot_replica_grouped_bars(output_dir)
    plot_render_localization(output_dir)
    plot_skeleton_mechanism(output_dir)
    plot_quantization(output_dir)
    plot_tum_sampler(output_dir)
    generate_table_figures(output_dir)
    print(f"Wrote RA-L figures to {output_dir}")


if __name__ == "__main__":
    main()
