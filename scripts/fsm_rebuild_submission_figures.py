import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
TABLES = MVP / "reviewer_paper_tables_v2"
OUT = ROOT.parent / "C_GLS/ralsubmission/figures/final"
ROOM2_MAP = MVP / "runtime_reserve_v1/maps/replica_room2_0/logdet/fsm_map.npz"
ROOM2_GEOMETRY = MVP / "runtime_reserve_v1/maps/replica_room2_0/logdet/activation_geometry.npz"
ROOM2_RUN = MVP / "reviewer_ablation_suite_v2/runs/replica_room2_corrweak128/localize/replica_room2_corrweak128"
ROOM2_EXAMPLE_FRAME = 66

COLORS = {
    "gt": "#202020",
    "base": "#4C78A8",
    "ours": "#E45756",
    "accent": "#F2A541",
    "good": "#2A9D8F",
    "neutral": "#7A7A7A",
    "bad": "#8E5EA2",
}


def load_trajectory(path):
    rows = np.loadtxt(path)
    return rows[:, 1:4]


def trajectory_path(suite, case, name):
    return MVP / suite / "runs" / case / "localize" / case / name


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def representative_trajectories():
    scenes = [
        ("Room2", "+2.93 pp JTS", "reviewer_ablation_suite_v2", "replica_room2", ROOT / "experiments/FSM/audit_v2_full_rerun/main/replica_room2_0/full/maps/full_ratio_1/fsm_map.npz"),
        ("Office0", "+0.02 pp JTS", "reviewer_ablation_suite_v2", "replica_office0", ROOT / "experiments/FSM/audit_v2_full_rerun/main/replica_office0_0/full/maps/full_ratio_1/fsm_map.npz"),
        ("Office4 held-out", "-1.22 pp JTS", "reviewer_ablation_suite_v2", "replica_office4", ROOT.parent / "SplaTAM/experiments/Replica/office4_0/params.npz"),
        ("TUM fr3/office", "+0.21 pp JTS", "reviewer_ablation_suite_v2_tum", "tum_freiburg3_long_office_household", ROOT / "experiments/FSM/audit_v2_full_rerun/main/tum_freiburg3_long_office_household/full/maps/full_ratio_1/fsm_map.npz"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.35), constrained_layout=True)
    for axis, (title, delta, suite, prefix, map_path) in zip(axes, scenes):
        baseline_case = f"{prefix}_baseline"
        corrweak_case = f"{prefix}_corrweak128"
        gt = load_trajectory(trajectory_path(suite, baseline_case, "trajectory_gt.txt"))
        baseline = load_trajectory(trajectory_path(suite, baseline_case, "trajectory_est.txt"))
        corrweak = load_trajectory(trajectory_path(suite, corrweak_case, "trajectory_est.txt"))
        map_data = np.load(map_path)
        map_points = map_data["means"] if "means" in map_data.files else map_data["means3D"]
        horizontal = map_points[:, [0, 2]]
        trajectories = np.concatenate([gt[:, [0, 2]], baseline[:, [0, 2]], corrweak[:, [0, 2]]], axis=0)
        lower = np.minimum(np.percentile(horizontal, 1, axis=0), trajectories.min(axis=0))
        upper = np.maximum(np.percentile(horizontal, 99, axis=0), trajectories.max(axis=0))
        margin = np.maximum((upper - lower) * 0.04, 0.05)
        lower -= margin
        upper += margin
        density, x_edges, z_edges = np.histogram2d(
            horizontal[:, 0], horizontal[:, 1], bins=220,
            range=[[lower[0], upper[0]], [lower[1], upper[1]]],
        )
        density = np.log1p(density.T)
        positive = density[density > 0]
        if positive.size:
            density = np.clip((density - np.percentile(positive, 15)) / max(np.percentile(positive, 99) - np.percentile(positive, 15), 1e-9), 0, 1)
        axis.imshow(
            density, origin="lower", extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
            cmap="Greys", vmin=0, vmax=1, alpha=0.58, interpolation="bilinear", zorder=0,
        )
        axis.plot(gt[:, 0], gt[:, 2], color=COLORS["gt"], linewidth=1.15, label="GT", zorder=3)
        axis.plot(
            baseline[:, 0], baseline[:, 2], color=COLORS["base"], linewidth=1.0,
            linestyle="--", label="Compact", zorder=2,
        )
        axis.plot(
            corrweak[:, 0], corrweak[:, 2], color=COLORS["ours"], linewidth=1.0,
            label="Intervention", zorder=2,
        )
        axis.scatter(baseline[0, 0], baseline[0, 2], color=COLORS["base"], marker="o", s=10, zorder=4)
        axis.scatter(corrweak[0, 0], corrweak[0, 2], color=COLORS["ours"], marker="s", s=10, zorder=4)
        axis.set_title(f"{title}\n{delta}", fontsize=8.5, pad=3)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, linewidth=0.3, alpha=0.22)
        axis.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig_representative_trajectories")


def add_box(axis, xy, width, height, text, face, edge="#444444", fontsize=7.1, linewidth=0.9):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(axis, start, end, color="#444444", style="-", label=None, label_offset=(0, 0)):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.95, color=color, linestyle=style)
    axis.add_patch(patch)
    if label:
        midpoint = ((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1])
        axis.text(
            *midpoint, label, ha="center", va="center", fontsize=6.2, color="#333333",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
            zorder=5,
        )


def intervention_pipeline():
    map_data = np.load(ROOM2_MAP)
    geometry = np.load(ROOM2_GEOMETRY)
    base = map_data["means"]
    reserve = geometry["reserve_means"]
    reserve_ids = geometry["reserve_global_ids"].astype(np.int64)
    events = list(csv.DictReader(open(ROOM2_RUN / "reserve_selection_events.csv", encoding="utf-8")))
    selected_rows = [row for row in events if int(row["frame_id"]) == ROOM2_EXAMPLE_FRAME]
    selected_ids = np.array([int(row["selected_global_id"]) for row in selected_rows], dtype=np.int64)
    lookup = {int(identifier): index for index, identifier in enumerate(reserve_ids)}
    selected = reserve[[lookup[int(identifier)] for identifier in selected_ids if int(identifier) in lookup]]
    camera = np.loadtxt(ROOM2_RUN / "trajectory_est.txt")[ROOM2_EXAMPLE_FRAME, 1:4]
    local_base = base[np.linalg.norm(base[:, :2] - camera[:2], axis=1) < 3.0]
    local_reserve = reserve[np.linalg.norm(reserve[:, :2] - camera[:2], axis=1) < 3.0]
    rng = np.random.default_rng(20260915)
    if len(local_base) > 12000:
        local_base = local_base[rng.choice(len(local_base), 12000, replace=False)]

    fig = plt.figure(figsize=(7.15, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, 0.9])
    visual = fig.add_subplot(grid[0, 0])
    flow = fig.add_subplot(grid[0, 1])
    charts = grid[0, 2].subgridspec(2, 1, hspace=0.48)
    score_axis = fig.add_subplot(charts[0, 0])
    compat_axis = fig.add_subplot(charts[1, 0])

    visual.scatter(local_base[:, 0], local_base[:, 1], s=0.35, color="#8DB5D8", alpha=0.24, rasterized=True, label="Compact")
    visual.scatter(local_reserve[:, 0], local_reserve[:, 1], s=5, facecolors="none", edgecolors="#8E5EA2", linewidths=0.35, alpha=0.6, label="Reserve")
    visual.scatter(selected[:, 0], selected[:, 1], s=22, color="#E45756", marker="*", linewidths=0.2, edgecolors="#6B1D1B", label="Selected")
    visual.scatter(camera[0], camera[1], s=45, color="#202020", marker="^", label="Camera")
    visual.set_aspect("equal", adjustable="datalim")
    visual.set_title(f"(a) Real support proposal\nRoom2 frame {ROOM2_EXAMPLE_FRAME}", fontsize=8)
    visual.set_xlabel("world x (m)")
    visual.set_ylabel("world y (m)")
    visual.grid(True, linewidth=0.3, alpha=0.3)
    visual.legend(frameon=False, fontsize=5.8, loc="best", ncol=2, handletextpad=0.3, columnspacing=0.5)
    visual.tick_params(labelsize=6)

    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    flow.set_title("(b) State-isolated target reconstruction", fontsize=8, pad=2)
    add_box(flow, (0.03, 0.74), 0.25, 0.13, "Compact target\nprimary solve", "#D9EAF7", fontsize=6.4)
    add_box(flow, (0.38, 0.74), 0.22, 0.13, "Weak-state\ndiagnostic", "#FFF1CC", fontsize=6.4)
    add_box(flow, (0.70, 0.74), 0.26, 0.13, "Primary accepted\nor recovered", "#E8EEF2", fontsize=6.4)
    add_box(flow, (0.03, 0.42), 0.25, 0.14, "Disjoint reserve\nCorr-Weak proposal", "#E5D9F2", fontsize=6.4)
    add_box(flow, (0.38, 0.42), 0.22, 0.14, "Rebuild target\nnormals + matches", "#F8D4BC", fontsize=6.4)
    add_box(flow, (0.70, 0.42), 0.26, 0.14, "Candidate solve\nfrom primary pose", "#F4C4A5", fontsize=6.4)
    add_box(flow, (0.26, 0.10), 0.22, 0.13, "Tracking + V4\nlocal gate", "#F9E7B2", fontsize=6.4)
    add_box(flow, (0.56, 0.10), 0.18, 0.13, "Reject\nretain primary", "#E8EEF2", fontsize=6.2)
    add_box(flow, (0.80, 0.10), 0.17, 0.13, "Adopt\ncandidate", "#D9EFE7", fontsize=6.2)

    # Primary lane: no crossings.
    arrow(flow, (0.28, 0.805), (0.38, 0.805))
    arrow(flow, (0.60, 0.805), (0.70, 0.805), label="not triggered", label_offset=(0, 0.055))

    # Triggered branch uses the empty channel between the top and middle rows.
    flow.plot([0.49, 0.49, 0.16], [0.74, 0.64, 0.64], color="#444444", linewidth=0.95, zorder=1)
    arrow(flow, (0.16, 0.64), (0.16, 0.56))
    flow.text(
        0.35, 0.64, "triggered", ha="center", va="center", fontsize=6.2,
        color="#333333", bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
        zorder=5,
    )

    # Candidate reconstruction lane.
    arrow(flow, (0.28, 0.49), (0.38, 0.49))
    arrow(flow, (0.60, 0.49), (0.70, 0.49))
    flow.plot([0.83, 0.83, 0.37], [0.42, 0.31, 0.31], color="#444444", linewidth=0.95, zorder=1)
    arrow(flow, (0.37, 0.31), (0.37, 0.23))

    # Gate outcomes occupy separate bottom channels and never return through the primary lane.
    arrow(flow, (0.48, 0.165), (0.56, 0.165), label="reject", label_offset=(0, 0.05))
    flow.plot([0.37, 0.37, 0.885], [0.10, 0.055, 0.055], color="#444444", linewidth=0.95, zorder=1)
    arrow(flow, (0.885, 0.055), (0.885, 0.10), label="adopt", label_offset=(0.05, 0.0))
    flow.text(0.50, 0.005, "Only adoption replaces the active pose history", ha="center", fontsize=5.8, color="#555555")

    scores = np.array([float(row["selection_score"]) for row in selected_rows])
    compatibility = np.array([float(row["p_corr"]) for row in selected_rows])
    ranks = np.arange(len(scores))
    score_axis.plot(ranks, scores, color="#E45756", linewidth=1.0)
    score_axis.fill_between(ranks, scores, color="#E45756", alpha=0.16)
    score_axis.set_title("(c) Selected-score decay", fontsize=7.5)
    score_axis.set_xlabel("rank", fontsize=6.5)
    score_axis.set_ylabel("score", fontsize=6.5)
    compat_axis.scatter(compatibility, scores, c=ranks, cmap="viridis_r", s=10, edgecolors="none")
    compat_axis.set_title("(d) Compatibility vs score", fontsize=7.5)
    compat_axis.set_xlabel("compatibility", fontsize=6.5)
    compat_axis.set_ylabel("score", fontsize=6.5)
    for axis in (score_axis, compat_axis):
        axis.grid(True, linewidth=0.3, alpha=0.3)
        axis.tick_params(labelsize=5.8)
    save(fig, "fig_intervention_pipeline")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def intervention_sensitivity():
    budget_rows = read_rows(TABLES / "table_budget_sensitivity.csv")
    normal_rows = read_rows(TABLES / "table_normal_recomputation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)

    budgets = [32, 64, 128, 256]
    markers = {"room2": "o", "office0": "s", "office4": "^"}
    labels = {"room2": "Room2", "office0": "Office0", "office4": "Office4 held-out"}
    scene_colors = {"room2": COLORS["good"], "office0": COLORS["accent"], "office4": COLORS["bad"]}
    for scene in ("room2", "office0", "office4"):
        rows = [row for row in budget_rows if row["scene"] == scene]
        baseline = float(next(row for row in rows if row["budget"] == "baseline")["jts_percent"])
        values = [
            float(next(row for row in rows if row["budget"] != "baseline" and int(row["budget"]) == budget)["jts_percent"])
            - baseline
            for budget in budgets
        ]
        axes[0].plot(
            budgets, values, marker=markers[scene], markersize=4.5, linewidth=1.4,
            color=scene_colors[scene], label=labels[scene],
        )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].axvline(128, color="#777777", linewidth=0.7, linestyle=":")
    axes[0].text(128, axes[0].get_ylim()[1] * 0.82, "frozen K", rotation=90, fontsize=6.7, ha="right", va="top")
    axes[0].set_xticks(budgets)
    axes[0].set_xlabel("Candidate budget K")
    axes[0].set_ylabel("Delta JTS vs compact (pp)")
    axes[0].set_title("(a) Non-monotonic budget response", fontsize=8.5)
    axes[0].grid(True, linewidth=0.35, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=6.7, loc="best")

    scenes = ["room2", "office0", "office4", "fr3"]
    display = ["Room2", "Office0", "Office4", "fr3"]
    x = np.arange(len(scenes))
    width = 0.34
    trans_ratios, rot_ratios, jts_deltas = [], [], []
    for scene in scenes:
        recomputed = next(row for row in normal_rows if row["scene"] == scene and row["normals"] == "recomputed")
        frozen = next(row for row in normal_rows if row["scene"] == scene and row["normals"] == "frozen")
        trans_ratios.append(float(frozen["common_translation_rmse_m"]) / float(recomputed["common_translation_rmse_m"]))
        rot_ratios.append(float(frozen["common_rotation_rmse_deg"]) / float(recomputed["common_rotation_rmse_deg"]))
        jts_deltas.append(float(frozen["jts_percent"]) - float(recomputed["jts_percent"]))
    axes[1].bar(x - width / 2, trans_ratios, width, color=COLORS["base"], label="Translation RMSE")
    axes[1].bar(x + width / 2, rot_ratios, width, color=COLORS["accent"], hatch="//", label="Rotation RMSE")
    axes[1].axhline(1, color="#333333", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, display)
    axes[1].set_ylabel("Frozen / recomputed RMSE (x)")
    axes[1].set_title("(b) Effect of freezing target normals", fontsize=8.5)
    axes[1].grid(True, axis="y", which="both", linewidth=0.35, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=6.7, loc="lower right")
    annotation_factors = [1.18, 1.28, 1.16, 1.55]
    for index, delta in enumerate(jts_deltas):
        height = max(trans_ratios[index], rot_ratios[index])
        axes[1].text(
            index, height * annotation_factors[index], f"dJTS {delta:+.2f}",
            ha="center", va="bottom", fontsize=6.2,
        )
    axes[1].set_ylim(0.8, max(max(trans_ratios), max(rot_ratios)) * 2.1)
    save(fig, "fig_intervention_sensitivity")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    representative_trajectories()
    intervention_pipeline()
    intervention_sensitivity()
    print(f"Wrote submission figures to {OUT}")


if __name__ == "__main__":
    main()
