import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
SUBMISSION = ROOT.parent / "C_GLS/ralsubmission"
ASSETS = SUBMISSION / "figure_assets/figure2_assets_v2"
FIGURES = SUBMISSION / "figures/final"
TABLES = MVP / "reviewer_paper_tables_v2"

MAP = MVP / "runtime_reserve_v1/maps/replica_room2_0/logdet/fsm_map.npz"
GEOM = MVP / "runtime_reserve_v1/maps/replica_room2_0/logdet/activation_geometry.npz"
RUN = MVP / "reviewer_ablation_suite_v2/runs/replica_room2_corrweak128/localize/replica_room2_corrweak128"
ADOPTED_FRAME = 66


def reset():
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    for name in ("01_pipeline", "02_chart_visuals", "03_geometry_visuals", "04_source_data"):
        (ASSETS / name).mkdir(parents=True, exist_ok=True)


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def copy_core_assets():
    shutil.copy2(FIGURES / "fig_intervention_pipeline.pdf", ASSETS / "01_pipeline/complete_pipeline.pdf")
    shutil.copy2(FIGURES / "fig_intervention_pipeline.png", ASSETS / "01_pipeline/complete_pipeline.png")
    shutil.copy2(FIGURES / "fig_representative_trajectories.pdf", ASSETS / "02_chart_visuals/mixed_trajectories.pdf")
    shutil.copy2(FIGURES / "fig_representative_trajectories.png", ASSETS / "02_chart_visuals/mixed_trajectories.png")
    shutil.copy2(FIGURES / "fig_intervention_sensitivity.pdf", ASSETS / "02_chart_visuals/budget_normal_sensitivity.pdf")
    shutil.copy2(FIGURES / "fig_intervention_sensitivity.png", ASSETS / "02_chart_visuals/budget_normal_sensitivity.png")


def load_pose(path, frame_id):
    row = np.loadtxt(path)[frame_id]
    position = row[1:4]
    quaternion = row[4:8]
    return position, quaternion


def scene_geometry():
    map_data = np.load(MAP)
    geometry = np.load(GEOM)
    base = map_data["means"]
    reserve = geometry["reserve_means"]
    reserve_ids = geometry["reserve_global_ids"].astype(np.int64)
    events = list(csv.DictReader(open(RUN / "reserve_selection_events.csv", encoding="utf-8")))
    selected_rows = [row for row in events if int(row["frame_id"]) == ADOPTED_FRAME]
    selected_ids = np.array([int(row["selected_global_id"]) for row in selected_rows], dtype=np.int64)
    selected_lookup = {int(identifier): index for index, identifier in enumerate(reserve_ids)}
    selected = reserve[[selected_lookup[int(identifier)] for identifier in selected_ids if int(identifier) in selected_lookup]]
    camera, _ = load_pose(RUN / "trajectory_est.txt", ADOPTED_FRAME)

    center = camera[:2]
    radius = 3.0
    base_mask = np.linalg.norm(base[:, :2] - center, axis=1) < radius
    reserve_mask = np.linalg.norm(reserve[:, :2] - center, axis=1) < radius
    local_base = base[base_mask]
    local_reserve = reserve[reserve_mask]
    rng = np.random.default_rng(20260915)
    if len(local_base) > 25000:
        local_base = local_base[rng.choice(len(local_base), 25000, replace=False)]

    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.scatter(local_base[:, 0], local_base[:, 1], s=0.35, color="#8DB5D8", alpha=0.24, rasterized=True, label="Compact local support")
    axis.scatter(local_reserve[:, 0], local_reserve[:, 1], s=6, facecolors="none", edgecolors="#8E5EA2", linewidths=0.45, alpha=0.65, label="Visible reserve pool")
    axis.scatter(selected[:, 0], selected[:, 1], s=23, color="#E45756", marker="*", linewidths=0.2, edgecolors="#6B1D1B", label="Selected reserve")
    axis.scatter(camera[0], camera[1], s=60, color="#202020", marker="^", label="Estimated camera")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title(f"Room2 adopted intervention, frame {ADOPTED_FRAME}")
    axis.legend(frameon=False, fontsize=7, loc="best")
    axis.grid(True, linewidth=0.35, alpha=0.3)
    save(fig, ASSETS / "03_geometry_visuals", "room2_frame66_support_topdown")

    fig = plt.figure(figsize=(5.2, 4.2))
    axis = fig.add_subplot(111, projection="3d")
    base3 = local_base
    if len(base3) > 10000:
        base3 = base3[rng.choice(len(base3), 10000, replace=False)]
    axis.scatter(base3[:, 0], base3[:, 1], base3[:, 2], s=0.35, color="#8DB5D8", alpha=0.18, rasterized=True)
    axis.scatter(local_reserve[:, 0], local_reserve[:, 1], local_reserve[:, 2], s=5, facecolors="none", edgecolors="#8E5EA2", linewidths=0.35, alpha=0.5)
    axis.scatter(selected[:, 0], selected[:, 1], selected[:, 2], s=28, color="#E45756", marker="*")
    axis.scatter(camera[0], camera[1], camera[2], s=50, color="#202020", marker="^")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Compact support, reserve pool, and adopted proposal")
    save(fig, ASSETS / "03_geometry_visuals", "room2_frame66_support_3d")

    scores = np.array([float(row["selection_score"]) for row in selected_rows])
    compatibility = np.array([float(row["p_corr"]) for row in selected_rows])
    ranks = np.arange(len(scores))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), constrained_layout=True)
    axes[0].plot(ranks, scores, color="#E45756", linewidth=1.25)
    axes[0].fill_between(ranks, scores, color="#E45756", alpha=0.18)
    axes[0].set_xlabel("selected rank")
    axes[0].set_ylabel("Corr-Weak score")
    axes[0].set_title("Proposal score decay")
    axes[1].scatter(compatibility, scores, c=ranks, cmap="viridis_r", s=18, edgecolors="none")
    axes[1].set_xlabel("compatibility")
    axes[1].set_ylabel("Corr-Weak score")
    axes[1].set_title("Compatibility vs final score")
    for axis in axes:
        axis.grid(True, linewidth=0.35, alpha=0.3)
    save(fig, ASSETS / "02_chart_visuals", "room2_frame66_selection_diagnostics")

    with open(ASSETS / "04_source_data/room2_frame66_selected.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected_rows)
    np.savez_compressed(
        ASSETS / "04_source_data/room2_frame66_geometry.npz",
        camera_position=camera,
        local_base_means=local_base,
        visible_reserve_means=local_reserve,
        selected_reserve_means=selected,
        selected_global_ids=selected_ids,
    )


def copy_source_tables():
    for name in (
        "table_closed_loop_interventions.csv",
        "table_budget_sensitivity.csv",
        "table_normal_recomputation.csv",
        "table_first_pose_perturbations.csv",
    ):
        shutil.copy2(TABLES / name, ASSETS / "04_source_data" / name)


def write_readme():
    text = f"""Figure 2 visual asset pack
==========================

This folder separates conceptual workflow artwork from visualizations generated from frozen experiment data.

01_pipeline
  complete_pipeline.pdf/png
  Editable vector pipeline used by the paper. It shows the compact primary path, weak-state branch,
  persistent reserve, Corr-Weak proposal, full normal recomputation, reassociation, candidate solve,
  adopt path, and reject-to-primary path.

02_chart_visuals
  mixed_trajectories.pdf/png
    Four nominal outcomes: Room2, Office0, policy-held-out Office4, and TUM fr3/office.
  budget_normal_sensitivity.pdf/png
    Candidate-budget response and frozen-versus-recomputed normal sensitivity.
  room2_frame66_selection_diagnostics.pdf/png
    Real selected-point score and compatibility diagnostics for an adopted Room2 event.

03_geometry_visuals
  room2_frame66_support_topdown.pdf/png
  room2_frame66_support_3d.pdf/png
    Pure geometric visualizations generated from the frozen Room2 compact map, 4096-element reserve,
    estimated camera pose, and selected global Gaussian IDs at adopted frame {ADOPTED_FRAME}.
    These are data visualizations, not rendered RGB camera images.

04_source_data
  Frozen CSV tables, selected event rows, and a compact NPZ containing the plotted geometry.

Provenance
  Map: {MAP}
  Reserve: {GEOM}
  Run: {RUN}
  Adopted example frame: {ADOPTED_FRAME}

Recommended Figure 2 composition
  Left: compact-map / reserve partition visual from 03_geometry_visuals.
  Center: complete_pipeline.
  Right-top: room2_frame66_selection_diagnostics.
  Right-bottom: budget_normal_sensitivity or mixed_trajectories.

Do not present the geometric panels as photorealistic Gaussian rendering. They show Gaussian centers and
selected support only. The pipeline is conceptual; the chart and geometry panels are generated from frozen data.
"""
    (ASSETS / "README.txt").write_text(text)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    reset()
    copy_core_assets()
    scene_geometry()
    copy_source_tables()
    write_readme()
    print(json.dumps({"asset_root": str(ASSETS), "adopted_frame": ADOPTED_FRAME}, indent=2))


if __name__ == "__main__":
    main()
