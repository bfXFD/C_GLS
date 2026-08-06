import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PARAMS = Path("/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz")
DEFAULT_MAP = Path(
    "/home/ras/xbf/C_GLS_code/experiments/FSM/replica_room0_0/"
    "batch_500_pareto_v1_1/maps/voxel_scale_opacity_ratio_0p2/fsm_map.npz"
)
DEFAULT_RGB = Path("/home/ras/xbf/SplaTAM/data/Replica/room0/results/frame000250.jpg")
DEFAULT_OUTPUT = Path("/home/ras/xbf/C_GLS/ralsubmission/figures/source_assets/legacy_scene_map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an RGB scene and Gaussian-map visualization for the C-GLS paper.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _sample_indices(count: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    sample_count = min(count, max(1, int(round(count * fraction))))
    return np.sort(rng.choice(count, size=sample_count, replace=False))


def _fit_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = axes[:2].T
    projected = centered @ projection
    # Keep the dominant scene extent horizontal for a compact paper panel.
    if np.ptp(projected[:, 1]) > np.ptp(projected[:, 0]):
        projection = projection[:, [1, 0]]
    return center, projection


def _project(points: np.ndarray, center: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return (points - center) @ projection


def _plot_map(
    ax: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    projection: np.ndarray,
    limits: tuple[float, float, float, float],
    title: str,
) -> None:
    projected = _project(points, center, projection)
    ax.set_facecolor("#111318")
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=np.clip(colors, 0.0, 1.0),
        s=0.18,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.6)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be in (0, 1]")

    params = np.load(args.params, allow_pickle=True)
    skeleton = np.load(args.map, allow_pickle=True)
    full_points = np.asarray(params["means3D"], dtype=np.float32)
    full_colors = np.asarray(params["rgb_colors"], dtype=np.float32)
    skeleton_indices = np.asarray(skeleton["indices"], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    full_keep = _sample_indices(len(full_points), args.sample_fraction, rng)
    skeleton_keep = _sample_indices(len(skeleton_indices), args.sample_fraction, rng)
    full_sample = full_points[full_keep]
    full_color_sample = full_colors[full_keep]
    skeleton_sample = full_points[skeleton_indices[skeleton_keep]]
    skeleton_color_sample = full_colors[skeleton_indices[skeleton_keep]]

    center, projection = _fit_projection(full_sample)
    projected_full = _project(full_sample, center, projection)
    x_min, x_max = np.quantile(projected_full[:, 0], [0.005, 0.995])
    y_min, y_max = np.quantile(projected_full[:, 1], [0.005, 0.995])
    x_pad = 0.03 * (x_max - x_min)
    y_pad = 0.03 * (y_max - y_min)
    limits = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

    image = mpimg.imread(args.rgb)
    fig, (ax_rgb, ax_full, ax_skeleton) = plt.subplots(
        1,
        3,
        figsize=(3.45, 1.12),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
    )
    ax_rgb.imshow(image)
    ax_rgb.set_title("(a) RGB scene", fontsize=7, pad=2)
    ax_rgb.axis("off")
    _plot_map(
        ax_full,
        full_sample,
        full_color_sample,
        center,
        projection,
        limits,
        "(b) Full map",
    )
    _plot_map(
        ax_skeleton,
        skeleton_sample,
        skeleton_color_sample,
        center,
        projection,
        limits,
        "(c) C-GLS 20%",
    )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.80, wspace=0.06)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "fig_scene_map_visualization.pdf"
    png_path = args.output_dir / "fig_scene_map_visualization.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Displayed samples: full={len(full_sample)}, C-GLS={len(skeleton_sample)} at fraction={args.sample_fraction:.3f}")


if __name__ == "__main__":
    main()
