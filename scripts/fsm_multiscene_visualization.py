import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.fsm_dataset import get_frame, load_dataset_from_config
from utils.fsm_map import load_splatam_params
from utils.recon_helpers import setup_camera
from utils.slam_helpers import params2rendervar


OUTPUT_DIR = Path("/home/ras/xbf/C_GLS/ralsubmission/figures")
MAIN_FIGURE_DIR = OUTPUT_DIR / "main"
ASSET_DIR = OUTPUT_DIR / "source_assets" / "fig1"
CACHE_DIR = ASSET_DIR / "cache"

SCENES = [
    {
        "label": "Replica room0",
        "short": "replica_room0",
        "config": Path("/home/ras/xbf/C_GLS_code/configs/fsm/replica_fsm.py"),
        "params": Path("/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz"),
        "map": Path("/home/ras/xbf/C_GLS_code/experiments/FSM/replica_room0_0/batch_500_pareto_v1_1/maps/voxel_scale_opacity_ratio_0p2/fsm_map.npz"),
        "method": "C-GLS-Voxel 20%",
        "frame": 250,
    },
    {
        "label": "Replica office0",
        "short": "replica_office0",
        "config": Path("/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_office0_fsm.py"),
        "params": Path("/home/ras/xbf/SplaTAM/experiments/Replica/office0_0/params.npz"),
        "map": Path("/home/ras/xbf/C_GLS_code/experiments/FSM/replica_office0_0/batch_500_pareto_v1_1/maps/voxel_scale_opacity_ratio_0p2/fsm_map.npz"),
        "method": "C-GLS-Voxel 20%",
        "frame": 250,
    },
    {
        "label": "TUM fr1/desk",
        "short": "tum_fr1_desk",
        "config": Path("/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_freiburg1_desk_fsm.py"),
        "params": Path("/home/ras/xbf/C_GLS_code/experiments/TUM/freiburg1_desk_seed0/params.npz"),
        "map": Path("/home/ras/xbf/C_GLS_code/experiments/FSM/tum_freiburg1_desk/batch_500_tum_pareto_v1_1/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz"),
        "method": "FPS-C-GLS 10%",
        "frame": 250,
    },
    {
        "label": "TUM fr2/xyz",
        "short": "tum_fr2_xyz",
        "config": Path("/home/ras/xbf/C_GLS_code/configs/fsm/generated_tum/tum_freiburg2_xyz_fsm.py"),
        "params": Path("/home/ras/xbf/C_GLS_code/experiments/TUM/freiburg2_xyz_seed0/params.npz"),
        "map": Path("/home/ras/xbf/C_GLS_code/experiments/FSM/tum_freiburg2_xyz/batch_500_tum_pareto_v1_1/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz"),
        "method": "FPS-C-GLS 10%",
        "frame": 250,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the C-GLS workflow and four-scene 4x5 visualization matrix.")
    parser.add_argument("--output-dir", type=Path, default=MAIN_FIGURE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--top-width", type=int, default=1200)
    parser.add_argument("--top-height", type=int, default=760)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def _to_torch_params(params: dict[str, Any], indices: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    required = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]
    if indices is None:
        indices = np.arange(np.asarray(params["means3D"]).shape[0], dtype=np.int64)
    return {key: torch.tensor(np.asarray(params[key])[indices]).cuda().float() for key in required}


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.transpose(image[:3], (1, 2, 0))
    image = image[..., :3].astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def _render_view(params: dict[str, Any], frame: dict[str, Any], indices: np.ndarray | None) -> np.ndarray:
    render_params = _to_torch_params(params, indices)
    height, width = _normalize_rgb(frame["color"]).shape[:2]
    camera = setup_camera(w=width, h=height, k=frame["intrinsics"], w2c=frame["w2c"])
    with torch.no_grad():
        image, _, _ = Renderer(raster_settings=camera)(**params2rendervar(render_params))
    result = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
    del render_params, image, camera
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _fit_top_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    count = min(len(points), 300000)
    sample = points[rng.choice(len(points), size=count, replace=False)]
    center = np.median(sample, axis=0)
    centered = sample - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = axes[:2].T
    projected = centered @ projection
    if np.ptp(projected[:, 1]) > np.ptp(projected[:, 0]):
        projection = projection[:, [1, 0]]
    return center, projection


def _top_view(
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    projection: np.ndarray,
    limits: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    projected = (points - center) @ projection
    x0, x1, y0, y1 = limits
    valid = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= x0)
        & (projected[:, 0] <= x1)
        & (projected[:, 1] >= y0)
        & (projected[:, 1] <= y1)
    )
    projected = projected[valid]
    colors = np.clip(colors[valid], 0.0, 1.0)
    px = np.clip(((projected[:, 0] - x0) / (x1 - x0) * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((y1 - projected[:, 1]) / (y1 - y0) * (height - 1)).astype(np.int32), 0, height - 1)
    flat = py * width + px
    counts = np.bincount(flat, minlength=width * height).astype(np.float32)
    image = np.zeros((height * width, 3), dtype=np.float32)
    for channel in range(3):
        image[:, channel] = np.bincount(flat, weights=colors[:, channel], minlength=width * height)
    occupied = counts > 0
    image[occupied] /= counts[occupied, None]
    image = image.reshape(height, width, 3)
    density = np.log1p(counts.reshape(height, width))
    density /= max(float(np.quantile(density[density > 0], 0.98)) if np.any(density > 0) else 1.0, 1e-6)
    density = np.clip(density, 0.0, 1.0)[..., None]
    background = np.full_like(image, 0.055)
    return np.clip(background * (1.0 - density) + image * density, 0.0, 1.0)


def _scene_cache_paths(cache_dir: Path, short: str) -> dict[str, Path]:
    root = cache_dir / short
    return {
        "root": root,
        "rgb": root / "rgb.png",
        "top_full": root / "top_full.png",
        "top_cgls": root / "top_cgls.png",
        "render_full": root / "render_full.png",
        "render_cgls": root / "render_cgls.png",
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def generate_scene_assets(scene: dict[str, Any], cache_dir: Path, width: int, height: int, overwrite: bool) -> dict[str, Path]:
    paths = _scene_cache_paths(cache_dir, scene["short"])
    if not overwrite and all(path.exists() for key, path in paths.items() if key != "root"):
        return paths

    print(f"Generating assets for {scene['label']}", flush=True)
    params = load_splatam_params(str(scene["params"]))
    map_data = np.load(scene["map"], allow_pickle=True)
    indices = np.asarray(map_data["indices"], dtype=np.int64)
    dataset = load_dataset_from_config(str(scene["config"]))
    frame_id = min(int(scene["frame"]), len(dataset) - 1)
    frame = get_frame(dataset, frame_id)
    rgb = _normalize_rgb(frame["color"])

    points = np.asarray(params["means3D"], dtype=np.float32)
    colors = np.asarray(params["rgb_colors"], dtype=np.float32)
    center, projection = _fit_top_projection(points)
    rng = np.random.default_rng(23)
    sample = points[rng.choice(len(points), size=min(len(points), 400000), replace=False)]
    projected = (sample - center) @ projection
    x0, x1 = np.quantile(projected[:, 0], [0.003, 0.997])
    y0, y1 = np.quantile(projected[:, 1], [0.003, 0.997])
    x_pad = 0.025 * (x1 - x0)
    y_pad = 0.025 * (y1 - y0)
    limits = (x0 - x_pad, x1 + x_pad, y0 - y_pad, y1 + y_pad)

    top_full = _top_view(points, colors, center, projection, limits, width, height)
    top_cgls = _top_view(points[indices], colors[indices], center, projection, limits, width, height)
    render_full = _render_view(params, frame, None)
    render_cgls = _render_view(params, frame, indices)

    _save_image(paths["rgb"], rgb)
    _save_image(paths["top_full"], top_full)
    _save_image(paths["top_cgls"], top_cgls)
    _save_image(paths["render_full"], render_full)
    _save_image(paths["render_cgls"], render_cgls)
    del params, map_data, dataset, frame, points, colors, top_full, top_cgls, render_full, render_cgls
    gc.collect()
    return paths


def _draw_gaussians(ax: plt.Axes, center: tuple[float, float], dense: bool, color_mode: str) -> None:
    rng = np.random.default_rng(5 if dense else 9)
    count = 85 if dense else 24
    palette = ["#0072B2", "#009E73"] if color_mode == "skeleton" else ["#56B4E9", "#009E73", "#E69F00", "#CC79A7"]
    for _ in range(count):
        x = center[0] + rng.uniform(-0.34, 0.34)
        y = center[1] + rng.uniform(-0.12, 0.12)
        patch = Ellipse(
            (x, y),
            rng.uniform(0.018, 0.045),
            rng.uniform(0.008, 0.022),
            angle=float(rng.uniform(0, 180)),
            facecolor=palette[int(rng.integers(0, len(palette)))],
            edgecolor="none",
            alpha=0.75,
        )
        ax.add_patch(patch)


def _draw_workflow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = ["Dense 3DGS map", "Coverage partition", "Scale-opacity saliency", "Localization skeleton", "RGB-D registration"]
    ys = [0.91, 0.71, 0.51, 0.31, 0.11]
    for idx, (title, y) in enumerate(zip(titles, ys)):
        box = FancyBboxPatch((0.05, y - 0.07), 0.90, 0.14, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor="#F7F9FB", edgecolor="#4D4D4D", linewidth=0.8)
        ax.add_patch(box)
        ax.text(0.5, y + 0.052, title, ha="center", va="center", fontsize=6.4, weight="bold", color="#222222")
        if idx == 0:
            _draw_gaussians(ax, (0.5, y - 0.018), dense=True, color_mode="dense")
        elif idx == 1:
            for gx in np.linspace(0.18, 0.82, 5):
                ax.plot([gx, gx], [y - 0.052, y + 0.02], color="#00A6B2", linewidth=0.45, alpha=0.8)
            for gy in np.linspace(y - 0.052, y + 0.02, 4):
                ax.plot([0.16, 0.84], [gy, gy], color="#00A6B2", linewidth=0.45, alpha=0.8)
            _draw_gaussians(ax, (0.5, y - 0.018), dense=False, color_mode="dense")
        elif idx == 2:
            _draw_gaussians(ax, (0.5, y - 0.018), dense=False, color_mode="dense")
            for x, color in [(0.30, "#E69F00"), (0.50, "#009E73"), (0.70, "#E69F00")]:
                ax.add_patch(Ellipse((x, y - 0.02), 0.095, 0.038, facecolor=color, edgecolor="#333333", linewidth=0.4))
        elif idx == 3:
            _draw_gaussians(ax, (0.5, y - 0.018), dense=False, color_mode="skeleton")
        else:
            camera = Rectangle((0.15, y - 0.047), 0.14, 0.075, facecolor="#4D4D4D", edgecolor="#222222", linewidth=0.5)
            lens = Circle((0.30, y - 0.009), 0.027, facecolor="#56B4E9", edgecolor="#222222", linewidth=0.5)
            ax.add_patch(camera)
            ax.add_patch(lens)
            for target in [(0.55, y + 0.025), (0.72, y - 0.01), (0.84, y + 0.02), (0.60, y - 0.045)]:
                ax.plot([0.33, target[0]], [y - 0.009, target[1]], color="#666666", linestyle="--", linewidth=0.55)
                ax.add_patch(Circle(target, 0.012, facecolor="#009E73", edgecolor="none"))
        if idx < len(ys) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.075), (0.5, ys[idx + 1] + 0.075), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#444444"))


def export_workflow(asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.45, 5.35))
    _draw_workflow(ax)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.005, top=0.995)
    fig.savefig(asset_dir / "00_cgls_workflow.pdf", bbox_inches="tight")
    fig.savefig(asset_dir / "00_cgls_workflow.png", bbox_inches="tight", dpi=500)
    plt.close(fig)


def collect_assets(scene_assets: list[tuple[dict[str, Any], dict[str, Path]]], output_dir: Path, asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        source = output_dir / f"fig_cgls_system_multiscene.{suffix}"
        shutil.copy2(source, asset_dir / f"fig1_final_composite.{suffix}")

    keys = ["rgb", "top_full", "top_cgls", "render_full", "render_cgls"]
    labels = ["rgb_observation", "full_map_top_view", "cgls_map_top_view", "full_map_camera_view", "cgls_map_camera_view"]
    manifest_lines = [
        "Fig.1 asset manifest",
        "====================",
        "",
        "Column order: RGB observation | Full-map top view | C-GLS top view | Full-map camera view | C-GLS camera view",
        "",
    ]
    for row, (scene, paths) in enumerate(scene_assets, start=1):
        manifest_lines.append(f"Row {row}: {scene['label']} ({scene['method']}, frame {scene['frame']})")
        for col, (key, label) in enumerate(zip(keys, labels), start=1):
            destination = asset_dir / f"row{row}_col{col}_{scene['short']}_{label}.png"
            shutil.copy2(paths[key], destination)
            manifest_lines.append(f"  {destination.name}")
        manifest_lines.append("")
    (asset_dir / "ASSET_MANIFEST.txt").write_text("\n".join(manifest_lines), encoding="utf-8")


def compose_figure(scene_assets: list[tuple[dict[str, Any], dict[str, Path]]], output_dir: Path) -> None:
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 5.65))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 5.0], wspace=0.035)
    ax_flow = fig.add_subplot(outer[0, 0])
    _draw_workflow(ax_flow)
    matrix = outer[0, 1].subgridspec(4, 5, wspace=0.025, hspace=0.09)
    column_titles = ["RGB observation", "Full map\ntop view", "C-GLS map\ntop view", "Full map\ncamera view", "C-GLS map\ncamera view"]

    for row, (scene, paths) in enumerate(scene_assets):
        images = [plt.imread(paths[key]) for key in ["rgb", "top_full", "top_cgls", "render_full", "render_cgls"]]
        for col, image in enumerate(images):
            ax = fig.add_subplot(matrix[row, col])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.45)
            if row == 0:
                ax.set_title(column_titles[col], fontsize=6.8, pad=3, weight="bold")
            if col == 0:
                ax.text(-0.07, 0.5, scene["label"], transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.8, weight="bold")
            if col in (2, 4):
                ax.text(0.98, 0.03, scene["method"], transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="white", bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.2, "edgecolor": "none"})

    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.015, top=0.95)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_cgls_system_multiscene.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig_cgls_system_multiscene.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    assets = []
    for scene in SCENES:
        for key in ["config", "params", "map"]:
            if not scene[key].exists():
                raise FileNotFoundError(scene[key])
        assets.append((scene, generate_scene_assets(scene, args.cache_dir, args.top_width, args.top_height, args.overwrite_cache)))
    compose_figure(assets, args.output_dir)
    export_workflow(args.asset_dir)
    collect_assets(assets, args.output_dir, args.asset_dir)
    print(f"Wrote {args.output_dir / 'fig_cgls_system_multiscene.pdf'}")
    print(f"Wrote {args.output_dir / 'fig_cgls_system_multiscene.png'}")
    print(f"Collected Fig.1 assets in {args.asset_dir}")


if __name__ == "__main__":
    main()
