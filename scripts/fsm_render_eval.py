import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from pytorch_msssim import ms_ssim
from utils.fsm_dataset import get_frame, load_dataset_from_config
from utils.fsm_map import load_splatam_params
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_psnr
from utils.slam_helpers import params2depthplussilhouette, params2rendervar


FRAME_FIELDS = [
    "frame_id",
    "psnr",
    "ms_ssim",
    "depth_l1",
    "valid_depth_pixels",
    "presence_pixels",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rendering quality for full or FSM-compressed SplaTAM maps.")
    parser.add_argument("--config", required=True, help="FSM dataset config path.")
    parser.add_argument("--params", required=True, help="Full SplaTAM params.npz path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for rendering metrics.")
    parser.add_argument("--map", default=None, help="Optional fsm_map.npz. If omitted, evaluate the full map.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=500, help="End frame index, exclusive. Use -1 for dataset end.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every Nth frame in the selected range.")
    parser.add_argument("--sil-thres", type=float, default=0.5, help="Rendered silhouette threshold for masked RGB metrics.")
    parser.add_argument("--no-sil-mask", action="store_true", help="Do not apply rendered silhouette mask to RGB metrics.")
    return parser.parse_args()


def _to_torch_params(params: dict[str, Any], indices: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    required = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing required render params: {missing}")

    if indices is None:
        indices = np.arange(np.asarray(params["means3D"]).shape[0], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)

    render_params = {}
    for key in required:
        value = np.asarray(params[key])
        render_params[key] = torch.tensor(value[indices]).cuda().float()
    return render_params


def _load_map_indices(map_path: str | None) -> np.ndarray | None:
    if map_path is None:
        return None
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    data = np.load(map_path, allow_pickle=True)
    if "indices" not in data:
        raise KeyError(f"FSM map {map_path} does not contain indices")
    return np.asarray(data["indices"], dtype=np.int64)


def _frame_indices(dataset_len: int, start: int, end: int, eval_every: int) -> list[int]:
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    start = int(start)
    end = dataset_len if int(end) < 0 else min(int(end), dataset_len)
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame range start={start}, end={end}")
    return list(range(start, end, int(eval_every)))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_render_eval(
    config_path: str,
    params_path: str,
    output_dir: str,
    map_path: str | None = None,
    start: int = 0,
    end: int = 500,
    eval_every: int = 10,
    sil_thres: float = 0.5,
    use_sil_mask: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    full_params = load_splatam_params(params_path)
    indices = _load_map_indices(map_path)
    render_params = _to_torch_params(full_params, indices=indices)
    num_render_gaussians = int(render_params["means3D"].shape[0])

    dataset = load_dataset_from_config(config_path)
    selected_frames = _frame_indices(len(dataset), start=start, end=end, eval_every=eval_every)
    if not selected_frames:
        raise ValueError("No frames selected for rendering evaluation")

    rows = []
    psnr_values = []
    ssim_values = []
    depth_l1_values = []

    with torch.no_grad():
        rendervar = params2rendervar(render_params)
        for frame_id in tqdm(selected_frames, desc="render_eval"):
            frame = get_frame(dataset, frame_id)
            color = torch.tensor(frame["color"]).cuda().float()
            if color.max() > 1.0:
                color = color / 255.0
            color = color.permute(2, 0, 1)
            depth = torch.tensor(frame["depth"]).cuda().float()
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth.permute(2, 0, 1)
            elif depth.ndim == 3 and depth.shape[0] != 1:
                raise ValueError(f"Unexpected depth shape for frame {frame_id}: {tuple(depth.shape)}")

            w2c = frame["w2c"]
            cam = setup_camera(
                w=color.shape[2],
                h=color.shape[1],
                k=frame["intrinsics"],
                w2c=w2c,
            )

            im, _, _ = Renderer(raster_settings=cam)(**rendervar)
            depth_rendervar = params2depthplussilhouette(render_params, torch.tensor(w2c).cuda().float())
            depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
            rendered_depth = depth_sil[0:1]
            silhouette = depth_sil[1]

            valid_depth = depth > 0
            presence = silhouette > float(sil_thres)
            rgb_mask = valid_depth
            if use_sil_mask:
                rgb_mask = rgb_mask & presence.unsqueeze(0)
            if int(rgb_mask.sum().item()) == 0:
                rgb_mask = valid_depth

            weighted_im = torch.clamp(im, 0.0, 1.0) * rgb_mask
            weighted_gt = torch.clamp(color, 0.0, 1.0) * rgb_mask
            psnr = calc_psnr(weighted_im, weighted_gt).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt.unsqueeze(0).cpu(), data_range=1.0, size_average=True)

            depth_l1 = torch.abs(rendered_depth - depth)[valid_depth].mean()

            row = {
                "frame_id": str(frame_id),
                "psnr": f"{float(psnr.detach().cpu()):.12g}",
                "ms_ssim": f"{float(ssim):.12g}",
                "depth_l1": f"{float(depth_l1.detach().cpu()):.12g}",
                "valid_depth_pixels": str(int(valid_depth.sum().item())),
                "presence_pixels": str(int(presence.sum().item())),
            }
            rows.append(row)
            psnr_values.append(float(row["psnr"]))
            ssim_values.append(float(row["ms_ssim"]))
            depth_l1_values.append(float(row["depth_l1"]))

    summary = {
        "params_path": params_path,
        "map_path": map_path or "",
        "num_full_gaussians": int(np.asarray(full_params["means3D"]).shape[0]),
        "num_render_gaussians": num_render_gaussians,
        "compression_ratio": float(num_render_gaussians / int(np.asarray(full_params["means3D"]).shape[0])),
        "num_eval_frames": len(selected_frames),
        "start": int(start),
        "end": int(end),
        "eval_every": int(eval_every),
        "use_sil_mask": bool(use_sil_mask),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ms_ssim_mean": float(np.mean(ssim_values)),
        "ms_ssim_std": float(np.std(ssim_values)),
        "depth_l1_mean": float(np.mean(depth_l1_values)),
        "depth_l1_std": float(np.std(depth_l1_values)),
    }

    _write_csv(output_root / "render_metrics.csv", rows)
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_render_eval(
        config_path=args.config,
        params_path=args.params,
        output_dir=args.output_dir,
        map_path=args.map,
        start=args.start,
        end=args.end,
        eval_every=args.eval_every,
        sil_thres=args.sil_thres,
        use_sil_mask=not args.no_sil_mask,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
