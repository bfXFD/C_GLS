import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from scripts.fsm_localize import load_config, run_localization
from utils.fsm_dataset import load_dataset_from_config
from utils.fsm_map import build_fsm_map, expand_log_scales, load_splatam_params, save_fsm_map
from utils.fsm_pruning import resolve_budget, select_topk


REPLICA_SCENES = {
    "room0_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/replica_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz",
    },
    "office0_0": {
        "config": "/home/ras/xbf/C_GLS_code/configs/fsm/generated/replica_office0_fsm.py",
        "params": "/home/ras/xbf/SplaTAM/experiments/Replica/office0_0/params.npz",
    },
}


SUMMARY_FIELDS = [
    "experiment",
    "scene",
    "strategy",
    "compression_ratio",
    "gamma",
    "beta",
    "voxel_multiplier",
    "N_full",
    "N_star",
    "ate_rmse",
    "rpe_rmse",
    "latency_mean_ms",
    "latency_p95_ms",
    "map_size_mb",
    "occupied_voxel_ratio",
    "map_path",
    "localize_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P1 C-GLS ablations for saliency exponents and voxel sensitivity.")
    parser.add_argument("--experiment", choices=("saliency", "voxel"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenes", nargs="+", default=["room0_0", "office0_0"], choices=sorted(REPLICA_SCENES))
    parser.add_argument("--ratios", nargs="+", type=float, default=None)
    parser.add_argument("--gamma-beta", nargs=2, type=float, action="append", metavar=("GAMMA", "BETA"))
    parser.add_argument("--voxel-multipliers", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))


def saliency_scores(params: dict[str, Any], gamma: float, beta: float) -> np.ndarray:
    log_scales = expand_log_scales(np.asarray(params["log_scales"], dtype=np.float64))
    logits = np.asarray(params["logit_opacities"], dtype=np.float64)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(f"Expected logit_opacities with shape [N] or [N, 1], got {logits.shape}")
    log_opacity = np.log(np.clip(_sigmoid(logits), 1e-12, 1.0))
    log_score = float(gamma) * np.sum(log_scales, axis=1) + float(beta) * log_opacity
    log_score = log_score - float(np.max(log_score))
    return np.exp(log_score).astype(np.float64)


def select_voxel_grid_with_multiplier(params: dict[str, Any], k: int, fill_scores: np.ndarray, multiplier: float) -> np.ndarray:
    means = np.asarray(params["means3D"], dtype=np.float64)
    num_items = means.shape[0]
    k = min(int(k), int(num_items))
    if k == num_items:
        return np.arange(num_items, dtype=np.int64)
    span = np.ptp(means, axis=0)
    positive_span = span[span > 0]
    if positive_span.size == 0:
        return select_topk(fill_scores, k, largest=True)
    volume = float(np.prod(np.maximum(span, np.min(positive_span))))
    base_voxel_size = max((volume / max(k, 1)) ** (1.0 / 3.0), 1e-12)
    voxel_size = base_voxel_size * float(multiplier)
    grid = np.floor((means - np.min(means, axis=0)) / voxel_size).astype(np.int64)

    order = np.argsort(-fill_scores, kind="mergesort")
    seen = set()
    selected = []
    for idx in order:
        key = tuple(int(v) for v in grid[idx])
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(idx))
        if len(selected) == k:
            break
    if len(selected) < k:
        selected_set = set(selected)
        for idx in order:
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            if len(selected) == k:
                break
    return np.asarray(selected, dtype=np.int64)


def occupied_voxel_ratio(params: dict[str, Any], indices: np.ndarray, target_voxels: int = 20000) -> float:
    means = np.asarray(params["means3D"], dtype=np.float64)
    span = np.ptp(means, axis=0)
    positive_span = span[span > 0]
    if positive_span.size == 0:
        return 1.0
    volume = float(np.prod(np.maximum(span, np.min(positive_span))))
    voxel_size = max((volume / max(int(target_voxels), 1)) ** (1.0 / 3.0), 1e-12)
    grid = np.floor((means - np.min(means, axis=0)) / voxel_size).astype(np.int64)
    full_count = np.unique(grid, axis=0).shape[0]
    retained_count = np.unique(grid[np.asarray(indices, dtype=np.int64)], axis=0).shape[0]
    return float(retained_count / full_count) if full_count else 1.0


def _format_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    experiment: str,
    scene: str,
    scene_cfg: dict[str, str],
    output_root: Path,
    ratio: float,
    gamma: float,
    beta: float,
    voxel_multiplier: float,
    start: int,
    end: int,
    skip_existing: bool,
    overwrite: bool,
) -> dict[str, Any]:
    params = load_splatam_params(scene_cfg["params"])
    num_full = int(np.asarray(params["means3D"]).shape[0])
    k = resolve_budget(num_full, compression_ratio=ratio)
    scores = saliency_scores(params, gamma=gamma, beta=beta)
    indices = select_voxel_grid_with_multiplier(params, k, fill_scores=scores, multiplier=voxel_multiplier)

    variant = (
        f"{experiment}_{scene}_ratio_{_format_float(ratio)}_"
        f"g_{_format_float(gamma)}_b_{_format_float(beta)}_vm_{_format_float(voxel_multiplier)}"
    )
    map_dir = output_root / scene / "maps" / variant
    localize_dir = output_root / scene / "localize" / variant
    if map_dir.exists() and overwrite:
        shutil.rmtree(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_dir / "fsm_map.npz"
    if overwrite or not map_path.exists():
        fsm_map = build_fsm_map(
            params,
            indices=indices,
            scores=scores,
            strategy=f"cgls_{experiment}",
            compression_ratio=float(indices.size / num_full),
            source_params_path=scene_cfg["params"],
        )
        save_fsm_map(fsm_map, str(map_path))

    local_summary_path = localize_dir / "summary.json"
    if skip_existing and local_summary_path.exists():
        local_summary = _read_json(local_summary_path)
    else:
        config = load_config(scene_cfg["config"])
        dataset = load_dataset_from_config(scene_cfg["config"])
        local_summary = run_localization(
            config=config,
            map_path=str(map_path),
            output_dir=str(localize_dir),
            dataset=dataset,
            start=start,
            end=end,
        )

    return {
        "experiment": experiment,
        "scene": scene,
        "strategy": f"cgls_{experiment}",
        "compression_ratio": f"{float(indices.size / num_full):.12g}",
        "gamma": f"{float(gamma):.12g}",
        "beta": f"{float(beta):.12g}",
        "voxel_multiplier": f"{float(voxel_multiplier):.12g}",
        "N_full": str(num_full),
        "N_star": str(indices.size),
        "ate_rmse": local_summary.get("ate_rmse", ""),
        "rpe_rmse": local_summary.get("rpe_trans_rmse", ""),
        "latency_mean_ms": local_summary.get("latency_mean_ms", ""),
        "latency_p95_ms": local_summary.get("latency_p95_ms", ""),
        "map_size_mb": f"{map_path.stat().st_size / (1024 * 1024):.12g}" if map_path.exists() else "",
        "occupied_voxel_ratio": f"{occupied_voxel_ratio(params, indices):.12g}",
        "map_path": str(map_path),
        "localize_dir": str(localize_dir),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if args.experiment == "saliency":
        ratios = args.ratios or [0.10, 0.20]
        gamma_beta = args.gamma_beta or [(1.0 / 3.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0 / 3.0, 0.5)]
        jobs = [(ratio, gamma, beta, 1.0) for ratio in ratios for gamma, beta in gamma_beta]
    else:
        ratios = args.ratios or [0.20]
        jobs = [(ratio, 1.0 / 3.0, 1.0, multiplier) for ratio in ratios for multiplier in args.voxel_multipliers]

    rows = []
    for scene in args.scenes:
        for ratio, gamma, beta, voxel_multiplier in jobs:
            print(
                f"[fsm_p1_ablation] {args.experiment} scene={scene} ratio={ratio} "
                f"gamma={gamma} beta={beta} voxel_multiplier={voxel_multiplier}",
                flush=True,
            )
            rows.append(
                run_one(
                    experiment=args.experiment,
                    scene=scene,
                    scene_cfg=REPLICA_SCENES[scene],
                    output_root=output_root,
                    ratio=ratio,
                    gamma=gamma,
                    beta=beta,
                    voxel_multiplier=voxel_multiplier,
                    start=args.start,
                    end=args.end,
                    skip_existing=args.skip_existing,
                    overwrite=args.overwrite,
                )
            )
            _write_csv(output_root / "summary_all.csv", rows)
    print(f"Wrote {len(rows)} rows to {output_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
