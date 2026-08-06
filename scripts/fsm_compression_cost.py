import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from scripts.fsm_lightgaussian_scores import compute_lightgaussian_proxy_scores
from utils.fsm_map import build_fsm_map, load_splatam_params, save_fsm_map
from utils.fsm_pruning import resolve_budget, select_by_strategy, select_topk


REPLICA_SCENES = {
    "room0_0": "/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz",
    "room1_0": "/home/ras/xbf/SplaTAM/experiments/Replica/room1_0/params.npz",
    "room2_0": "/home/ras/xbf/SplaTAM/experiments/Replica/room2_0/params.npz",
    "office0_0": "/home/ras/xbf/SplaTAM/experiments/Replica/office0_0/params.npz",
}


FIELDS = [
    "scene",
    "strategy",
    "compression_ratio",
    "N_full",
    "N_star",
    "compression_time_s",
    "map_size_mb",
    "map_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure offline compression/export cost for FSM map variants.")
    parser.add_argument("--output-dir", default="/home/ras/xbf/C_GLS_code/experiments/FSM/compression_cost_v1_0")
    parser.add_argument("--scenes", nargs="+", default=list(REPLICA_SCENES), choices=sorted(REPLICA_SCENES))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["voxel_scale_opacity", "fps_scale_opacity", "lightgaussian_proxy"],
        choices=["voxel_scale_opacity", "fps_scale_opacity", "lightgaussian_proxy", "random", "scale_large"],
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.20, 0.10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _format_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def compress_once(
    params_path: str,
    output_root: Path,
    scene: str,
    strategy: str,
    ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    params = load_splatam_params(params_path)
    num_full = int(np.asarray(params["means3D"]).shape[0])
    variant = f"{scene}_{strategy}_ratio_{_format_float(ratio)}"
    map_dir = output_root / "maps" / variant
    map_path = map_dir / "fsm_map.npz"
    if map_path.exists() and not overwrite:
        return {
            "scene": scene,
            "strategy": strategy,
            "compression_ratio": f"{float(ratio):.12g}",
            "N_full": str(num_full),
            "N_star": str(np.load(map_path, allow_pickle=True)["indices"].shape[0]),
            "compression_time_s": "",
            "map_size_mb": f"{map_path.stat().st_size / (1024 * 1024):.12g}",
            "map_path": str(map_path),
        }

    map_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    if strategy == "lightgaussian_proxy":
        scores, _ = compute_lightgaussian_proxy_scores(params, v_pow=0.1, opacity_power=1.0, normalize="max")
        indices = select_topk(scores, resolve_budget(num_full, compression_ratio=ratio), largest=True)
        score_array = scores
    else:
        indices, score_array = select_by_strategy(params, strategy=strategy, compression_ratio=ratio, seed=seed)
    fsm_map = build_fsm_map(
        params,
        indices=indices,
        scores=score_array,
        strategy=strategy,
        compression_ratio=float(indices.size / num_full),
        source_params_path=params_path,
    )
    save_fsm_map(fsm_map, str(map_path))
    elapsed = time.perf_counter() - start_time
    return {
        "scene": scene,
        "strategy": strategy,
        "compression_ratio": f"{float(indices.size / num_full):.12g}",
        "N_full": str(num_full),
        "N_star": str(indices.size),
        "compression_time_s": f"{elapsed:.12g}",
        "map_size_mb": f"{map_path.stat().st_size / (1024 * 1024):.12g}",
        "map_path": str(map_path),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    rows = []
    for scene in args.scenes:
        params_path = REPLICA_SCENES[scene]
        if not Path(params_path).exists():
            raise FileNotFoundError(params_path)
        for strategy in args.strategies:
            for ratio in args.ratios:
                print(f"[fsm_compression_cost] scene={scene} strategy={strategy} ratio={ratio}", flush=True)
                rows.append(compress_once(params_path, output_root, scene, strategy, ratio, args.seed, args.overwrite))
                _write_csv(output_root / "summary_all.csv", rows)
    print(f"Wrote {len(rows)} rows to {output_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
