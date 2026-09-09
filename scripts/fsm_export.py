import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_map import build_fsm_map, load_splatam_params, save_fsm_map
from utils.fsm_pruning import SUPPORTED_STRATEGIES, select_by_strategy


SUMMARY_FIELDS = [
    "strategy",
    "compression_ratio",
    "count",
    "N_full",
    "N_star",
    "map_path",
    "source_params_path",
    "seed",
    "fps_candidate_cap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export Fisher Skeleton Map variants from a SplaTAM params.npz file.")
    parser.add_argument("--params", required=True, help="Path to SplaTAM params.npz.")
    parser.add_argument("--output-dir", required=True, help="Directory where FSM maps and summary CSV are written.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full"],
        choices=SUPPORTED_STRATEGIES,
        help="Pruning strategies to export.",
    )
    parser.add_argument(
        "--compression-ratios",
        nargs="*",
        type=float,
        default=None,
        help="Compression ratios to export, e.g. 0.01 0.05 1.0.",
    )
    parser.add_argument(
        "--counts",
        nargs="*",
        type=int,
        default=None,
        help="Absolute Gaussian counts to export. Mutually exclusive with --compression-ratios.",
    )
    parser.add_argument(
        "--external-scores",
        default=None,
        help="Optional .npz/.npy file with external scores for strategy external_scores.",
    )
    parser.add_argument("--external-scores-key", default="scores", help="Key used when --external-scores is an .npz file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for random pruning.")
    parser.add_argument(
        "--fps-candidate-cap",
        type=int,
        default=20000,
        help="Maximum random candidate pool for approximate FPS; use 0 for exact full-map FPS.",
    )
    parser.add_argument("--covariance-eps", type=float, default=1e-6, help="Positive diagonal covariance jitter.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directories/files.")
    return parser.parse_args()


def load_external_scores(path: str | None, key: str = "scores") -> np.ndarray | None:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float64)
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"External score file '{path}' does not contain key '{key}'")
    return np.asarray(data[key], dtype=np.float64)


def _format_ratio(ratio: float) -> str:
    return f"{ratio:.6g}".replace(".", "p")


def _variant_name(strategy: str, compression_ratio: float | None, count: int | None) -> str:
    if count is not None:
        return f"{strategy}_count_{count}"
    if compression_ratio is not None:
        return f"{strategy}_ratio_{_format_ratio(compression_ratio)}"
    return f"{strategy}_full"


def _prepare_variant_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_fsm_maps(
    params_path: str,
    output_dir: str,
    strategies: list[str],
    compression_ratios: list[float] | None = None,
    counts: list[int] | None = None,
    external_scores: np.ndarray | None = None,
    seed: int = 0,
    fps_candidate_cap: int | None = 20000,
    covariance_eps: float = 1e-6,
    overwrite: bool = False,
) -> list[dict]:
    if compression_ratios and counts:
        raise ValueError("compression_ratios and counts are mutually exclusive")

    params = load_splatam_params(params_path)
    num_full = int(np.asarray(params["means3D"]).shape[0])
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if counts:
        budgets = [(None, int(count)) for count in counts]
    else:
        ratios = compression_ratios if compression_ratios else [1.0]
        budgets = [(float(ratio), None) for ratio in ratios]

    rows = []
    for strategy in strategies:
        for compression_ratio, count in budgets:
            selected_indices, score_array = select_by_strategy(
                params,
                strategy=strategy,
                compression_ratio=compression_ratio,
                count=count,
                scores=external_scores,
                seed=seed,
                fps_candidate_cap=fps_candidate_cap,
            )
            effective_ratio = float(selected_indices.size / num_full)
            variant_dir = output_root / _variant_name(strategy, compression_ratio, count)
            _prepare_variant_dir(variant_dir, overwrite=overwrite)
            map_path = variant_dir / "fsm_map.npz"

            fsm_map = build_fsm_map(
                params,
                indices=selected_indices,
                scores=score_array,
                strategy=strategy,
                compression_ratio=effective_ratio,
                source_params_path=params_path,
                covariance_eps=covariance_eps,
            )
            save_fsm_map(fsm_map, str(map_path))

            rows.append(
                {
                    "strategy": strategy,
                    "compression_ratio": f"{effective_ratio:.12g}",
                    "count": "" if count is None else str(count),
                    "N_full": str(num_full),
                    "N_star": str(selected_indices.size),
                    "map_path": str(map_path),
                    "source_params_path": params_path,
                    "seed": str(seed),
                    "fps_candidate_cap": "" if fps_candidate_cap is None else str(fps_candidate_cap),
                }
            )

    _write_summary(output_root / "export_summary.csv", rows)
    return rows


def main() -> None:
    args = parse_args()
    external_scores = load_external_scores(args.external_scores, key=args.external_scores_key)
    rows = export_fsm_maps(
        params_path=args.params,
        output_dir=args.output_dir,
        strategies=args.strategies,
        compression_ratios=args.compression_ratios,
        counts=args.counts,
        external_scores=external_scores,
        seed=args.seed,
        fps_candidate_cap=None if args.fps_candidate_cap == 0 else args.fps_candidate_cap,
        covariance_eps=args.covariance_eps,
        overwrite=args.overwrite,
    )
    print(f"Exported {len(rows)} FSM map variants to {args.output_dir}")


if __name__ == "__main__":
    main()
