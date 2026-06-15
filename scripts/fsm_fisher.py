import argparse
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_fisher import compute_geometric_fisher, compute_pruning_residuals, predict_error_increment
from utils.fsm_map import load_fsm_map, load_splatam_params


def parse_args():
    parser = argparse.ArgumentParser(description="Compute geometric Fisher scores for SplaTAM/FSM maps.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--params", help="Path to SplaTAM params.npz.")
    source.add_argument("--map", help="Path to fsm_map.npz.")
    parser.add_argument("--output", required=True, help="Output .npz path for Fisher scores and FIM statistics.")
    parser.add_argument("--retained-indices", default=None, help="Optional .npy/.npz retained indices for residual statistics.")
    parser.add_argument("--retained-indices-key", default="indices", help="Key for retained indices when using .npz.")
    parser.add_argument("--normal-k", type=int, default=20, help="Number of neighbors for normal estimation.")
    parser.add_argument("--regularization", type=float, default=1e-9, help="FIM regularization for inverse statistics.")
    parser.add_argument("--max-points", type=int, default=None, help="Optional deterministic point subsample for large maps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used with --max-points.")
    return parser.parse_args()


def load_points(params_path: str | None = None, map_path: str | None = None) -> tuple[np.ndarray, str]:
    if params_path is not None:
        params = load_splatam_params(params_path)
        return np.asarray(params["means3D"], dtype=np.float64), params_path
    if map_path is not None:
        fsm_map = load_fsm_map(map_path)
        return np.asarray(fsm_map["means"], dtype=np.float64), map_path
    raise ValueError("Either params_path or map_path must be provided")


def load_retained_indices(path: str | None, key: str = "indices") -> np.ndarray | None:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.int64)
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Retained index file '{path}' does not contain key '{key}'")
    return np.asarray(data[key], dtype=np.int64)


def compute_and_save_fisher(
    output_path: str,
    params_path: str | None = None,
    map_path: str | None = None,
    retained_indices: np.ndarray | None = None,
    normal_k: int = 20,
    regularization: float = 1e-9,
    max_points: int | None = None,
    seed: int = 0,
) -> dict:
    points, source_path = load_points(params_path=params_path, map_path=map_path)
    original_num_points = points.shape[0]
    sample_indices = None
    retained_mask_sample = None
    if max_points is not None and original_num_points > int(max_points):
        if int(max_points) <= 0:
            raise ValueError("max_points must be positive")
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(rng.choice(original_num_points, size=int(max_points), replace=False)).astype(np.int64)
        points = points[sample_indices]
        if retained_indices is not None:
            retained_mask = np.zeros(original_num_points, dtype=bool)
            retained_mask[np.asarray(retained_indices, dtype=np.int64)] = True
            retained_mask_sample = retained_mask[sample_indices]

    fisher = compute_geometric_fisher(points, normals=None, weights=None, normal_k=normal_k)
    output = {
        "points": points.astype(np.float32),
        "per_gaussian_fim": fisher["per_gaussian_fim"].astype(np.float32),
        "scores": fisher["scores"].astype(np.float32),
        "full_fim": fisher["full_fim"].astype(np.float32),
        "lambda_min": np.asarray(fisher["lambda_min"], dtype=np.float32),
        "eigvals": fisher["eigvals"].astype(np.float32),
        "normals": fisher["normals"].astype(np.float32),
        "jacobians": fisher["jacobians"].astype(np.float32),
        "source_path": np.asarray(source_path),
        "original_num_points": np.asarray(original_num_points, dtype=np.int64),
        "sample_indices": np.asarray([], dtype=np.int64) if sample_indices is None else sample_indices,
    }

    if retained_indices is not None:
        retained_indices_for_fisher = np.asarray(retained_indices, dtype=np.int64)
        if sample_indices is not None:
            retained_indices_for_fisher = np.flatnonzero(retained_mask_sample).astype(np.int64)
            if retained_indices_for_fisher.size == 0:
                raise ValueError("No retained indices are present in the Fisher point subsample")
        residuals = compute_pruning_residuals(fisher["per_gaussian_fim"], retained_indices_for_fisher)
        prediction = predict_error_increment(
            residuals["full_fim"], residuals["pruned_fim"], regularization=regularization
        )
        output.update(
            {
                "retained_indices": retained_indices_for_fisher.astype(np.int64),
                "original_retained_indices": np.asarray(retained_indices, dtype=np.int64),
                "pruned_indices": residuals["pruned_indices"].astype(np.int64),
                "retained_fim": residuals["retained_fim"].astype(np.float32),
                "pruned_fim": residuals["pruned_fim"].astype(np.float32),
                "delta_trace": np.asarray(residuals["delta_trace"], dtype=np.float32),
                "delta_spectral": np.asarray(residuals["delta_spectral"], dtype=np.float32),
                "trace_inv_full": np.asarray(prediction["trace_inv_full"], dtype=np.float32),
                "pred_delta_trace": np.asarray(prediction["pred_delta_trace"], dtype=np.float32),
                "pred_delta_spectral": np.asarray(prediction["pred_delta_spectral"], dtype=np.float32),
                "valid_trace_bound": np.asarray(prediction["valid_trace_bound"]),
                "valid_spectral_bound": np.asarray(prediction["valid_spectral_bound"]),
            }
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **output)
    return output


def main() -> None:
    args = parse_args()
    retained_indices = load_retained_indices(args.retained_indices, key=args.retained_indices_key)
    output = compute_and_save_fisher(
        output_path=args.output,
        params_path=args.params,
        map_path=args.map,
        retained_indices=retained_indices,
        normal_k=args.normal_k,
        regularization=args.regularization,
        max_points=args.max_points,
        seed=args.seed,
    )
    print(f"Saved Fisher scores for {output['scores'].shape[0]} Gaussians to {args.output}")


if __name__ == "__main__":
    main()
