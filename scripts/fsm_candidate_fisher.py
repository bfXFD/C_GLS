import argparse
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from scripts.fsm_fisher import compute_and_save_fisher
from utils.fsm_map import build_fsm_map, load_splatam_params, save_fsm_map
from utils.fsm_pruning import select_by_strategy


def parse_args():
    parser = argparse.ArgumentParser(description="Compute full-length Fisher scores on a scale-aware candidate subset.")
    parser.add_argument("--params", required=True, help="Path to SplaTAM params.npz.")
    parser.add_argument("--output", required=True, help="Output .npz containing full-length scores.")
    parser.add_argument("--candidate-ratio", type=float, default=0.25, help="Candidate ratio selected before Fisher scoring.")
    parser.add_argument("--candidate-strategy", default="scale_large", help="Strategy used for candidate selection.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for stochastic candidate strategies.")
    parser.add_argument("--normal-k", type=int, default=20, help="Number of neighbors for candidate normal estimation.")
    return parser.parse_args()


def compute_candidate_fisher_scores(
    params_path: str,
    output_path: str,
    candidate_ratio: float = 0.25,
    candidate_strategy: str = "scale_large",
    seed: int = 0,
    normal_k: int = 20,
) -> dict:
    params = load_splatam_params(params_path)
    num_full = int(np.asarray(params["means3D"]).shape[0])
    candidate_indices, _ = select_by_strategy(
        params,
        strategy=candidate_strategy,
        compression_ratio=float(candidate_ratio),
        seed=int(seed),
    )

    temp_map_path = f"{output_path}.candidate_map.tmp.npz"
    candidate_map = build_fsm_map(
        params,
        indices=candidate_indices,
        scores=np.ones(num_full, dtype=np.float64),
        strategy=f"candidate_{candidate_strategy}",
        compression_ratio=float(candidate_indices.size / num_full),
        source_params_path=params_path,
    )
    save_fsm_map(candidate_map, temp_map_path)
    try:
        fisher = compute_and_save_fisher(
            f"{output_path}.candidate_fisher.tmp.npz",
            map_path=temp_map_path,
            normal_k=int(normal_k),
        )
    finally:
        if os.path.exists(temp_map_path):
            os.remove(temp_map_path)

    scores = np.zeros(num_full, dtype=np.float32)
    scores[candidate_indices] = np.asarray(fisher["scores"], dtype=np.float32)
    output = {
        "scores": scores,
        "candidate_indices": candidate_indices.astype(np.int64),
        "candidate_ratio": np.asarray(candidate_indices.size / num_full, dtype=np.float32),
        "candidate_strategy": np.asarray(candidate_strategy),
        "source_params_path": np.asarray(params_path),
        "normal_k": np.asarray(int(normal_k), dtype=np.int64),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "N_full": np.asarray(num_full, dtype=np.int64),
        "N_candidate": np.asarray(candidate_indices.size, dtype=np.int64),
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **output)
    temp_fisher_path = f"{output_path}.candidate_fisher.tmp.npz"
    if os.path.exists(temp_fisher_path):
        os.remove(temp_fisher_path)
    return output


def main() -> None:
    args = parse_args()
    output = compute_candidate_fisher_scores(
        params_path=args.params,
        output_path=args.output,
        candidate_ratio=args.candidate_ratio,
        candidate_strategy=args.candidate_strategy,
        seed=args.seed,
        normal_k=args.normal_k,
    )
    print(
        f"Saved full-length candidate Fisher scores to {args.output}: "
        f"N_full={int(output['N_full'])}, N_candidate={int(output['N_candidate'])}"
    )


if __name__ == "__main__":
    main()
