import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from utils.fsm_map import expand_log_scales, load_splatam_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LightGaussian-inspired proxy importance scores for SplaTAM params.npz. "
            "The score is opacity^opacity_power * volume^v_pow and is intended for "
            "fsm_eval.py --strategies external_scores."
        )
    )
    parser.add_argument("--params", required=True, help="Path to SplaTAM params.npz.")
    parser.add_argument("--output", required=True, help="Output .npz path containing key 'scores'.")
    parser.add_argument("--v-pow", type=float, default=0.1, help="Volume exponent used by the LightGaussian proxy score.")
    parser.add_argument("--opacity-power", type=float, default=1.0, help="Opacity exponent used by the proxy score.")
    parser.add_argument(
        "--normalize",
        choices=("max", "minmax", "none"),
        default="max",
        help="Score normalization before saving. Ranking is unchanged for max/minmax.",
    )
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    return parser.parse_args()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))


def compute_lightgaussian_proxy_scores(
    params: dict,
    v_pow: float = 0.1,
    opacity_power: float = 1.0,
    normalize: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    if "log_scales" not in params:
        raise KeyError("Missing required field 'log_scales'")
    if "logit_opacities" not in params:
        raise KeyError("Missing required field 'logit_opacities'")

    log_scales = expand_log_scales(np.asarray(params["log_scales"], dtype=np.float64))
    logits = np.asarray(params["logit_opacities"], dtype=np.float64)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(f"Expected logit_opacities with shape [N] or [N, 1], got {logits.shape}")
    if log_scales.shape[0] != logits.shape[0]:
        raise ValueError(f"log_scales and logit_opacities have mismatched N: {log_scales.shape[0]} != {logits.shape[0]}")

    opacity = np.clip(_sigmoid(logits), 1e-12, 1.0)
    log_volume = np.sum(log_scales, axis=1)
    raw_log_scores = float(opacity_power) * np.log(opacity) + float(v_pow) * log_volume
    if np.any(~np.isfinite(raw_log_scores)):
        raise ValueError("Computed non-finite LightGaussian proxy log scores")

    # Stabilize exponentiation. This preserves ranking and avoids overflow on large scenes.
    scores = np.exp(raw_log_scores - float(np.max(raw_log_scores)))
    if normalize == "minmax":
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        denom = max(max_score - min_score, 1e-12)
        scores = (scores - min_score) / denom
    elif normalize == "none":
        scores = np.exp(raw_log_scores)
    elif normalize != "max":
        raise ValueError(f"Unsupported normalization: {normalize}")

    if np.any(~np.isfinite(scores)):
        raise ValueError("Computed non-finite LightGaussian proxy scores")
    return scores.astype(np.float32), raw_log_scores.astype(np.float32)


def main() -> None:
    args = parse_args()
    params = load_splatam_params(args.params)
    scores, raw_log_scores = compute_lightgaussian_proxy_scores(
        params,
        v_pow=args.v_pow,
        opacity_power=args.opacity_power,
        normalize=args.normalize,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        scores=scores,
        raw_log_scores=raw_log_scores,
        method=np.asarray("lightgaussian_proxy"),
        v_pow=np.asarray(float(args.v_pow), dtype=np.float32),
        opacity_power=np.asarray(float(args.opacity_power), dtype=np.float32),
        normalize=np.asarray(args.normalize),
        source_params_path=np.asarray(args.params),
    )

    metadata = {
        "method": "lightgaussian_proxy",
        "definition": "score = sigmoid(logit_opacity)^opacity_power * exp(sum(log_scales))^v_pow",
        "params": args.params,
        "output": str(output_path),
        "num_gaussians": int(scores.shape[0]),
        "v_pow": float(args.v_pow),
        "opacity_power": float(args.opacity_power),
        "normalize": args.normalize,
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
    }
    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
