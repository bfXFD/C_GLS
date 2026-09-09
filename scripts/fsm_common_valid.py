#!/usr/bin/env python3
"""CLI for strict common-valid multi-method offline trajectory comparison."""

import argparse
import json
import os
import sys
from typing import Sequence


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_common_valid import CommonValidConfig, parse_method_specs, run_common_valid


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare localization methods on identical common-valid frame masks."
    )
    parser.add_argument(
        "--method", action="append", required=True, metavar="NAME=RUN_DIR",
        help="Method name and localization run directory; repeat for each method.",
    )
    parser.add_argument("--gt", default=None, help="Optional external TUM GT trajectory for strict validation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-deltas", nargs="+", type=int, default=[1, 10, 30, 100])
    parser.add_argument("--align-with-scale", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    methods = parse_method_specs(args.method)
    config = CommonValidConfig(
        frame_deltas=tuple(args.frame_deltas), align_with_scale=args.align_with_scale,
        bootstrap_replicates=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed,
    )
    payload = run_common_valid(methods, args.output_dir, gt=args.gt, config=config)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
