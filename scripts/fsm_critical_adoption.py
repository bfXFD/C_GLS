"""CLI for offline FSM critical-adoption analysis."""

import argparse
import json
import os
import sys
from typing import Sequence


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.fsm_critical_adoption import CriticalAdoptionConfig, run_analysis


def _name_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=RUN_DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=RUN_DIR")
    return name, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze critical reserve adoptions from existing localization outputs.")
    parser.add_argument("--baseline", required=True, type=_name_path, metavar="NAME=RUN_DIR")
    parser.add_argument("--run", required=True, action="append", type=_name_path, metavar="NAME=RUN_DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 30, 100])
    parser.add_argument("--rpe-deltas", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--burst-gaps", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--temporal-windows", nargs="+", type=int, default=[1, 2, 5, 10, 30, 100])
    parser.add_argument(
        "--compare-run", action="append", default=[], metavar="NAME",
        help="Run used as temporal comparator; repeatable, e.g. random128_seed2.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = CriticalAdoptionConfig(
        horizons=tuple(args.horizons), rpe_deltas=tuple(args.rpe_deltas),
        burst_gaps=tuple(args.burst_gaps), temporal_windows=tuple(args.temporal_windows),
    )
    payload = run_analysis(args.baseline[0], args.baseline[1], args.run, args.output_dir,
                           config, tuple(args.compare_run))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
