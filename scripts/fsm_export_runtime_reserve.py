"""Export a small runtime reserve from an existing, intact activation audit archive."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.fsm_map import export_runtime_reserve_geometry, load_fsm_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--audit-sidecar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_runtime_reserve_geometry(str(args.audit_sidecar), load_fsm_map(str(args.map)), str(args.output))
    old, new = args.audit_sidecar.stat().st_size, args.output.stat().st_size
    print(f"{args.output}: {old} -> {new} bytes ({100*new/old:.2f}% of audit archive)")


if __name__ == "__main__":
    main()
