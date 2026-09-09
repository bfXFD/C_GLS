import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from scripts.fsm_build_observable_map import (
    CANDIDATE_SOURCE_NAMES,
    RESERVE_POLICY,
    _build_activation_geometry,
    _cheap_scores,
    _reserve_normals,
    _quota_select,
)
from utils.fsm_map import load_fsm_map, load_splatam_params, save_fsm_map


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Attach a bounded independent reserve to a fixed FSM base map.")
    parser.add_argument("--params", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--map-output", required=True)
    parser.add_argument("--activation-geometry-output", required=True)
    parser.add_argument("--reserve-count", type=int, default=4096)
    parser.add_argument("--fill-strategy", choices=("opacity", "scale_opacity"), default="scale_opacity")
    parser.add_argument("--normal-k", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _controls(args, selected_ids):
    value = {
        "builder": "fixed_map_independent_reserve_v1",
        "base_map_path": str(Path(args.base_map).resolve()),
        "base_map_sha256": _sha256(args.base_map),
        "selected_ids_sha256": hashlib.sha256(np.ascontiguousarray(selected_ids).tobytes()).hexdigest(),
        "reserve_count": int(args.reserve_count),
        "reserve_policy": "half_scale_opacity_half_deterministic_spatial_outside_fixed_map",
        "fill_strategy": args.fill_strategy,
        "normal_k": int(args.normal_k),
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.reserve_count <= 0 or args.normal_k < 3:
        raise ValueError("reserve-count must be positive and normal-k at least 3")
    params = load_splatam_params(args.params)
    base = load_fsm_map(args.base_map)
    selected = np.asarray(base["indices"], dtype=np.int64)
    means = np.asarray(params["means3D"], dtype=np.float64)
    if np.any(selected < 0) or np.any(selected >= means.shape[0]):
        raise ValueError("fixed map indices are outside the source params")
    params_hash = _sha256(args.params)
    controls, controls_hash = _controls(args, selected)
    plan = {
        **controls,
        "params_path": str(Path(args.params).resolve()),
        "params_sha256": params_hash,
        "controls_sha256": controls_hash,
        "selected_count": int(selected.size),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    done_path = Path(f"{args.activation_geometry_output}.done.json")
    if args.skip_existing and all(Path(path).is_file() for path in (args.map_output, args.activation_geometry_output, done_path)):
        stored = json.loads(done_path.read_text())
        if stored.get("provenance") == plan:
            print(json.dumps({"status": "skipped", **plan}, indent=2, sort_keys=True))
            return

    cheap_scores = _cheap_scores(params, args.fill_strategy)
    selected_mask = np.zeros(means.shape[0], dtype=bool)
    selected_mask[selected] = True
    outside = np.flatnonzero(~selected_mask)
    reserve_ids, reserve_sources = _quota_select(means, outside, cheap_scores, args.reserve_count)
    reserve_normals, reserve_normal_source = _reserve_normals(params, means, reserve_ids, args.normal_k)
    empty_ids = np.empty(0, dtype=np.int64)
    provenance = {"params_sha256": params_hash, "controls_sha256": controls_hash}
    geometry = _build_activation_geometry(
        params, selected, selected, selected, empty_ids, empty_ids,
        np.empty((0, 3), dtype=np.float64), "fixed_map_no_replacement",
        reserve_ids, reserve_normals, reserve_sources,
        np.zeros(reserve_ids.size, dtype=np.float64), reserve_normal_source,
        cheap_scores, provenance,
    )
    enhanced = dict(base)
    enhanced.update(
        source_params_path=np.asarray(str(Path(args.params).resolve())),
        source_params_sha256=np.asarray(params_hash),
        selection_controls_sha256=np.asarray(controls_hash),
        fixed_base_map_sha256=np.asarray(controls["base_map_sha256"]),
    )
    for output in (args.map_output, args.activation_geometry_output):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_fsm_map(enhanced, args.map_output)
    np.savez(args.activation_geometry_output, **geometry)
    done = {"provenance": plan, "reserve_count": int(reserve_ids.size), "reserve_source_names": CANDIDATE_SOURCE_NAMES}
    done_path.write_text(json.dumps(done, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", **done}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
