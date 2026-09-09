#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/task1_strict_baseline}"
CONDA_ENV="${CONDA_ENV:-splatam}"
FIRST_POSE_PROTOCOL="${FIRST_POSE_PROTOCOL:-gt_first_frame}"
MODE="${1:-smoke}"

ROOM0_CONFIG="$ROOT/configs/fsm/replica_fsm.py"
OFFICE0_CONFIG="$ROOT/configs/fsm/generated/replica_office0_fsm.py"
ROOM0_MAP="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_room0_0/full/maps/full_ratio_1/fsm_map.npz"
OFFICE0_MAP="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_office0_0/full/maps/full_ratio_1/fsm_map.npz"

if [[ "$OUT_ROOT" == *"/audit_v2_full_rerun/"* || "$OUT_ROOT" == *"/audit_v2_full_rerun" ]]; then
  printf 'Refusing to write strict results inside audit_v2_full_rerun: %s\n' "$OUT_ROOT" >&2
  exit 2
fi

case "$FIRST_POSE_PROTOCOL" in
  gt_first_frame|identity) ;;
  *)
    printf 'FIRST_POSE_PROTOCOL must be gt_first_frame or identity for this baseline runner\n' >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke|full|all) ;;
  *)
    printf 'Usage: %s [smoke|full|all]\n' "$0" >&2
    exit 2
    ;;
esac

for input_path in "$ROOM0_CONFIG" "$OFFICE0_CONFIG" "$ROOM0_MAP" "$OFFICE0_MAP"; do
  if [[ ! -f "$input_path" ]]; then
    printf 'Missing required input: %s\n' "$input_path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT/config_snapshot" "$OUT_ROOT/localize" "$OUT_ROOT/logs" "$OUT_ROOT/.state"
cp "$ROOM0_CONFIG" "$OUT_ROOT/config_snapshot/replica_room0_fsm.py"
cp "$OFFICE0_CONFIG" "$OUT_ROOT/config_snapshot/replica_office0_fsm.py"

export ROOT OUT_ROOT FIRST_POSE_PROTOCOL ROOM0_CONFIG OFFICE0_CONFIG ROOM0_MAP OFFICE0_MAP
conda run -n "$CONDA_ENV" python -c '
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = os.environ["ROOT"]
status = subprocess.run(
    ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
).stdout.splitlines()
manifest = {
    "experiment": "mvp_observable_v1_task1_strict_baseline",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "evaluation_protocol": "mvp_strict",
    "first_pose_protocol": os.environ["FIRST_POSE_PROTOCOL"],
    "uses_gt_first_pose": os.environ["FIRST_POSE_PROTOCOL"] == "gt_first_frame",
    "uses_gt_current_frame_initialization": False,
    "uses_gt_fallback": False,
    "motion_prior": "last_valid_pose",
    "loop_closure": False,
    "relocalization": False,
    "backend": "numpy_point_to_point",
    "git_status_short": status,
    "python": platform.python_version(),
    "inputs": {
        "replica_room0_0": {
            "config": os.environ["ROOM0_CONFIG"],
            "map": os.environ["ROOM0_MAP"],
        },
        "replica_office0_0": {
            "config": os.environ["OFFICE0_CONFIG"],
            "map": os.environ["OFFICE0_MAP"],
        },
    },
}
Path(os.environ["OUT_ROOT"], "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
'

run_case() {
  local name="$1"
  local config="$2"
  local map="$3"
  local start="$4"
  local end="$5"
  local output="$OUT_ROOT/localize/$name"
  local log="$OUT_ROOT/logs/$name.log"
  local state="$OUT_ROOT/.state/$name.done"

  if [[ -f "$state" ]]; then
    printf '[skip] %s\n' "$name"
    return
  fi

  printf '[run] %s frames=[%s,%s)\n' "$name" "$start" "$end"
  conda run -n "$CONDA_ENV" python scripts/fsm_localize.py \
    --config "$config" \
    --map "$map" \
    --output-dir "$output" \
    --protocol mvp_strict \
    --first-pose-protocol "$FIRST_POSE_PROTOCOL" \
    --backend numpy_point_to_point \
    --start "$start" \
    --end "$end" >"$log" 2>&1
  touch "$state"
  printf '[done] %s\n' "$name"
}

if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  run_case "replica_room0_0_smoke_0_50" "$ROOM0_CONFIG" "$ROOM0_MAP" 0 50
fi

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
  run_case "replica_room0_0_full" "$ROOM0_CONFIG" "$ROOM0_MAP" 0 -1
  run_case "replica_office0_0_full" "$OFFICE0_CONFIG" "$OFFICE0_MAP" 0 -1
fi

printf 'Task 1 strict baseline mode=%s output=%s\n' "$MODE" "$OUT_ROOT"
