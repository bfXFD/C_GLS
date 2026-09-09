#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SCENES="${SCENES:-both}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/core_route_v8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-500}"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2
    exit 2
  fi
done
if [[ "$SCENES" != "room0" && "$SCENES" != "office0" && "$SCENES" != "both" ]]; then
  printf 'SCENES must be room0, office0, or both\n' >&2
  exit 2
fi
if ! [[ "$FRAME_START" =~ ^[0-9]+$ && "$FRAME_END" =~ ^[0-9]+$ ]] || (( FRAME_END <= FRAME_START )); then
  printf 'FRAME_START and FRAME_END must be integers with FRAME_END > FRAME_START\n' >&2
  exit 2
fi
if [[ "$OUT_ROOT" == *"/audit_v2_full_rerun/"* || "$OUT_ROOT" == *"/audit_v2_full_rerun" ]]; then
  printf 'Refusing to write inside frozen audit output: %s\n' "$OUT_ROOT" >&2
  exit 2
fi

run_scene() {
  local scene="$1"
  local dataset="replica_${scene}_0"
  local map="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/$dataset/logdet/fsm_map.npz"
  local geometry="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/$dataset/logdet/activation_geometry.npz"
  for path in "$map" "$geometry"; do
    if [[ ! -f "$path" ]]; then
      printf 'Missing required map or sidecar: %s\n' "$path" >&2
      exit 1
    fi
  done
  local -a cmd=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
    --output-root "$OUT_ROOT/$scene" --start "$FRAME_START" --end "$FRAME_END"
    --case "baseline,$ROOT/configs/fsm/mvp_observable/replica_${scene}.py,$map"
    --case "corrweak128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_corr_weak128_info_v3.py,$map,$geometry"
    --case "weak128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_weak128_info_v3.py,$map,$geometry"
    --case "saliency128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_saliency128_info_v3.py,$map,$geometry")
  for seed in 0 1 2 3 4; do
    cmd+=(--case "random128_seed${seed},$ROOT/configs/fsm/mvp_observable/replica_${scene}_random128_info_v3_seed${seed}.py,$map,$geometry")
  done
  if [[ "$SKIP_EXISTING" == "1" ]]; then cmd+=(--skip-existing); fi
  printf '[plan]'; printf ' %q' "${cmd[@]}"; printf '\n'
  if [[ "$PLAN_ONLY" == "0" ]]; then "${cmd[@]}"; fi
}

if [[ "$SCENES" == "room0" || "$SCENES" == "both" ]]; then run_scene room0; fi
if [[ "$SCENES" == "office0" || "$SCENES" == "both" ]]; then run_scene office0; fi
