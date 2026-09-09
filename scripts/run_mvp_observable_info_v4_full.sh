#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SCENES="${SCENES:-both}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/core_route_v10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-500}"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$SCENES" == "room0" || "$SCENES" == "office0" || "$SCENES" == "both" ]] || { printf 'SCENES must be room0, office0, or both\n' >&2; exit 2; }
if ! [[ "$FRAME_START" =~ ^[0-9]+$ && "$FRAME_END" =~ ^[0-9]+$ ]] || (( FRAME_END <= FRAME_START )); then
  printf 'FRAME_START and FRAME_END must be integers with FRAME_END > FRAME_START\n' >&2
  exit 2
fi
[[ "$OUT_ROOT" != *"/audit_v2_full_rerun/"* && "$OUT_ROOT" != *"/audit_v2_full_rerun" ]] || { printf 'Refusing frozen audit output\n' >&2; exit 2; }

run_scene() {
  local scene="$1" dataset="replica_${1}_0"
  local map="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/$dataset/logdet/fsm_map.npz"
  local geometry="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/$dataset/logdet/activation_geometry.npz"
  [[ -f "$map" && -f "$geometry" ]] || { printf 'Missing map or sidecar for %s\n' "$scene" >&2; exit 1; }
  local -a cmd=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
    --output-root "$OUT_ROOT/$scene" --start "$FRAME_START" --end "$FRAME_END"
    --case "baseline,$ROOT/configs/fsm/mvp_observable/replica_${scene}.py,$map"
    --case "corrweak128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_corr_weak128_info_v4.py,$map,$geometry"
    --case "weak128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_weak128_info_v4.py,$map,$geometry"
    --case "saliency128,$ROOT/configs/fsm/mvp_observable/replica_${scene}_saliency128_info_v4.py,$map,$geometry")
  for seed in 0 1 2 3 4; do
    cmd+=(--case "random128_seed${seed},$ROOT/configs/fsm/mvp_observable/replica_${scene}_random128_info_v4_seed${seed}.py,$map,$geometry")
  done
  [[ "$SKIP_EXISTING" == "0" ]] || cmd+=(--skip-existing)
  printf '[plan]'; printf ' %q' "${cmd[@]}"; printf '\n'
  [[ "$PLAN_ONLY" == "1" ]] || "${cmd[@]}"
}

[[ "$SCENES" == "office0" ]] || run_scene room0
[[ "$SCENES" == "room0" ]] || run_scene office0
