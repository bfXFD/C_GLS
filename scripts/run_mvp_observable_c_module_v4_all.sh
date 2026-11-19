#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
DATASETS_FILTER="${DATASETS_FILTER:-all}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-full}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/c_module_runtimeweak_v4_all7}"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$FRAME_START" =~ ^[0-9]+$ ]] || { printf 'FRAME_START must be non-negative\n' >&2; exit 2; }
if [[ "$FRAME_END" != "full" ]]; then
  [[ "$FRAME_END" =~ ^[0-9]+$ ]] && (( FRAME_END > FRAME_START )) || {
    printf 'FRAME_END must be full or an integer greater than FRAME_START\n' >&2; exit 2;
  }
fi
[[ "$OUT_ROOT" != *"/audit_v2_full_rerun/"* && "$OUT_ROOT" != *"/audit_v2_full_rerun" ]] || {
  printf 'Refusing frozen audit output\n' >&2; exit 2;
}

enabled() {
  [[ "$DATASETS_FILTER" == "all" || ",${DATASETS_FILTER}," == *",$1,"* ]]
}

run_pair() {
  local key="$1" output="$2" corr_config="$3" weak_config="$4" map="$5" geometry="$6"
  enabled "$key" || return
  for path in "$corr_config" "$weak_config" "$map" "$geometry"; do
    [[ -f "$path" ]] || { printf 'Missing required input: %s\n' "$path" >&2; exit 1; }
  done
  local -a command=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
    --output-root "$OUT_ROOT/$output" --start "$FRAME_START")
  [[ "$FRAME_END" == "full" ]] || command+=(--end "$FRAME_END")
  command+=(
    --case "corrweak128,$corr_config,$map,$geometry"
    --case "weak128,$weak_config,$map,$geometry"
  )
  [[ "$SKIP_EXISTING" == "0" ]] || command+=(--skip-existing)
  printf '[plan]'; printf ' %q' "${command[@]}"; printf '\n'
  [[ "$PLAN_ONLY" == "1" ]] || "${command[@]}"
}

for scene in room0 room1 room2 office0; do
  map_root="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_${scene}_0/logdet"
  run_pair "replica_${scene}_0" "replica_${scene}_0" \
    "$ROOT/configs/fsm/mvp_observable/replica_${scene}_corr_weak128_info_v4.py" \
    "$ROOT/configs/fsm/mvp_observable/replica_${scene}_weak128_info_v4.py" \
    "$map_root/fsm_map.npz" "$map_root/activation_geometry.npz"
done

for sequence in freiburg1_desk freiburg2_xyz freiburg3_long_office_household; do
  map_root="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/$sequence"
  run_pair "tum_${sequence}" "tum_${sequence}" \
    "$ROOT/configs/fsm/mvp_observable/tum_${sequence}_corr_weak128_info_v4.py" \
    "$ROOT/configs/fsm/mvp_observable/tum_${sequence}_weak128_info_v4.py" \
    "$map_root/fsm_map.npz" "$map_root/activation_geometry.npz"
done

printf 'C-module V4 panel plan_only=%s datasets=%s frame_end=%s output=%s\n' \
  "$PLAN_ONLY" "$DATASETS_FILTER" "$FRAME_END" "$OUT_ROOT"
