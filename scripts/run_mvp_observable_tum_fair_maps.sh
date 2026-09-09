#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SEQUENCES="${SEQUENCES:-all}"
STAGE="${STAGE:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_fair_maps_v1}"
OBS_MAP_ROOT="${OBS_MAP_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps}"
AUDIT_ROOT="$ROOT/experiments/FSM/audit_v2_full_rerun/main"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$STAGE" == "localize" || "$STAGE" == "analysis" || "$STAGE" == "all" ]] || {
  printf 'STAGE must be localize, analysis, or all\n' >&2
  exit 2
}
[[ "$OUT_ROOT" != *"/audit_v2_full_rerun/"* && "$OUT_ROOT" != *"/audit_v2_full_rerun" ]] || {
  printf 'Refusing frozen audit output path: %s\n' "$OUT_ROOT" >&2
  exit 2
}

declare -a ALL_SEQUENCES=(freiburg1_desk freiburg2_xyz freiburg3_long_office_household)
declare -a SELECTED=()
for sequence in "${ALL_SEQUENCES[@]}"; do
  [[ "$SEQUENCES" == "all" || ",${SEQUENCES}," == *",$sequence,"* ]] && SELECTED+=("$sequence")
done
[[ "${#SELECTED[@]}" -gt 0 ]] || { printf 'SEQUENCES selected no known TUM sequence\n' >&2; exit 2; }

print_command() { printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; [[ "$PLAN_ONLY" == "1" ]] || "$@"; }
require_file() { [[ "$PLAN_ONLY" == "1" || -f "$1" ]] || { printf 'Missing input: %s\n' "$1" >&2; exit 1; }; }

run_sequence() {
  local sequence="$1"
  local config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}.py"
  local corr_config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}_corr_weak128_info_v4.py"
  local audit="$AUDIT_ROOT/tum_${sequence}"
  local full="$audit/full/maps/full_ratio_1/fsm_map.npz"
  local voxel="$audit/pareto/maps/voxel_scale_opacity_ratio_0p1/fsm_map.npz"
  local fps="$audit/pareto/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz"
  local observable="$OBS_MAP_ROOT/tum_${sequence}/logdet/fsm_map.npz"
  local geometry="$OBS_MAP_ROOT/tum_${sequence}/logdet/activation_geometry.npz"
  local output="$OUT_ROOT/$sequence"
  local closed="$output/closed_loop"
  local analysis="$output/common_valid"
  local -a skip=()
  [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)

  for path in "$config" "$corr_config" "$full" "$voxel" "$fps" "$observable" "$geometry"; do require_file "$path"; done
  printf '[sequence] %s\n' "$sequence"
  if [[ "$STAGE" == "localize" || "$STAGE" == "all" ]]; then
    run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
      --output-root "$closed" \
      --case "full,$config,$full" \
      --case "cgls_voxel10,$config,$voxel" \
      --case "cgls_fps10,$config,$fps" \
      --case "observable10,$config,$observable" \
      --case "corrweak128,$corr_config,$observable,$geometry" "${skip[@]}"
  fi
  if [[ "$STAGE" == "analysis" || "$STAGE" == "all" ]]; then
    for case in full cgls_voxel10 cgls_fps10 observable10 corrweak128; do
      require_file "$closed/localize/$case/summary.json"
    done
    if [[ "$SKIP_EXISTING" == "1" && -f "$analysis/result.json" ]]; then
      printf '[skip] existing %s\n' "$analysis/result.json"
    else
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_common_valid.py" \
        --method "full=$closed/localize/full" \
        --method "cgls_voxel10=$closed/localize/cgls_voxel10" \
        --method "cgls_fps10=$closed/localize/cgls_fps10" \
        --method "observable10=$closed/localize/observable10" \
        --method "corrweak128=$closed/localize/corrweak128" \
        --output-dir "$analysis" --frame-deltas 1 10 30 100 \
        --bootstrap-reps 1000 --bootstrap-seed 0
    fi
  fi
}

for sequence in "${SELECTED[@]}"; do run_sequence "$sequence"; done
printf 'TUM fair maps stage=%s plan_only=%s sequences=%s output=%s\n' "$STAGE" "$PLAN_ONLY" "$SEQUENCES" "$OUT_ROOT"
