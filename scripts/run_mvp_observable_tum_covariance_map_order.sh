#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SEQUENCES="${SEQUENCES:-freiburg2_xyz,freiburg3_long_office_household}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_covariance_map_order_v1}"
AUDIT="$ROOT/experiments/FSM/audit_v2_full_rerun/main"
OBS="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps"

[[ "$PLAN_ONLY" == "0" || "$PLAN_ONLY" == "1" ]] || exit 2
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path\n' >&2; exit 2; }
ALL=(freiburg1_desk freiburg2_xyz freiburg3_long_office_household)
SELECTED=()
for sequence in "${ALL[@]}"; do
  [[ "$SEQUENCES" == "all" || ",$SEQUENCES," == *",$sequence,"* ]] && SELECTED+=("$sequence")
done
[[ "${#SELECTED[@]}" -gt 0 ]] || exit 2

for sequence in "${SELECTED[@]}"; do
  config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}_covariance.py"
  full="$AUDIT/tum_${sequence}/full/maps/full_ratio_1/fsm_map.npz"
  fps="$AUDIT/tum_${sequence}/pareto/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz"
  observable="$OBS/tum_${sequence}/logdet/fsm_map.npz"
  cmd=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
    --output-root "$OUT_ROOT/$sequence"
    --case "full,$config,$full"
    --case "fps10,$config,$fps"
    --case "observable10,$config,$observable"
    --skip-existing)
  printf '[plan]'; printf ' %q' "${cmd[@]}"; printf '\n'
  [[ "$PLAN_ONLY" == "1" ]] || "${cmd[@]}"
done
