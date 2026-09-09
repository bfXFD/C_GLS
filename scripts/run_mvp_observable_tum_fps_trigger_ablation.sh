#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_trigger_ablation_v1/freiburg3_long_office_household}"
MAP_ROOT="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg3_long_office_household"
BASE_CONFIG="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household.py"
MID_CONFIG="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household_corr_weak128_info_v4_gate_mid.py"
STRICT_CONFIG="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household_corr_weak128_info_v4_gate_strict.py"
MAP="$MAP_ROOT/fsm_map.npz"
GEOMETRY="$MAP_ROOT/activation_geometry.npz"

[[ "$PLAN_ONLY" == "0" || "$PLAN_ONLY" == "1" ]] || exit 2
[[ "$SKIP_EXISTING" == "0" || "$SKIP_EXISTING" == "1" ]] || exit 2
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path\n' >&2; exit 2; }
for path in "$BASE_CONFIG" "$MID_CONFIG" "$STRICT_CONFIG" "$MAP" "$GEOMETRY"; do
  [[ "$PLAN_ONLY" == "1" || -f "$path" ]] || { printf 'Missing %s\n' "$path" >&2; exit 1; }
done
cmd=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
  --output-root "$OUT_ROOT"
  --case "fps10,$BASE_CONFIG,$MAP"
  --case "gate_mid,$MID_CONFIG,$MAP,$GEOMETRY"
  --case "gate_strict,$STRICT_CONFIG,$MAP,$GEOMETRY")
[[ "$SKIP_EXISTING" == "0" ]] || cmd+=(--skip-existing)
printf '[plan]'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "$PLAN_ONLY" == "1" ]] || "${cmd[@]}"
