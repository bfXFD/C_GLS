#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/core_route_v6/office0_0_500_closed_loop}"
MAP="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/fsm_map.npz"
GEOMETRY="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/activation_geometry.npz"

if [[ "$PLAN_ONLY" != "0" && "$PLAN_ONLY" != "1" ]]; then
  printf 'PLAN_ONLY must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$OUT_ROOT" == *"/audit_v2_full_rerun/"* || "$OUT_ROOT" == *"/audit_v2_full_rerun" ]]; then
  printf 'Refusing to write inside frozen audit output: %s\n' "$OUT_ROOT" >&2
  exit 2
fi
for path in "$MAP" "$GEOMETRY"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required input: %s\n' "$path" >&2
    exit 1
  fi
done

cmd=(conda run -n "$CONDA_ENV" python scripts/fsm_eval_long_horizon.py
  --output-root "$OUT_ROOT" --start 0 --end 500
  --case "baseline,$ROOT/configs/fsm/mvp_observable/replica_office0.py,$MAP"
  --case "corr_weak64,$ROOT/configs/fsm/mvp_observable/replica_office0_corr_weak64.py,$MAP,$GEOMETRY"
  --case "corr_weak128,$ROOT/configs/fsm/mvp_observable/replica_office0_corr_weak128.py,$MAP,$GEOMETRY"
  --case "weak128,$ROOT/configs/fsm/mvp_observable/replica_office0_core_route_weak128.py,$MAP,$GEOMETRY"
  --case "saliency128,$ROOT/configs/fsm/mvp_observable/replica_office0_core_route_saliency128.py,$MAP,$GEOMETRY")
for seed in 0 1 2 3 4; do
  cmd+=(--case "random128_seed${seed},$ROOT/configs/fsm/mvp_observable/replica_office0_core_route_random128_seed${seed}.py,$MAP,$GEOMETRY")
done

if [[ "$PLAN_ONLY" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
