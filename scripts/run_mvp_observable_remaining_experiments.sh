#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STAGE="${STAGE:-all}"
TIMING_ROOT="${TIMING_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/timing_v1}"
COV_ROOT="${COV_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_covariance_map_order_v1}"

[[ "$PLAN_ONLY" == "0" || "$PLAN_ONLY" == "1" ]] || { printf 'PLAN_ONLY must be 0 or 1\n' >&2; exit 2; }
[[ "$SKIP_EXISTING" == "0" || "$SKIP_EXISTING" == "1" ]] || { printf 'SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
[[ "$STAGE" == "all" || "$STAGE" == "covariance" || "$STAGE" == "office0" || "$STAGE" == "tum_fr2" ]] || {
  printf 'STAGE must be all, covariance, office0, or tum_fr2\n' >&2
  exit 2
}
for path in "$TIMING_ROOT" "$COV_ROOT"; do
  [[ "$path" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path: %s\n' "$path" >&2; exit 2; }
done

print_command() { printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; [[ "$PLAN_ONLY" == "1" ]] || "$@"; }
require_file() { [[ "$PLAN_ONLY" == "1" || -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }

run_analysis() {
  local expected="$1"
  shift
  if [[ "$SKIP_EXISTING" == "1" && -f "$expected" ]]; then
    printf '[skip] existing %s\n' "$expected"
  else
    run_command "$@"
  fi
}

run_case() {
  local output="$1"
  local case_spec="$2"
  local case_name="${case_spec%%,*}"
  local done="$output/.state/$case_name.done.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$done" ]]; then
    printf '[skip] existing %s\n' "$done"
    return
  fi
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$output" --case "$case_spec" --skip-existing
}

run_covariance_analysis() {
  local sequence="$1"
  local base="$COV_ROOT/$sequence"
  for method in full fps10 observable10; do
    require_file "$base/localize/$method/summary.json"
  done
  run_analysis "$base/common_valid/result.json" \
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_common_valid.py" \
    --method "full=$base/localize/full" \
    --method "fps10=$base/localize/fps10" \
    --method "observable10=$base/localize/observable10" \
    --output-dir "$base/common_valid" --frame-deltas 1 10 30 100 \
    --bootstrap-reps 1000 --bootstrap-seed 0
}

run_office0_timing() {
  local map="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/fsm_map.npz"
  local geometry="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/activation_geometry.npz"
  local baseline_config="$ROOT/configs/fsm/mvp_observable/replica_office0.py"
  local corrweak_config="$ROOT/configs/fsm/mvp_observable/replica_office0_corr_weak128_info_v4.py"
  for path in "$map" "$geometry" "$baseline_config" "$corrweak_config"; do require_file "$path"; done
  run_case "$TIMING_ROOT/office0_repeat1_baseline" "baseline,$baseline_config,$map"
  run_case "$TIMING_ROOT/office0_repeat1_corrweak" "corrweak128,$corrweak_config,$map,$geometry"
  run_case "$TIMING_ROOT/office0_repeat2_corrweak" "corrweak128,$corrweak_config,$map,$geometry"
  run_case "$TIMING_ROOT/office0_repeat2_baseline" "baseline,$baseline_config,$map"
  run_case "$TIMING_ROOT/office0_repeat3_baseline" "baseline,$baseline_config,$map"
  run_case "$TIMING_ROOT/office0_repeat3_corrweak" "corrweak128,$corrweak_config,$map,$geometry"
}

run_tum_fr2_timing() {
  local map="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg2_xyz/fsm_map.npz"
  local geometry="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg2_xyz/activation_geometry.npz"
  local baseline_config="$ROOT/configs/fsm/mvp_observable/tum_freiburg2_xyz.py"
  local corrweak_config="$ROOT/configs/fsm/mvp_observable/tum_freiburg2_xyz_corr_weak128_info_v4.py"
  for path in "$map" "$geometry" "$baseline_config" "$corrweak_config"; do require_file "$path"; done
  run_case "$TIMING_ROOT/tum_fr2_repeat1_baseline" "fps10,$baseline_config,$map"
  run_case "$TIMING_ROOT/tum_fr2_repeat1_corrweak" "fps_corrweak128,$corrweak_config,$map,$geometry"
  run_case "$TIMING_ROOT/tum_fr2_repeat2_corrweak" "fps_corrweak128,$corrweak_config,$map,$geometry"
  run_case "$TIMING_ROOT/tum_fr2_repeat2_baseline" "fps10,$baseline_config,$map"
  run_case "$TIMING_ROOT/tum_fr2_repeat3_baseline" "fps10,$baseline_config,$map"
  run_case "$TIMING_ROOT/tum_fr2_repeat3_corrweak" "fps_corrweak128,$corrweak_config,$map,$geometry"
}

printf '[stage] %s plan_only=%s skip_existing=%s\n' "$STAGE" "$PLAN_ONLY" "$SKIP_EXISTING"
if [[ "$STAGE" == "all" || "$STAGE" == "covariance" ]]; then
  run_covariance_analysis freiburg2_xyz
  run_covariance_analysis freiburg3_long_office_household
fi
if [[ "$STAGE" == "all" || "$STAGE" == "office0" ]]; then
  run_office0_timing
fi
if [[ "$STAGE" == "all" || "$STAGE" == "tum_fr2" ]]; then
  run_tum_fr2_timing
fi
printf 'Remaining experiments completed: stage=%s plan_only=%s\n' "$STAGE" "$PLAN_ONLY"
