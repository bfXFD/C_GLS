#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SEQUENCES="${SEQUENCES:-all}"
STAGE="${STAGE:-all}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_core_v1}"
MAP_ROOT="${MAP_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PAIRED_FRAME_END="${PAIRED_FRAME_END:-500}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$STAGE" == "maps" || "$STAGE" == "paired" || "$STAGE" == "score" || "$STAGE" == "stats" || "$STAGE" == "closed_loop" || "$STAGE" == "analysis" || "$STAGE" == "all" ]] || {
  printf 'STAGE must be maps, paired, score, stats, closed_loop, analysis, or all\n' >&2
  exit 2
}
if ! [[ "$PAIRED_FRAME_END" =~ ^[1-9][0-9]*$ && "$BOOTSTRAP_REPLICATES" =~ ^[1-9][0-9]*$ && "$BOOTSTRAP_SEED" =~ ^[0-9]+$ ]]; then
  printf 'PAIRED_FRAME_END and BOOTSTRAP_REPLICATES must be positive; BOOTSTRAP_SEED must be non-negative\n' >&2
  exit 2
fi
for path in "$OUT_ROOT" "$MAP_ROOT"; do
  [[ "$path" != *"/audit_v2_full_rerun/"* && "$path" != *"/audit_v2_full_rerun" ]] || {
    printf 'Refusing frozen audit path: %s\n' "$path" >&2
    exit 2
  }
done

declare -a ALL_SEQUENCES=(
  freiburg1_desk
  freiburg2_xyz
  freiburg3_long_office_household
)

sequence_enabled() {
  [[ "$SEQUENCES" == "all" || ",${SEQUENCES}," == *",$1,"* ]]
}

declare -a SELECTED_SEQUENCES=()
for sequence in "${ALL_SEQUENCES[@]}"; do
  sequence_enabled "$sequence" && SELECTED_SEQUENCES+=("$sequence")
done
[[ "${#SELECTED_SEQUENCES[@]}" -gt 0 ]] || { printf 'SEQUENCES selected no known TUM sequence\n' >&2; exit 2; }

print_command() {
  printf '[plan]'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  print_command "$@"
  [[ "$PLAN_ONLY" == "1" ]] || "$@"
}

require_file() {
  [[ "$PLAN_ONLY" == "1" || -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }
}

run_output_command() {
  local expected="$1"
  shift
  if [[ "$SKIP_EXISTING" == "1" && -f "$expected" ]]; then
    printf '[skip] existing %s\n' "$expected"
    return
  fi
  run_command "$@"
}

run_maps() {
  local params="$1" map_dir="$2" map="$3" geometry="$4"
  local -a skip=()
  [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
    --params "$params" --ratio 0.1 --max-reference-poses 64 \
    --candidate-cap 4096 --pair-cap 200000 --fill-strategy scale_opacity \
    --greedy-core-cap 128 --replacement-count 128 --reserve-count 4096 \
    --normal-k 20 --regularization 1e-6 --min-eigen-every 0 --min-eigen-bonus 0 \
    --output "$map_dir/selection.npz" --map-output "$map" \
    --activation-geometry-output "$geometry" "${skip[@]}"
}

run_paired() {
  local paired_config="$1" map="$2" geometry="$3" paired_root="$4"
  local -a skip=()
  [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  require_file "$map"
  require_file "$geometry"
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$paired_root" --start 0 --end "$PAIRED_FRAME_END" \
    --case "paired,$paired_config,$map,$geometry" "${skip[@]}"
}

run_score() {
  local paired_case="$1" score_root="$2"
  require_file "$paired_case/paired_reserve/paired_reserve_ids.npz"
  require_file "$paired_case/trajectory_gt.txt"
  require_file "$paired_case/trajectory_est.txt"
  run_output_command "$score_root/paired_reserve_gt_summary.json" \
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_score_paired_reserve_gt.py" \
    --paired-npz "$paired_case/paired_reserve/paired_reserve_ids.npz" \
    --trajectory-gt "$paired_case/trajectory_gt.txt" \
    --trajectory-est "$paired_case/trajectory_est.txt" \
    --output-dir "$score_root"
}

run_stats() {
  local paired_case="$1" score_root="$2" stats_root="$3"
  require_file "$paired_case/paired_reserve/paired_reserve_rows.csv"
  require_file "$score_root/paired_reserve_gt_rows.csv"
  run_output_command "$stats_root/summary.json" \
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_paired_frame_stats.py" \
    --paired-rows "$paired_case/paired_reserve/paired_reserve_rows.csv" \
    --gt-rows "$score_root/paired_reserve_gt_rows.csv" \
    --output-dir "$stats_root" --budgets 64 128 256 \
    --expected-random-seeds 0 1 2 3 4 --require-complete-random-panel \
    --complete-across-budgets --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
}

run_closed_loop() {
  local config="$1" corrweak_config="$2" map="$3" geometry="$4" closed_root="$5"
  local -a skip=()
  [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  require_file "$map"
  require_file "$geometry"
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$closed_root" \
    --case "baseline,$config,$map" \
    --case "corrweak128,$corrweak_config,$map,$geometry" "${skip[@]}"
}

run_analysis() {
  local closed_root="$1" analysis_root="$2"
  local baseline="$closed_root/localize/baseline"
  local corrweak="$closed_root/localize/corrweak128"
  require_file "$baseline/summary.json"
  require_file "$corrweak/summary.json"
  run_output_command "$analysis_root/common_valid/result.json" \
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_common_valid.py" \
    --method "baseline=$baseline" --method "corrweak128=$corrweak" \
    --output-dir "$analysis_root/common_valid" --frame-deltas 1 10 30 100 \
    --bootstrap-reps 1000 --bootstrap-seed 0
  run_output_command "$analysis_root/critical_adoption/summary.json" \
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_critical_adoption.py" \
    --baseline "baseline=$baseline" --run "corrweak128=$corrweak" \
    --output-dir "$analysis_root/critical_adoption" --horizons 10 30 100 \
    --rpe-deltas 1 5 10 --burst-gaps 1 2 5 10 \
    --temporal-windows 1 2 5 10 30 100
}

run_sequence() {
  local sequence="$1"
  local params="$ROOT/experiments/TUM/${sequence}_seed0/params.npz"
  local config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}.py"
  local paired_config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}_paired_core_route.py"
  local corrweak_config="$ROOT/configs/fsm/mvp_observable/tum_${sequence}_corr_weak128_info_v4.py"
  local map_dir="$MAP_ROOT/tum_${sequence}/logdet"
  local map="$map_dir/fsm_map.npz"
  local geometry="$map_dir/activation_geometry.npz"
  local sequence_root="$OUT_ROOT/$sequence"
  local paired_root="$sequence_root/paired"
  local paired_case="$paired_root/localize/paired"
  local score_root="$sequence_root/paired_gt"
  local stats_root="$sequence_root/paired_stats"
  local closed_root="$sequence_root/closed_loop"
  local analysis_root="$sequence_root/analysis"

  require_file "$params"
  require_file "$config"
  require_file "$paired_config"
  require_file "$corrweak_config"
  printf '[sequence] %s\n' "$sequence"

  case "$STAGE" in
    maps) run_maps "$params" "$map_dir" "$map" "$geometry" ;;
    paired) run_paired "$paired_config" "$map" "$geometry" "$paired_root" ;;
    score) run_score "$paired_case" "$score_root" ;;
    stats) run_stats "$paired_case" "$score_root" "$stats_root" ;;
    closed_loop) run_closed_loop "$config" "$corrweak_config" "$map" "$geometry" "$closed_root" ;;
    analysis) run_analysis "$closed_root" "$analysis_root" ;;
    all)
      run_maps "$params" "$map_dir" "$map" "$geometry"
      run_paired "$paired_config" "$map" "$geometry" "$paired_root"
      run_score "$paired_case" "$score_root"
      run_stats "$paired_case" "$score_root" "$stats_root"
      run_closed_loop "$config" "$corrweak_config" "$map" "$geometry" "$closed_root"
      run_analysis "$closed_root" "$analysis_root"
      ;;
  esac
}

for sequence in "${SELECTED_SEQUENCES[@]}"; do
  run_sequence "$sequence"
done

printf 'TUM core stage=%s plan_only=%s sequences=%s output=%s\n' "$STAGE" "$PLAN_ONLY" "$SEQUENCES" "$OUT_ROOT"
