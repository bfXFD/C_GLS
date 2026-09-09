#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SCENES="${SCENES:-both}"
STAGE="${STAGE:-maps}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/room12_generalization_v1}"
MAP_ROOT="${MAP_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-500}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$SCENES" == "room1" || "$SCENES" == "room2" || "$SCENES" == "both" ]] || { printf 'SCENES must be room1, room2, or both\n' >&2; exit 2; }
[[ "$STAGE" == "maps" || "$STAGE" == "paired" || "$STAGE" == "score" || "$STAGE" == "stats" || "$STAGE" == "closed_loop" ]] || {
  printf 'STAGE must be maps, paired, score, stats, or closed_loop\n' >&2
  exit 2
}
if ! [[ "$FRAME_START" =~ ^[0-9]+$ && "$FRAME_END" =~ ^[0-9]+$ ]] || (( FRAME_END <= FRAME_START )); then
  printf 'FRAME_START and FRAME_END must be integers with FRAME_END > FRAME_START\n' >&2
  exit 2
fi
if ! [[ "$BOOTSTRAP_REPLICATES" =~ ^[1-9][0-9]*$ && "$BOOTSTRAP_SEED" =~ ^[0-9]+$ ]]; then
  printf 'BOOTSTRAP_REPLICATES must be positive and BOOTSTRAP_SEED non-negative\n' >&2
  exit 2
fi
for path in "$OUT_ROOT" "$MAP_ROOT"; do
  [[ "$path" != *"/audit_v2_full_rerun/"* && "$path" != *"/audit_v2_full_rerun" ]] || {
    printf 'Refusing frozen audit path: %s\n' "$path" >&2
    exit 2
  }
done

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

run_scene() {
  local scene="$1"
  local dataset="replica_${scene}_0"
  local params="/home/ras/xbf/SplaTAM/experiments/Replica/${scene}_0/params.npz"
  local config="$ROOT/configs/fsm/mvp_observable/replica_${scene}.py"
  local paired_config="$ROOT/configs/fsm/mvp_observable/replica_${scene}_paired_core_route.py"
  local corrweak_config="$ROOT/configs/fsm/mvp_observable/replica_${scene}_corr_weak128_info_v4.py"
  local map_dir="$MAP_ROOT/$dataset/logdet"
  local map="$map_dir/fsm_map.npz"
  local geometry="$map_dir/activation_geometry.npz"
  local paired_root="$OUT_ROOT/paired/$scene"
  local paired_case="$paired_root/localize/paired"
  local score_root="$OUT_ROOT/paired_gt/$scene"
  local stats_root="$OUT_ROOT/paired_stats/$scene"
  local closed_loop_root="$OUT_ROOT/closed_loop/$scene"
  local -a skip=()
  [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)

  require_file "$params"
  require_file "$config"
  require_file "$paired_config"
  require_file "$corrweak_config"

  case "$STAGE" in
    maps)
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
        --params "$params" --ratio 0.2 --max-reference-poses 64 \
        --candidate-cap 4096 --pair-cap 200000 --fill-strategy scale_opacity \
        --greedy-core-cap 128 --replacement-count 128 --reserve-count 4096 \
        --normal-k 20 --regularization 1e-6 --min-eigen-every 0 --min-eigen-bonus 0 \
        --output "$map_dir/selection.npz" --map-output "$map" \
        --activation-geometry-output "$geometry" "${skip[@]}"
      ;;
    paired)
      require_file "$map"
      require_file "$geometry"
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
        --output-root "$paired_root" --start "$FRAME_START" --end "$FRAME_END" \
        --case "paired,$paired_config,$map,$geometry" "${skip[@]}"
      ;;
    score)
      require_file "$paired_case/paired_reserve/paired_reserve_ids.npz"
      require_file "$paired_case/trajectory_gt.txt"
      require_file "$paired_case/trajectory_est.txt"
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_score_paired_reserve_gt.py" \
        --paired-npz "$paired_case/paired_reserve/paired_reserve_ids.npz" \
        --trajectory-gt "$paired_case/trajectory_gt.txt" \
        --trajectory-est "$paired_case/trajectory_est.txt" \
        --output-dir "$score_root"
      ;;
    stats)
      require_file "$paired_case/paired_reserve/paired_reserve_rows.csv"
      require_file "$score_root/paired_reserve_gt_rows.csv"
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_paired_frame_stats.py" \
        --paired-rows "$paired_case/paired_reserve/paired_reserve_rows.csv" \
        --gt-rows "$score_root/paired_reserve_gt_rows.csv" \
        --output-dir "$stats_root" --budgets 64 128 256 \
        --expected-random-seeds 0 1 2 3 4 --require-complete-random-panel \
        --complete-across-budgets --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
        --bootstrap-seed "$BOOTSTRAP_SEED"
      ;;
    closed_loop)
      require_file "$map"
      require_file "$geometry"
      run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
        --output-root "$closed_loop_root" --start "$FRAME_START" --end "$FRAME_END" \
        --case "baseline,$config,$map" \
        --case "corrweak128,$corrweak_config,$map,$geometry" "${skip[@]}"
      ;;
  esac
}

[[ "$SCENES" == "room2" ]] || run_scene room1
[[ "$SCENES" == "room1" ]] || run_scene room2

printf 'Room1/Room2 generalization stage=%s plan_only=%s output=%s\n' "$STAGE" "$PLAN_ONLY" "$OUT_ROOT"
