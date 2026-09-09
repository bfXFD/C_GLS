#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1}"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
OBSERVABLE_RATIO="${OBSERVABLE_RATIO:-0.2}"
DATASETS_FILTER="${DATASETS_FILTER:-}"
STAGES="${STAGES:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_REFERENCE_POSES="${MAX_REFERENCE_POSES:-64}"
CANDIDATE_CAP="${CANDIDATE_CAP:-4096}"
GREEDY_CORE_CAP="${GREEDY_CORE_CAP:-128}"
WEAK_GREEDY_CORE_CAP="${WEAK_GREEDY_CORE_CAP:-64}"
LOGDET_REPLACEMENT_COUNT="${LOGDET_REPLACEMENT_COUNT:-128}"
WEAK_REPLACEMENT_COUNT="${WEAK_REPLACEMENT_COUNT:-64}"
PAIR_CAP="${PAIR_CAP:-200000}"
MIN_EIGEN_EVERY="${MIN_EIGEN_EVERY:-16}"
RESERVE_COUNT="${RESERVE_COUNT:-4096}"
ESTIMATED_OUTPUT_GIB="${ESTIMATED_OUTPUT_GIB:-8}"

if [[ "$OUT_ROOT" == *"/audit_v2_full_rerun/"* || "$OUT_ROOT" == *"/audit_v2_full_rerun" ]]; then
  printf 'Refusing to use frozen audit output as OUT_ROOT: %s\n' "$OUT_ROOT" >&2
  exit 2
fi
if [[ "$PLAN_ONLY" != "0" && "$PLAN_ONLY" != "1" ]]; then
  printf 'PLAN_ONLY must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$STAGES" != "maps" && "$STAGES" != "eval" && "$STAGES" != "all" ]]; then
  printf 'STAGES must be maps, eval, or all\n' >&2
  exit 2
fi
if ! [[ "$RESERVE_COUNT" =~ ^[0-9]+$ ]]; then
  printf 'RESERVE_COUNT must be a non-negative integer\n' >&2
  exit 2
fi

declare -a DATASETS=(
  replica_room0_0
  replica_room1_0
  replica_room2_0
  replica_office0_0
  tum_freiburg1_desk
  tum_freiburg2_xyz
  tum_freiburg3_long_office_household
)

declare -A CONFIGS=(
  [replica_room0_0]="$ROOT/configs/fsm/mvp_observable/replica_room0.py"
  [replica_room1_0]="$ROOT/configs/fsm/mvp_observable/replica_room1.py"
  [replica_room2_0]="$ROOT/configs/fsm/mvp_observable/replica_room2.py"
  [replica_office0_0]="$ROOT/configs/fsm/mvp_observable/replica_office0.py"
  [tum_freiburg1_desk]="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk.py"
  [tum_freiburg2_xyz]="$ROOT/configs/fsm/mvp_observable/tum_freiburg2_xyz.py"
  [tum_freiburg3_long_office_household]="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household.py"
)

declare -A PARAMS=(
  [replica_room0_0]="/home/ras/xbf/SplaTAM/experiments/Replica/room0_0/params.npz"
  [replica_room1_0]="/home/ras/xbf/SplaTAM/experiments/Replica/room1_0/params.npz"
  [replica_room2_0]="/home/ras/xbf/SplaTAM/experiments/Replica/room2_0/params.npz"
  [replica_office0_0]="/home/ras/xbf/SplaTAM/experiments/Replica/office0_0/params.npz"
  [tum_freiburg1_desk]="$ROOT/experiments/TUM/freiburg1_desk_seed0/params.npz"
  [tum_freiburg2_xyz]="$ROOT/experiments/TUM/freiburg2_xyz_seed0/params.npz"
  [tum_freiburg3_long_office_household]="$ROOT/experiments/TUM/freiburg3_long_office_household_seed0/params.npz"
)

declare -A BASELINE_MAPS=(
  [replica_room0_0]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_room0_0/full/maps/full_ratio_1/fsm_map.npz"
  [replica_room1_0]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_room1_0/full/maps/full_ratio_1/fsm_map.npz"
  [replica_room2_0]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_room2_0/full/maps/full_ratio_1/fsm_map.npz"
  [replica_office0_0]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_office0_0/full/maps/full_ratio_1/fsm_map.npz"
  [tum_freiburg1_desk]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/tum_freiburg1_desk/full/maps/full_ratio_1/fsm_map.npz"
  [tum_freiburg2_xyz]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/tum_freiburg2_xyz/full/maps/full_ratio_1/fsm_map.npz"
  [tum_freiburg3_long_office_household]="$ROOT/experiments/FSM/audit_v2_full_rerun/main/tum_freiburg3_long_office_household/full/maps/full_ratio_1/fsm_map.npz"
)

print_command() {
  printf '[plan]'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  print_command "$@"
  if [[ "$PLAN_ONLY" == "0" ]]; then
    "$@"
  fi
}

dataset_enabled() {
  [[ -z "$DATASETS_FILTER" || ",${DATASETS_FILTER}," == *",$1,"* ]]
}

declare -a SELECTED_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  if dataset_enabled "$dataset"; then
    SELECTED_DATASETS+=("$dataset")
  fi
done
if [[ "${#SELECTED_DATASETS[@]}" -eq 0 ]]; then
  printf 'DATASETS_FILTER selected no known datasets\n' >&2
  exit 2
fi

if [[ "$PLAN_ONLY" == "0" ]]; then
  lock_dir="$OUT_ROOT/.runner.lock"
  if ! mkdir -p "$OUT_ROOT" || ! mkdir "$lock_dir"; then
    printf 'another MVP observable runner is active: %s\n' "$lock_dir" >&2
    exit 2
  fi
  trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT INT TERM
  for path in "$ROOT/scripts/fsm_build_observable_map.py" "$ROOT/scripts/fsm_eval_long_horizon.py" "$ROOT/scripts/fsm_mvp_preflight.py"; do
    test -f "$path"
  done
  preflight=(conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_mvp_preflight.py" --output-root "$OUT_ROOT" --estimated-output-bytes "$((ESTIMATED_OUTPUT_GIB * 1024 * 1024 * 1024))")
  for dataset in "${SELECTED_DATASETS[@]}"; do
    test -f "${CONFIGS[$dataset]}"
    test -f "${PARAMS[$dataset]}"
    test -f "${BASELINE_MAPS[$dataset]}"
    preflight+=(--input "${CONFIGS[$dataset]}" --npz-input "${PARAMS[$dataset]}" --npz-input "${BASELINE_MAPS[$dataset]}")
  done
  "${preflight[@]}"
fi

printf 'safe defaults: poses=%s candidates=%s core=%s weak_core=%s pairs=%s min_eigen_every=%s reserve=%s; upper candidate scans=%s factor recomputations=%s\n' \
  "$MAX_REFERENCE_POSES" "$CANDIDATE_CAP" "$GREEDY_CORE_CAP" "$WEAK_GREEDY_CORE_CAP" "$PAIR_CAP" "$MIN_EIGEN_EVERY" "$RESERVE_COUNT" \
  "$((MAX_REFERENCE_POSES * CANDIDATE_CAP))" "$((CANDIDATE_CAP + GREEDY_CORE_CAP * (CANDIDATE_CAP - 1)))"

for dataset in "${SELECTED_DATASETS[@]}"; do
  scene_root="$OUT_ROOT/maps/$dataset"
  cheap_map="$scene_root/cheap_baseline/fsm_map.npz"
  logdet_map="$scene_root/logdet/fsm_map.npz"
  min_eigen_map="$scene_root/min_eigen/fsm_map.npz"

  skip_builder=()
  [[ "$SKIP_EXISTING" == "1" ]] && skip_builder=(--skip-existing)
  if [[ "$STAGES" == "maps" || "$STAGES" == "all" ]]; then
    run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
      --params "${PARAMS[$dataset]}" --ratio "$OBSERVABLE_RATIO" --max-reference-poses "$MAX_REFERENCE_POSES" \
      --candidate-cap "$CANDIDATE_CAP" --pair-cap "$PAIR_CAP" "${skip_builder[@]}" \
      --output "$scene_root/cheap_baseline/selection.npz" --map-output "$cheap_map" \
      --greedy-core-cap 0 --replacement-count 0 --min-eigen-every 0 --min-eigen-bonus 0
    run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
      --params "${PARAMS[$dataset]}" --ratio "$OBSERVABLE_RATIO" --max-reference-poses "$MAX_REFERENCE_POSES" \
      --candidate-cap "$CANDIDATE_CAP" --pair-cap "$PAIR_CAP" "${skip_builder[@]}" \
      --output "$scene_root/logdet/selection.npz" --map-output "$logdet_map" \
      --activation-geometry-output "$scene_root/logdet/activation_geometry.npz" \
      --greedy-core-cap "$GREEDY_CORE_CAP" --replacement-count "$LOGDET_REPLACEMENT_COUNT" \
      --min-eigen-every 0 --min-eigen-bonus 0 --reserve-count "$RESERVE_COUNT"
    run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
      --params "${PARAMS[$dataset]}" --ratio "$OBSERVABLE_RATIO" --max-reference-poses "$MAX_REFERENCE_POSES" \
      --candidate-cap "$CANDIDATE_CAP" --pair-cap "$PAIR_CAP" "${skip_builder[@]}" \
      --output "$scene_root/min_eigen/selection.npz" --map-output "$min_eigen_map" \
      --activation-geometry-output "$scene_root/min_eigen/activation_geometry.npz" \
      --greedy-core-cap "$WEAK_GREEDY_CORE_CAP" --replacement-count "$WEAK_REPLACEMENT_COUNT" \
      --min-eigen-every "$MIN_EIGEN_EVERY" --min-eigen-bonus 0.1 --reserve-count "$RESERVE_COUNT"
  fi

  if [[ "$STAGES" == "eval" || "$STAGES" == "all" ]]; then
    eval_skip=()
    [[ "$SKIP_EXISTING" == "1" ]] && eval_skip=(--skip-existing)
    run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
      --output-root "$OUT_ROOT/long_horizon/$dataset" "${eval_skip[@]}" \
      --case "baseline,${CONFIGS[$dataset]},${BASELINE_MAPS[$dataset]}" \
      --case "cheap_scale_opacity_baseline,${CONFIGS[$dataset]},$cheap_map" \
      --case "observable_core_cheap_fill,${CONFIGS[$dataset]},$logdet_map" \
      --case "observable_core_weak_cheap_fill,${CONFIGS[$dataset]},$min_eigen_map"
  fi
done

printf 'mvp_observable stages planned=%s output=%s; frozen maps are referenced read-only and never copied\n' "$PLAN_ONLY" "$OUT_ROOT"
printf 'recommended first run: DATASETS_FILTER=replica_room0_0 STAGES=maps PLAN_ONLY=0 bash %s\n' "$0"
