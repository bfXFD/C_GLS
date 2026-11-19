#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SUITE="${SUITE:-essential}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-full}"
SCENES="${SCENES:-replica:room2,replica:office0,replica:office4,tum:freiburg3_long_office_household}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/reviewer_ablation_suite_v2}"
CONFIG="$ROOT/configs/fsm/mvp_observable/reviewer_ablation_case.py"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
case "$SUITE" in
  essential|fixed_normals|budgets|primary_v4|perturb4|perturb8|all) ;;
  *) printf 'SUITE must be essential, fixed_normals, budgets, primary_v4, perturb4, perturb8, or all\n' >&2; exit 2 ;;
esac
[[ "$FRAME_START" =~ ^[0-9]+$ ]] || { printf 'FRAME_START must be non-negative\n' >&2; exit 2; }
if [[ "$FRAME_END" != "full" ]]; then
  [[ "$FRAME_END" =~ ^[0-9]+$ ]] && (( FRAME_END > FRAME_START )) || { printf 'Invalid FRAME_END\n' >&2; exit 2; }
fi
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path\n' >&2; exit 2; }
[[ -f "$CONFIG" ]] || { printf 'Missing config: %s\n' "$CONFIG" >&2; exit 1; }

write_suite_manifest() {
  [[ "$PLAN_ONLY" == "1" ]] && return
  mkdir -p "$OUT_ROOT"
  local manifest="$OUT_ROOT/suite_invocation.json"
  local payload
  payload="$(python - "$SUITE" "$SCENES" "$FRAME_START" "$FRAME_END" "$CONDA_ENV" <<'PY'
import json
import sys
print(json.dumps({
    "schema_version": 1,
    "suite": sys.argv[1],
    "scenes": sys.argv[2].split(","),
    "frame_start": int(sys.argv[3]),
    "frame_end": None if sys.argv[4] == "full" else int(sys.argv[4]),
    "conda_env": sys.argv[5],
    "serial_execution": True,
}, sort_keys=True))
PY
)"
  if [[ -f "$manifest" ]]; then
    [[ "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),sort_keys=True,separators=(",",":")))' "$manifest")" == \
       "$(python -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]),sort_keys=True,separators=(",",":")))' "$payload")" ]] || {
      printf 'Existing suite invocation differs: %s\n' "$manifest" >&2
      exit 2
    }
  else
    python - "$manifest" "$payload" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps(json.loads(sys.argv[2]), indent=2, sort_keys=True) + "\n")
PY
  fi
}

map_and_geometry() {
  local dataset="$1" scene="$2"
  if [[ "$dataset" == "replica" ]]; then
    local base
    if [[ "$scene" == office1 || "$scene" == office2 || "$scene" == office3 || "$scene" == office4 ]]; then
      base="$ROOT/experiments/FSM/mvp_observable_v1/replica_office1_4_untouched_v1/maps/replica_${scene}_0/logdet"
    else
      base="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_${scene}_0/logdet"
    fi
    MAP="$base/fsm_map.npz"
    GEOMETRY="$base/activation_geometry.npz"
  elif [[ "$dataset" == "tum" ]]; then
    local base="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/$scene"
    MAP="$base/fsm_map.npz"
    GEOMETRY="$base/activation_geometry.npz"
  else
    printf 'Unknown dataset: %s\n' "$dataset" >&2
    exit 2
  fi
}

run_case() {
  local dataset="$1" scene="$2" name="$3" method="$4" budget="$5" seed="$6"
  local target_mode="$7" normal_mode="$8" use_v4="$9" twist="${10:-null}"
  map_and_geometry "$dataset" "$scene"
  for path in "$MAP"; do [[ -f "$path" ]] || { printf 'Missing input: %s\n' "$path" >&2; exit 1; }; done
  [[ "$method" == "baseline" || -f "$GEOMETRY" ]] || { printf 'Missing sidecar: %s\n' "$GEOMETRY" >&2; exit 1; }
  local key="${dataset}_${scene}_${name}"
  local output="$OUT_ROOT/runs/$key"
  local -a command=(env
    "CGLS_REVIEW_DATASET=$dataset"
    "CGLS_REVIEW_SCENE=$scene"
    "CGLS_REVIEW_METHOD=$method"
    "CGLS_REVIEW_BUDGET=$budget"
    "CGLS_REVIEW_SEED=$seed"
    "CGLS_REVIEW_TARGET_MODE=$target_mode"
    "CGLS_REVIEW_NORMAL_MODE=$normal_mode"
    "CGLS_REVIEW_USE_V4=$use_v4"
    "CGLS_REVIEW_FIRST_POSE_TWIST=$twist"
    conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py"
    --output-root "$output" --start "$FRAME_START")
  [[ "$FRAME_END" == "full" ]] || command+=(--end "$FRAME_END")
  if [[ "$method" == "baseline" ]]; then
    command+=(--case "$key,$CONFIG,$MAP")
  else
    command+=(--case "$key,$CONFIG,$MAP,$GEOMETRY")
  fi
  [[ "$SKIP_EXISTING" == "0" ]] || command+=(--skip-existing)
  printf '[plan]'; printf ' %q' "${command[@]}"; printf '\n'
  [[ "$PLAN_ONLY" == "1" ]] || "${command[@]}"
}

run_scene() {
  local dataset="$1" scene="$2"
  printf '[scene] %s:%s\n' "$dataset" "$scene"
  case "$SUITE" in
    essential)
      run_case "$dataset" "$scene" baseline baseline 128 0 append augmented_local_pca 1
      run_case "$dataset" "$scene" rerun0 corrweak 128 0 rerun_primary frozen_concat 0
      for seed in 0 1 2 3 4; do
        run_case "$dataset" "$scene" "random128_s${seed}" random 128 "$seed" append augmented_local_pca 1
      done
      run_case "$dataset" "$scene" saliency128 saliency 128 0 append augmented_local_pca 1
      run_case "$dataset" "$scene" weak128 weak 128 0 append augmented_local_pca 1
      run_case "$dataset" "$scene" corrweak128 corrweak 128 0 append augmented_local_pca 1
      run_case "$dataset" "$scene" replace128 corrweak 128 0 replace augmented_local_pca 1
      run_case "$dataset" "$scene" corrweak128_no_v4 corrweak 128 0 append augmented_local_pca 0
      ;;
    fixed_normals)
      run_case "$dataset" "$scene" corrweak128_frozen_normals corrweak 128 0 append frozen_concat 1
      ;;
    budgets)
      for budget in 32 64 256; do
        run_case "$dataset" "$scene" "corrweak${budget}" corrweak "$budget" 0 append augmented_local_pca 1
      done
      ;;
    primary_v4)
      run_case "$dataset" "$scene" primary_only_v4 corrweak 128 0 rerun_primary frozen_concat 1
      ;;
    all)
      local original_suite="$SUITE"
      SUITE=essential; run_scene "$dataset" "$scene"
      SUITE=fixed_normals; run_scene "$dataset" "$scene"
      SUITE=budgets; run_scene "$dataset" "$scene"
      SUITE=primary_v4; run_scene "$dataset" "$scene"
      SUITE="$original_suite"
      ;;
    perturb4|perturb8) ;;
  esac
}

write_suite_manifest
IFS=',' read -r -a scene_specs <<< "$SCENES"
for spec in "${scene_specs[@]}"; do
  dataset="${spec%%:*}"
  scene="${spec#*:}"
  [[ "$dataset" != "$scene" ]] || { printf 'Scene must be dataset:scene, got %s\n' "$spec" >&2; exit 2; }
  run_scene "$dataset" "$scene"
done

if [[ "$SUITE" == "perturb4" || "$SUITE" == "perturb8" || "$SUITE" == "all" ]]; then
  perturb_names=(txp02 txn02 yawp1 yawn1 typ02 tyn02 txp05_yawp2 txn05_yawn2)
  perturb_twists=(
    '[0,0,0,0.02,0,0]' '[0,0,0,-0.02,0,0]'
    '[0,0,0.017453292519943295,0,0,0]' '[0,0,-0.017453292519943295,0,0,0]'
    '[0,0,0,0,0.02,0]' '[0,0,0,0,-0.02,0]'
    '[0,0,0.03490658503988659,0.05,0,0]' '[0,0,-0.03490658503988659,-0.05,0,0]'
  )
  perturb_count=8
  [[ "$SUITE" == "perturb4" ]] && perturb_count=4
  for spec in "${scene_specs[@]}"; do
    dataset="${spec%%:*}"; scene="${spec#*:}"
    for ((index=0; index<perturb_count; index++)); do
      name="${perturb_names[$index]}"; twist="${perturb_twists[$index]}"
      run_case "$dataset" "$scene" "perturb_${name}_baseline" baseline 128 0 append augmented_local_pca 1 "$twist"
      run_case "$dataset" "$scene" "perturb_${name}_corrweak128" corrweak 128 0 append augmented_local_pca 1 "$twist"
    done
  done
fi

if [[ "$PLAN_ONLY" == "0" ]]; then
  conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_collect_reviewer_ablation.py" \
    --root "$OUT_ROOT" --output "$OUT_ROOT/summary/run_index.csv"
fi
printf 'Reviewer suite complete: suite=%s plan_only=%s output=%s\n' "$SUITE" "$PLAN_ONLY" "$OUT_ROOT"
