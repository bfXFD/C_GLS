#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STAGE="${STAGE:-all}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:--1}"
HORIZONS="${HORIZONS:-1,3,5,10}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/required_acceptance_experiments_v1}"
REVERSE_ROOT="$OUT_ROOT/reverse_revisit"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do [[ "$value" == 0 || "$value" == 1 ]] || exit 2; done
case "$STAGE" in preflight|reverse|paired|horizon|label|freeze|evaluate|acceptance|all) ;; *) printf 'Invalid STAGE\n' >&2; exit 2;; esac
[[ "$FRAME_START" =~ ^[0-9]+$ ]] || exit 2
[[ "$FRAME_END" == -1 || "$FRAME_END" =~ ^[0-9]+$ ]] || exit 2
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output\n' >&2; exit 2; }

print_command(){ printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run(){ print_command "$@"; [[ "$PLAN_ONLY" == 1 ]] || "$@"; }
need(){ [[ "$PLAN_ONLY" == 1 || -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }
run_if_missing(){ local expected="$1"; shift; if [[ "$SKIP_EXISTING" == 1 && -f "$expected" ]]; then printf '[skip] %s\n' "$expected"; else run "$@"; fi; }

IFS=',' read -r -a horizon_values <<< "$HORIZONS"
[[ "${horizon_values[*]}" == "1 3 5 10" ]] || { printf 'HORIZONS must be exactly 1,3,5,10\n' >&2; exit 2; }

write_invocation(){
  [[ "$PLAN_ONLY" == 1 ]] && return
  mkdir -p "$OUT_ROOT"
  local path="$OUT_ROOT/suite_invocation.json"
  local payload
  payload="$(python - "$FRAME_START" "$FRAME_END" "$CONDA_ENV" <<'PY'
import json,sys
print(json.dumps({"schema_version":1,"frame_start":int(sys.argv[1]),"frame_end":int(sys.argv[2]),"conda_env":sys.argv[3],"horizons":[1,3,5,10],"development_scene":"office0","test_scenes":["room2","office4","fr3","reverse_desk2_to_desk"],"serial_execution":True},sort_keys=True))
PY
)"
  if [[ -f "$path" ]]; then
    [[ "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),sort_keys=True))' "$path")" == "$payload" ]] || { printf 'Invocation mismatch: %s\n' "$path" >&2; exit 2; }
  else
    python - "$path" "$payload" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps(json.loads(sys.argv[2]),indent=2,sort_keys=True)+"\n")
PY
  fi
}

declare -A CFG MAP GEOM
CFG[office0]="$ROOT/configs/fsm/mvp_observable/replica_office0_horizon_paired.py"
MAP[office0]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/fsm_map.npz"
GEOM[office0]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/activation_geometry.npz"
CFG[room2]="$ROOT/configs/fsm/mvp_observable/replica_room2_horizon_paired.py"
MAP[room2]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_room2_0/logdet/fsm_map.npz"
GEOM[room2]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_room2_0/logdet/activation_geometry.npz"
CFG[office4]="$ROOT/configs/fsm/mvp_observable/replica_office4_horizon_paired.py"
MAP[office4]="$ROOT/experiments/FSM/mvp_observable_v1/replica_office1_4_untouched_v1/maps/replica_office4_0/logdet/fsm_map.npz"
GEOM[office4]="$ROOT/experiments/FSM/mvp_observable_v1/replica_office1_4_untouched_v1/maps/replica_office4_0/logdet/activation_geometry.npz"
CFG[fr3]="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household_horizon_paired.py"
MAP[fr3]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/tum_freiburg3_long_office_household/logdet/fsm_map.npz"
GEOM[fr3]="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/tum_freiburg3_long_office_household/logdet/activation_geometry.npz"
CFG[reverse]="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_horizon_paired.py"
MAP[reverse]="$REVERSE_ROOT/map_artifacts/reserve/fsm_map.npz"
GEOM[reverse]="$REVERSE_ROOT/map_artifacts/reserve/activation_geometry.npz"

preflight(){
  for scene in office0 room2 office4 fr3; do need "${CFG[$scene]}"; need "${MAP[$scene]}"; need "${GEOM[$scene]}"; done
  for script in fsm_eval_long_horizon.py fsm_verify_hypotheses.py fsm_label_horizon_gt.py fsm_freeze_horizon_thresholds.py fsm_evaluate_horizon_test.py; do need "$ROOT/scripts/$script"; done
}

run_reverse(){
  run env PLAN_ONLY="$PLAN_ONLY" SKIP_EXISTING="$SKIP_EXISTING" FRAME_START="$FRAME_START" FRAME_END="$FRAME_END" \
    OUT_ROOT="$REVERSE_ROOT" bash "$ROOT/scripts/run_fsm_independent_revisit_desk2_to_desk.sh"
}

run_paired_scene(){
  local scene="$1"
  local output="$OUT_ROOT/paired/$scene"
  local expected="$output/localize/paired/paired_reserve/paired_reserve_ids.npz"
  need "${CFG[$scene]}"; need "${MAP[$scene]}"; need "${GEOM[$scene]}"
  local -a skip=(); [[ "$SKIP_EXISTING" == 0 ]] || skip=(--skip-existing)
  run_if_missing "$expected" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$output" --start "$FRAME_START" --end "$FRAME_END" \
    --case "paired,${CFG[$scene]},${MAP[$scene]},${GEOM[$scene]}" "${skip[@]}"
}

run_horizon_scene(){
  local scene="$1"
  local case="$OUT_ROOT/paired/$scene/localize/paired"
  need "$case/paired_reserve/paired_reserve_ids.npz"; need "$case/trajectory_gt.txt"
  for h in "${horizon_values[@]}"; do
    local min=2; [[ "$h" == 1 ]] && min=1
    local out="$OUT_ROOT/horizon_gtfree/$scene/h$h"
    run_if_missing "$out/horizon_branch_poses.npz" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_verify_hypotheses.py" \
      --config "${CFG[$scene]}" --map "${MAP[$scene]}" --localize-dir "$case" \
      --output-dir "$out" --policy runtime_corr_weak --budget 128 --horizon "$h" --min-common-valid-frames "$min"
  done
}

run_label_scene(){
  local scene="$1"
  local gt="$OUT_ROOT/paired/$scene/localize/paired/trajectory_gt.txt"
  need "$gt"
  for h in "${horizon_values[@]}"; do
    local horizon="$OUT_ROOT/horizon_gtfree/$scene/h$h" out="$OUT_ROOT/horizon_labels/$scene/h$h"
    need "$horizon/horizon_branch_poses.npz"
    run_if_missing "$out/horizon_gt_labels.csv" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_label_horizon_gt.py" \
      --horizon-dir "$horizon" --trajectory-gt "$gt" --output-dir "$out" \
      --translation-neutral-m 0.0 --rotation-neutral-deg 0.0
  done
}

run_freeze(){
  local output="$OUT_ROOT/frozen/horizon_thresholds_office0.json"
  if [[ "$SKIP_EXISTING" == 1 && -f "$output" ]]; then printf '[skip] immutable %s\n' "$output"; return; fi
  run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_freeze_horizon_thresholds.py" \
    --development "1:$OUT_ROOT/horizon_gtfree/office0/h1:$OUT_ROOT/horizon_labels/office0/h1" \
    --development "3:$OUT_ROOT/horizon_gtfree/office0/h3:$OUT_ROOT/horizon_labels/office0/h3" \
    --development "5:$OUT_ROOT/horizon_gtfree/office0/h5:$OUT_ROOT/horizon_labels/office0/h5" \
    --development "10:$OUT_ROOT/horizon_gtfree/office0/h10:$OUT_ROOT/horizon_labels/office0/h10" --output "$output"
}

run_evaluate_scene(){
  local scene="$1"
  local frozen="$OUT_ROOT/frozen/horizon_thresholds_office0.json"
  local output="$OUT_ROOT/evaluation/$scene.json"
  need "$frozen"
  run_if_missing "$output" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_evaluate_horizon_test.py" \
    --frozen-json "$frozen" \
    --test "1:$OUT_ROOT/horizon_gtfree/$scene/h1:$OUT_ROOT/horizon_labels/$scene/h1" \
    --test "3:$OUT_ROOT/horizon_gtfree/$scene/h3:$OUT_ROOT/horizon_labels/$scene/h3" \
    --test "5:$OUT_ROOT/horizon_gtfree/$scene/h5:$OUT_ROOT/horizon_labels/$scene/h5" \
    --test "10:$OUT_ROOT/horizon_gtfree/$scene/h10:$OUT_ROOT/horizon_labels/$scene/h10" --output "$output"
}

run_acceptance(){
  run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_required_acceptance_report.py" --root "$OUT_ROOT" --output "$OUT_ROOT/acceptance.json" --require-all
}

write_invocation
case "$STAGE" in
  preflight) preflight ;;
  reverse) run_reverse ;;
  paired) preflight; for s in office0 room2 office4 fr3 reverse; do run_paired_scene "$s"; done ;;
  horizon) for s in office0 room2 office4 fr3 reverse; do run_horizon_scene "$s"; done ;;
  label) for s in office0 room2 office4 fr3 reverse; do run_label_scene "$s"; done ;;
  freeze) run_freeze ;;
  evaluate) for s in room2 office4 fr3 reverse; do run_evaluate_scene "$s"; done ;;
  acceptance) run_acceptance ;;
  all)
    preflight
    run_reverse
    for s in office0 room2 office4 fr3 reverse; do run_paired_scene "$s"; done
    for s in office0 room2 office4 fr3 reverse; do run_horizon_scene "$s"; done
    for s in office0 room2 office4 fr3 reverse; do run_label_scene "$s"; done
    run_freeze
    for s in room2 office4 fr3 reverse; do run_evaluate_scene "$s"; done
    run_acceptance
    ;;
esac
printf 'Required acceptance experiments stage=%s plan_only=%s output=%s\n' "$STAGE" "$PLAN_ONLY" "$OUT_ROOT"
