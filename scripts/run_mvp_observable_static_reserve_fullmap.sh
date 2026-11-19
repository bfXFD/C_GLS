#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
STAGES="${STAGES:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/static_reserve_fullmap_v1}"
PROTOCOL="$ROOT/configs/fsm/mvp_observable/static_reserve_fullmap_frozen_protocol.json"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$STAGES" == "maps" || "$STAGES" == "eval" || "$STAGES" == "analysis" || "$STAGES" == "all" ]] || {
  printf 'STAGES must be maps, eval, analysis, or all\n' >&2; exit 2;
}
[[ "$OUT_ROOT" != *"/audit_v2_full_rerun/"* && "$OUT_ROOT" != *"/audit_v2_full_rerun" ]] || {
  printf 'Refusing frozen audit output: %s\n' "$OUT_ROOT" >&2; exit 2;
}
[[ -f "$PROTOCOL" ]] || { printf 'Missing frozen protocol: %s\n' "$PROTOCOL" >&2; exit 1; }

print_command() { printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; [[ "$PLAN_ONLY" == "1" ]] || "$@"; }
require_file() { [[ -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }

freeze_sources() {
  [[ "$PLAN_ONLY" == "1" ]] && return
  mkdir -p "$OUT_ROOT"
  if [[ ! -f "$OUT_ROOT/frozen_protocol.json" ]]; then
    cp "$PROTOCOL" "$OUT_ROOT/frozen_protocol.json"
    python -c 'import hashlib,json,pathlib,sys; ps=[pathlib.Path(x) for x in sys.argv[2:]]; pathlib.Path(sys.argv[1]).write_text(json.dumps([{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in ps],indent=2)+"\n")' \
      "$OUT_ROOT/frozen_source_manifest.json" "$PROTOCOL" \
      "$ROOT/scripts/fsm_materialize_static_reserve_map.py" "$ROOT/scripts/fsm_eval_long_horizon.py" \
      "$ROOT/scripts/fsm_common_valid.py" "$ROOT/configs/fsm/mvp_observable/common.py" \
      "$ROOT/utils/fsm_runtime_reserve.py" "$ROOT/utils/fsm_correspondence_weak.py" "$ROOT/utils/fsm_map.py"
  fi
}

scene_values() {
  case "$1" in
    replica_room2_0)
      CONFIG="$ROOT/configs/fsm/mvp_observable/replica_room2.py"
      CORR_CONFIG="$ROOT/configs/fsm/mvp_observable/replica_room2_corr_weak128_info_v4.py"
      BASE="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_room2_0/logdet/fsm_map.npz"
      GEOMETRY="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_room2_0/logdet/activation_geometry.npz"
      FULL="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_room2_0/full/maps/full_ratio_1/fsm_map.npz"
      ;;
    replica_office0_0)
      CONFIG="$ROOT/configs/fsm/mvp_observable/replica_office0.py"
      CORR_CONFIG="$ROOT/configs/fsm/mvp_observable/replica_office0_corr_weak128_info_v4.py"
      BASE="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/fsm_map.npz"
      GEOMETRY="$ROOT/experiments/FSM/mvp_observable_v1/runtime_reserve_v1/maps/replica_office0_0/logdet/activation_geometry.npz"
      FULL="$ROOT/experiments/FSM/audit_v2_full_rerun/main/replica_office0_0/full/maps/full_ratio_1/fsm_map.npz"
      ;;
    tum_freiburg3_long_office_household)
      CONFIG="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household.py"
      CORR_CONFIG="$ROOT/configs/fsm/mvp_observable/tum_freiburg3_long_office_household_corr_weak128_info_v4.py"
      BASE="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg3_long_office_household/fsm_map.npz"
      GEOMETRY="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg3_long_office_household/activation_geometry.npz"
      FULL="$ROOT/experiments/FSM/audit_v2_full_rerun/main/tum_freiburg3_long_office_household/full/maps/full_ratio_1/fsm_map.npz"
      ;;
    *) printf 'Unknown scene: %s\n' "$1" >&2; exit 2 ;;
  esac
  STATIC="$OUT_ROOT/maps/$1/static_all_reserve/fsm_map.npz"
  EVAL_ROOT="$OUT_ROOT/eval/$1"
}

run_map() {
  local scene="$1"; scene_values "$scene"
  [[ "$PLAN_ONLY" == "1" ]] || { require_file "$BASE"; require_file "$GEOMETRY"; }
  local -a skip=(); [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_materialize_static_reserve_map.py" \
    --base-map "$BASE" --activation-geometry "$GEOMETRY" --output "$STATIC" \
    --expected-reserve-count 4096 "${skip[@]}"
}

run_eval() {
  local scene="$1"; scene_values "$scene"
  if [[ "$PLAN_ONLY" == "0" ]]; then
    for path in "$CONFIG" "$CORR_CONFIG" "$BASE" "$GEOMETRY" "$STATIC" "$FULL"; do require_file "$path"; done
  fi
  local -a skip=(); [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$EVAL_ROOT" --start 0 \
    --case "compact,$CONFIG,$BASE" \
    --case "corrweak128,$CORR_CONFIG,$BASE,$GEOMETRY" \
    --case "static4096,$CONFIG,$STATIC" \
    --case "full,$CONFIG,$FULL" "${skip[@]}"
}

run_analysis() {
  local scene="$1"; scene_values "$scene"
  if [[ "$PLAN_ONLY" == "0" ]]; then
    for method in compact corrweak128 static4096 full; do require_file "$EVAL_ROOT/localize/$method/summary.json"; done
  fi
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_common_valid.py" \
    --method "compact=$EVAL_ROOT/localize/compact" \
    --method "corrweak128=$EVAL_ROOT/localize/corrweak128" \
    --method "static4096=$EVAL_ROOT/localize/static4096" \
    --method "full=$EVAL_ROOT/localize/full" \
    --output-dir "$OUT_ROOT/analysis/$scene/common_valid" --bootstrap-reps 1000 --bootstrap-seed 0
}

freeze_sources
for scene in replica_room2_0 replica_office0_0 tum_freiburg3_long_office_household; do
  printf '[scene] %s\n' "$scene"
  [[ "$STAGES" != "maps" && "$STAGES" != "all" ]] || run_map "$scene"
  [[ "$STAGES" != "eval" && "$STAGES" != "all" ]] || run_eval "$scene"
  [[ "$STAGES" != "analysis" && "$STAGES" != "all" ]] || run_analysis "$scene"
done
printf 'Static-reserve/full-map panel plan_only=%s stages=%s output=%s\n' "$PLAN_ONLY" "$STAGES" "$OUT_ROOT"
