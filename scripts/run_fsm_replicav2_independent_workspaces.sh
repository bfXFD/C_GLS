#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
cd "$ROOT"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SCENES="${SCENES:-room_0,office_0}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:--1}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/Replica_V2}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/replicav2_independent_workspaces_v1}"
CFG="$ROOT/configs/fsm/mvp_observable/replicav2_independent_case.py"

for value in "$PLAN_ONLY" "$SKIP_EXISTING" "$AUTO_DOWNLOAD"; do [[ "$value" == 0 || "$value" == 1 ]] || exit 2; done
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || exit 2
IFS=',' read -r -a scenes <<< "$SCENES"
print_command(){ printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run(){ print_command "$@"; [[ "$PLAN_ONLY" == 1 ]] || "$@"; }
run_if_missing(){ local expected="$1"; shift; if [[ "$SKIP_EXISTING" == 1 && -f "$expected" ]]; then printf '[skip] %s\n' "$expected"; else run "$@"; fi; }

if [[ "$PLAN_ONLY" == 0 && ! -d "$DATA_ROOT" ]]; then
  if [[ "$AUTO_DOWNLOAD" == 1 ]]; then
    run env DATA_ROOT="$DATA_ROOT" bash "$ROOT/bash_scripts/download_replicav2.sh"
  else
    printf 'Replica V2 data is missing: %s\n' "$DATA_ROOT" >&2
    printf 'Run: DATA_ROOT=%q bash %q\n' "$DATA_ROOT" "$ROOT/bash_scripts/download_replicav2.sh" >&2
    printf 'Or rerun this command with AUTO_DOWNLOAD=1.\n' >&2
    exit 2
  fi
fi
if [[ "$PLAN_ONLY" == 0 ]]; then
  run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_replicav2_preflight.py" --data-root "$DATA_ROOT" --scenes "${scenes[@]}" --output "$OUT_ROOT/data_preflight.json"
else
  print_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_replicav2_preflight.py" --data-root "$DATA_ROOT" --scenes "${scenes[@]}" --output "$OUT_ROOT/data_preflight.json"
fi

for scene in "${scenes[@]}"; do
  params="$ROOT/experiments/Replica_V2/${scene}_0/params.npz"
  maps="$OUT_ROOT/$scene/maps"
  full="$maps/full_ratio_1/fsm_map.npz"
  fps="$maps/fps_ratio_0p1/fsm_map.npz"
  reserve_map="$maps/reserve/fsm_map.npz"
  geometry="$maps/reserve/activation_geometry.npz"
  run_if_missing "$params" env WANDB_MODE=disabled SCENE_NAME="$scene" conda run -n "$CONDA_ENV" python "$ROOT/scripts/splatam.py" "$ROOT/configs/replica_v2/splatam.py"
  run_if_missing "$full" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_export.py" --params "$params" --output-dir "$maps" --strategies full --compression-ratios 1.0
  if [[ ! -f "$fps" || "$SKIP_EXISTING" == 0 ]]; then
    run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_export.py" --params "$params" --output-dir "$maps/fps_export" --strategies fps --compression-ratios 0.1
    if [[ "$PLAN_ONLY" == 0 ]]; then mkdir -p "$(dirname "$fps")"; cp "$maps/fps_export/fps_ratio_0p1/fsm_map.npz" "$fps"; fi
  else printf '[skip] %s\n' "$fps"; fi
  run_if_missing "$geometry" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_fixed_map_reserve.py" --params "$params" --base-map "$fps" --map-output "$reserve_map" --activation-geometry-output "$geometry" --reserve-count 4096 --fill-strategy scale_opacity --normal-k 20 --skip-existing
  for arm in full compact corrweak replace; do
    map="$fps"; geom=()
    [[ "$arm" == full ]] && map="$full"
    [[ "$arm" == corrweak || "$arm" == replace ]] && { map="$reserve_map"; geom=("$geometry"); }
    output="$OUT_ROOT/$scene/$arm"
    skip=(); [[ "$SKIP_EXISTING" == 0 ]] || skip=(--skip-existing)
    case_spec="${scene}_${arm},$CFG,$map"
    [[ "${#geom[@]}" -eq 0 ]] || case_spec="$case_spec,${geom[0]}"
    run env CGLS_REPLICAV2_SCENE="$scene" CGLS_REPLICAV2_METHOD="$arm" conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$output" --start "$FRAME_START" --end "$FRAME_END" --case "$case_spec" "${skip[@]}"
  done
done
run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_replicav2_acceptance.py" --root "$OUT_ROOT" --scenes "${scenes[@]}" --output "$OUT_ROOT/acceptance.json"
printf 'ReplicaV2 independent workspaces plan_only=%s scenes=%s output=%s\n' "$PLAN_ONLY" "$SCENES" "$OUT_ROOT"
