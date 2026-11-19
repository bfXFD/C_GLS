#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
SPLATAM_ROOT="/home/ras/xbf/SplaTAM"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
STAGES="${STAGES:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/replica_office1_4_untouched_v1}"
PROTOCOL="$ROOT/configs/fsm/mvp_observable/replica_office1_4_frozen_protocol.json"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$STAGES" == "splatam" || "$STAGES" == "maps" || "$STAGES" == "eval" || "$STAGES" == "all" ]] || {
  printf 'STAGES must be splatam, maps, eval, or all\n' >&2; exit 2;
}
[[ "$OUT_ROOT" != *"/audit_v2_full_rerun/"* && "$OUT_ROOT" != *"/audit_v2_full_rerun" ]] || {
  printf 'Refusing frozen audit output: %s\n' "$OUT_ROOT" >&2; exit 2;
}
[[ -f "$PROTOCOL" ]] || { printf 'Missing frozen protocol: %s\n' "$PROTOCOL" >&2; exit 1; }

print_command() { printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; [[ "$PLAN_ONLY" == "1" ]] || "$@"; }
require_file() { [[ -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }

write_freeze_manifest() {
  [[ "$PLAN_ONLY" == "1" ]] && return
  mkdir -p "$OUT_ROOT"
  if [[ ! -f "$OUT_ROOT/frozen_protocol.json" ]]; then
    cp "$PROTOCOL" "$OUT_ROOT/frozen_protocol.json"
    python -c 'import hashlib,json,pathlib,sys; paths=[pathlib.Path(p) for p in sys.argv[2:]]; rows=[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]; pathlib.Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+"\n")' \
      "$OUT_ROOT/frozen_source_manifest.json" \
      "$PROTOCOL" \
      "$ROOT/configs/fsm/mvp_observable/common.py" \
      "$ROOT/utils/fsm_runtime_reserve.py" \
      "$ROOT/utils/fsm_correspondence_weak.py" \
      "$ROOT/scripts/fsm_build_observable_map.py" \
      "$ROOT/scripts/fsm_eval_long_horizon.py" \
      "$SPLATAM_ROOT/configs/replica/splatam.py" \
      "$SPLATAM_ROOT/scripts/splatam.py"
  fi
}

run_splatam() {
  local scene="$1"
  local config="$SPLATAM_ROOT/configs/replica/generated/splatam_${scene}.py"
  local run_dir="$SPLATAM_ROOT/experiments/Replica/${scene}_0"
  local params="$run_dir/params.npz"
  require_file "$config"
  if [[ -f "$params" ]]; then printf '[skip] existing SplaTAM parameters: %s\n' "$params"; return; fi
  if [[ -d "$run_dir" ]]; then printf 'Refusing partial SplaTAM directory without params: %s\n' "$run_dir" >&2; exit 1; fi
  print_command conda run -n "$CONDA_ENV" python "$SPLATAM_ROOT/scripts/splatam.py" "$config"
  if [[ "$PLAN_ONLY" == "0" ]]; then
    (cd "$SPLATAM_ROOT" && conda run -n "$CONDA_ENV" python scripts/splatam.py "configs/replica/generated/splatam_${scene}.py")
  fi
}

run_map() {
  local scene="$1"
  local params="$SPLATAM_ROOT/experiments/Replica/${scene}_0/params.npz"
  local map_dir="$OUT_ROOT/maps/replica_${scene}_0/logdet"
  local -a skip=(); [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  [[ "$PLAN_ONLY" == "1" ]] || require_file "$params"
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_observable_map.py" \
    --params "$params" --ratio 0.2 --max-reference-poses 64 --candidate-cap 4096 \
    --pair-cap 200000 --fill-strategy scale_opacity --greedy-core-cap 128 \
    --replacement-count 128 --reserve-count 4096 --normal-k 20 --regularization 1e-6 \
    --min-eigen-every 0 --min-eigen-bonus 0 --output "$map_dir/selection.npz" \
    --map-output "$map_dir/fsm_map.npz" --activation-geometry-output "$map_dir/activation_geometry.npz" \
    "${skip[@]}"
}

run_eval() {
  local scene="$1"
  local config_root="$ROOT/configs/fsm/mvp_observable"
  local map_dir="$OUT_ROOT/maps/replica_${scene}_0/logdet"
  local map="$map_dir/fsm_map.npz" geometry="$map_dir/activation_geometry.npz"
  local baseline="$config_root/replica_${scene}.py"
  local corr="$config_root/replica_${scene}_corr_weak128_info_v4.py"
  local weak="$config_root/replica_${scene}_weak128_info_v4.py"
  local -a skip=(); [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)
  if [[ "$PLAN_ONLY" == "0" ]]; then
    require_file "$map"; require_file "$geometry"; require_file "$baseline"; require_file "$corr"; require_file "$weak"
  fi
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" \
    --output-root "$OUT_ROOT/eval/replica_${scene}_0" --start 0 \
    --case "baseline,$baseline,$map" \
    --case "corrweak128,$corr,$map,$geometry" \
    --case "weak128,$weak,$map,$geometry" "${skip[@]}"
}

write_freeze_manifest
for scene in office1 office2 office3 office4; do
  printf '[scene] %s\n' "$scene"
  [[ "$STAGES" != "splatam" && "$STAGES" != "all" ]] || run_splatam "$scene"
  [[ "$STAGES" != "maps" && "$STAGES" != "all" ]] || run_map "$scene"
  [[ "$STAGES" != "eval" && "$STAGES" != "all" ]] || run_eval "$scene"
done
printf 'Office1-4 frozen pipeline plan_only=%s stages=%s output=%s\n' "$PLAN_ONLY" "$STAGES" "$OUT_ROOT"
