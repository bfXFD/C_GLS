#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:--1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/independent_revisit_desk2_v1}"
MAP_ROOT="${MAP_ROOT:-$OUT_ROOT/map_artifacts}"
TABLE_OUTPUT="${TABLE_OUTPUT:-$OUT_ROOT/independent_revisit_table.csv}"
PROTOCOL="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_protocol.json"
SPLATAM_CONFIG="$ROOT/configs/tum/tum_eval.py"
SPLATAM_SCRIPT="$ROOT/scripts/splatam.py"
PARAMS="${PARAMS:-$ROOT/experiments/TUM/freiburg1_desk2_seed0/params.npz}"
FULL_MAP="$MAP_ROOT/full/full_ratio_1/fsm_map.npz"
FPS_MAP="$MAP_ROOT/fps10/fps_scale_opacity_ratio_0p1/fsm_map.npz"
RESERVE_MAP="$MAP_ROOT/reserve/fsm_map.npz"
GEOMETRY="$MAP_ROOT/reserve/activation_geometry.npz"

for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do
  [[ "$value" == "0" || "$value" == "1" ]] || { printf 'PLAN_ONLY and SKIP_EXISTING must be 0 or 1\n' >&2; exit 2; }
done
[[ "$FRAME_START" =~ ^[0-9]+$ ]] || { printf 'FRAME_START must be non-negative\n' >&2; exit 2; }
[[ "$FRAME_END" == "-1" || "$FRAME_END" =~ ^[0-9]+$ ]] || { printf 'FRAME_END must be -1 or non-negative\n' >&2; exit 2; }
if [[ "$FRAME_END" != "-1" ]] && (( FRAME_END <= FRAME_START )); then
  printf 'FRAME_END must exceed FRAME_START\n' >&2
  exit 2
fi
for path in "$OUT_ROOT" "$MAP_ROOT" "$TABLE_OUTPUT"; do
  [[ "$path" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path: %s\n' "$path" >&2; exit 2; }
done

cd "$ROOT"
print_command() { printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; [[ "$PLAN_ONLY" == "1" ]] || "$@"; }
require_file() { [[ "$PLAN_ONLY" == "1" || -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }
skip=(); [[ "$SKIP_EXISTING" == "0" ]] || skip=(--skip-existing)

for path in "$PROTOCOL" "$SPLATAM_CONFIG" "$SPLATAM_SCRIPT"; do require_file "$path"; done

if [[ -f "$PARAMS" ]]; then
  printf '[skip] existing desk2 SplaTAM parameters: %s\n' "$PARAMS"
else
  run_command env WANDB_MODE=disabled SEED=0 SCENE_NUM=1 conda run -n "$CONDA_ENV" python "$SPLATAM_SCRIPT" "$SPLATAM_CONFIG"
fi
require_file "$PARAMS"

if [[ -f "$FULL_MAP" ]]; then
  printf '[skip] existing full desk2 map: %s\n' "$FULL_MAP"
else
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_export.py" --params "$PARAMS" --output-dir "$MAP_ROOT/full" --strategies full --compression-ratios 1 --seed 0 --fps-candidate-cap 20000 --overwrite
fi
if [[ -f "$FPS_MAP" ]]; then
  printf '[skip] existing FPS10 desk2 map: %s\n' "$FPS_MAP"
else
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_export.py" --params "$PARAMS" --output-dir "$MAP_ROOT/fps10" --strategies fps_scale_opacity --compression-ratios 0.1 --seed 0 --fps-candidate-cap 20000 --overwrite
fi
require_file "$FULL_MAP"; require_file "$FPS_MAP"

if [[ -f "$RESERVE_MAP" && -f "$GEOMETRY" && -f "$GEOMETRY.done.json" ]]; then
  printf '[skip] existing deterministic desk2 reserve artifacts\n'
else
  run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_fixed_map_reserve.py" --params "$PARAMS" --base-map "$FPS_MAP" --map-output "$RESERVE_MAP" --activation-geometry-output "$GEOMETRY" --reserve-count 4096 --fill-strategy scale_opacity --normal-k 20 --skip-existing
fi
require_file "$RESERVE_MAP"; require_file "$GEOMETRY"

full_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_full.py"
compact_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_compact.py"
corr_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_corr_weak128_v4.py"
replace_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_replace128_v4.py"
for path in "$full_cfg" "$compact_cfg" "$corr_cfg" "$replace_cfg"; do require_file "$path"; done

run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_independent_revisit_provenance.py" \
  --output "$OUT_ROOT/pipeline_provenance.json" --frame-start "$FRAME_START" --frame-end "$FRAME_END" \
  --input "$PROTOCOL" --input "$SPLATAM_CONFIG" --input "$SPLATAM_SCRIPT" --input "$PARAMS" \
  --input "$FULL_MAP" --input "$FPS_MAP" --input "$RESERVE_MAP" --input "$GEOMETRY" \
  --input "$full_cfg" --input "$compact_cfg" --input "$corr_cfg" --input "$replace_cfg"

run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/full_map" --start "$FRAME_START" --end "$FRAME_END" --case "full_map,$full_cfg,$FULL_MAP" "${skip[@]}"
run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/compact_fps10" --start "$FRAME_START" --end "$FRAME_END" --case "compact_fps10,$compact_cfg,$FPS_MAP" "${skip[@]}"
run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/corr_weak128_v4" --start "$FRAME_START" --end "$FRAME_END" --case "corr_weak128_v4,$corr_cfg,$RESERVE_MAP,$GEOMETRY" "${skip[@]}"
run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/replace128_v4" --start "$FRAME_START" --end "$FRAME_END" --case "replace128_v4,$replace_cfg,$RESERVE_MAP,$GEOMETRY" "${skip[@]}"
run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_independent_revisit_acceptance.py" --root "$OUT_ROOT" --require-all --output "$OUT_ROOT/acceptance.json"
run_command conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_independent_revisit_table.py" --root "$OUT_ROOT" --output "$TABLE_OUTPUT"

printf 'Reverse independent revisit plan_only=%s map=rgbd_dataset_freiburg1_desk2 query=rgbd_dataset_freiburg1_desk output=%s\n' "$PLAN_ONLY" "$OUT_ROOT"
