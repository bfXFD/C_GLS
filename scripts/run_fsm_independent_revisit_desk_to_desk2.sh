#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:--1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/independent_revisit_desk_to_desk2_v1}"
FULL_MAP="$ROOT/experiments/FSM/audit_v2_full_rerun/main/tum_freiburg1_desk/full/maps/full_ratio_1/fsm_map.npz"
COMPACT_MAP="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg1_desk/fsm_map.npz"
GEOMETRY="$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps/freiburg1_desk/activation_geometry.npz"
full_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_full_forward.py"
compact_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_compact_forward.py"
corr_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_corr_weak128_v4_forward.py"
replace_cfg="$ROOT/configs/fsm/mvp_observable/tum_freiburg1_desk2_independent_replace128_v4_forward.py"
for value in "$PLAN_ONLY" "$SKIP_EXISTING"; do [[ "$value" == 0 || "$value" == 1 ]] || exit 2; done
[[ "$OUT_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output\n' >&2; exit 2; }
print_command(){ printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run(){ print_command "$@"; [[ "$PLAN_ONLY" == 1 ]] || "$@"; }
need(){ [[ "$PLAN_ONLY" == 1 || -f "$1" ]] || { printf 'Missing required input: %s\n' "$1" >&2; exit 1; }; }
for path in "$FULL_MAP" "$COMPACT_MAP" "$GEOMETRY" "$full_cfg" "$compact_cfg" "$corr_cfg" "$replace_cfg"; do need "$path"; done
skip=(); [[ "$SKIP_EXISTING" == 0 ]] || skip=(--skip-existing)
run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/full_map" --start "$FRAME_START" --end "$FRAME_END" --case "full_map,$full_cfg,$FULL_MAP" "${skip[@]}"
run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/compact_fps10" --start "$FRAME_START" --end "$FRAME_END" --case "compact_fps10,$compact_cfg,$COMPACT_MAP" "${skip[@]}"
run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/corr_weak128_v4" --start "$FRAME_START" --end "$FRAME_END" --case "corr_weak128_v4,$corr_cfg,$COMPACT_MAP,$GEOMETRY" "${skip[@]}"
run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$OUT_ROOT/replace128_v4" --start "$FRAME_START" --end "$FRAME_END" --case "replace128_v4,$replace_cfg,$COMPACT_MAP,$GEOMETRY" "${skip[@]}"
printf 'Forward independent revisit plan_only=%s map=desk query=desk2 output=%s\n' "$PLAN_ONLY" "$OUT_ROOT"
