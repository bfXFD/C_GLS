#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
CONDA_ENV="${CONDA_ENV:-splatam}"
PLAN_ONLY="${PLAN_ONLY:-1}"
SEQUENCES="${SEQUENCES:-all}"
STAGE="${STAGE:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1}"
MAP_ROOT="${MAP_ROOT:-$ROOT/experiments/FSM/mvp_observable_v1/tum_fps_hybrid_v1/maps}"
AUDIT_ROOT="$ROOT/experiments/FSM/audit_v2_full_rerun/main"

[[ "$PLAN_ONLY" == "0" || "$PLAN_ONLY" == "1" ]] || exit 2
[[ "$SKIP_EXISTING" == "0" || "$SKIP_EXISTING" == "1" ]] || exit 2
[[ "$STAGE" == "build" || "$STAGE" == "paired" || "$STAGE" == "closed_loop" || "$STAGE" == "analysis" || "$STAGE" == "all" ]] || exit 2
[[ "$OUT_ROOT" != *audit_v2_full_rerun* && "$MAP_ROOT" != *audit_v2_full_rerun* ]] || { printf 'Refusing frozen audit output path\n' >&2; exit 2; }

ALL=(freiburg1_desk freiburg2_xyz freiburg3_long_office_household)
SELECTED=()
for s in "${ALL[@]}"; do [[ "$SEQUENCES" == all || ",$SEQUENCES," == *",$s,"* ]] && SELECTED+=("$s"); done
[[ "${#SELECTED[@]}" -gt 0 ]] || exit 2
print_command(){ printf '[plan]'; printf ' %q' "$@"; printf '\n'; }
run(){ print_command "$@"; [[ "$PLAN_ONLY" == 1 ]] || "$@"; }
need(){ [[ "$PLAN_ONLY" == 1 || -f "$1" ]] || { printf 'Missing %s\n' "$1" >&2; exit 1; }; }

for s in "${SELECTED[@]}"; do
  printf '[sequence] %s\n' "$s"
  params="$ROOT/experiments/TUM/${s}_seed0/params.npz"
  source="$AUDIT_ROOT/tum_${s}/pareto/maps/fps_scale_opacity_ratio_0p1/fsm_map.npz"
  mapdir="$MAP_ROOT/$s"
  map="$mapdir/fsm_map.npz"
  geom="$mapdir/activation_geometry.npz"
  basecfg="$ROOT/configs/fsm/mvp_observable/tum_${s}.py"
  paircfg="$ROOT/configs/fsm/mvp_observable/tum_${s}_paired_core_route.py"
  corrcfg="$ROOT/configs/fsm/mvp_observable/tum_${s}_corr_weak128_info_v4.py"
  seqroot="$OUT_ROOT/$s"
  skip=(); [[ "$SKIP_EXISTING" == 0 ]] || skip=(--skip-existing)
  need "$params"; need "$source"; need "$basecfg"; need "$paircfg"; need "$corrcfg"
  if [[ "$STAGE" == build || "$STAGE" == all ]]; then
    run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_build_fixed_map_reserve.py" --params "$params" --base-map "$source" --map-output "$map" --activation-geometry-output "$geom" --reserve-count 4096 --fill-strategy scale_opacity --normal-k 20 "${skip[@]}"
  fi
  if [[ "$STAGE" == paired || "$STAGE" == all ]]; then
    need "$map"; need "$geom"
    run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$seqroot/paired" --start 0 --end 500 --case "paired,$paircfg,$map,$geom" "${skip[@]}"
    need "$seqroot/paired/localize/paired/paired_reserve/paired_reserve_ids.npz"
    need "$seqroot/paired/localize/paired/trajectory_gt.txt"
    need "$seqroot/paired/localize/paired/trajectory_est.txt"
    if [[ "$SKIP_EXISTING" == 1 && -f "$seqroot/paired_gt/paired_reserve_gt_summary.json" ]]; then printf '[skip] paired-gt %s\n' "$s"; else
      run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_score_paired_reserve_gt.py" --paired-npz "$seqroot/paired/localize/paired/paired_reserve/paired_reserve_ids.npz" --trajectory-gt "$seqroot/paired/localize/paired/trajectory_gt.txt" --trajectory-est "$seqroot/paired/localize/paired/trajectory_est.txt" --output-dir "$seqroot/paired_gt"
    fi
    need "$seqroot/paired/localize/paired/paired_reserve/paired_reserve_rows.csv"
    need "$seqroot/paired_gt/paired_reserve_gt_rows.csv"
    if [[ "$SKIP_EXISTING" == 1 && -f "$seqroot/paired_stats/summary.json" ]]; then printf '[skip] paired-stats %s\n' "$s"; else
      run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_paired_frame_stats.py" --paired-rows "$seqroot/paired/localize/paired/paired_reserve/paired_reserve_rows.csv" --gt-rows "$seqroot/paired_gt/paired_reserve_gt_rows.csv" --output-dir "$seqroot/paired_stats" --budgets 64 128 256 --expected-random-seeds 0 1 2 3 4 --require-complete-random-panel --complete-across-budgets --bootstrap-replicates 10000 --bootstrap-seed 0
    fi
  fi
  if [[ "$STAGE" == closed_loop || "$STAGE" == all ]]; then
    need "$map"; need "$geom"
    run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_eval_long_horizon.py" --output-root "$seqroot/closed_loop" --case "fps10,$basecfg,$map" --case "fps_corrweak128,$corrcfg,$map,$geom" "${skip[@]}"
  fi
  if [[ "$STAGE" == analysis || "$STAGE" == all ]]; then
    need "$seqroot/closed_loop/localize/fps10/summary.json"; need "$seqroot/closed_loop/localize/fps_corrweak128/summary.json"
    if [[ "$SKIP_EXISTING" == 1 && -f "$seqroot/common_valid/result.json" ]]; then printf '[skip] common-valid %s\n' "$s"; else
      run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_common_valid.py" --method "fps10=$seqroot/closed_loop/localize/fps10" --method "fps_corrweak128=$seqroot/closed_loop/localize/fps_corrweak128" --output-dir "$seqroot/common_valid" --frame-deltas 1 10 30 100 --bootstrap-reps 1000 --bootstrap-seed 0
    fi
    if [[ "$SKIP_EXISTING" == 1 && -f "$seqroot/critical_adoption/summary.json" ]]; then printf '[skip] critical-adoption %s\n' "$s"; else
      run conda run -n "$CONDA_ENV" python "$ROOT/scripts/fsm_critical_adoption.py" --baseline "fps10=$seqroot/closed_loop/localize/fps10" --run "fps_corrweak128=$seqroot/closed_loop/localize/fps_corrweak128" --output-dir "$seqroot/critical_adoption" --horizons 10 30 100 --rpe-deltas 1 5 10 --burst-gaps 1 2 5 10 --temporal-windows 1 2 5 10 30 100
    fi
  fi
done
