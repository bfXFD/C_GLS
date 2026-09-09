#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ras/xbf/C_GLS_code"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/FSM/audit_v2_full_rerun}"
CONDA_ENV="${CONDA_ENV:-splatam}"
RENDER_EVERY="${RENDER_EVERY:-1}"
RESUME="${RESUME:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_LIGHTGAUSSIAN_PROXY="${RUN_LIGHTGAUSSIAN_PROXY:-1}"
RUN_COVARIANCE_WEIGHTED="${RUN_COVARIANCE_WEIGHTED:-1}"
FISHER_MAX_POINTS="${FISHER_MAX_POINTS:-200000}"

STATE_DIR="$OUT_ROOT/.state"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"
cd "$ROOT"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

declare -a DATASETS=(
  "replica_room0_0"
  "replica_room1_0"
  "replica_room2_0"
  "replica_office0_0"
  "tum_freiburg1_desk"
  "tum_freiburg2_xyz"
  "tum_freiburg3_long_office_household"
)

declare -A CONFIGS=(
  [replica_room0_0]="$ROOT/configs/fsm/replica_fsm.py"
  [replica_room1_0]="$ROOT/configs/fsm/generated/replica_room1_fsm.py"
  [replica_room2_0]="$ROOT/configs/fsm/generated/replica_room2_fsm.py"
  [replica_office0_0]="$ROOT/configs/fsm/generated/replica_office0_fsm.py"
  [tum_freiburg1_desk]="$ROOT/configs/fsm/generated_tum/tum_freiburg1_desk_fsm.py"
  [tum_freiburg2_xyz]="$ROOT/configs/fsm/generated_tum/tum_freiburg2_xyz_fsm.py"
  [tum_freiburg3_long_office_household]="$ROOT/configs/fsm/generated_tum/tum_freiburg3_long_office_household_fsm.py"
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

declare -A PRIMARY_STRATEGY=(
  [replica_room0_0]="voxel_scale_opacity"
  [replica_room1_0]="voxel_scale_opacity"
  [replica_room2_0]="voxel_scale_opacity"
  [replica_office0_0]="voxel_scale_opacity"
  [tum_freiburg1_desk]="fps_scale_opacity"
  [tum_freiburg2_xyz]="fps_scale_opacity"
  [tum_freiburg3_long_office_household]="fps_scale_opacity"
)

declare -A PRIMARY_RATIO=(
  [replica_room0_0]="0.20"
  [replica_room1_0]="0.20"
  [replica_room2_0]="0.20"
  [replica_office0_0]="0.20"
  [tum_freiburg1_desk]="0.10"
  [tum_freiburg2_xyz]="0.10"
  [tum_freiburg3_long_office_household]="0.10"
)

variant_ratio() {
  local ratio="$1"
  ratio="${ratio%0}"
  ratio="${ratio/./p}"
  printf '%s' "$ratio"
}

run_step() {
  local name="$1"
  shift
  local marker="$STATE_DIR/$name.done"
  local log="$LOG_DIR/$name.log"
  if [[ "$RESUME" == "1" && -f "$marker" ]]; then
    printf '[skip] %s\n' "$name"
    return 0
  fi
  printf '[run] %s\n' "$name"
  "$@" 2>&1 | tee "$log"
  touch "$marker"
  printf '[done] %s\n' "$name"
}

run_eval() {
  local dataset="$1"
  local output="$2"
  local seed="$3"
  local cap="$4"
  shift 4
  conda run -n "$CONDA_ENV" python scripts/fsm_eval.py \
    --config "${CONFIGS[$dataset]}" \
    --params "${PARAMS[$dataset]}" \
    --output-dir "$output" \
    --start 0 --end -1 \
    --fisher-max-points "$FISHER_MAX_POINTS" \
    --seed "$seed" \
    --fps-candidate-cap "$cap" \
    --overwrite \
    "$@"
}

run_localize() {
  local dataset="$1"
  local map="$2"
  local output="$3"
  local backend="${4:-numpy_point_to_point}"
  conda run -n "$CONDA_ENV" python scripts/fsm_localize.py \
    --config "${CONFIGS[$dataset]}" \
    --map "$map" \
    --output-dir "$output" \
    --backend "$backend" \
    --start 0 --end -1
}

check_inputs() {
  for dataset in "${DATASETS[@]}"; do
    test -f "${CONFIGS[$dataset]}"
    test -f "${PARAMS[$dataset]}"
  done
  conda run -n "$CONDA_ENV" python -m pytest tests/test_fsm_*.py
}

run_main_dataset() {
  local dataset="$1"
  local base="$OUT_ROOT/main/$dataset"
  run_eval "$dataset" "$base/full" 0 20000 \
    --strategies full --compression-ratios 1
  run_eval "$dataset" "$base/pareto" 0 20000 \
    --strategies random scale_large voxel_grid_pure voxel_scale_opacity fps_scale_opacity \
    --compression-ratios 0.25 0.20 0.15 0.10 0.05
  conda run -n "$CONDA_ENV" python scripts/fsm_merge_summaries.py \
    --output "$base/summary_all.csv" \
    --input "$base/full/summary_all.csv" "$base/pareto/summary_all.csv"
}

run_seed_dataset() {
  local dataset="$1"
  local seed="$2"
  local base="$OUT_ROOT/seeds/$dataset/seed_$seed"
  run_eval "$dataset" "$base" "$seed" 20000 \
    --strategies random fps_scale_opacity \
    --compression-ratios "${PRIMARY_RATIO[$dataset]}"
}

run_cap_dataset() {
  local dataset="$1"
  local cap="$2"
  local base="$OUT_ROOT/fps_caps/$dataset/cap_$cap"
  run_eval "$dataset" "$base" 0 "$cap" \
    --strategies fps_scale_opacity \
    --compression-ratios "${PRIMARY_RATIO[$dataset]}"
}

run_storage_dataset() {
  local dataset="$1"
  local strategy="${PRIMARY_STRATEGY[$dataset]}"
  local ratio="${PRIMARY_RATIO[$dataset]}"
  local token
  token="$(variant_ratio "$ratio")"
  local base_map="$OUT_ROOT/main/$dataset/pareto/maps/${strategy}_ratio_${token}/fsm_map.npz"
  local base="$OUT_ROOT/storage/$dataset"
  mkdir -p "$base"

  conda run -n "$CONDA_ENV" python scripts/fsm_map_storage.py \
    --map "$base_map" --output "$base/base_field_sizes.json"

  conda run -n "$CONDA_ENV" python scripts/fsm_quantize_map.py \
    --map "$base_map" --output "$base/drop_scores_fp32.npz" \
    --float-dtype float32 --fields means --drop-scores
  conda run -n "$CONDA_ENV" python scripts/fsm_quantize_map.py \
    --map "$base_map" --output "$base/means_fp16_keep_scores.npz" \
    --float-dtype float16 --fields means
  conda run -n "$CONDA_ENV" python scripts/fsm_quantize_map.py \
    --map "$base_map" --output "$base/means_fp16_drop_scores.npz" \
    --float-dtype float16 --fields means --drop-scores

  for variant in drop_scores_fp32 means_fp16_keep_scores means_fp16_drop_scores; do
    conda run -n "$CONDA_ENV" python scripts/fsm_map_storage.py \
      --map "$base/${variant}.npz" --output "$base/${variant}_field_sizes.json"
    run_localize "$dataset" "$base/${variant}.npz" "$base/localize_$variant"
  done
}

run_render_dataset() {
  local dataset="$1"
  local strategy="${PRIMARY_STRATEGY[$dataset]}"
  local ratio="${PRIMARY_RATIO[$dataset]}"
  local token
  token="$(variant_ratio "$ratio")"
  local full_map="$OUT_ROOT/main/$dataset/full/maps/full_ratio_1/fsm_map.npz"
  local primary_map="$OUT_ROOT/main/$dataset/pareto/maps/${strategy}_ratio_${token}/fsm_map.npz"
  local base="$OUT_ROOT/render/$dataset"

  conda run -n "$CONDA_ENV" python scripts/fsm_render_eval.py \
    --config "${CONFIGS[$dataset]}" --params "${PARAMS[$dataset]}" \
    --map "$full_map" --output-dir "$base/full" \
    --start 0 --end -1 --eval-every "$RENDER_EVERY"
  conda run -n "$CONDA_ENV" python scripts/fsm_render_eval.py \
    --config "${CONFIGS[$dataset]}" --params "${PARAMS[$dataset]}" \
    --map "$primary_map" --output-dir "$base/primary" \
    --start 0 --end -1 --eval-every "$RENDER_EVERY"
}

run_covariance_dataset() {
  local dataset="$1"
  local strategy="${PRIMARY_STRATEGY[$dataset]}"
  local ratio="${PRIMARY_RATIO[$dataset]}"
  local token
  token="$(variant_ratio "$ratio")"
  local full_map="$OUT_ROOT/main/$dataset/full/maps/full_ratio_1/fsm_map.npz"
  local primary_map="$OUT_ROOT/main/$dataset/pareto/maps/${strategy}_ratio_${token}/fsm_map.npz"
  local base="$OUT_ROOT/covariance_weighted/$dataset"
  run_localize "$dataset" "$full_map" "$base/full" numpy_covariance_weighted
  run_localize "$dataset" "$primary_map" "$base/primary" numpy_covariance_weighted
}

run_lightgaussian_dataset() {
  local dataset="$1"
  local base="$OUT_ROOT/lightgaussian_proxy/$dataset"
  local scores="$base/scores/lightgaussian_proxy_scores.npz"
  local metadata="$base/scores/lightgaussian_proxy_scores.json"
  mkdir -p "$base/scores"
  conda run -n "$CONDA_ENV" python scripts/fsm_lightgaussian_scores.py \
    --params "${PARAMS[$dataset]}" --output "$scores" --metadata "$metadata" \
    --v-pow 0.1 --opacity-power 1
  conda run -n "$CONDA_ENV" python scripts/fsm_eval.py \
    --config "${CONFIGS[$dataset]}" --params "${PARAMS[$dataset]}" \
    --output-dir "$base/eval" --strategies external_scores \
    --compression-ratios 0.20 0.10 0.05 \
    --external-scores "$scores" --external-scores-key scores \
    --start 0 --end -1 --fisher-max-points "$FISHER_MAX_POINTS" \
    --seed 0 --overwrite
}

run_step "000_check_inputs" check_inputs

for dataset in "${DATASETS[@]}"; do
  run_step "100_main_$dataset" run_main_dataset "$dataset"
done

run_step "200_saliency_ablation" \
  conda run -n "$CONDA_ENV" python scripts/fsm_p1_ablation_eval.py \
    --experiment saliency --output-dir "$OUT_ROOT/ablations/saliency" \
    --scenes room0_0 office0_0 --ratios 0.10 0.20 \
    --gamma-beta 0.3333333333333333 1 \
    --gamma-beta 1 0 --gamma-beta 0 1 --gamma-beta 1 1 \
    --gamma-beta 0.3333333333333333 0.5 --gamma-beta 0 0 \
    --start 0 --end -1 --overwrite

run_step "210_voxel_ablation" \
  conda run -n "$CONDA_ENV" python scripts/fsm_p1_ablation_eval.py \
    --experiment voxel --output-dir "$OUT_ROOT/ablations/voxel" \
    --scenes room0_0 office0_0 --ratios 0.10 0.20 \
    --voxel-multipliers 0.25 0.5 1 2 4 \
    --start 0 --end -1 --overwrite

for dataset in "${DATASETS[@]}"; do
  for seed in 0 1 2; do
    run_step "300_seed_${dataset}_$seed" run_seed_dataset "$dataset" "$seed"
  done
done

for dataset in "${DATASETS[@]}"; do
  for cap in 5000 20000; do
    run_step "400_cap_${dataset}_$cap" run_cap_dataset "$dataset" "$cap"
  done
done

cat > "$OUT_ROOT/fps_exact_skip.txt" <<'EOF'
Exact full-map FPS was intentionally not run. Available maps contain about 0.8M to 6.7M Gaussians;
the current exact FPS implementation is O(N*K) and is not computationally tractable at the paper's
10%-20% budgets. The audited experiment runs complete 5k and 20k candidate-cap comparisons instead.
EOF

for dataset in "${DATASETS[@]}"; do
  run_step "500_storage_$dataset" run_storage_dataset "$dataset"
done

run_step "600_compression_cost" \
  conda run -n "$CONDA_ENV" python scripts/fsm_compression_cost.py \
    --output-dir "$OUT_ROOT/compression_cost" \
    --scenes room0_0 room1_0 room2_0 office0_0 \
    --strategies voxel_scale_opacity fps_scale_opacity lightgaussian_proxy \
    --ratios 0.20 0.10 --seed 0 --overwrite

if [[ "$RUN_LIGHTGAUSSIAN_PROXY" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    run_step "700_lightgaussian_$dataset" run_lightgaussian_dataset "$dataset"
  done
fi

if [[ "$RUN_COVARIANCE_WEIGHTED" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    run_step "800_covariance_$dataset" run_covariance_dataset "$dataset"
  done
fi

if [[ "$RUN_RENDER" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    run_step "900_render_$dataset" run_render_dataset "$dataset"
  done
fi

run_step "950_replica_pareto_summary" \
  conda run -n "$CONDA_ENV" python scripts/fsm_pareto_summary.py \
    --output "$OUT_ROOT/replica_pareto_summary.csv" \
    --summary room0_0 "$OUT_ROOT/main/replica_room0_0/summary_all.csv" \
    --summary room1_0 "$OUT_ROOT/main/replica_room1_0/summary_all.csv" \
    --summary room2_0 "$OUT_ROOT/main/replica_room2_0/summary_all.csv" \
    --summary office0_0 "$OUT_ROOT/main/replica_office0_0/summary_all.csv"

run_step "960_tum_pareto_summary" \
  conda run -n "$CONDA_ENV" python scripts/fsm_pareto_summary.py \
    --output "$OUT_ROOT/tum_pareto_summary.csv" \
    --summary freiburg1_desk "$OUT_ROOT/main/tum_freiburg1_desk/summary_all.csv" \
    --summary freiburg2_xyz "$OUT_ROOT/main/tum_freiburg2_xyz/summary_all.csv" \
    --summary freiburg3_long_office_household "$OUT_ROOT/main/tum_freiburg3_long_office_household/summary_all.csv"

printf 'All audited experiments completed. Output: %s\n' "$OUT_ROOT"
