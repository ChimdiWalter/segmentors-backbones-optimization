#!/usr/bin/env bash
# run_ablations_parallel.sh  –  Launch all 11 ablation variants in parallel.
#
# Usage:  source ~/.venvs/lesegenv/bin/activate && bash run_ablations_parallel.sh
#
# Each variant uses --workers 2 and --per-image-seconds 300 for bounded runtime.
# All 11 variants run concurrently; total ~22 workers.
# Expected wall time: ~45-90 min on a loaded machine.
# -----------------------------------------------------------------------
set -uo pipefail

BASE="$(cd "$(dirname "$0")/../../.." && pwd)"
INPUT="$BASE/experiments/exp_v1/configs/ablation_input"
OBJ_OUT="$BASE/experiments/exp_v1/outputs/ablations_objective"
SEARCH_OUT="$BASE/experiments/exp_v1/outputs/ablations_search"
VARIANT_DIR="$BASE/experiments/exp_v1/scripts/obj_variants"
OPTIMIZER="$BASE/navier_optimize_hellbender_updatecsv.py"
LOGDIR="$BASE/experiments/exp_v1/logs/ablations"
W=2  # workers per variant

mkdir -p "$LOGDIR"

echo "============================================================"
echo "  PARALLEL ABLATION PIPELINE  –  $(date)"
echo "  BASE:    $BASE"
echo "  WORKERS per variant: $W"
echo "============================================================"

PIDS=()
NAMES=()

run_obj() {
    local name="$1"
    local script="$2"
    local outdir="$OBJ_OUT/$name"
    mkdir -p "$outdir"
    echo "[OBJ/$name] Starting ..."
    python3 "$script" \
        --input "$INPUT" --output "$outdir" --mode fused \
        --budget-random 64 --topk-refine 4 --refine-steps 12 \
        --downscale 0.75 --workers "$W" --seed 1337 \
        --per-image-seconds 300 --log opt_summary.csv \
        > "$LOGDIR/obj_${name}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("obj/$name")
}

run_search() {
    local name="$1"; shift
    local outdir="$SEARCH_OUT/$name"
    mkdir -p "$outdir"
    echo "[SEARCH/$name] Starting ..."
    python3 "$OPTIMIZER" \
        --input "$INPUT" --output "$outdir" --mode fused \
        --workers "$W" --seed 1337 --log opt_summary.csv \
        "$@" \
        > "$LOGDIR/search_${name}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("search/$name")
}

# ── Objective ablations (6) ───────────────────────────────────────
run_obj "full"           "$VARIANT_DIR/optimizer_full.py"
run_obj "no_color_dist"  "$VARIANT_DIR/optimizer_no_color_dist.py"
run_obj "no_grad_align"  "$VARIANT_DIR/optimizer_no_grad_align.py"
run_obj "no_contour"     "$VARIANT_DIR/optimizer_no_contour.py"
run_obj "no_area"        "$VARIANT_DIR/optimizer_no_area.py"
run_obj "no_small_base"  "$VARIANT_DIR/optimizer_no_small_base.py"

# ── Search ablations (5) ─────────────────────────────────────────
# default (full pipeline)
run_search "default" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 0.75 \
    --per-image-seconds 300

# random-only: no refinement
run_search "random_only" \
    --budget-random 64 --topk-refine 0 --refine-steps 0 --downscale 0.75 \
    --per-image-seconds 300

# low-budget: fewer random samples
run_search "low_budget" \
    --budget-random 16 --topk-refine 4 --refine-steps 12 --downscale 0.75 \
    --per-image-seconds 300

# no-downscale: full resolution search
run_search "no_downscale" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 1.0 \
    --per-image-seconds 300

# fast: aggressive timeout
run_search "fast" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 0.75 \
    --per-image-seconds 60

echo ""
echo "Launched ${#PIDS[@]} variants.  PIDs: ${PIDS[*]}"
echo "Logs: $LOGDIR/"
echo ""
echo "Waiting for all to finish ..."

FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "$pid"; then
        echo "  [DONE] $name (PID $pid) – OK"
    else
        echo "  [FAIL] $name (PID $pid) – exit $?"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================================"
echo "  ALL VARIANTS COMPLETE  ($FAIL failures)  –  $(date)"
echo "============================================================"

# ── Summary of outputs ────────────────────────────────────────────
echo ""
echo "=== Output CSVs ==="
for csv in "$OBJ_OUT"/*/opt_summary.csv "$SEARCH_OUT"/*/opt_summary.csv; do
    if [ -f "$csv" ]; then
        rows=$(tail -n +2 "$csv" | wc -l)
        echo "  $csv : $rows data rows"
    fi
done

exit $FAIL
