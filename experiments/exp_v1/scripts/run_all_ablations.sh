#!/usr/bin/env bash
# run_all_ablations.sh  –  Full ablation pipeline for Paper 3 tables.
#
# Usage:  source ~/venvs/lesegenv/bin/activate && bash run_all_ablations.sh
#
# Steps:
#   1. Create 30-image stratified subset + symlinks
#   2. Create objective-ablation script copies (one weight zeroed each)
#   3. Run objective ablations (6 variants)
#   4. Run search ablations  (5 variants)
#   5. Summarise → LaTeX → patch paper → rebuild PDF
# -----------------------------------------------------------------------
set -euo pipefail

BASE="$(cd "$(dirname "$0")/../../.." && pwd)"   # segmentors_backbones/
SCRIPTS="$BASE/experiments/exp_v1/scripts"
INPUT="$BASE/experiments/exp_v1/configs/ablation_input"
OBJ_OUT="$BASE/experiments/exp_v1/outputs/ablations_objective"
SEARCH_OUT="$BASE/experiments/exp_v1/outputs/ablations_search"
OPTIMIZER="$BASE/navier_optimize_hellbender_updatecsv.py"
WORKERS="${ABLATION_WORKERS:-4}"                 # override with env var

echo "============================================================"
echo "  ABLATION PIPELINE  –  $(date)"
echo "  BASE:    $BASE"
echo "  WORKERS: $WORKERS"
echo "============================================================"

# ── 1. Create 30-image subset ─────────────────────────────────────
echo ""
echo ">>> Step 1: Creating 30-image stratified subset ..."
python3 "$SCRIPTS/create_ablation_subset.py"

if [ ! -d "$INPUT" ] || [ -z "$(ls -A "$INPUT" 2>/dev/null)" ]; then
    echo "ERROR: ablation_input is empty"; exit 1
fi
N_IMAGES=$(ls "$INPUT" | wc -l)
echo "  $N_IMAGES images in ablation_input"

# ── 2. Create objective-ablation optimizer copies ─────────────────
echo ""
echo ">>> Step 2: Creating objective-ablation script copies ..."
VARIANT_DIR="$BASE/experiments/exp_v1/scripts/obj_variants"
mkdir -p "$VARIANT_DIR"

# The original weights in the optimizer (line-matched):
#   W_AREA         = 0.05
#   W_SMALL_BASE   = 0.1
#   W_GRAD_ALIGN   = 0.5
#   W_COLOR_DIST   = 4.0
#   W_CONTOUR      = 2.0

declare -A OBJ_VARIANTS
OBJ_VARIANTS=(
    [no_color_dist]="W_COLOR_DIST"
    [no_grad_align]="W_GRAD_ALIGN"
    [no_contour]="W_CONTOUR"
    [no_area]="W_AREA"
    [no_small_base]="W_SMALL_BASE"
)

for variant in "${!OBJ_VARIANTS[@]}"; do
    weight="${OBJ_VARIANTS[$variant]}"
    dst="$VARIANT_DIR/optimizer_${variant}.py"
    # Regex: match the line "W_WEIGHT_NAME  = <number>" and set it to 0
    sed -E "s/^(${weight}[[:space:]]*=)[[:space:]]*[0-9.]+/\1 0.0/" "$OPTIMIZER" > "$dst"
    # Verify the change
    val=$(grep -oP "^${weight}\s*=\s*\K[0-9.]+" "$dst" || echo "?")
    echo "  $variant: $weight -> $val  ($dst)"
done

# Also copy the unmodified optimizer for the 'full' baseline
cp "$OPTIMIZER" "$VARIANT_DIR/optimizer_full.py"
echo "  full: all weights unchanged"

# ── 3. Run objective ablations ────────────────────────────────────
echo ""
echo ">>> Step 3: Running objective ablations ..."

run_variant() {
    local name="$1"
    local script="$2"
    local outdir="$3/$name"
    mkdir -p "$outdir"

    echo "  [$name] Starting ... -> $outdir"
    python3 "$script" \
        --input  "$INPUT" \
        --output "$outdir" \
        --mode   fused \
        --budget-random 64 \
        --topk-refine   4 \
        --refine-steps  12 \
        --downscale     0.75 \
        --workers       "$WORKERS" \
        --seed          1337 \
        --log           opt_summary.csv \
        2>&1 | tail -3
    echo "  [$name] Done."
}

for variant in full no_color_dist no_grad_align no_contour no_area no_small_base; do
    run_variant "$variant" "$VARIANT_DIR/optimizer_${variant}.py" "$OBJ_OUT"
done

# ── 4. Run search ablations ──────────────────────────────────────
echo ""
echo ">>> Step 4: Running search ablations ..."

run_search() {
    local name="$1"; shift
    local outdir="$SEARCH_OUT/$name"
    mkdir -p "$outdir"

    echo "  [$name] Starting ... -> $outdir"
    python3 "$OPTIMIZER" \
        --input  "$INPUT" \
        --output "$outdir" \
        --mode   fused \
        --log    opt_summary.csv \
        --workers "$WORKERS" \
        --seed   1337 \
        "$@" \
        2>&1 | tail -3
    echo "  [$name] Done."
}

# default (full pipeline, same as objective/full but stored under search/)
run_search "default" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 0.75

# random-only: no refinement phase
run_search "random_only" \
    --budget-random 64 --topk-refine 0 --refine-steps 0 --downscale 0.75

# low-budget: fewer random samples
run_search "low_budget" \
    --budget-random 16 --topk-refine 4 --refine-steps 12 --downscale 0.75

# no-downscale: full resolution during search
run_search "no_downscale" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 1.0

# fast: hard timeout per image
run_search "fast" \
    --budget-random 64 --topk-refine 4 --refine-steps 12 --downscale 0.75 \
    --per-image-seconds 60

# ── 5. Summarise, generate LaTeX, patch paper, build PDF ─────────
echo ""
echo ">>> Step 5: Summarising and generating LaTeX tables ..."
python3 "$SCRIPTS/summarize_and_latex.py"

echo ""
echo ">>> Step 6: Rebuilding PDF ..."
cd "$BASE/Paper_3"
if command -v latexmk &>/dev/null; then
    latexmk -pdf -interaction=nonstopmode paper3_exp_v1.tex 2>&1 | tail -5
else
    pdflatex -interaction=nonstopmode paper3_exp_v1.tex 2>&1 | tail -5
    pdflatex -interaction=nonstopmode paper3_exp_v1.tex 2>&1 | tail -5
fi

echo ""
if [ -f paper3_exp_v1.pdf ]; then
    echo "============================================================"
    echo "  SUCCESS – Paper_3/paper3_exp_v1.pdf updated"
    echo "  $(ls -lh paper3_exp_v1.pdf)"
    echo "============================================================"
else
    echo "ERROR: PDF not found after build"
    exit 1
fi
