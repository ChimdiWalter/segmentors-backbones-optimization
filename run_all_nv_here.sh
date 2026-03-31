#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG (edit these two to your paths) ---
BESTCH_DIR="${1:-./navier_output}"      # "bestch"
FUSED_DIR="${2:-./navier_output_2}"     # "fused"
# --------------------------------------------

# Make a timestamped root so each run is clean
ROOT="nv_plots_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT"

echo "[info] bestch: $BESTCH_DIR"
echo "[info] fused : $FUSED_DIR"
echo "[info] saving under: $ROOT"

# ---------- analyze_navier_logs.py ----------
ANALYZE_DIR="$ROOT/analyze_navier_logs"
mkdir -p "$ANALYZE_DIR"
python3 analyze_navier_logs.py \
  "$BESTCH_DIR" "$FUSED_DIR" \
  --split-by-mode \
  --save "$ANALYZE_DIR"

# ---------- nv4d_plot.py ----------
NV4D1_DIR="$ROOT/nv4d_plot"
mkdir -p "$NV4D1_DIR"
# by n_contours (avg and split)
python3 nv4d_plot.py "$BESTCH_DIR" "$FUSED_DIR" \
  --by n_contours --split-by-mode --save "$NV4D1_DIR"
# by total_area_px (avg and split)
python3 nv4d_plot.py "$BESTCH_DIR" "$FUSED_DIR" \
  --by total_area_px --split-by-mode --save "$NV4D1_DIR"

# ---------- nv4d_plot_2.py (discernible colors & size-by) ----------
NV4D2_DIR="$ROOT/nv4d_plot_2"
mkdir -p "$NV4D2_DIR"
# color by n_contours, size by total_area_px
python3 nv4d_plot_2.py "$BESTCH_DIR" "$FUSED_DIR" \
  --by n_contours --split-by-mode --save "$NV4D2_DIR" \
  --norm quantile --clip 2 98 --alpha 0.95 --size-by total_area_px
# color by total_area_px, size by n_contours
python3 nv4d_plot_2.py "$BESTCH_DIR" "$FUSED_DIR" \
  --by total_area_px --split-by-mode --save "$NV4D2_DIR" \
  --norm quantile --clip 2 98 --alpha 0.95 --size-by n_contours

# ---------- nv2d_scatter.py (both union & intersection) ----------
NV2D_UNION_DIR="$ROOT/nv2d_union"
NV2D_INTER_DIR="$ROOT/nv2d_intersection"
mkdir -p "$NV2D_UNION_DIR" "$NV2D_INTER_DIR"

python3 nv2d_scatter.py \
  "$BESTCH_DIR" "$FUSED_DIR" \
  --union \
  --out "$NV2D_UNION_DIR/nv2d_union.png"

python3 nv2d_scatter.py \
  "$BESTCH_DIR" "$FUSED_DIR" \
  --intersection \
  --out "$NV2D_INTER_DIR/nv2d_intersection.png"

echo ""
echo "[done] Plots and tables saved under: $ROOT"
find "$ROOT" -maxdepth 2 -type f | sed 's/^/  /'
