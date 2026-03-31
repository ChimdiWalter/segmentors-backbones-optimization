#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP="experiments/exp_v1"
LEAVES="/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"
OPT_CSV="$EXP/outputs/opt_summary_local.csv"
MODEL="$EXP/outputs/unet_patches/lesion_unet_patches_best.pth"

mkdir -p "$EXP/logs" "$EXP/outputs"

echo "=== (1) ROI-gated inference (thr=0.30) ==="
OUT_STUDENT="$EXP/outputs/unet_infer_on_leaves_roi_thr0p30"
mkdir -p "$OUT_STUDENT"
python3 infer_unet_patchify_cli.py \
  --model "$MODEL" \
  --input_dir "$LEAVES" \
  --output_dir "$OUT_STUDENT" \
  --patch_size 512 --overlap 64 --thresh 0.30 \
  --apply_leaf_roi --leaf_gray_thresh 10 \
  2>&1 | tee "$EXP/logs/infer_on_leaves_roi_thr0p30.log"

echo "=== (2) Evaluate teacher vs ROI-student ==="
python3 eval_teacher_student.py \
  --opt_csv "$OPT_CSV" \
  --student_dir "$OUT_STUDENT" \
  --out_csv "$EXP/outputs/teacher_student_metrics_roi_thr0p30.csv" \
  2>&1 | tee "$EXP/logs/eval_teacher_student_roi_thr0p30.log"

echo "=== (3) Failure taxonomy summary ==="
python3 - << 'PY' | tee "$EXP/logs/failure_taxonomy_roi_thr0p30.log"
import pandas as pd
df = pd.read_csv("experiments/exp_v1/outputs/teacher_student_metrics_roi_thr0p30.csv")
bad = df[df["dice"] < 0.3].copy()
bad["teacher_empty"] = bad["teacher_area_px"] == 0
bad["student_empty"] = bad["student_area_px"] == 0
print("Bad cases (<0.3):", len(bad))
print("Teacher empty:", bad["teacher_empty"].sum())
print("Teacher empty & student nonempty:", ((bad["teacher_empty"]) & (~bad["student_empty"])).sum())
print("Teacher nonempty & student empty:", ((~bad["teacher_empty"]) & (bad["student_empty"])).sum())
print("Both nonempty mismatch:", ((~bad["teacher_empty"]) & (~bad["student_empty"])).sum())
PY

echo "=== (4) Build disagreement montage (paper figure) ==="
python3 make_disagreement_montage.py \
  --metrics_csv "$EXP/outputs/teacher_student_metrics_roi_thr0p30.csv" \
  --teacher_dir "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_3" \
  --student_dir "$OUT_STUDENT" \
  --raw_dir "$LEAVES" \
  --out_png "$EXP/outputs/paper_disagreement_montage_roi_thr0p30.png" \
  --n 6

echo "DONE: "
echo "  - $EXP/outputs/teacher_student_metrics_roi_thr0p30.csv"
echo "  - $EXP/outputs/paper_disagreement_montage_roi_thr0p30.png"