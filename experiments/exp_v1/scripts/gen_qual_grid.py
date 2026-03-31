#!/usr/bin/env python3
"""Generate qual_grid_teacher_student.png: 6 rows x 5 columns montage.

Columns: Input | Teacher overlay | Student overlay | Teacher mask | Student mask

Selects 6 representative images from teacher_student_metrics_roi_thr0p30.csv,
spread across the Dice range.
"""

import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent.parent.parent
LEAVES_DIR = BASE / "leaves"
UNET_DIR = BASE / "experiments" / "exp_v1" / "outputs" / "unet_infer_on_leaves_roi_thr0p30"
OPT_CSV = BASE / "experiments" / "exp_v1" / "outputs" / "opt_summary_local.csv"
TS_CSV = BASE / "experiments" / "exp_v1" / "outputs" / "teacher_student_metrics_roi_thr0p30.csv"
OUT_PATH = BASE / "Paper_3" / "figs" / "exp_v1" / "qual_grid_teacher_student.png"

N_ROWS = 6
THUMB_W = 400
THUMB_H = 200
PAD = 4
COL_LABELS = ["Input", "Teacher overlay", "Student overlay", "Teacher mask", "Student mask"]
LABEL_H = 24


def load_opt_summary():
    """Map filename base -> overlay16_path, mask16_path."""
    data = {}
    with open(OPT_CSV) as f:
        for r in csv.DictReader(f):
            base = r["filename"].replace(".tif", "")
            data[base] = r
    return data


def main():
    # Read teacher_student_metrics
    ts_rows = []
    with open(TS_CSV) as f:
        for r in csv.DictReader(f):
            ts_rows.append(r)

    # Sort by Dice and pick 6 spread across range
    ts_rows.sort(key=lambda r: float(r["dice"]))
    n = len(ts_rows)
    indices = [int(i * (n - 1) / (N_ROWS - 1)) for i in range(N_ROWS)]
    selected = [ts_rows[i] for i in indices]

    opt_data = load_opt_summary()

    # Build grid
    ncols = 5
    grid_w = ncols * THUMB_W + (ncols + 1) * PAD
    grid_h = LABEL_H + N_ROWS * THUMB_H + (N_ROWS + 1) * PAD
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column labels
    for col_idx, label in enumerate(COL_LABELS):
        x = PAD + col_idx * (THUMB_W + PAD) + THUMB_W // 2
        draw.text((x, 2), label, fill=(0, 0, 0), anchor="mt")

    for row_idx, ts_row in enumerate(selected):
        base = ts_row["base"]
        leaf_path = LEAVES_DIR / (base + ".tif")
        teacher_overlay_path = Path(opt_data[base]["overlay16_path"])
        teacher_mask_path = Path(opt_data[base]["mask16_path"])
        student_overlay_path = UNET_DIR / (base + "_overlay.jpg")
        student_mask_path = UNET_DIR / (base + "_mask.png")

        paths = [leaf_path, teacher_overlay_path, student_overlay_path,
                 teacher_mask_path, student_mask_path]

        y_offset = LABEL_H + PAD + row_idx * (THUMB_H + PAD)

        for col_idx, p in enumerate(paths):
            x_offset = PAD + col_idx * (THUMB_W + PAD)
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
                canvas.paste(img, (x_offset, y_offset))
            except Exception as e:
                # Draw placeholder
                draw.rectangle([x_offset, y_offset, x_offset + THUMB_W, y_offset + THUMB_H],
                               fill=(200, 200, 200))
                draw.text((x_offset + 4, y_offset + 4), f"Missing: {p.name}", fill=(128, 0, 0))

    canvas.save(OUT_PATH, quality=92)
    print(f"Wrote {OUT_PATH} ({grid_w}x{grid_h})")
    print(f"Selected images (by Dice):")
    for ts_row in selected:
        print(f"  {ts_row['base']}: Dice={float(ts_row['dice']):.3f}")


if __name__ == "__main__":
    main()
