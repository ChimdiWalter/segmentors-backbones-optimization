#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np

EX_DIR = "experiments/exp_v1/outputs/teacher_student_figures/examples"
OUT_PATH = "experiments/exp_v1/outputs/teacher_student_figures/montage_teacher_student.png"

FONT = cv2.FONT_HERSHEY_SIMPLEX

def read_metrics_txt(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k.strip()] = v.strip()
    # dice/iou may be missing if format changes, default to '?'
    return d.get("base",""), d.get("dice","?"), d.get("iou","?"), d.get("teacher_area_px","?"), d.get("student_area_px","?")

def put_label(img, text, x=10, y=30, scale=0.9, thick=2):
    # black outline + white text for readability
    cv2.putText(img, text, (x, y), FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, (255,255,255), thick, cv2.LINE_AA)

def load_pair(folder, base):
    t = os.path.join(folder, f"{base}__teacher_overlay.png")
    s = os.path.join(folder, f"{base}__student_overlay.jpg")
    mt = os.path.join(folder, f"{base}__metrics.txt")
    if not (os.path.exists(t) and os.path.exists(s) and os.path.exists(mt)):
        raise FileNotFoundError(f"Missing one of: {t}, {s}, {mt}")
    timg = cv2.imread(t, cv2.IMREAD_COLOR)
    simg = cv2.imread(s, cv2.IMREAD_COLOR)
    return timg, simg, mt

def pick_one(folder):
    # just pick the first metrics file in that category folder
    mfiles = sorted(glob.glob(os.path.join(folder, "*__metrics.txt")))
    if not mfiles:
        raise RuntimeError(f"No metrics files found in {folder}")
    base = os.path.basename(mfiles[0]).replace("__metrics.txt", "")
    return base

def resize_same_height(a, b, H=420):
    def rs(img):
        h, w = img.shape[:2]
        scale = H / max(1, h)
        return cv2.resize(img, (int(w*scale), H), interpolation=cv2.INTER_AREA)
    return rs(a), rs(b)

def make_row(category):
    folder = os.path.join(EX_DIR, category)
    base = pick_one(folder)
    timg, simg, mt = load_pair(folder, base)
    b, dice, iou, ta, sa = read_metrics_txt(mt)

    # resize
    timg, simg = resize_same_height(timg, simg, H=420)

    # add headers
    put_label(timg, "Teacher (Optimizer)", y=30)
    put_label(simg, "Student (UNet)", y=30)

    # add metrics (on student)
    put_label(simg, f"{category.upper()} | Dice={dice}  IoU={iou}", y=65, scale=0.8)
    put_label(simg, f"Teacher area={ta}  Student area={sa}", y=95, scale=0.7)

    # concat side-by-side
    gap = np.zeros((timg.shape[0], 20, 3), dtype=np.uint8)
    row = np.concatenate([timg, gap, simg], axis=1)
    return row

def main():
    rows = [
        make_row("worst"),
        make_row("median"),
        make_row("best"),
    ]

    # pad to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        pad = max_w - r.shape[1]
        if pad > 0:
            r = np.concatenate([r, np.zeros((r.shape[0], pad, 3), dtype=np.uint8)], axis=1)
        padded.append(r)

    # stack with separators
    sep = np.zeros((25, max_w, 3), dtype=np.uint8)
    montage = np.concatenate([padded[0], sep, padded[1], sep, padded[2]], axis=0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ok = cv2.imwrite(OUT_PATH, montage)
    if not ok:
        raise RuntimeError(f"Failed to write montage: {OUT_PATH}")
    print("Saved montage:", OUT_PATH)

if __name__ == "__main__":
    main()