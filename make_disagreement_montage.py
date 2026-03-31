#!/usr/bin/env python3
import os, argparse, glob
import csv
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

def put(img, txt, x=10, y=28, s=0.8, t=2):
    cv2.putText(img, txt, (x,y), FONT, s, (0,0,0), t+2, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y), FONT, s, (255,255,255), t, cv2.LINE_AA)

def read_any8(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.dtype == np.uint16:
        im = (im / 257).astype(np.uint8)
    elif im.dtype != np.uint8:
        im = im.astype(np.uint8)
    return im

def find_teacher_overlay(teacher_dir, base):
    pref = base + "__"
    cands = sorted([p for p in glob.glob(os.path.join(teacher_dir, pref + "*__overlay16.tif"))])
    return cands[0] if cands else None

def resize_h(im, H=360):
    h,w = im.shape[:2]
    if h == 0: return im
    s = H / h
    return cv2.resize(im, (max(1,int(w*s)), H), interpolation=cv2.INTER_AREA)

def read_metrics_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row["teacher_area_px"] = int(float(r.get("teacher_area_px", 0) or 0))
            row["student_area_px"] = int(float(r.get("student_area_px", 0) or 0))
            row["dice"] = float(r.get("dice", 0.0) or 0.0)
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--student_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    rows = read_metrics_csv(args.metrics_csv)
    # disagreement = teacher empty, student non-empty
    dis = [r for r in rows if r["teacher_area_px"] == 0 and r["student_area_px"] > 0]
    dis = sorted(dis, key=lambda r: r["student_area_px"], reverse=True)[:args.n]

    if len(dis) == 0:
        raise RuntimeError("No teacher-empty/student-nonempty cases found.")

    rows = []
    for r in dis:
        base = r["base"]
        raw_path = os.path.join(args.raw_dir, base + ".tif")
        raw = read_any8(raw_path)
        if raw is None:
            raw = np.zeros((360,600,3), dtype=np.uint8)

        t_overlay_path = find_teacher_overlay(args.teacher_dir, base)
        t_overlay = read_any8(t_overlay_path) if t_overlay_path else np.zeros_like(raw)

        s_overlay_path = os.path.join(args.student_dir, base + "_overlay.jpg")
        s_overlay = read_any8(s_overlay_path)
        if s_overlay is None:
            s_overlay = np.zeros_like(raw)

        raw = resize_h(raw, 360)
        t_overlay = resize_h(t_overlay, 360)
        s_overlay = resize_h(s_overlay, 360)

        put(raw, "Raw")
        put(t_overlay, "Teacher overlay (optimizer)")
        put(s_overlay, f"Student overlay (UNet)  Dice={r['dice']:.3f}")

        gap = np.zeros((raw.shape[0], 18, 3), dtype=np.uint8)
        row = np.concatenate([raw, gap, t_overlay, gap, s_overlay], axis=1)
        rows.append(row)

    max_w = max(x.shape[1] for x in rows)
    padded = []
    for x in rows:
        if x.shape[1] < max_w:
            pad = np.zeros((x.shape[0], max_w-x.shape[1], 3), dtype=np.uint8)
            x = np.concatenate([x, pad], axis=1)
        padded.append(x)

    sep = np.zeros((22, max_w, 3), dtype=np.uint8)
    montage = padded[0]
    for x in padded[1:]:
        montage = np.concatenate([montage, sep, x], axis=0)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    cv2.imwrite(args.out_png, montage)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
