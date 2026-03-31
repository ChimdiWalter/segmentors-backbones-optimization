#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn

COLOR_PINK = (255, 100, 255)

# ---- Model must match training ----
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cbr(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.e1 = cbr(3, 32)
        self.e2 = cbr(32, 64)
        self.e3 = cbr(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = cbr(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = cbr(64, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x = self.up2(x3)
        x = self.d2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.d1(torch.cat([x, x1], dim=1))
        return self.final(x)

def get_starts(length, patch_size, stride):
    starts = list(range(0, max(1, length), stride))
    if not starts:
        starts = [0]
    if starts[-1] + patch_size < length:
        starts.append(length - patch_size)
    return sorted(set(max(0, s) for s in starts))

def pad_patch(img_patch, patch_size):
    h, w = img_patch.shape[:2]
    pad_bottom = max(0, patch_size - h)
    pad_right = max(0, patch_size - w)
    if pad_bottom == 0 and pad_right == 0:
        return img_patch, h, w
    padded = cv2.copyMakeBorder(
        img_patch, 0, pad_bottom, 0, pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )
    return padded, h, w

def filter_components(binary_mask_u8, min_area):
    if min_area <= 0:
        return binary_mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(binary_mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument("--max_area_frac", type=float, default=1.0)
    ap.add_argument("--apply_leaf_roi", action="store_true")
    ap.add_argument("--leaf_gray_thresh", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    stride = args.patch_size - args.overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])
    if args.limit:
        files = files[:args.limit]

    print("Found images:", len(files))

    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("[WARN] unreadable:", fname)
            continue

        h, w = img_bgr.shape[:2]
        y_starts = get_starts(h, args.patch_size, stride)
        x_starts = get_starts(w, args.patch_size, stride)

        prob_sum = np.zeros((h, w), dtype=np.float32)
        count_sum = np.zeros((h, w), dtype=np.float32)

        for y in y_starts:
            for x in x_starts:
                patch = img_bgr[y:y+args.patch_size, x:x+args.patch_size]
                patch_padded, valid_h, valid_w = pad_patch(patch, args.patch_size)

                patch_rgb = cv2.cvtColor(patch_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                patch_chw = np.transpose(patch_rgb, (2, 0, 1))
                inp = torch.from_numpy(patch_chw).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

                probs = probs[:valid_h, :valid_w]
                prob_sum[y:y+valid_h, x:x+valid_w] += probs
                count_sum[y:y+valid_h, x:x+valid_w] += 1.0

        pred_prob = prob_sum / np.maximum(count_sum, 1e-8)
        binary_mask = (pred_prob > args.thresh).astype(np.uint8) * 255

        if args.apply_leaf_roi:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            leaf_mask = (gray > args.leaf_gray_thresh).astype(np.uint8) * 255
            binary_mask = cv2.bitwise_and(binary_mask, leaf_mask)

        binary_mask = filter_components(binary_mask, min_area=args.min_area)

        area_frac = (binary_mask > 0).mean()
        if area_frac > args.max_area_frac:
            binary_mask[:] = 0

        prob_vis = (pred_prob * 255).clip(0, 255).astype(np.uint8)

        overlay = img_bgr.copy()
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_PINK, 2)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_mask.png"), binary_mask)
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_prob.png"), prob_vis)
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_overlay.jpg"), overlay)

        if idx % 10 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] {fname}")

    print("Done. Output:", args.output_dir)

if __name__ == "__main__":
    main()