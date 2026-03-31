import os
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

# ================= CONFIG =================
MODEL_PATH = "experiments/exp_v1/outputs/unet_patches/lesion_unet_patches_best.pth"
INPUT_DIR  = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"
OUTPUT_DIR = "experiments/exp_v1/outputs/unet_infer_on_leaves"

PATCH_SIZE = 512
OVERLAP = 64
THRESH = 0.5

COLOR_PINK = (255, 100, 255)


# ================= MODEL (must match training) =================
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

        return self.final(x)  # logits


# ================= HELPERS =================
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


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    stride = PATCH_SIZE - OVERLAP
    if stride <= 0:
        raise ValueError("OVERLAP must be smaller than PATCH_SIZE")

    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    print(f"Found {len(files)} images")

    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(INPUT_DIR, fname)
        img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Skipping unreadable image: {fname}")
            continue

        h, w = img_bgr.shape[:2]
        y_starts = get_starts(h, PATCH_SIZE, stride)
        x_starts = get_starts(w, PATCH_SIZE, stride)

        # Accumulate probabilities and counts for overlap averaging
        prob_sum = np.zeros((h, w), dtype=np.float32)
        count_sum = np.zeros((h, w), dtype=np.float32)

        for y in y_starts:
            for x in x_starts:
                patch = img_bgr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_padded, valid_h, valid_w = pad_patch(patch, PATCH_SIZE)

                # BGR -> RGB, normalize to [0,1], CHW
                patch_rgb = cv2.cvtColor(patch_padded, cv2.COLOR_BGR2RGB)
                patch_rgb = patch_rgb.astype(np.float32) / 255.0
                patch_chw = np.transpose(patch_rgb, (2, 0, 1))
                inp = torch.from_numpy(patch_chw).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # PATCH_SIZE x PATCH_SIZE

                # Crop back to valid (unpadded) region before stitching
                probs = probs[:valid_h, :valid_w]

                prob_sum[y:y+valid_h, x:x+valid_w] += probs
                count_sum[y:y+valid_h, x:x+valid_w] += 1.0

        pred_prob = prob_sum / np.maximum(count_sum, 1e-8)
        binary_mask = (pred_prob > THRESH).astype(np.uint8) * 255

        # Save probability map (optional, scaled to 0-255)
        prob_vis = (pred_prob * 255).clip(0, 255).astype(np.uint8)

        # Overlay contours
        overlay = img_bgr.copy()
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_PINK, 2)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_mask.png"), binary_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_prob.png"), prob_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_overlay.jpg"), overlay)

        print(f"[{idx}/{len(files)}] Done: {fname}")

    print(f"\nDone. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()