import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import tifffile as tiff
except ImportError:
    tiff = None

# ================= CONFIG =================
PATCH_METADATA_CSV = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves_patches/patch_metadata.csv"
MODEL_SAVE_PATH = "lesion_unet_patches_best.pth"

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
VAL_SPLIT = 0.15
SEED = 42

# Keep some empty patches so the model learns background too.
# 1.0 = keep all empty patches
# 0.25 = keep 25% of empty patches
KEEP_EMPTY_PROB = 0.35

NUM_WORKERS = 4
PIN_MEMORY = True


# ================= DATASET =================
def _read_any_image(path):
    ext = Path(path).suffix.lower()
    if ext in [".tif", ".tiff"] and tiff is not None:
        arr = tiff.imread(path)
        return arr
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return arr


class PatchLesionSegmentationDataset(Dataset):
    """
    Uses patch_metadata.csv from the patchify script.
    Expects columns:
      - patch_image
      - patch_mask
    """
    def __init__(self, metadata_csv, keep_empty_prob=1.0):
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.strip()

        required = ["patch_image", "patch_mask"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in patch metadata CSV")

        # Keep only rows with existing files
        valid_rows = []
        dropped = 0
        for _, row in df.iterrows():
            img_p = str(row["patch_image"])
            msk_p = str(row["patch_mask"]) if not pd.isna(row["patch_mask"]) else ""
            if not img_p or not os.path.exists(img_p):
                dropped += 1
                continue
            if not msk_p or not os.path.exists(msk_p):
                dropped += 1
                continue
            valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # Optionally subsample empty-mask patches to reduce class imbalance
        if keep_empty_prob < 1.0 and len(self.df) > 0:
            kept = []
            for _, row in self.df.iterrows():
                m = _read_any_image(row["patch_mask"])
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[..., 0]
                has_lesion = np.any(m > 0)
                if has_lesion:
                    kept.append(row)
                else:
                    if random.random() < keep_empty_prob:
                        kept.append(row)

            self.df = pd.DataFrame(kept).reset_index(drop=True)

        print(f"[Dataset] Loaded {len(self.df)} patch pairs (dropped {dropped}).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["patch_image"])
        mask_path = str(row["patch_mask"])

        # ---- Image ----
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image patch: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        # ---- Mask ----
        mask = _read_any_image(mask_path)
        if mask is None:
            raise RuntimeError(f"Failed to read mask patch: {mask_path}")

        if mask.ndim == 3:
            # if mask saved as RGB/3-channel PNG, use first channel
            mask = mask[..., 0]

        # Binary mask
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # H,W -> 1,H,W

        return torch.from_numpy(img), torch.from_numpy(mask)


# ================= MODEL (UNet-like) =================
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
        x1 = self.e1(x)               # H, W
        x2 = self.e2(self.pool(x1))   # H/2, W/2
        x3 = self.e3(self.pool(x2))   # H/4, W/4

        x = self.up2(x3)              # H/2, W/2
        x = self.d2(torch.cat([x, x2], dim=1))

        x = self.up1(x)               # H, W
        x = self.d1(torch.cat([x, x1], dim=1))

        return self.final(x)          # logits


# ================= LOSSES / METRICS =================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


@torch.no_grad()
def batch_iou(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


# ================= TRAINING =================
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = PatchLesionSegmentationDataset(
        PATCH_METADATA_CSV,
        keep_empty_prob=KEEP_EMPTY_PROB
    )

    if len(dataset) == 0:
        print("No patch pairs found. Check patch_metadata.csv and patch paths.")
        return

    # Train/val split
    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)

            # Combined loss: BCE + Dice
            loss_bce = bce(logits, masks)
            loss_dice = dice(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_iou_sum += batch_iou(logits, masks)

        train_loss = train_loss_sum / max(1, len(train_loader))
        train_iou = train_iou_sum / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(imgs)
                loss_bce = bce(logits, masks)
                loss_dice = dice(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice

                val_loss_sum += loss.item()
                val_iou_sum += batch_iou(logits, masks)

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_iou = val_iou_sum / max(1, len(val_loader))

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [Saved best] {MODEL_SAVE_PATH}")

    # Save final too (optional)
    final_path = MODEL_SAVE_PATH.replace(".pth", "_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Best model: {MODEL_SAVE_PATH}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    train()