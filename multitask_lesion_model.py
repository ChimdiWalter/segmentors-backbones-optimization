#!/usr/bin/env python3
"""
multitask_lesion_model.py

Multi-task model that learns:
  - Lesion segmentation mask
  - Navier+snake parameters [mu, lambda, diffusion_rate, alpha, beta, gamma, energy_threshold]

Uses:
  - CSV from Navier optimizer (cpu_output_3/opt_summary.csv)
  - Original leaf images (leaves/)
  - 16-bit mask TIFs located in the *same directory* as the CSV,
    ignoring any /cluster/VAST/... prefix in mask16_path.
"""

import os
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import tifffile as tiff
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

# ----------------------------- CONFIG ----------------------------------------

@dataclass
class Config:
    csv_path: str = (
        "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/"
        "cpu_output_3/opt_summary.csv"
    )
    img_dir: str = (
        "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"
    )
    out_dir: str = "multitask_out"

    img_size: int = 256
    in_channels: int = 3

    batch_size: int = 8
    num_epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-5

    num_workers: int = 4
    seed: int = 1337

    # Train/val/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Multi-task loss weighting
    lambda_params: float = 1.0  # weight for parameter regression vs segmentation


PARAM_COLS = [
    "mu",
    "lambda",
    "diffusion_rate",
    "alpha",
    "beta",
    "gamma",
    "energy_threshold",
]

# ----------------------------- UTILS -----------------------------------------


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_dataframe(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load CSV, filter for good rows, and fix mask paths so they are
    resolved relative to the CSV directory.

    Returns:
        df_filtered, param_means, param_stds
    """
    print(f"[prepare_dataframe] Loaded CSV from {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path)
    print(f"[prepare_dataframe] Loaded {len(df)} rows from {cfg.csv_path}")

    # Basic filter: keep good rows
    df = df[(df["status"] == "ok") & (df["score"] > 0.0)].reset_index(drop=True)
    print(
        f"[prepare_dataframe] Filtered dataset size: {len(df)} rows (status=ok & score>0.0)"
    )

    csv_dir = os.path.dirname(cfg.csv_path)
    resolved_mask_paths: List[str] = []
    keep_indices: List[int] = []

    missing = 0
    for idx, row in df.iterrows():
        raw_mask_path = str(row["mask16_path"])
        base = os.path.basename(raw_mask_path)
        local_path = os.path.join(csv_dir, base)
        if not os.path.exists(local_path):
            missing += 1
            continue
        keep_indices.append(idx)
        resolved_mask_paths.append(local_path)

    if missing > 0:
        print(f"[prepare_dataframe] WARNING: {missing} rows dropped due to missing masks")

    df = df.iloc[keep_indices].reset_index(drop=True)
    df["mask16_resolved"] = resolved_mask_paths
    print(f"[prepare_dataframe] After resolving masks, {len(df)} rows remain")

    # Parameter stats
    params = df[PARAM_COLS].to_numpy(dtype=np.float32)
    param_means = params.mean(axis=0)
    param_stds = params.std(axis=0) + 1e-6  # avoid divide-by-zero
    print(f"[Param stats] means: {param_means}")
    print(f"[Param stats] stds : {param_stds}")

    return df, param_means, param_stds


# ----------------------------- DATASET ---------------------------------------


class LesionMultiTaskDataset(Dataset):
    """
    Each sample:
        img_t       : [C, H, W] float32
        mask_t      : [1, H, W] float32
        params_norm : [P]       float32 (normalized)
        meta        : dict with filename and mask paths
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        param_means: np.ndarray,
        param_stds: np.ndarray,
        img_size: int = 256,
        in_channels: int = 3,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.param_means = param_means.astype(np.float32)
        self.param_stds = param_stds.astype(np.float32)
        self.augment = augment

        print(
            f"[LesionMultiTaskDataset] N={len(self.df)}, "
            f"augment={self.augment}, in_channels={self.in_channels}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str) -> np.ndarray:
        """
        Load RGB leaf image and return [C, H, W] float32 in [0,1].
        """
        img_path = os.path.join(self.img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  # [H, W, C]
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        return img

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load 16-bit mask TIFF, binarize, resize, shape [1, H, W].
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = tiff.imread(mask_path)
        mask = (mask > 0).astype(np.float32)  # [H, W]
        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = np.expand_dims(mask, axis=0)  # [1, H, W]
        return mask

    def _augment(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple spatial augmentations (flips). These create negative strides,
        so we will explicitly copy before returning.
        """
        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, :, ::-1]
            mask = mask[:, :, ::-1]
        # Vertical flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1, :]

        # You can add brightness/contrast or small rotations if desired.

        # IMPORTANT: copy to ensure positive strides (avoid torch.from_numpy error)
        img = img.copy()
        mask = mask.copy()
        return img, mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["filename"]

        img = self._load_image(filename)  # [C, H, W]
        mask_path = row["mask16_resolved"]
        mask = self._load_mask(mask_path)  # [1, H, W]

        if self.augment:
            img, mask = self._augment(img, mask)

        # Extra safety: ensure contiguous arrays with positive strides
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        # Params -> normalized
        params = row[PARAM_COLS].to_numpy(dtype=np.float32)
        params_norm = (params - self.param_means) / self.param_stds

        img_t = torch.from_numpy(img)          # [C, H, W]
        mask_t = torch.from_numpy(mask)        # [1, H, W]
        params_t = torch.from_numpy(params_norm)  # [P]

        meta = {
            "filename": filename,
            "raw_mask16_path": row["mask16_path"],
            "mask16_resolved": mask_path,
        }

        return img_t, mask_t, params_t, meta


# ----------------------------- MODEL -----------------------------------------


class MultiTaskLesionNet(nn.Module):
    """
    ResNet18 backbone with:
      - a parameter regression head (7 outputs)
      - a segmentation head that upsamples backbone features to 256x256
    """

    def __init__(self, in_channels: int = 3, num_params: int = 7):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt first conv if needed
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # We'll use all layers except fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,   # -> 64 x 64 x 64 for 256x256 input
            backbone.layer1,    # 64 x 64 x 64
            backbone.layer2,    # 128 x 32 x 32
            backbone.layer3,    # 256 x 16 x 16
            backbone.layer4,    # 512 x 8 x 8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Param head
        self.param_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_params),
        )

        # Simple decoder for segmentation
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 8 -> 16
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 16 -> 32
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 32 -> 64
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4)   # 64 -> 256
        self.dec0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)  # final logits

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W] -> seg_logits: [B, 1, H, W], params: [B, P]
        """
        feats = self.backbone(x)  # [B, 512, 8, 8] for 256x256

        # Param regression
        pooled = self.global_pool(feats)
        params = self.param_head(pooled)  # [B, P]

        # Segmentation
        d3 = self.dec3(self.up3(feats))   # [B, 256, 16, 16]
        d2 = self.dec2(self.up2(d3))      # [B, 128, 32, 32]
        d1 = self.dec1(self.up1(d2))      # [B, 64, 64, 64]
        d0 = self.dec0(self.up0(d1))      # [B, 32, 256, 256]
        seg_logits = self.seg_head(d0)    # [B, 1, 256, 256]

        return seg_logits, params


# ----------------------------- TRAIN / EVAL ----------------------------------


def dice_coefficient(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    pred_logits: [B, 1, H, W] (raw logits)
    target:      [B, 1, H, W] (0 or 1)
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def train_multitask(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_multitask] Using device: {device}")

    ensure_dir(cfg.out_dir)

    df, param_means, param_stds = prepare_dataframe(cfg)

    # Split into train/val/test
    total_n = len(df)
    n_train = int(total_n * cfg.train_ratio)
    n_val = int(total_n * cfg.val_ratio)
    n_test = total_n - n_train - n_val
    print(f"[Split] train={n_train}, val={n_val}, test={n_test}")

    df_shuffled = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df_train = df_shuffled.iloc[:n_train].reset_index(drop=True)
    df_val = df_shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df_shuffled.iloc[n_train + n_val :].reset_index(drop=True)

    train_ds = LesionMultiTaskDataset(
        df_train,
        cfg.img_dir,
        param_means,
        param_stds,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        augment=True,
    )
    val_ds = LesionMultiTaskDataset(
        df_val,
        cfg.img_dir,
        param_means,
        param_stds,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        augment=False,
    )
    test_ds = LesionMultiTaskDataset(
        df_test,
        cfg.img_dir,
        param_means,
        param_stds,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = MultiTaskLesionNet(
        in_channels=cfg.in_channels,
        num_params=len(PARAM_COLS),
    ).to(device)

    seg_criterion = nn.BCEWithLogitsLoss()
    param_criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.out_dir, "multitask_lesion_best.pth")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_seg_loss = 0.0
        running_param_loss = 0.0

        for imgs, masks, params_norm, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            params_norm = params_norm.to(device, non_blocking=True)

            optimizer.zero_grad()
            seg_logits, param_pred_norm = model(imgs)

            # Segmentation loss
            seg_loss = seg_criterion(seg_logits, masks)

            # Parameter regression loss (in normalized space)
            param_loss = param_criterion(param_pred_norm, params_norm)

            loss = seg_loss + cfg.lambda_params * param_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_seg_loss += seg_loss.item() * imgs.size(0)
            running_param_loss += param_loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        train_seg_loss = running_seg_loss / len(train_ds)
        train_param_loss = running_param_loss / len(train_ds)

        # ---------------- VAL ----------------
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_param_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks, params_norm, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                params_norm = params_norm.to(device, non_blocking=True)

                seg_logits, param_pred_norm = model(imgs)
                seg_loss = seg_criterion(seg_logits, masks)
                param_loss = param_criterion(param_pred_norm, params_norm)
                loss = seg_loss + cfg.lambda_params * param_loss

                val_loss += loss.item() * imgs.size(0)
                val_seg_loss += seg_loss.item() * imgs.size(0)
                val_param_loss += param_loss.item() * imgs.size(0)

                val_dice += dice_coefficient(seg_logits, masks) * imgs.size(0)

        val_loss /= len(val_ds)
        val_seg_loss /= len(val_ds)
        val_param_loss /= len(val_ds)
        val_dice /= len(val_ds)

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={train_loss:.4f} "
            f"(seg={train_seg_loss:.4f}, param={train_param_loss:.4f}) | "
            f"ValLoss={val_loss:.4f} "
            f"(seg={val_seg_loss:.4f}, param={val_param_loss:.4f}), "
            f"ValDice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "param_means": param_means,
                    "param_stds": param_stds,
                    "cfg": cfg.__dict__,
                },
                best_model_path,
            )
            print(f"  -> New best model saved to {best_model_path}")

    # ---------------- TEST ----------------
    print("\n[TEST] Loading best model and evaluating on test set...")
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()


    test_loss = 0.0
    test_seg_loss = 0.0
    test_param_loss = 0.0
    test_dice = 0.0
    with torch.no_grad():
        for imgs, masks, params_norm, _ in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            params_norm = params_norm.to(device, non_blocking=True)

            seg_logits, param_pred_norm = model(imgs)
            seg_loss = seg_criterion(seg_logits, masks)
            param_loss = param_criterion(param_pred_norm, params_norm)
            loss = seg_loss + cfg.lambda_params * param_loss

            test_loss += loss.item() * imgs.size(0)
            test_seg_loss += seg_loss.item() * imgs.size(0)
            test_param_loss += param_loss.item() * imgs.size(0)
            test_dice += dice_coefficient(seg_logits, masks) * imgs.size(0)

    test_loss /= len(test_ds)
    test_seg_loss /= len(test_ds)
    test_param_loss /= len(test_ds)
    test_dice /= len(test_ds)

    print(
        f"[TEST RESULTS] Loss={test_loss:.4f} "
        f"(seg={test_seg_loss:.4f}, param={test_param_loss:.4f}), "
        f"Dice={test_dice:.4f}"
    )


# ----------------------------- INFERENCE HELPERS -----------------------------


def load_trained_model(ckpt_path: str, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    
    in_channels = cfg_dict.get("in_channels", 3)
    num_params = len(PARAM_COLS)

    model = MultiTaskLesionNet(in_channels=in_channels, num_params=num_params)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    param_means = ckpt["param_means"]
    param_stds = ckpt["param_stds"]

    return model, param_means, param_stds, cfg_dict


def infer_on_image(
    model: nn.Module,
    param_means: np.ndarray,
    param_stds: np.ndarray,
    img_path: str,
    img_size: int = 256,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # [C, H, W]
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits, param_pred_norm = model(img_t)
        seg_probs = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        param_pred_norm = param_pred_norm[0].cpu().numpy()

    # denormalize params
    param_pred = param_pred_norm * param_stds + param_means

    params_dict = {name: float(val) for name, val in zip(PARAM_COLS, param_pred)}

    seg_mask_bin = (seg_probs > 0.5).astype(np.uint8)

    return seg_probs, seg_mask_bin, params_dict


# ----------------------------- MAIN ------------------------------------------


def main():
    cfg = Config()
    set_seed(cfg.seed)
    train_multitask(cfg)


if __name__ == "__main__":
    main()
