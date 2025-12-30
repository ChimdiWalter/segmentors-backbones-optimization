#!/usr/bin/env python3
# multitask_infer_new.py
#
# Inference for MultiTaskLesionNet:
#   - Two modes: "highres" (resize whole image) and "patch" (sliding window)
#   - Loads model checkpoint with weights_only=False (PyTorch 2.6 safe)
#   - Produces 16-bit overlay + mask for each image
#   - Writes a CSV with predicted Navier params for each image

import os
import argparse
import csv

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd

from multitask_lesion_model import MultiTaskLesionNet  # must match your training file


ALLOWED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# -------------------- Param stats from training CSV -------------------- #
def compute_param_stats_from_csv(csv_path: str):
    """Recompute mean/std of physics params from the training CSV."""
    print(f"[compute_param_stats_from_csv] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Use same filter as training: status=ok & score>0
    df = df[(df["status"] == "ok") & (df["score"] > 0.0)].reset_index(drop=True)

    cols = ["mu", "lambda", "diffusion_rate",
            "alpha", "beta", "gamma", "energy_threshold"]
    params = df[cols].to_numpy(dtype=np.float32)

    mean = params.mean(axis=0)
    std = params.std(axis=0) + 1e-8

    print("[compute_param_stats_from_csv] mean:", mean)
    print("[compute_param_stats_from_csv] std :", std)
    return mean, std


# -------------------- Model loading -------------------- #
def load_trained_model(ckpt_path: str,
                       device: torch.device,
                       train_csv: str):
    """Load MultiTaskLesionNet and param stats from checkpoint or CSV."""
    print(f"[load_trained_model] Loading checkpoint: {ckpt_path}")
    # PyTorch 2.6 default is weights_only=True; we override to allow normal pickle
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # For older PyTorch that doesn't know weights_only
        ckpt = torch.load(ckpt_path, map_location=device)

    # Two possibilities:
    # 1) checkpoint is {"model_state": ..., "param_mean": ..., "param_std": ...}
    # 2) checkpoint is just model.state_dict()
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        print("[load_trained_model] Found 'model_state' in checkpoint dict.")
        state_dict = ckpt["model_state"]
        param_mean = ckpt.get("param_mean", None)
        param_std = ckpt.get("param_std", None)
    else:
        print("[load_trained_model] Checkpoint looks like raw state_dict.")
        state_dict = ckpt
        param_mean, param_std = None, None

    # If param stats not in ckpt, recompute from CSV
    if param_mean is None or param_std is None:
        print("[load_trained_model] 'param_mean/std' NOT in checkpoint, recomputing from CSV...")
        param_mean, param_std = compute_param_stats_from_csv(train_csv)

    param_mean = np.asarray(param_mean, dtype=np.float32)
    param_std = np.asarray(param_std, dtype=np.float32)

    # Build model *exactly* as in training
    model = MultiTaskLesionNet(in_channels=3, num_params=7)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, param_mean, param_std


# -------------------- Image utilities -------------------- #
def to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """Convert any uint16/float/gray to uint8 BGR for visualization."""
    if img is None:
        return None

    if img.dtype == np.uint8:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    if img.dtype == np.uint16:
        img8 = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    else:
        img_f = img.astype(np.float32)
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx <= mn + 1e-8:
            img8 = np.zeros_like(img_f, dtype=np.uint8)
        else:
            img8 = ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)

    if img8.ndim == 2:
        img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8


def make_overlay16_from_mask(orig_img: np.ndarray,
                             mask_bool: np.ndarray,
                             color_bgr=(0, 0, 255),
                             alpha: float = 0.6):
    """
    Overlay a binary mask on top of the original image.
    - orig_img: any depth, any channels (we convert to 8-bit BGR).
    - mask_bool: H x W boolean.
    Returns (overlay16, mask16) both uint16.
    """
    base8 = to_uint8_bgr(orig_img)
    h, w = mask_bool.shape
    if base8.shape[:2] != (h, w):
        base8 = cv2.resize(base8, (w, h), interpolation=cv2.INTER_AREA)

    color_mask = np.zeros_like(base8, dtype=np.uint8)
    color_mask[mask_bool] = color_bgr  # red lesions, rest untouched

    overlay8 = cv2.addWeighted(base8, 1.0, color_mask, alpha, 0.0)

    # Convert to 16-bit for TIF (no weird green cast; just scaled up)
    overlay16 = overlay8.astype(np.uint16) * 257
    mask16 = mask_bool.astype(np.uint16) * 65535
    return overlay16, mask16


def preprocess_image_for_model(bgr_img: np.ndarray,
                               target_size: int = 256) -> np.ndarray:
    """
    Convert BGR -> RGB, resize, scale to [0,1], return CHW float32.
    """
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_f, (2, 0, 1))  # HWC -> CHW
    return img_chw


# -------------------- Inference: high-res mode -------------------- #
def infer_image_highres(model,
                        device,
                        img_path: str,
                        param_mean: np.ndarray,
                        param_std: np.ndarray,
                        resize_long: int = 1024,
                        prob_thresh: float = 0.5,
                        target_size: int = 256):
    """
    Run the model on a resized version of the full image, then upsample
    the segmentation back to the original resolution.
    """
    orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if orig is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    H0, W0 = orig.shape[:2]

    # Rescale long side if needed (for speed)
    bgr = to_uint8_bgr(orig)
    if resize_long is not None:
        scale = resize_long / max(H0, W0)
        if scale < 1.0:
            H1 = int(round(H0 * scale))
            W1 = int(round(W0 * scale))
            bgr = cv2.resize(bgr, (W1, H1), interpolation=cv2.INTER_AREA)

    # Prepare for the network
    img_chw = preprocess_image_for_model(bgr, target_size=target_size)
    img_t = torch.from_numpy(img_chw).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        seg_logits, param_norm = model(img_t)  # <- multi-task output
        seg_prob = torch.sigmoid(seg_logits)   # [1,1,h,w]

    # Upsample back to original resolution
    seg_prob_up = F.interpolate(
        seg_prob,
        size=(H0, W0),
        mode="bilinear",
        align_corners=False,
    )
    seg_prob_np = seg_prob_up[0, 0].cpu().numpy()
    seg_mask = seg_prob_np >= prob_thresh

    # Denormalize physics params
    param_norm_np = param_norm[0].cpu().numpy()
    params = param_norm_np * param_std + param_mean

    overlay16, mask16 = make_overlay16_from_mask(orig, seg_mask)
    return overlay16, mask16, params


# -------------------- Inference: patch mode -------------------- #
def infer_image_patches(model,
                        device,
                        img_path: str,
                        param_mean: np.ndarray,
                        param_std: np.ndarray,
                        patch_size: int = 512,
                        patch_stride: int = 384,
                        prob_thresh: float = 0.5,
                        target_size: int = 256):
    """
    Sliding-window patch inference for very large images.

    Each patch:
      - extracted in RGB
      - padded by reflection if near border
      - resized to target_size x target_size
      - fed through model -> seg_prob_patch
    Then we stitch all patch probabilities with simple averaging on overlaps.
    Params are averaged over all patches.
    """
    orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if orig is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    base8 = to_uint8_bgr(orig)
    H0, W0 = base8.shape[:2]

    # Work in RGB float space for patch extraction
    base_rgb = cv2.cvtColor(base8, cv2.COLOR_BGR2RGB)
    base_f = base_rgb.astype(np.float32) / 255.0

    prob_accum = np.zeros((H0, W0), dtype=np.float32)
    weight_accum = np.zeros((H0, W0), dtype=np.float32)
    params_list = []

    ys = list(range(0, H0, patch_stride))
    xs = list(range(0, W0, patch_stride))

    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, H0)
            x1 = min(x + patch_size, W0)

            patch = base_f[y:y1, x:x1, :]  # RGB float
            ph, pw = patch.shape[:2]

            # Pad to patch_size by reflection (avoids border artifacts)
            pad_h = patch_size - ph
            pad_w = patch_size - pw
            if pad_h > 0 or pad_w > 0:
                patch = np.pad(
                    patch,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="reflect",
                )

            # Convert patch RGB -> BGR uint8, then standard preprocessing
            patch_rgb_uint8 = (patch * 255.0).astype(np.uint8)
            patch_bgr = cv2.cvtColor(patch_rgb_uint8, cv2.COLOR_RGB2BGR)
            patch_chw = preprocess_image_for_model(patch_bgr,
                                                   target_size=target_size)
            patch_t = torch.from_numpy(patch_chw).unsqueeze(0).to(device)

            with torch.no_grad():
                seg_logits, param_norm = model(patch_t)
                seg_prob_patch = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
                params_list.append(param_norm[0].cpu().numpy())

            # Resize patch prob back to original patch spatial size
            seg_prob_patch = cv2.resize(
                seg_prob_patch,
                (pw, ph),
                interpolation=cv2.INTER_LINEAR,
            )

            prob_accum[y:y1, x:x1] += seg_prob_patch
            weight_accum[y:y1, x:x1] += 1.0

    # Average where overlapped
    prob_final = prob_accum / np.maximum(weight_accum, 1e-6)
    seg_mask = prob_final >= prob_thresh

    # Average predicted params across patches
    params_mean_norm = np.mean(params_list, axis=0)
    params = params_mean_norm * param_std + param_mean

    overlay16, mask16 = make_overlay16_from_mask(orig, seg_mask)
    return overlay16, mask16, params


# -------------------- Main CLI -------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Multi-task lesion inference (high-res / patch)."
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to multitask_lesion_best.pth (checkpoint from training).",
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory with unseen leaf images.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Where to write overlays/masks/CSV.",
    )
    ap.add_argument(
        "--train-csv",
        required=True,
        help="Training CSV (opt_summary.csv) to recompute param mean/std.",
    )
    ap.add_argument(
        "--mode",
        choices=["highres", "patch"],
        default="highres",
        help="Inference mode: 'highres' or 'patch'.",
    )
    ap.add_argument(
        "--resize-long",
        type=int,
        default=1024,
        help="Max long side for highres mode (default 1024).",
    )
    ap.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size for patch mode (default 512).",
    )
    ap.add_argument(
        "--patch-stride",
        type=int,
        default=384,
        help="Stride for patch mode (default 384).",
    )
    ap.add_argument(
        "--prob-thresh",
        type=float,
        default=0.5,
        help="Sigmoid threshold for mask binarization (default 0.5).",
    )
    ap.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Network input size (must match training, default 256).",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model + param stats
    model, param_mean, param_std = load_trained_model(
        args.ckpt,
        device,
        args.train_csv,
    )

    # Collect images
    img_files = [
        f for f in sorted(os.listdir(args.input_dir))
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT
    ]
    print(f"[infer] Found {len(img_files)} images in {args.input_dir}")

    # Prepare summary CSV
    csv_out_path = os.path.join(args.output_dir, "multitask_unseen_summary.csv")
    header = [
        "filename",
        "mode",
        "overlay16_path",
        "mask16_path",
        "mu",
        "lambda",
        "diffusion_rate",
        "alpha",
        "beta",
        "gamma",
        "energy_threshold",
    ]
    with open(csv_out_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)

        # Process each image
        for i, fname in enumerate(img_files, 1):
            in_path = os.path.join(args.input_dir, fname)
            print(f"[{i}/{len(img_files)}] {fname}")
            try:
                if args.mode == "highres":
                    overlay16, mask16, params = infer_image_highres(
                        model,
                        device,
                        in_path,
                        param_mean,
                        param_std,
                        resize_long=args.resize_long,
                        prob_thresh=args.prob_thresh,
                        target_size=args.target_size,
                    )
                else:
                    overlay16, mask16, params = infer_image_patches(
                        model,
                        device,
                        in_path,
                        param_mean,
                        param_std,
                        patch_size=args.patch_size,
                        patch_stride=args.patch_stride,
                        prob_thresh=args.prob_thresh,
                        target_size=args.target_size,
                    )
            except Exception as e:
                print(f"  -> ERROR on {fname}: {e}")
                continue

            base = os.path.splitext(fname)[0]
            overlay_path = os.path.join(
                args.output_dir,
                f"{base}__multitask_overlay16.tif",
            )
            mask_path = os.path.join(
                args.output_dir,
                f"{base}__multitask_mask16.tif",
            )

            cv2.imwrite(overlay_path, overlay16)
            cv2.imwrite(mask_path, mask16)

            mu, lam, dr, alpha, beta, gamma, eth = params.tolist()
            writer.writerow([
                fname,
                args.mode,
                overlay_path,
                mask_path,
                mu,
                lam,
                dr,
                alpha,
                beta,
                gamma,
                eth,
            ])

    print(f"[infer] Done. Summary CSV -> {csv_out_path}")


if __name__ == "__main__":
    main()
