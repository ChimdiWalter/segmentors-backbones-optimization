#!/usr/bin/env python3
"""
build_nav_dataset.py

Precompute a compact PyTorch dataset from the outputs of nv_optimize_robust_v2.py.

Inputs:
  - opt_summary.csv (or any CSV with the same schema)
  - mask16_path files written by nv_optimize_robust_v2.py

Outputs:
  - a single .pt file containing:
      * masks:          [N, 1, H, W] float32 in {0,1}
      * tabular:        [N, F] float32 (numeric features from CSV)
      * timeout_labels: [N] int64 (1 if status == "timeout", else 0)
      * meta:           list of dicts (filename, status, etc.)

Usage example:
  python build_nav_dataset.py \
      --csv /cluster/.../navier_output_refined/opt_summary_refined.csv \
      --out /cluster/.../navier_output_refined/navier_dataset.pt \
      --mask-size 256
"""

import os
import csv
import json
import argparse

import cv2
import numpy as np
import torch


TABULAR_KEYS_DEFAULT = [
    "score",
    "n_contours",
    "area_px",
    "area_frac",
    "small_count",
    "mu",
    "lambda",
    "diffusion_rate",
    "alpha",
    "beta",
    "gamma",
    "energy_threshold",
]


def parse_float(row, key, default=0.0):
    """Safe float parse from CSV row."""
    v = row.get(key, "")
    if v == "" or v is None:
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def load_mask_as_binary(mask_path, mask_size=None):
    """
    Load 16-bit mask and return binary float32 image [1, H, W] in {0,1}.

    If mask_size is not None, resize to (mask_size, mask_size) with nearest neighbor.
    """
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    # anything >0 is lesion
    mask_bin = (mask_raw > 0).astype(np.float32)

    if mask_size is not None:
        mask_bin = cv2.resize(
            mask_bin,
            (mask_size, mask_size),
            interpolation=cv2.INTER_NEAREST
        )

    # add channel dimension: [1, H, W]
    if mask_bin.ndim == 2:
        mask_bin = mask_bin[None, ...]
    elif mask_bin.ndim == 3:
        # in case mask is already multi-channel; take first channel
        mask_bin = mask_bin[..., 0][None, ...]

    return mask_bin


def build_dataset_from_csv(
    csv_path: str,
    mask_size: int = 256,
    tabular_keys=None,
    include_timeouts: bool = True,
):
    """
    Core routine that reads opt_summary.csv and builds tensors.

    Returns:
      masks:          torch.FloatTensor [N, 1, H, W]
      tabular:        torch.FloatTensor [N, F]
      timeout_labels: torch.LongTensor  [N]
      meta:           list[dict]
      tabular_keys:   list[str]
    """
    if tabular_keys is None:
        tabular_keys = TABULAR_KEYS_DEFAULT

    masks = []
    tabular_rows = []
    timeout_labels = []
    meta = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "")
            if status not in ("ok", "timeout"):
                # skip load_error, worker_error, etc.
                continue

            if (not include_timeouts) and status == "timeout":
                continue

            mask_path = row.get("mask16_path", "")
            if not mask_path:
                # nothing to load
                continue
            if not os.path.exists(mask_path):
                print(f"[WARN] mask path does not exist, skipping: {mask_path}")
                continue

            try:
                mask_bin = load_mask_as_binary(mask_path, mask_size=mask_size)
            except Exception as e:
                print(f"[WARN] failed to load mask {mask_path}: {e}")
                continue

            # build tabular feature vector
            feats = []
            for k in tabular_keys:
                feats.append(parse_float(row, k, default=0.0))
            feats = np.array(feats, dtype=np.float32)

            # timeout label: 1 if status == "timeout", else 0
            timeout_label = 1 if status == "timeout" else 0

            # meta info for debugging / analysis
            meta_row = {
                "filename": row.get("filename", ""),
                "status": status,
                "mode": row.get("mode", ""),
                "mask16_path": mask_path,
                "overlay16_path": row.get("overlay16_path", ""),
            }

            masks.append(mask_bin)
            tabular_rows.append(feats)
            timeout_labels.append(timeout_label)
            meta.append(meta_row)

    if not masks:
        raise RuntimeError("No usable rows found in CSV (masks list is empty).")

    masks_np = np.stack(masks, axis=0)           # [N, 1, H, W]
    tabular_np = np.stack(tabular_rows, axis=0)  # [N, F]
    timeout_np = np.array(timeout_labels, dtype=np.int64)

    masks_t = torch.from_numpy(masks_np).float()
    tabular_t = torch.from_numpy(tabular_np).float()
    timeout_t = torch.from_numpy(timeout_np).long()

    return masks_t, tabular_t, timeout_t, meta, tabular_keys


def main():
    ap = argparse.ArgumentParser(description="Build Navier+Snake lesion dataset from opt_summary.csv")
    ap.add_argument("--csv", required=True, help="Path to opt_summary.csv from nv_optimize_robust_v2.py")
    ap.add_argument("--out", required=True, help="Output .pt filename (e.g., navier_dataset.pt)")
    ap.add_argument("--mask-size", type=int, default=256, help="Resize masks to (mask_size, mask_size)")
    ap.add_argument("--no-timeouts", action="store_true", help="Drop timeout samples (status == 'timeout')")
    args = ap.parse_args()

    include_timeouts = not args.no_timeouts

    print("=== build_nav_dataset ===")
    print(f"CSV:          {args.csv}")
    print(f"Out:          {args.out}")
    print(f"Mask size:    {args.mask_size} x {args.mask_size}")
    print(f"Timeouts?     {'included' if include_timeouts else 'excluded'}")
    print("====================================", flush=True)

    masks_t, tabular_t, timeout_t, meta, tabular_keys = build_dataset_from_csv(
        csv_path=args.csv,
        mask_size=args.mask_size,
        include_timeouts=include_timeouts,
    )

    state = {
        "masks": masks_t,
        "tabular": tabular_t,
        "timeout_labels": timeout_t,
        "meta": meta,
        "tabular_keys": tabular_keys,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state, args.out)

    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "csv": args.csv,
                "out": args.out,
                "mask_size": args.mask_size,
                "include_timeouts": include_timeouts,
                "n_samples": masks_t.shape[0],
                "tabular_keys": tabular_keys,
            },
            f,
            indent=2,
        )

    print(f"[DONE] Saved dataset -> {args.out}")
    print(f"[INFO] Metadata JSON  -> {meta_path}")


if __name__ == "__main__":
    main()
