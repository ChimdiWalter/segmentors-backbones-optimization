import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import tifffile as tiff
except ImportError:
    tiff = None


# =========================
# I/O helpers
# =========================
def read_image_any(path: str):
    """Read image, return RGB for color images."""
    ext = Path(path).suffix.lower()

    if ext in [".tif", ".tiff"] and tiff is not None:
        arr = tiff.imread(path)
        # tifffile often returns RGB already; leave as is
        return arr

    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None

    # Convert BGR -> RGB for color images
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def write_image_any(path: str, img: np.ndarray):
    """Write image, assuming RGB for color arrays."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = Path(path).suffix.lower()

    if ext in [".tif", ".tiff"]:
        if tiff is None:
            raise RuntimeError("tifffile is required to save TIFF. Install: pip install tifffile")
        tiff.imwrite(path, img)
        return

    out = img
    # Convert RGB -> BGR for cv2.imwrite on color images
    if img.ndim == 3 and img.shape[2] >= 3:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if ext == ".png":
        ok = cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif ext in [".jpg", ".jpeg"]:
        ok = cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        ok = cv2.imwrite(path, out)

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


# =========================
# Patching helpers
# =========================
def get_starts(length: int, patch: int, stride: int):
    """Generate start indices that cover the full dimension."""
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be > 0")

    starts = list(range(0, max(1, length), stride))
    if not starts:
        starts = [0]

    # Ensure tail is covered
    if starts[-1] + patch < length:
        starts.append(max(0, length - patch))

    return sorted(set(starts))


def pad_patch(arr: np.ndarray, patch_h: int, patch_w: int, is_mask: bool):
    """Pad patch to exact patch size without resizing."""
    h, w = arr.shape[:2]
    pad_bottom = max(0, patch_h - h)
    pad_right = max(0, patch_w - w)

    if pad_bottom == 0 and pad_right == 0:
        return arr, 0, 0

    if arr.ndim == 2:
        pad_val = 0
    else:
        pad_val = [0] * arr.shape[2]

    if is_mask:
        padded = cv2.copyMakeBorder(
            arr, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_val
        )
    else:
        padded = cv2.copyMakeBorder(
            arr, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

    return padded, pad_bottom, pad_right


# =========================
# Mask matching
# =========================
def build_mask_map_from_csv(csv_path: str, mask_dir_override: str = None):
    """
    Build mapping: image filename -> mask path using optimizer CSV.
    Expects columns: filename, mask16_path (and optionally status).
    Handles path repair if CSV paths are old cluster paths.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = ["filename", "mask16_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    csv_dir = os.path.dirname(csv_path)
    mask_map = {}

    for _, row in df.iterrows():
        img_name = str(row["filename"]).strip()
        raw_mask = str(row["mask16_path"]).strip()

        if not img_name or img_name.lower() == "nan":
            continue

        candidate_paths = []

        # Raw path from CSV
        if raw_mask and raw_mask.lower() != "nan":
            candidate_paths.append(raw_mask)
            # Local repair using CSV directory + basename
            candidate_paths.append(os.path.join(csv_dir, os.path.basename(raw_mask)))

            # If user gave a mask_dir override, try that too
            if mask_dir_override:
                candidate_paths.append(os.path.join(mask_dir_override, os.path.basename(raw_mask)))

        # Pick first existing
        found = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                found = p
                break

        if found:
            mask_map[img_name] = found

    return mask_map


def find_mask_by_stem(image_filename: str, mask_dir: str, mask_suffix: str = ""):
    """
    Match by stem:
      leaf001.jpg -> leaf001.tif / leaf001.png
      or leaf001_mask.tif if mask_suffix='_mask'
    """
    stem = Path(image_filename).stem
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        cand = os.path.join(mask_dir, f"{stem}{mask_suffix}{ext}")
        if os.path.exists(cand):
            return cand
    return None


# =========================
# Core patchify
# =========================
def patchify_one_image(
    img_path: str,
    patch_h: int,
    patch_w: int,
    stride_h: int,
    stride_w: int,
    image_out_dir: str,
    out_ext: str = ".png",
    mask_path: str = None,
    mask_out_dir: str = None,
):
    img = read_image_any(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    y_starts = get_starts(h, patch_h, stride_h)
    x_starts = get_starts(w, patch_w, stride_w)

    mask = None
    if mask_path:
        mask = read_image_any(mask_path)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        if mask.shape[:2] != (h, w):
            raise ValueError(
                f"Mask size mismatch for {Path(img_path).name}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}"
            )

    stem = Path(img_path).stem
    rows = []

    for y in y_starts:
        for x in x_starts:
            # Image patch
            patch = img[y:y + patch_h, x:x + patch_w]
            patch, pad_b, pad_r = pad_patch(patch, patch_h, patch_w, is_mask=False)

            patch_name = f"{stem}__y{y:05d}_x{x:05d}{out_ext}"
            patch_image_path = os.path.join(image_out_dir, patch_name)
            write_image_any(patch_image_path, patch)

            patch_mask_path = ""
            if mask is not None and mask_out_dir is not None:
                mpatch = mask[y:y + patch_h, x:x + patch_w]
                mpatch, _, _ = pad_patch(mpatch, patch_h, patch_w, is_mask=True)

                # Keep masks lossless
                mask_ext = out_ext if out_ext in [".png", ".tif", ".tiff"] else ".png"
                patch_mask_name = f"{stem}__y{y:05d}_x{x:05d}{mask_ext}"
                patch_mask_path = os.path.join(mask_out_dir, patch_mask_name)
                write_image_any(patch_mask_path, mpatch)

            rows.append({
                "source_image": img_path,
                "patch_image": patch_image_path,
                "source_mask": mask_path if mask_path else "",
                "patch_mask": patch_mask_path,
                "orig_h": h,
                "orig_w": w,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "x": x,
                "y": y,
                "x_end": min(x + patch_w, w),
                "y_end": min(y + patch_h, h),
                "pad_right": pad_r,
                "pad_bottom": pad_b,
                "stride_h": stride_h,
                "stride_w": stride_w,
            })

    return rows


def patchify_folder(
    input_dir: str,
    output_dir: str,
    patch_size: int,
    overlap: int,
    out_ext: str,
    mask_dir: str = None,
    mask_output_dir: str = None,
    mask_suffix: str = "",
    csv_path: str = None,
    metadata_name: str = "patch_metadata.csv",
):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input image folder not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    if mask_output_dir:
        os.makedirs(mask_output_dir, exist_ok=True)

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    # Optional exact mapping via CSV
    mask_map = {}
    if csv_path:
        print(f"[Info] Building image->mask mapping from CSV: {csv_path}")
        mask_map = build_mask_map_from_csv(csv_path, mask_dir_override=mask_dir)
        print(f"[Info] CSV mask mappings found: {len(mask_map)}")

    img_files = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            img_files.append(os.path.join(input_dir, f))

    if not img_files:
        print(f"No images found in: {input_dir}")
        return

    all_rows = []
    num_with_masks = 0
    num_without_masks = 0
    num_skipped = 0

    for i, img_path in enumerate(img_files, 1):
        try:
            img_name = os.path.basename(img_path)

            # Resolve mask path
            mask_path = None
            if csv_path:
                mask_path = mask_map.get(img_name, None)
            elif mask_dir:
                mask_path = find_mask_by_stem(img_name, mask_dir, mask_suffix=mask_suffix)

            if mask_path:
                num_with_masks += 1
            else:
                num_without_masks += 1

            rows = patchify_one_image(
                img_path=img_path,
                patch_h=patch_size,
                patch_w=patch_size,
                stride_h=stride,
                stride_w=stride,
                image_out_dir=output_dir,
                out_ext=out_ext,
                mask_path=mask_path,
                mask_out_dir=mask_output_dir,
            )
            all_rows.extend(rows)
            print(f"[{i}/{len(img_files)}] {img_name} -> {len(rows)} patches" + (" (mask)" if mask_path else " (no mask)"))

        except Exception as e:
            num_skipped += 1
            print(f"[WARN] Skipping {img_path}: {e}")

    # Save metadata
    meta_path = os.path.join(output_dir, metadata_name)
    if all_rows:
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        # create empty file with basic columns if nothing processed
        with open(meta_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_image", "patch_image", "source_mask", "patch_mask"])

    print("\n===== Patchify complete =====")
    print(f"Images processed: {len(img_files) - num_skipped}")
    print(f"Images skipped:   {num_skipped}")
    print(f"With masks:       {num_with_masks}")
    print(f"Without masks:    {num_without_masks}")
    print(f"Image patches:    {output_dir}")
    if mask_output_dir:
        print(f"Mask patches:     {mask_output_dir}")
    print(f"Metadata CSV:     {meta_path}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Patchify full-resolution images (and optional masks) without resizing."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with original full-size images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save image patches")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size (square). Default: 512")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between patches. Default: 64")
    parser.add_argument("--out_ext", type=str, default=".png",
                        choices=[".png", ".tif", ".tiff", ".jpg", ".jpeg"],
                        help="Patch file format (PNG/TIFF recommended)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Folder with masks (used for stem-based matching if --csv_path not provided)")
    parser.add_argument("--mask_output_dir", type=str, default=None,
                        help="Folder to save mask patches")
    parser.add_argument("--mask_suffix", type=str, default="",
                        help="Optional suffix for mask files, e.g. '_mask'")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional optimizer CSV (recommended). Uses filename->mask16_path mapping.")
    parser.add_argument("--metadata_name", type=str, default="patch_metadata.csv",
                        help="Output metadata CSV filename (saved inside output_dir)")

    args = parser.parse_args()

    patchify_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        out_ext=args.out_ext,
        mask_dir=args.mask_dir,
        mask_output_dir=args.mask_output_dir,
        mask_suffix=args.mask_suffix,
        csv_path=args.csv_path,
        metadata_name=args.metadata_name,
    )


if __name__ == "__main__":
    main()