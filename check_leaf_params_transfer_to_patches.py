#!/usr/bin/env python3
import os, csv, random
import numpy as np
import cv2
from pathlib import Path

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

ROOT = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones"
OPT_CSV = f"{ROOT}/experiments/exp_v1/outputs/opt_summary_local.csv"
PATCH_META = f"{ROOT}/leaves_patches/patch_metadata.csv"   # adjust if different
N_LEAVES = 10
PATCHES_PER_LEAF = 40
SEED = 0

# --- helpers copied/minimized from optimizer ---
SMALL_MIN_AREA = 16

def _to_float01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def apply_mask(u8, mask_u8):
    return cv2.bitwise_and(u8, u8, mask=mask_u8) if mask_u8 is not None else u8

def binary_threshold_mask(gray8, threshold=127):
    _, m = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return m

def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[...,1], gray, lab[...,1], lab[...,2]]
    fused = None
    for ch in candidates:
        chm = apply_mask(ch, binary_mask) if binary_mask is not None else ch
        chf = chm.astype(np.float32)
        mx = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(fused)
    return apply_mask(enh, binary_mask) if binary_mask is not None else enh

def compute_grads(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return gx, gy, mag

def build_seed(gx, gy, sigma=1.5):
    sgx = cv2.GaussianBlur(gx, (0,0), sigmaX=sigma, sigmaY=sigma)
    sgy = cv2.GaussianBlur(gy, (0,0), sigmaX=sigma, sigmaY=sigma)
    sm  = np.sqrt(sgx*sgx + sgy*sgy)
    return sgx, sgy, sm

def elastic_energy(seed_mag, img_mag, img_gx, img_gy, seed_gx, seed_gy,
                  iters, diffusion_rate, mu, lambda_param, edge_thr):
    diff = seed_mag.astype(np.float64).copy()
    edge_mask = (img_mag > edge_thr).astype(np.float64)
    for _ in range(int(iters)):
        lap = laplace(diff)
        gy, gx = np.gradient(diff)
        div = gx + gy
        ftx = img_gx - seed_gx
        fty = img_gy - seed_gy
        ftm = np.sqrt(ftx*ftx + fty*fty)
        diff += diffusion_rate*(mu*lap + (lambda_param+mu)*div + edge_mask*(ftm - diff))
    diff = np.maximum(diff, 0.0)
    mx = float(diff.max())
    if mx < 1e-8:
        return np.zeros_like(diff, dtype=np.uint8)
    return np.uint8(255.0*diff/mx)

def snake_seg(img_u8, energy_u8, its, alpha, beta, gamma,
              min_blob_frac=1e-7, max_blob_frac=0.85, thr=50):
    h,w = img_u8.shape
    img_area = h*w
    l_size = max(SMALL_MIN_AREA, int(min_blob_frac*img_area))
    u_size = int(max_blob_frac*img_area)

    labeled = label(energy_u8 > thr)
    props = regionprops(labeled)
    out = np.zeros((h,w), dtype=bool)

    for r in props:
        if not (l_size < r.area < u_size):
            continue
        minr,minc,maxr,maxc = r.bbox
        if (maxr-minr) < 5 or (maxc-minc) < 5:
            continue
        crop = img_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop), 3)

        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr-minr)/2.0
        cc = (maxc-minc)/2.0
        init = np.vstack([rr*np.sin(s)+rr, cc*np.cos(s)+cc]).T

        try:
            snake = active_contour(crop_f, init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=its)
        except TypeError:
            snake = active_contour(crop_f, init, alpha=alpha, beta=beta, gamma=gamma, max_iterations=its)

        si = np.round(snake).astype(int)
        si[:,0] = np.clip(si[:,0], 0, crop.shape[0]-1)
        si[:,1] = np.clip(si[:,1], 0, crop.shape[1]-1)
        pr, pc = si[:,0], si[:,1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop.shape)
        out[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out

def dice(a, b, eps=1e-6):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum()
    return (2*inter + eps) / (a.sum() + b.sum() + eps)

def finalize_on_patch(patch_bgr, params, mode="fused"):
    # work channel
    gray8 = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    leaf_mask = binary_threshold_mask(gray8, 127)

    if mode == "bestch":
        g = patch_bgr[...,1]
        work = apply_mask(g, leaf_mask)
    else:
        work = fused_channel_u8(patch_bgr, leaf_mask)

    gx, gy, mag = compute_grads(work)
    sgx, sgy, smag = build_seed(gx, gy, 1.5)

    energy = elastic_energy(
        smag, mag, gx, gy, sgx, sgy,
        iters=params["iters_final"],
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda_param"],
        edge_thr=int(params["energy_threshold"]),
    )

    # IMPORTANT for patch test: avoid rejecting lesions that fill the patch
    seg = snake_seg(
        work, energy,
        its=params["snake_iters_final"],
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        thr=int(params["energy_threshold"]),
        max_blob_frac=0.98
    )
    return seg.astype(np.uint8)

def main():
    random.seed(SEED)

    # load per-leaf params
    leaf_params = {}
    with open(OPT_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status","").strip().lower() != "ok":
                continue
            fn = os.path.basename(row["filename"])
            stem = Path(fn).stem

            leaf_params[stem] = {
                "mu": float(row["mu"]),
                "lambda_param": float(row["lambda"]),          # renamed key
                "diffusion_rate": float(row["diffusion_rate"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "gamma": float(row["gamma"]),
                "energy_threshold": int(float(row["energy_threshold"])),
                "iters_final": 30,         # match your optimizer defaults
                "snake_iters_final": 100,  # match your optimizer defaults
                "mode": (str(row.get("mode","fused")).strip() or "fused")
            }

    # load patch metadata and group by leaf stem
    by_leaf = {}
    with open(PATCH_META, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pimg = row["patch_image"]
            pmask = row["patch_mask"]
            stem = Path(pimg).name.split("__y")[0]  # leaf stem from patch filename
            by_leaf.setdefault(stem, []).append((pimg, pmask))

    leaves = [k for k in by_leaf.keys() if k in leaf_params]
    leaves = random.sample(leaves, min(N_LEAVES, len(leaves)))
    print("Testing leaves:", len(leaves))

    all_d = []
    per_leaf = []

    for stem in leaves:
        items = by_leaf[stem]
        items = random.sample(items, min(PATCHES_PER_LEAF, len(items)))
        p = leaf_params[stem]
        mode = p["mode"] if p["mode"] in ("fused","bestch") else "fused"

        ds = []
        for pimg, pmask in items:
            img = cv2.imread(pimg, cv2.IMREAD_COLOR)
            mgt = cv2.imread(pmask, cv2.IMREAD_UNCHANGED)
            if img is None or mgt is None:
                continue
            if mgt.ndim == 3:
                mgt = mgt[...,0]
            mgt = (mgt > 0).astype(np.uint8)

            mpred = finalize_on_patch(img, p, mode=mode)
            ds.append(dice(mpred, mgt))

        if ds:
            per_leaf.append((stem, float(np.mean(ds)), float(np.median(ds)), len(ds)))
            all_d.extend(ds)

    print("\nPer-leaf patch-consistency (Dice):")
    for stem, m, med, n in sorted(per_leaf, key=lambda t:t[1])[:5]:
        print("  worst", stem, "mean", m, "median", med, "n", n)
    for stem, m, med, n in sorted(per_leaf, key=lambda t:t[1], reverse=True)[:5]:
        print("  best ", stem, "mean", m, "median", med, "n", n)

    if all_d:
        print("\nOverall patch-consistency Dice:")
        print("  mean:", float(np.mean(all_d)))
        print("  median:", float(np.median(all_d)))
        print("  p10/p90:", float(np.quantile(all_d,0.1)), float(np.quantile(all_d,0.9)))
    else:
        print("No patch pairs evaluated. Check paths.")

if __name__ == "__main__":
    main()