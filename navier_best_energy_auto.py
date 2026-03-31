import os
import csv
import argparse
import numpy as np
import cv2
import sys
from datetime import datetime

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon

# ------------ CONFIG (edit these) ------------
INPUT_DIR  = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves'
OUTPUT_DIR = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/navier_output_bestch_auto'
LOG_NAME   = 'results_log.csv'
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
# ---------------------------------------------

PINK_16 = (65535, 5140, 37779)  # (B, G, R) uint16

# ---- curated list of 6 "good" combinations (mu, lam, dr, alpha, beta, gamma) ----
COMBOS = [
    (0.05, 0.05, 0.1, 0.01, 0.1, 0.01),
    (0.05, 0.05, 0.4, 0.01, 0.1, 0.1),

    (0.05, 0.50, 0.1, 0.10, 0.5, 0.01),
    (0.50, 0.05, 0.4, 0.10, 0.1, 0.1),

    (0.50, 0.50, 0.1, 0.50, 0.5, 0.01),
    (0.50, 0.50, 0.4, 0.50, 0.5, 0.1),
]

def fmt(v):
    return str(v).replace('.', 'p')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------- OpenCL / UMat accel helpers ----------------
cv2.ocl.setUseOpenCL(True)
USE_UMAT = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
cv2.setUseOptimized(True)

def to_umat(u8): return cv2.UMat(u8) if USE_UMAT else u8
def from_umat(u): return u.get() if isinstance(u, cv2.UMat) else u

def gaussian_blur_u(u8, k=(3,3), s=0):
    return from_umat(cv2.GaussianBlur(to_umat(u8), k, s))

def sobel_xy_u(u8):
    u = to_umat(u8)
    gx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    return from_umat(gx), from_umat(gy)

def laplacian_u(f64):
    # cv2 Laplacian on GPU (float32), convert back to float64
    if USE_UMAT:
        u32 = cv2.UMat(f64.astype(np.float32))
        lap = cv2.Laplacian(u32, cv2.CV_32F, ksize=3)
        return lap.get().astype(np.float64)
    return cv2.Laplacian(f64, cv2.CV_64F, ksize=3)

def bitwise_and_u(u8, m):
    return from_umat(cv2.bitwise_and(to_umat(u8), to_umat(u8), mask=to_umat(m)))

def clahe_u(u8, clip=2.0, tile=(8,8), mask=None):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    out = clahe.apply(to_umat(u8))
    out = from_umat(out)
    return cv2.bitwise_and(out, out, mask=mask) if mask is not None else out

def morph_close_u(u8, ksize=3, iters=1):
    k = np.ones((ksize,ksize), np.uint8)
    return from_umat(cv2.morphologyEx(to_umat(u8), cv2.MORPH_CLOSE, k, iterations=iters))

# ---- dtype-safe helpers ----
def read_color_anydepth(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None, None
    if raw.ndim == 2:
        raw_gray = raw
        bgr8 = _to_uint8(raw_gray)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw_gray, bgr8, gray8
    else:
        bgr8 = _to_uint8(raw)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw, bgr8, gray8

def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0)
        return (im * 255.0).astype(np.uint8)
    imin, imax = float(np.min(img)), float(np.max(img))
    if imax <= imin + 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)

def _to_float01(img):
    if img.dtype == np.uint8:
        return (img.astype(np.float32)) / 255.0
    if img.dtype == np.uint16:
        return (img.astype(np.float32)) / 65535.0
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

# ---- auto-threshold helpers ----
def masked_percentile(arr_u8, mask_u8, p):
    if mask_u8 is not None and mask_u8.any():
        vals = arr_u8[mask_u8 > 0]
        if vals.size >= 32:
            return int(np.percentile(vals, p))
    return int(np.percentile(arr_u8, p))

def _otsu_on_values(vals_u8):
    if vals_u8.size < 2:
        return int(np.median(vals_u8)) if vals_u8.size else 0
    arr = vals_u8.reshape(-1, 1)
    retval, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(retval)

def auto_eth_dual(grad_mag_u8, leaf_mask_u8):
    low_p, high_p = 60, 80
    Eth_q    = masked_percentile(grad_mag_u8, leaf_mask_u8, low_p)
    Eth_seed = masked_percentile(grad_mag_u8, leaf_mask_u8, high_p)
    Eth_seed = max(Eth_seed, Eth_q + 3)

    m = (leaf_mask_u8 > 0) if leaf_mask_u8 is not None else slice(None)
    vals = grad_mag_u8[m]
    rng = int(vals.max()) - int(vals.min()) if vals.size else 0
    if rng < 20 or Eth_seed <= Eth_q + 2:
        otsu = _otsu_on_values(vals.astype(np.uint8) if vals.size else grad_mag_u8.astype(np.uint8))
        Eth_q    = int(0.7 * otsu)
        Eth_seed = max(int(0.9 * otsu), Eth_q + 3)

    return int(np.clip(Eth_q, 0, 255)), int(np.clip(Eth_seed, 0, 255))

# ---- core ops ----
def binary_threshold_mask(gray8, threshold=127):
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8, mask_u8):
    return bitwise_and_u(image_u8, mask_u8)

def gradient_magnitude(u8):
    u8 = gaussian_blur_u(u8, (3,3), 0)
    gx, gy = sobel_xy_u(u8)
    mag = np.sqrt(gx**2 + gy**2)
    eps = 1e-8
    mag_u8 = np.uint8(255 * mag / max(float(mag.max()), eps))
    return mag_u8, gx, gy

def elastic_deformation_diffusion(image_u8, grad_x, grad_y,
                                  iterations=30, diffusion_rate=0.2,
                                  mu=0.5, lambda_param=0.5,
                                  edge_thresh=50):
    diffused = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)
    im_gx, im_gy = sobel_xy_u(image_u8)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)
    q = (im_mag > edge_thresh).astype(np.float64)

    for _ in range(iterations):
        lap = laplacian_u(diffused)
        div_v = np.gradient(diffused)[0]  # proxy
        ftx = im_gx - grad_x
        fty = im_gy - grad_y
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += diffusion_rate * (mu * lap + (lambda_param + mu) * div_v + q * (ftm - diffused))

    eps = 1e-8
    return np.uint8(255 * diffused / max(float(diffused.max()), eps))

def snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, l_size, u_size, energy_threshold):
    h, w = image_u8.shape
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    nprops = len(props)
    print(f"   [REGIONS] {nprops} at Eth_seed={energy_threshold}")
    if nprops == 0:
        return np.zeros((h, w), dtype=bool)

    out_mask = np.zeros((h, w), dtype=bool)

    for rgn in props:
        if not (l_size < rgn.area < u_size):
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)

        s = np.linspace(0, 2*np.pi, 200, endpoint=False)  # 200-point init
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_num_iter=50
            )
        except TypeError:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_iterations=50
            )

        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)
        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out_mask

# ---- best-channel selector ----
def best_single_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)

    candidates = {
        "B": bgr8[...,0],
        "G": bgr8[...,1],
        "R": bgr8[...,2],
        "GRAY": gray,
        "V": hsv[...,2],
        "a*": lab[...,1],
        "b*": lab[...,2],
    }

    def edge_score(ch):
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if binary_mask is not None:
            return float((mag * (binary_mask > 0)).sum())
        return float(mag.sum())

    best_name, best_ch, best_score = None, None, -1.0
    for name, ch in candidates.items():
        s = edge_score(ch)
        if s > best_score:
            best_name, best_ch, best_score = name, ch, s
    print(f"[CHANNEL] selected {best_name} (edge-score={best_score:.1f})")
    return best_ch

def _prepare_overlay16(raw):
    if raw.dtype == np.uint16:
        if raw.ndim == 2:
            return cv2.merge([raw, raw, raw])
        if raw.ndim == 3 and raw.shape[2] == 3:
            return raw.copy()
        if raw.ndim == 3 and raw.shape[2] == 1:
            g = raw[..., 0]
            return cv2.merge([g, g, g])
    if raw.ndim == 2:
        raw8 = raw if raw.dtype == np.uint8 else _to_uint8(raw)
        raw16 = (raw8.astype(np.uint16)) * 257
        return cv2.merge([raw16, raw16, raw16])
    if raw.ndim == 3 and raw.shape[2] == 3:
        raw8 = raw if raw.dtype == np.uint8 else _to_uint8(raw)
        return (raw8.astype(np.uint16)) * 257
    raw8 = _to_uint8(raw)
    if raw8.ndim == 2:
        raw16 = (raw8.astype(np.uint16)) * 257
        return cv2.merge([raw16, raw16, raw16])
    else:
        return (raw8.astype(np.uint16)) * 257

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-file", help="Process exactly this image and exit")
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--log-name", default=LOG_NAME)
    return ap.parse_args()

# ---- main processing ----
def process_images_with_grid_search(input_dir, output_dir, log_csv="results_log.csv", only_basename=None):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    if only_basename:
        files = [f for f in files if f == only_basename]

    print("========== navier_bestch AUTO DEBUG ==========")
    print(f"Start:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python:      {sys.executable}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"OpenCL on?   {USE_UMAT}")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Found files: {len(files)} (extensions: {ALLOWED_EXT})")
    print(f"Combos:      {len(COMBOS)} (curated)")
    print("==============================================", flush=True)
    if len(files) == 0:
        raise RuntimeError("Found 0 candidate images. Check INPUT_DIR or file extensions.")

    log_path = os.path.join(output_dir, log_csv)
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "filename","mu","lambda","diffusion_rate","alpha","beta","gamma",
                "Eth_q","Eth_seed","contour_mode","approx_mode",
                "n_contours","total_area_px","overlay16_path","mask16_path"
            ])

        for filename in files:
            in_path = os.path.join(input_dir, filename)
            print(f"\n[LOAD] {filename}", flush=True)
            raw, bgr8, gray8 = read_color_anydepth(in_path)
            if raw is None:
                print(f"[skip] Could not load {filename}", flush=True)
                continue

            print("[MASK] threshold gray for leaf mask...", flush=True)
            binary_mask = binary_threshold_mask(gray8, threshold=127)

            print("[SELECT] choose best working channel by edge energy...", flush=True)
            best_ch = best_single_channel_u8(bgr8, binary_mask)

            # Masked CLAHE to lift midtones on the leaf only (accelerated)
            leaf_enh = clahe_u(best_ch, clip=2.0, tile=(8,8), mask=None)
            masked_best = apply_mask(leaf_enh, binary_mask)

            print("[GRAD] precompute gradients...", flush=True)
            grad_mag_u8, gx, gy = gradient_magnitude(masked_best)

            Eth_q, Eth_seed = auto_eth_dual(grad_mag_u8, binary_mask)
            print(f"[AUTO] Eth_q={Eth_q}, Eth_seed={Eth_seed}", flush=True)

            overlay16 = _prepare_overlay16(raw)
            base = os.path.splitext(filename)[0]

            for (mu, lam, dr, alpha, beta, gamma) in COMBOS:
                param_str = f"mu{fmt(mu)}_lam{fmt(lam)}_d{fmt(dr)}_a{fmt(alpha)}_b{fmt(beta)}_g{fmt(gamma)}"
                print(f" ▶ {param_str} | Eth_q={Eth_q} Eth_seed={Eth_seed}", flush=True)
                try:
                    diff_map = elastic_deformation_diffusion(
                        masked_best, gx, gy,
                        iterations=30, diffusion_rate=dr, mu=mu, lambda_param=lam,
                        edge_thresh=Eth_q
                    )

                    seg_mask_bool = snake_seg(
                        masked_best, diff_map, its=50,
                        alpha=alpha, beta=beta, gamma=gamma,
                        l_size=25, u_size=50000,
                        energy_threshold=Eth_seed
                    )

                    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
                    seg_u8 = morph_close_u(seg_u8, ksize=3, iters=1)
                    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    out_overlay16 = overlay16.copy()
                    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)

                    mask16 = (seg_mask_bool.astype(np.uint16)) * 65535

                    overlay_path = os.path.join(output_dir, f"{base}__{param_str}__overlay16.tif")
                    mask_path    = os.path.join(output_dir, f"{base}__{param_str}__mask16.tif")
                    ensure_dir(os.path.dirname(overlay_path))
                    cv2.imwrite(overlay_path, out_overlay16)
                    cv2.imwrite(mask_path,    mask16)

                    n_contours = len(contours)
                    total_area = int(np.sum(seg_mask_bool))
                    writer.writerow([
                        filename, mu, lam, dr, alpha, beta, gamma,
                        Eth_q, Eth_seed, "RETR_EXTERNAL", "CHAIN_APPROX_SIMPLE",
                        n_contours, total_area, overlay_path, mask_path
                    ])
                    f.flush(); os.fsync(f.fileno())
                except cv2.error as e:
                    print(f"[skip][OpenCV] {filename} params={param_str}\n{e}", flush=True)
                except Exception as ex:
                    print(f"[skip][Error]  {filename} params={param_str}\n{ex}", flush=True)

def main():
    # avoid thread oversubscription inside each process
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    log_name = args.log_name

    if args.single_file:
        single = os.path.abspath(args.single_file)
        if not os.path.isfile(single):
            raise FileNotFoundError(single)
        ensure_dir(output_dir)
        only_base = os.path.basename(single)
        process_images_with_grid_search(os.path.dirname(single), output_dir, log_name, only_basename=only_base)
    else:
        process_images_with_grid_search(input_dir, output_dir, log_name)

if __name__ == "__main__":
    main()
