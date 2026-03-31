import os
import csv
import numpy as np
import cv2
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ------------ CONFIG (edit these) ------------
INPUT_DIR  = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves'
OUTPUT_DIR = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/navier_output'
LOG_NAME   = 'results_log.csv'
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
# ---------------------------------------------

# 16-bit pink (B,G,R)
PINK_16 = (65535, 5140, 37779)

# Fixed thresholds
ENERGY_THRESHOLDS = [30, 70]

# Curated TEN combos (mu, lam, dr, alpha, beta, gamma)
COMBOS = [
    (0.05, 0.05, 0.1, 0.01, 0.1, 0.01),
    (0.05, 0.05, 0.4, 0.01, 0.1, 0.1),
    (0.05, 0.50, 0.1, 0.01, 0.5, 0.01),
    (0.05, 0.50, 0.4, 0.01, 0.5, 0.1),

    (0.50, 0.05, 0.1, 0.10, 0.1, 0.01),
    (0.50, 0.05, 0.4, 0.10, 0.1, 0.1),
    (0.50, 0.50, 0.1, 0.10, 0.5, 0.01),
    (0.50, 0.50, 0.4, 0.10, 0.5, 0.1),

    (0.05, 0.05, 0.1, 0.50, 0.1, 0.01),
    (0.50, 0.50, 0.4, 0.50, 0.5, 0.1),
]

CONTOUR_MODE = cv2.RETR_EXTERNAL
APPROX_MODE  = cv2.CHAIN_APPROX_SIMPLE

# ---------------- I/O helpers ----------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def fmt(v): return str(v).replace('.', 'p')

def read_color_anydepth(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None: return None, None, None

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
    if img.dtype == np.uint8: return img
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0); return (im * 255.0).astype(np.uint8)
    imin, imax = float(np.min(img)), float(np.max(img))
    if imax <= imin + 1e-12: return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)

def _to_float01(img):
    if img.dtype == np.uint8: return (img.astype(np.float32))/255.0
    if img.dtype == np.uint16: return (img.astype(np.float32))/65535.0
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.float32)
    return (img - mn)/(mx - mn)

# ---------------- Core ops ----------------
def binary_threshold_mask(gray8, threshold=127):
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8, mask_u8):
    return cv2.bitwise_and(image_u8, image_u8, mask=mask_u8)

def gradient_magnitude(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    eps = 1e-8
    mag_u8 = np.uint8(255 * mag / max(np.max(mag), eps))
    return mag_u8, gx, gy

def elastic_deformation_diffusion(image_u8, grad_x, grad_y,
                                  iterations=30, diffusion_rate=0.2,
                                  mu=0.5, lambda_param=0.5,
                                  edge_thresh=50):
    diffused = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)

    im_gx = cv2.Sobel(image_u8, cv2.CV_64F, 1, 0, ksize=3)
    im_gy = cv2.Sobel(image_u8, cv2.CV_64F, 0, 1, ksize=3)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)

    q = (im_mag > edge_thresh).astype(np.float64)

    for _ in range(iterations):
        lap = laplace(diffused)
        div_v = np.gradient(diffused)[0]
        ftx = im_gx - grad_x
        fty = im_gy - grad_y
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += diffusion_rate * (mu * lap + (lambda_param + mu) * div_v + q * (ftm - diffused))

    eps = 1e-8
    return np.uint8(255 * diffused / max(np.max(diffused), eps))

def snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, l_size, u_size, energy_threshold):
    h, w = image_u8.shape
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)

    for rgn in props:
        if not (l_size < rgn.area < u_size):
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)

        # 200-point circular init
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_num_iter=100
            )
        except TypeError:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_iterations=100
            )

        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)

        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out_mask

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

# ---------------- Worker for one image ----------------
def process_single_image(filename):
    rows = []
    try:
        in_path = os.path.join(INPUT_DIR, filename)
        raw, bgr8, gray8 = read_color_anydepth(in_path)
        if raw is None:
            print(f"[skip] Could not load {filename}", flush=True)
            return rows

        # Use GREEN channel (G) as requested
        b8, g8, r8 = cv2.split(bgr8)
        binary_mask = binary_threshold_mask(gray8, threshold=127)
        masked_g = apply_mask(g8, binary_mask)

        # Precompute gradients once
        _, gx_g, gy_g = gradient_magnitude(masked_g)

        overlay16 = _prepare_overlay16(raw)
        base = os.path.splitext(filename)[0]

        for (mu, lam, dr, alpha, beta, gamma), eth in itertools.product(COMBOS, ENERGY_THRESHOLDS):
            param_str = f"mu{fmt(mu)}_lam{fmt(lam)}_d{fmt(dr)}_a{fmt(alpha)}_b{fmt(beta)}_g{fmt(gamma)}_e{eth}"
            try:
                diff_g = elastic_deformation_diffusion(
                    masked_g, gx_g, gy_g,
                    iterations=30, diffusion_rate=dr, mu=mu, lambda_param=lam,
                    edge_thresh=eth
                )

                seg_mask_bool = snake_seg(
                    masked_g, diff_g, its=100,
                    alpha=alpha, beta=beta, gamma=gamma,
                    l_size=5, u_size=50000, energy_threshold=eth
                )

                seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
                seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
                contours, _ = cv2.findContours(seg_u8, CONTOUR_MODE, APPROX_MODE)

                out_overlay16 = overlay16.copy()
                cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)

                mask16 = (seg_mask_bool.astype(np.uint16)) * 65535

                overlay_path = os.path.join(OUTPUT_DIR, f"{base}__{param_str}__overlay16.tif")
                mask_path    = os.path.join(OUTPUT_DIR, f"{base}__{param_str}__mask16.tif")
                ensure_dir(os.path.dirname(overlay_path))
                cv2.imwrite(overlay_path, out_overlay16)
                cv2.imwrite(mask_path,    mask16)

                n_contours = len(contours)
                total_area = int(np.sum(seg_mask_bool))
                rows.append([
                    filename, mu, lam, dr, alpha, beta, gamma, eth,
                    CONTOUR_MODE, APPROX_MODE, n_contours, total_area,
                    overlay_path, mask_path
                ])

            except cv2.error as e:
                print(f"[skip][OpenCV] {filename} params={param_str}\n{e}", flush=True)
            except Exception as ex:
                print(f"[skip][Error]  {filename} params={param_str}\n{ex}", flush=True)

    except Exception as ex:
        print(f"[worker error] {filename}: {ex}", flush=True)

    return rows

# ---------------- Driver (parallel) ----------------
def process_images_with_grid_search(input_dir, output_dir, log_csv="results_log.csv"):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    print("========== navier GREEN GRID PARALLEL ==========")
    print(f"Start:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python:      {sys.executable}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Found files: {len(files)} (extensions: {ALLOWED_EXT})")
    print(f"Workers:     {max_workers}")
    print(f"Combos:      {len(COMBOS)}  | Energy thresholds: {ENERGY_THRESHOLDS}")
    print("===============================================", flush=True)

    if len(files) == 0:
        raise RuntimeError("Found 0 candidate images.")

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_image, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            fn = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"[{i}/{len(files)}] done: {fn} -> {len(rows)} param-rows", flush=True)
            except Exception as e:
                print(f"[error] {fn}: {e}", flush=True)

    log_path = os.path.join(output_dir, log_csv)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "filename","mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold",
                "contour_mode","approx_mode","n_contours","total_area_px","overlay16_path","mask16_path"
            ])
        writer.writerows(all_rows)

    print(f"[DONE] wrote {len(all_rows)} rows -> {log_path}", flush=True)

def main():
    # Avoid thread oversubscription per worker
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    process_images_with_grid_search(INPUT_DIR, OUTPUT_DIR, LOG_NAME)

if __name__ == "__main__":
    main()
