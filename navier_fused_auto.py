import os
import csv
import numpy as np
import cv2
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ------------ CONFIG (edit these) ------------
INPUT_DIR  = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves'
OUTPUT_DIR = '/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/navier_output_fused_auto'
LOG_NAME   = 'results_log.csv'
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
# ---------------------------------------------

# Pink in BGR (8-bit) is (255,20,147). Scale to 16-bit by ×257.
PINK_16 = (65535, 5140, 37779)  # (B, G, R) uint16

# ---- curated list of 6 "good" combinations (mu, lam, dr, alpha, beta, gamma) ----
COMBOS = [
    (0.05, 0.05, 0.1, 0.01, 0.1, 0.01),
    (0.05, 0.05, 0.4, 0.01, 0.1, 0.1),

    (0.05, 0.50, 0.1, 0.10, 0.5, 0.01),
    (0.50, 0.05, 0.4, 0.10, 0.1, 0.1),

    (0.50, 0.50, 0.1, 0.50, 0.5, 0.01),  # tighter snake
    (0.50, 0.50, 0.4, 0.50, 0.5, 0.1),   # tighter + stronger pull
]

def fmt(v):
    return str(v).replace('.', 'p')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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
    arr = vals_u8.reshape(-1, 1)  # column vector
    retval, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(retval)

def auto_eth_dual(grad_mag_u8, leaf_mask_u8):
    # slightly permissive; adjust if too many/too few seeds
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
    return cv2.bitwise_and(image_u8, image_u8, mask=mask_u8)

def gradient_magnitude(u8):
    # light smoothing before Sobel for more connected edges
    u8 = cv2.GaussianBlur(u8, (3,3), 0)
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
        div_v = np.gradient(diffused)[0]  # proxy
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
    nprops = len(props)
    # Short-circuit: if no seeds/regions, skip snakes entirely
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

        # 200-point circular init (cheaper than 400)
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_num_iter=50  # cheaper snake
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

# ---- fused working channel ----
def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[..., 1], gray, lab[..., 1]]  # G, GRAY, a*
    fused = None
    for ch in candidates:
        chm = apply_mask(ch, binary_mask) if binary_mask is not None else ch
        chf = chm.astype(np.float32)
        mx = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, dtype=np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)
    return fused

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

# ------------------ worker ------------------
def process_single_image(filename):
    """
    Worker: processes one image for all COMBOS.
    Returns: list of CSV rows (one per combo) or [] if skipped/failed.
    """
    try:
        in_path = os.path.join(INPUT_DIR, filename)
        raw, bgr8, gray8 = read_color_anydepth(in_path)
        if raw is None:
            print(f"[skip] Could not load {filename}", flush=True)
            return []

        # Leaf mask
        binary_mask = binary_threshold_mask(gray8, threshold=127)

        # Fused working channel + masked CLAHE
        fused_u8 = fused_channel_u8(bgr8, binary_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        leaf_enh = clahe.apply(fused_u8)
        masked_fused = cv2.bitwise_and(leaf_enh, leaf_enh, mask=binary_mask)

        # Auto thresholds from gradient magnitude
        grad_mag_u8, gx, gy = gradient_magnitude(masked_fused)
        Eth_q, Eth_seed = auto_eth_dual(grad_mag_u8, binary_mask)

        overlay16 = _prepare_overlay16(raw)
        base = os.path.splitext(filename)[0]

        rows = []
        for (mu, lam, dr, alpha, beta, gamma) in COMBOS:
            param_str = f"mu{fmt(mu)}_lam{fmt(lam)}_d{fmt(dr)}_a{fmt(alpha)}_b{fmt(beta)}_g{fmt(gamma)}"
            try:
                diff_map = elastic_deformation_diffusion(
                    masked_fused, gx, gy,
                    iterations=30, diffusion_rate=dr, mu=mu, lambda_param=lam,
                    edge_thresh=Eth_q
                )

                seg_mask_bool = snake_seg(
                    masked_fused, diff_map, its=50,     # cheaper snake
                    alpha=alpha, beta=beta, gamma=gamma,
                    l_size=25, u_size=50000,            # skip tiny noise
                    energy_threshold=Eth_seed
                )

                seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
                seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
                contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    filename, mu, lam, dr, alpha, beta, gamma,
                    Eth_q, Eth_seed, "RETR_EXTERNAL", "CHAIN_APPROX_SIMPLE",
                    n_contours, total_area, overlay_path, mask_path
                ])
            except cv2.error as e:
                print(f"[skip][OpenCV] {filename} params={param_str}\n{e}", flush=True)
            except Exception as ex:
                print(f"[skip][Error]  {filename} params={param_str}\n{ex}", flush=True)

        return rows
    except Exception as ex:
        print(f"[skip][Error] {filename}\n{ex}", flush=True)
        return []

# ------------------ main ------------------
def process_images_with_grid_search(input_dir, output_dir, log_csv="results_log.csv"):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    print("========== navier_fused AUTO PARALLEL DEBUG ==========")
    print(f"Start:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python:      {sys.executable}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Found files: {len(files)} (extensions: {ALLOWED_EXT})")
    print(f"Combos:      {len(COMBOS)} (curated)")
    # choose sensible default: leave 1 core free
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"[PARALLEL]  launching {max_workers} workers", flush=True)
    print("=============================================", flush=True)
    if len(files) == 0:
        raise RuntimeError("Found 0 candidate images. Check INPUT_DIR or file extensions.")

    log_path = os.path.join(output_dir, log_csv)
    write_header = not os.path.exists(log_path)

    # Run workers
    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_image, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            fn = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"[{i}/{len(files)}] done: {fn} -> {len(rows)} rows", flush=True)
            except Exception as e:
                print(f"[error] {fn}: {e}", flush=True)

    # Write CSV once (main process) to avoid contention
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "filename","mu","lambda","diffusion_rate","alpha","beta","gamma",
                "Eth_q","Eth_seed","contour_mode","approx_mode",
                "n_contours","total_area_px","overlay16_path","mask16_path"
            ])
        writer.writerows(all_rows)

    print(f"[DONE] wrote {len(all_rows)} result rows to: {log_path}", flush=True)

def main():
    # Avoid thread oversubscription inside each worker
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    process_images_with_grid_search(INPUT_DIR, OUTPUT_DIR, LOG_NAME)

if __name__ == "__main__":
    main()
