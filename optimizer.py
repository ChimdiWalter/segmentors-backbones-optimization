# optimizer.py
import numpy as np
import cv2, math, random
from typing import Optional, Dict, Tuple

# --- helpers you already have or equivalent signatures:
# elastic_deformation_diffusion(image_u8, grad_x, grad_y, iterations, diffusion_rate, mu, lambda_param, edge_thresh)
# snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, l_size, u_size, energy_threshold)

def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return (2.0*inter / s) if s > 0 else 0.0

def boundary_hit_fraction(mask: np.ndarray, grad_mag_u8: np.ndarray, thr: int=40) -> float:
    # 8-neighborhood perimeter pixels
    m = mask.astype(np.uint8)
    edges = cv2.Canny(m*255, 50, 150)  # fast proxy for boundary
    strong = (grad_mag_u8 >= thr).astype(np.uint8)
    hit = np.logical_and(edges>0, strong>0).sum()
    tot = (edges>0).sum()
    return (hit / tot) if tot > 0 else 0.0

def mean_compactness(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    nlab, lab = cv2.connectedComponents(m)
    if nlab <= 1: return 0.0
    vals = []
    for lbl in range(1, nlab):
        comp = (lab==lbl).astype(np.uint8)
        area = int(comp.sum())
        if area < 5: 
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: 
            continue
        per = float(cv2.arcLength(contours[0], True))
        if per <= 1e-6: 
            continue
        vals.append((4.0*math.pi*area)/(per*per))
    return float(np.mean(vals)) if vals else 0.0

def gaussian_area_prior(area: int, target: float, sigma: float) -> float:
    # returns ~1 near target, down-weights extremes
    z = (area - target) / (sigma + 1e-8)
    return math.exp(-float(z*z))

def stability_score(mask_base: np.ndarray, make_mask_fn, jitters: int=2) -> float:
    # IoU vs jittered params/noise
    base = mask_base.astype(bool)
    ious = []
    for _ in range(jitters):
        m = make_mask_fn(jitter=True)
        if m is None: 
            continue
        other = m.astype(bool)
        inter = np.logical_and(base, other).sum()
        uni = np.logical_or(base, other).sum()
        ious.append((inter/uni) if uni>0 else 1.0)
    return float(np.mean(ious)) if ious else 0.0

def segment_with_params(image_u8, gx, gy, params):
    mu, lam, d, alpha, beta, gamma, eth = params
    energy = elastic_deformation_diffusion(
        image_u8, gx, gy,
        iterations=30, diffusion_rate=d, mu=mu, lambda_param=lam, edge_thresh=eth
    )
    mask = snake_seg(
        image_u8, energy, its=100,
        alpha=alpha, beta=beta, gamma=gamma,
        l_size=5, u_size=50000, energy_threshold=eth
    )
    return mask

def optimize_params_for_image(
    image_u8: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_mag_u8: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    area_target: Optional[float] = None,
    area_sigma: float = 3e5,
    energy_threshold_choices=(30, 50, 70),
    warmstart_params=(),
    n_random=24,
    n_refine=12,
    seed: int = 0
) -> Dict:
    rng = random.Random(seed)
    H, W = image_u8.shape
    total_pixels = H*W
    if area_target is None:
        area_target = 0.02 * total_pixels  # 2% of image as a soft prior

    # bounds
    MU = (0.01, 1.0)
    LA = (0.01, 1.0)
    D  = (0.05, 0.6)
    A  = (0.005, 0.6)
    B  = (0.05,  1.0)
    G  = (0.005, 0.5)

    def clip(v, lo, hi): return max(lo, min(hi, v))

    def rand_params():
        mu = 10**rng.uniform(np.log10(MU[0]), np.log10(MU[1]))
        la = 10**rng.uniform(np.log10(LA[0]), np.log10(LA[1]))
        d  = rng.uniform(*D)
        a  = 10**rng.uniform(np.log10(A[0]), np.log10(A[1]))
        b  = rng.uniform(*B)
        g  = 10**rng.uniform(np.log10(G[0]), np.log10(G[1]))
        eth = rng.choice(list(energy_threshold_choices))
        return (mu, la, d, a, b, g, eth)

    def objective(params):
        mask = segment_with_params(image_u8, grad_x, grad_y, params)
        if mask is None:
            return -1e9, None
        area = int(mask.sum())
        if gt_mask is not None:
            score = dice_coefficient(mask, gt_mask.astype(bool))
        else:
            # Unsupervised composite
            edge = boundary_hit_fraction(mask, grad_mag_u8, thr=40)
            comp = mean_compactness(mask)
            apr  = gaussian_area_prior(area, area_target, area_sigma)
            # stability via tiny jitters around params
            def mk(jitter=False):
                if not jitter:
                    return mask
                mu, la, d, a, b, g, eth = params
                j = lambda x, lo, hi: clip(x*(1.0 + rng.uniform(-0.08, 0.08)), lo, hi)
                pj = (j(mu,*MU), j(la,*LA), j(d,*D), j(a,*A), j(b,*B), j(g,*G), eth)
                return segment_with_params(image_u8, grad_x, grad_y, pj)
            stab = stability_score(mask, mk, jitters=2)
            score = 0.45*edge + 0.25*comp + 0.15*apr + 0.15*stab
        return float(score), mask

    # cache image-driven parts already computed in your pipeline
    # warmstart: your 10 curated combos (map to tuple order)
    tested = []
    best_score, best_params, best_mask = -1e9, None, None

    def try_params(p):
        nonlocal best_score, best_params, best_mask
        s, m = objective(p)
        tested.append((s, p))
        if s > best_score:
            best_score, best_params, best_mask = s, p, m

    for p in warmstart_params:
        try_params(p)

    for _ in range(n_random):
        try_params(rand_params())

    # local coordinate refine around best
    for _ in range(n_refine):
        mu, la, d, a, b, g, eth = best_params
        cand = []
        for (lo, hi, x) in [(MU,mu),(LA,la),(D,d),(A,a),(B,b),(G,g)]:
            step = (hi-lo)*0.15
            cand.extend([clip(x-step, lo, hi), clip(x+step, lo, hi)])
        # evaluate a small set of neighbors (keep eth fixed)
        for (mu2,la2,d2,a2,b2,g2) in [(cand[0],la,d,a,b,g),(mu,cand[2],d,a,b,g),
                                      (mu,la,cand[4],a,b,g),(mu,la,d,cand[6],b,g),
                                      (mu,la,d,a,cand[8],g),(mu,la,d,a,b,cand[10])]:
            try_params((mu2, la2, d2, a2, b2, g2, eth))

    return {
        "best_params": best_params,
        "best_score":  best_score,
        "best_mask":   best_mask,
        "evaluations": len(tested)
    }
