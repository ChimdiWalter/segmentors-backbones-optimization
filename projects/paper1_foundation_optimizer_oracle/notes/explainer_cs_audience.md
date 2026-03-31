# Paper 1 Deep Explainer — CS / ML Audience

**Paper:** *Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours for Unsupervised High-Throughput Phenotyping*
**Authors:** Chimdi Walter Ndubuisi, Toni Kazic
**Manuscript:** `manuscript/main_paper1.tex` → compiled PDF at `manuscript/main_paper1.pdf` / `pdf/paper1_final.pdf`

---

## 1. The Problem in One Paragraph

You have 174 high-resolution (~3900 × 1340 px), 16-bit TIFF leaf images showing fungal disease lesions. You need per-image binary masks. There is **no pixel-level ground truth**. Classical thresholding fails on heterogeneous images; training a UNet/SAM requires annotation you don't have. The paper's answer: treat the segmentation problem as a **black-box optimization** over the 6-dimensional physics parameter space of a Navier-Stokes–driven active-contour model, using a handcrafted no-ground-truth objective. Once you have pseudo-label masks from the optimizer (the "teacher"), you distill them into a compact UNet student for fast inference.

---

## 2. Technical Background: What You Need to Know

### 2.1 Active Contour Models (Snakes)
A snake is a parameterized curve `C(s) = (x(s), y(s))` that minimizes:

```
E_snake(C) = ∫ ½(α||C'||² + β||C''||²) ds  +  ∫ E_ext(C(s)) ds
```

- `α` (tension) penalizes stretching → shrinks the contour
- `β` (rigidity) penalizes curvature → keeps contour smooth
- `E_ext` is an external energy field derived from image gradients (low inside lesions, high at boundaries)

The evolution PDE adds a balloon term `γ`:
```
∂C/∂t = α·C'' − β·C'''' + γ·n(s) − ∇E_ext
```
where `n(s)` is the outward normal. Without `γ > 0`, snakes shrink to a point; with it, they inflate until edges are encountered.

### 2.2 Navier-Stokes–Inspired Diffusion (the External Force)
Raw Sobel gradients are noisy. This paper diffuses them using a scalar PDE inspired by elastic/viscous fluid mechanics:

```
E^(t+1) = E^(t) + Δt [ μ·Lap(E^(t))  +  (λ+μ)·div(∇E^(t))  +  q·(F − E^(t)) ]
```

- `μ` = viscosity → controls how aggressively high-freq noise is smoothed
- `(λ+μ)` = elastic coupling → propagates edge signal laterally
- `F = ||( I_x − S_x,  I_y − S_y )||₂` = forcing from fine-scale vs coarse-scale gradient discrepancy (marks lesion edges)
- `q = 1[G > T]` = edge gate; only strong-gradient regions drive the forcing
- Run for **80 explicit iterations**; stability requires `μ ≤ 0.25` (CFL: `Δt ≤ h²/(4μ)` with `Δt=1, h=1`)
- Result is a scalar energy map normalized to [0,255]

This plays the role of GVF (Gradient Vector Flow) but is tunable via the parameter `μ`.

### 2.3 Parameter Space θ = (μ, λ, α, β, γ, T) ∈ ℝ⁶
All six parameters are optimized **per image** over bounded search ranges. Mapping from `θ` to a mask `M` involves: diffusion → thresholding → connected-component filtering → snake evolution → finalization. This is non-differentiable at multiple steps, ruling out gradient-based methods.

---

## 3. The Heuristic Objective (No Ground Truth)

The score `S(θ)` the optimizer maximizes:

```
S(θ) = w_g·G(M,I) + w_c·C(M,I) + w_n·log(1+N(M)) − P_area(M) − P_noise(M)
```

| Term | Formula | Intuition |
|------|---------|-----------|
| Gradient alignment `G` | `E[G(x,y) | boundary∂M] / E[G(x,y) | all Ω]` | Mask boundaries should coincide with strong image edges |
| Chromatic separability `C` | `||μ_in^ab − μ_out^ab||₂ / (√2·255)` | Lesion interior vs exterior should be distinct in Lab ab-space |
| Topological reward | `log(1 + N(M))` | Reward multiple connected components; log prevents runaway fragmentation |
| Area penalty `P_area` | Penalizes empty masks and masks > 45% leaf area | Prevents trivial solutions |
| Noise penalty `P_noise` | Penalizes ultra-small components | Prevents dust/sensor speckle |

The full-objective ablation (Table I, `tab:abl_obj`) on 30 images shows:
- Without topology term `w_n=0`: median score drops from 14.80 → 3.68 (catastrophic—optimizer can't steer away from empty masks)
- Without gradient alignment `w_g=0`: median score 9.71 (significant)
- Without color distance `w_c=0`: median score 14.74 (small but masks are larger/noisier)
- Without area/noise penalties: score stays high but lesion count inflates (188 vs 134 median n_c without noise penalty), showing the penalties prevent degenerate fragmentation

**Where these numbers come from:**
Ablation runs are stored at `experiments/exp_v1/outputs` (ablation CSVs). Each ablation zeroes out one weight and reruns the full pipeline on a 30-image stratified subset. The 30-image subset is area-fraction stratified (5 quantile bins × 6 images).

---

## 4. The Search Algorithm

**Random-Restart Stochastic Hill Climbing** (Algorithm 1 in `main_paper1.tex`):

1. **Phase 1 – Random exploration:** Sample `K=64` parameter vectors uniformly from `Θ`; evaluate `S(I; θ_k)` on downscaled image (`s=0.75`); keep running best
2. **Phase 2 – Local refinement:** Take the top-4 random candidates; for each run `R=12` perturbation steps with cooling `σ_j = σ_0·0.85^j`; accept only improvements

**Search-strategy ablations (Table II, `tab:abl_search`):**
- Full pipeline: median score 14.95, ~4047 s median runtime
- Random-only (no refinement): 14.88 — barely worse; refinement gives marginal but consistent benefit on high-score leaves
- Low budget K=16: 14.91 — robust to budget reduction
- No downscale s=1.0: 16.58 score improvement but 5428 s vs 4047 s — full-res is better quality but ~35% slower
- Fast 60s timeout: 14.44 — tolerable quality loss for rapid screening

The deliberate simplicity is a feature: random-restart hill climbing is predictable, parallelizable on CPU clusters, and needs no gradients.

---

## 5. Teacher–Student Distillation

After the optimizer produces 174 masks ("teacher pseudo-labels"), a lightweight UNet student is trained on 4025 512×512 patch pairs:
- Patchification: 512×512, stride 448, reflection padding; 4599 total patches → keep all lesion-containing + 35% of empty-mask patches
- Architecture: 3-level encoder-decoder (channels 32→64→128), 2×(3×3 conv + BN + ReLU) per block, max-pool down, transposed-conv up
- Loss: ½ BCEWithLogits + ½ Dice
- Training: 40 epochs, Adam lr=1e-4, 85/15 split
- Inference: stitch patch predictions via overlap-averaging, threshold at 0.30 (tuned on val sweep)

**Results (from `experiments/exp_v1/outputs`):**
- All 174 images: mean Dice 0.598±0.226, mean IoU 0.457±0.195 (median Dice 0.682)
- Teacher-nonempty subset (N=162): mean Dice 0.642±0.163, median Dice 0.686
- ROI-gated at 0.30: nearly identical (mean Dice 0.599/0.644) → ROI gating doesn't help much, suggesting disagreements come from semantic mismatch (inner vs outer mask definition) not background leakage
- Threshold sweep peak: Dice 0.604 at threshold=0.30
- Failure tail: 18 images with Dice<0.30; 12 are teacher-empty/student-nonempty cases

**Figure pointers:**
| Figure | File | Path |
|--------|------|------|
| Teacher–student montage (worst/median/best) | `montage_teacher_student.png` | `figures/exp_v1/` |
| Disagreement cases | `montage_disagreement_roi.png` | `figures/exp_v1/` |
| Qualitative Dice-spectrum grid (6 rows) | `qual_grid_teacher_student.png` | `figures/exp_v1/` |
| Threshold sweep | `threshold_sweep_plot.png` | `figures/exp_v1/` |

---

## 6. Trace-Based Objective-Surface Analysis

To characterize the optimization landscape, the code re-ran the full optimizer on **9 selected leaves** (3 high-score, 3 median-score, 3 low-score from the production run of 174) with every candidate evaluation logged as a trace `(θ_k, s_k, t_k)`.

**Projected pairwise surface (Eq. 4):**
For any two parameters (p, q), scatter all `(θ_k[p], θ_k[q], s_k)` in 3D. The dominant pair is *energy threshold T vs. balloon force β* (highest R² in a linear regression of score on two-parameter model), followed by T vs. μ.

**Key findings:**
- **High-score leaves:** sharp landscape with distinct optima; refinement phase gives measurable score gain beyond random
- **Low-score leaves:** flat landscape; random phase already finds near-optimal; refinement contributes negligible gains
- **Multi-basin hints:** top-quartile candidates in high-score leaves spread across multiple parameter clusters, suggesting the landscape supports distinct good solutions (many small lesions vs. fewer coalesced regions)

**Figure pointers:**
| Figure | File | Path |
|--------|------|------|
| Objective surface, high-score leaf (DSC_0198) | `surface_DSC_0198_segment_1_segmented_smoothed__surface_energy_threshold_vs_beta.png` | `figures/9leaf/` |
| Objective surface, median-score (DSC_0379) | `surface_DSC_0379_...` | `figures/9leaf/` |
| Objective surface, low-score (DSC_0261) | `surface_DSC_0261_...` | `figures/9leaf/` |
| Convergence trajectory, high-score (DSC_0198) | `traj_DSC_0198_..._trajectory_best_score_so_far.png` | `figures/9leaf/` |
| Appendix trajectories (6 more leaves) | `traj_DSC_0166/0163/0274/0205/0185/0199_...` | `figures/9leaf/` |

---

## 7. Parameter-Cloud Geometry (Manifold Analysis)

Using **150 global–local parameter pairs** from 15 leaves (75 interior + 75 boundary 512×512 patches):
- **Global parameters** = per-leaf optimized θ*
- **Local parameters** = re-optimized on each patch independently
- **Shift** = δ_j = θ_j^local − θ_j^global ∈ ℝ⁷

**PCA results (Figure `pca2d_global_local_dD.png`, path `figures/manifold/`):**
- 2 PCs capture **46.5%** of total variance
- **PC1 separates** global from local clouds with centroid shift magnitude **1.616 standardized units** — confirms local re-optimization is a systematic displacement, not random noise
- Energy threshold dominates the mean shift profile (Figure `mean_delta_parameters.png`)

**Solid-3D covariance ellipsoids (Figure `pca3d_covariance_ellipsoids.png`, path `figures/solid3d/`):**
- 2σ ellipsoids of global (blue) vs local (red) are displaced but similarly oriented → comparable landscape curvature at both scales
- Patches where local improved Dice (ΔD > +0.02; n=73) vs. hurt (ΔD < −0.02; n=48) show partial PCA separation

**k-NN consistency test on the delta cloud:**
- Agreement 0.534 vs. random baseline 0.33 for ΔD groups → parameter-shift direction has predictive signal for which patches benefit from local re-optimization
- Interior vs. boundary label: agreement 0.488 vs. baseline 0.50 → spatial location within leaf does NOT predict shift direction

**Appendix figures:** t-SNE and UMAP of delta cloud (`tsne2d_delta_ddgroup.png`, `pca_centroid_shift_groups.png`) in `figures/manifold/`; convex hulls and pos/neg ΔD surfaces in `figures/solid3d/`.

---

## 8. How Figures Were Generated

| Figure set | Script / Source | Output location |
|------------|----------------|-----------------|
| Optimizer run status, runtime hist, score hist | `experiments/exp_v1/` run scripts | `figures/exp_v1/optimizer_*.png` |
| Threshold sweep | Post-processing on stitched UNet probability maps | `figures/exp_v1/threshold_sweep_plot.png` |
| Teacher–student montages | Qualitative comparison pipeline | `figures/exp_v1/montage_*.png` |
| Scatter (energy_threshold vs lesion_count, beta vs area_frac) | Correlation analysis on optimizer output CSV | `figures/exp_v1/scatter_*.png` |
| 9-leaf trace surfaces | 9-leaf trace rerun with logging enabled | `figures/9leaf/surface_*.png`, `traj_*.png` |
| PCA 2D global/local, mean delta | PCA on 150-patch parameter dataset | `figures/manifold/pca2d_*.png`, `mean_delta_*.png` |
| 3D covariance ellipsoids, convex hulls | 3D visualization pipeline | `figures/solid3d/pca3d_*.png` |
| Appendix param histograms | Optimizer output CSV | `figures/exp_v1/param_*_hist.png` |
| Appendix output distribution (n_contours, area_px) | Optimizer output CSV | `figures/exp_v1/n_contours_hist.png`, `area_px_hist.png` |

The figure provenance CSV at `notes/figure_provenance.csv` lists every figure label, filename, and source path.

---

## 9. Patch-Level Morphology Transfer (Table III)

Beyond Dice, the paper evaluates whether physics parameters transfer morphological fidelity:

| Metric | Global | Local | Teacher |
|--------|--------|-------|---------|
| Dice (all patches) | 0.430 | 0.443 | — |
| Area fraction | 0.026 | 0.028 | 0.169 |
| Lesion count | 39.4 | 29.7 | 23.9 |
| Count error | 19.1 | 12.4 | — |

**Key takeaway:** Both parameter sets severely underestimate area fraction (0.026 vs 0.169) because the two-stage inner-mask definition discards peripheral tissue. Local re-optimization cuts count error by 35% (19.1 → 12.4). Dice alone misleads — morphological metrics diverge. This table is at `experiments/exp_v1/outputs` (patch-level metrics CSV).

---

## 10. Multi-Stage Nested Inference

The paper proposes a two-stage extension:
1. **Stage-A (Outer):** full optimizer on entire leaf → outer lesion `M_outer`
2. **Stage-B (Inner):** restrict search to `M_outer` pixels; increase energy threshold T' = 1.3·T_A*; re-optimize → necrotic core `M_inner`

This yields the **Necrosis ratio = |M_inner| / (|M_outer| + ε)** — a biologically meaningful trait. In visualizations, outer (green) and inner (magenta) overlays are distinct. The two-stage algorithm is Algorithm 2 in the manuscript.

---

## 11. Discussion Highlights for CS Readers

- **Why no gradient?** The θ → M pipeline includes thresholding and connected-component filtering — non-differentiable. Gradient-free random-restart HC is the honest choice.
- **Why optimize per image?** The heuristic objective's color-separability and gradient-alignment terms rely on within-image contrast between lesion and background. Optimizing on patches risks degenerate single-class patches giving unreliable scores.
- **k-NN agreement 0.534 vs 0.33 baseline** establishes that the solution manifold has exploitable structure — the motivation for Paper 2's amortized predictor.
- **Failure modes:** 12/174 teacher-empty / student-nonempty disagreements. These are not threshold failures — they persist across all threshold values in the sweep. They reflect semantic ambiguity: the inner-mask definition can legitimately yield an empty mask for lesions without a necrotic core.
- **Scalability:** Downscaling to s=0.75 for search + timeout watchdogs makes the system practical on CPU clusters without GPUs. Single-thread BLAS/OpenCV enforced to avoid oversubscription.
