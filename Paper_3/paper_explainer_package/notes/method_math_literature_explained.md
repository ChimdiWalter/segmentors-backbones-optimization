# Methods, Mathematics, and Literature — Explained

**Date**: 2026-03-13
**Source**: Paper_3/manuscript_final_strengthened/tex/Latest_Paper_1_strengthened.tex

---

## 1. The Optimization Setup

### Plain Language
The paper treats each leaf image as its own optimization problem. Instead of using one fixed set of physics parameters for all images (which fails because lesions are so diverse), it searches for the best parameters *per image* using a trial-and-error approach. Think of it like tuning a radio dial for each station — each leaf needs its own settings.

### Technical Description
Given a preprocessed grayscale image $I: \Omega \subset \mathbb{R}^2 \to \mathbb{R}$, the framework seeks a parameter vector $\theta = (\mu, \lambda, \alpha, \beta, \gamma, T) \in \Theta$ (a bounded 7D search space) that maximizes a heuristic no-ground-truth objective:

$$\theta^* = \arg\max_{\theta \in \Theta} S(I; \theta)$$

The mapping $\theta \mapsto S(I; \theta)$ involves running a PDE-based physics engine, extracting candidate contours, and scoring the result — a non-differentiable pipeline that precludes gradient-based optimization.

### Mathematical Details
- **Parameters**: $\mu$ (viscosity/smoothing), $\lambda$ (elastic coupling), $\alpha$ (snake tension), $\beta$ (snake rigidity/balloon), $\gamma$ (inflation force), $T$ (energy threshold)
- **Search space**: Bounded box $\Theta$ with physically meaningful limits
- **Equations**: Eq. 1 (per-image optimization), Eq. 8 (same, restated in methodology)

### Literature Support
- Yu & Zhu (2020) [yu2020hyperparam]: HPO survey establishing that objectives and search-space design matter more than optimizer sophistication for small budgets
- Hansen & Ostermeier (2001) [hansen2001cma]: CMA-ES as a reference for evolutionary search
- Jones et al. (1998) [jones1998ego]: Bayesian optimization (EGO) as context for expensive black-box optimization

### Connected Outputs
- `opt_summary_local.csv`: The production run results
- `optimizer_summary/`: Distribution summaries

---

## 2. Navier-Stokes Diffusion Engine (PDE Component)

### Plain Language
Before the snake contours can find lesions, the image's edge information needs to be cleaned up and spread out. Raw edges are noisy (dust, leaf hairs, sensor grain). The paper uses a physics-inspired diffusion process — conceptually similar to how ink spreads in water — to create a smooth energy map that amplifies real lesion boundaries while dampening noise. The "Navier-Stokes" name comes from the fluid dynamics equations that inspire this diffusion.

### Technical Description
Starting from Sobel gradients $\nabla I = (I_x, I_y)$ and magnitude $G = \|\nabla I\|$, the method evolves a scalar energy field $E$ via an explicit diffusion-reaction scheme:

$$E^{(t+1)} = E^{(t)} + \Delta t \left( \mu \Delta E^{(t)} + (\lambda + \mu) \mathrm{div}(\nabla E^{(t)}) + q \cdot (F - E^{(t)}) \right)$$

This has three terms:
1. **$\mu \Delta E$**: Laplacian diffusion (viscosity-like smoothing)
2. **$(\lambda+\mu) \mathrm{div}(\nabla E)$**: Elastic coupling (promotes coherent field structure)
3. **$q(F - E)$**: Forcing term, gated by edge mask $q = \mathbf{1}[G > T]$

### Mathematical Details
- **Forcing**: $F = \|(I_x - S_x, I_y - S_y)\|_2$ where $(S_x, S_y)$ is a Gaussian-smoothed (seed) gradient. Forcing is strong where fine-scale edges deviate from coarse structure — exactly where lesion boundaries introduce new features.
- **Stability**: CFL condition $\Delta t \leq h^2/(4\mu)$. With $\Delta t = 1$ and $h = 1$, requires $\mu \leq 0.25$.
- **Implementation**: 80 iterations, normalize to [0, 255] after convergence.
- **Equation**: Eq. 4 (diffusion update), Eq. 5 (forcing term)

### Literature Support
- Bertalmio et al. (2001) [bertalmio2000navier]: Navier-Stokes inpainting — pioneered fluid-dynamics analogies in vision
- Xu & Prince (1998) [xu1998gvf]: Gradient Vector Flow — related diffusion approach for enlarging snake capture range
- Berry et al. (2018) [berry2018color]: Color calibration in phenotyping — motivates careful pre-processing

### Connected Outputs
- The energy field is computed inside `navier_optimize_robust_mostrecent.py`
- Visible indirectly in all mask/overlay outputs

---

## 3. Active Contour Dynamics (Snakes)

### Plain Language
After creating the smoothed energy map, the method places flexible curves (snakes) around candidate lesion regions and lets them evolve. The curves try to shrink onto boundaries (like rubber bands), while a balloon force pushes them outward to expand into lesion territory. The balance between shrinkage, smoothness, and expansion is controlled by the parameters being optimized.

### Technical Description
The snake energy functional is:

$$E_{\mathrm{snake}}(C) = \int_0^1 \frac{1}{2}\left(\alpha \|C'(s)\|^2 + \beta \|C''(s)\|^2\right) ds + \int_0^1 E_{\mathrm{ext}}(C(s)) ds$$

The curve evolves by:
$$\frac{\partial C}{\partial t} = \alpha C'' - \beta C'''' + \gamma \mathbf{n}(s) - \nabla E_{\mathrm{ext}}$$

- $\alpha$: Tension (penalizes stretching → shrinkage bias)
- $\beta$: Rigidity (penalizes curvature → smoothness/roundness)
- $\gamma$: Balloon force (outward expansion along normal)
- $E_{\mathrm{ext}}$: External energy from the diffused field

### Multi-Lesion Handling
Multiple candidate regions generated by thresholding $E$ and fitting snakes to each connected component satisfying scale constraints:
- $f_{\min} = 5 \times 10^{-4}$ (rejects dust below ~2,600 px at 5 MP)
- $f_{\max} = 0.45$ (prevents one component from engulfing entire leaf)

### Literature Support
- Kass et al. (1988) [kass1988snakes]: Original snakes formulation
- Caselles et al. (1997) [caselles1997geodesic]: Geodesic active contours
- Chan & Vese (2001) [chan2001active]: Region-based contours without edges

### Connected Outputs
- All mask16/overlay16 output files are the result of snake evolution
- Multi-lesion structure visible in `n_contours` column of `opt_summary_local.csv`

---

## 4. The Heuristic Objective Function (No-Ground-Truth Scoring)

### Plain Language
Since there are no human-drawn ground-truth masks, the paper invents a scoring function that estimates how good a segmentation is. It rewards three things: (1) boundaries that sit on real image edges, (2) color difference between lesion interior and healthy tissue, and (3) multiple distinct lesion regions (since real leaves typically have many spots). It penalizes degenerate solutions like empty masks or noisy speckle.

### Technical Description
$$S(\theta) = w_g \mathcal{G}(M, I) + w_c \mathcal{C}(M, I) + w_n \log(1 + N(M)) - \mathcal{P}_{\mathrm{area}}(M) - \mathcal{P}_{\mathrm{noise}}(M)$$

### Component Breakdown

**Gradient Alignment $\mathcal{G}$ (Eq. 10)**:
$$\mathcal{G}(M, I) = \frac{\mathbb{E}_{(x,y) \sim \partial M}[G(x,y)]}{\mathbb{E}_{(x,y) \sim \Omega}[G(x,y)] + \varepsilon}$$
Rewards boundaries on stronger-than-average gradients. If the mask boundary sits on real edges, this ratio is high.

**Chromatic Separability $\mathcal{C}$ (Eq. 11)**:
$$\mathcal{C}(M, I) = \frac{\|\boldsymbol{\mu}_{\mathrm{in}}^{ab} - \boldsymbol{\mu}_{\mathrm{out}}^{ab}\|_2}{\sqrt{2} \cdot 255}$$
Compares mean chromaticity (a*, b* channels in Lab space) inside vs outside the mask. Lesions (yellow/brown/red) should differ from healthy tissue (green). Normalized to [0, 1].

**Topological Reward**: $\log(1 + N(M))$ rewards multiple connected components but with diminishing returns (log prevents runaway fragmentation).

**Penalties**: $\mathcal{P}_{\mathrm{area}}$ (empty or too-large masks) and $\mathcal{P}_{\mathrm{noise}}$ (ultra-small dust components).

### Literature Support
- Berry et al. (2018) [berry2018color]: Color calibration motivates Lab-space comparison
- PlantCV (2025) [plantcv2025]: Phenotyping pipelines emphasize calibrated color analysis

### Connected Outputs
- `ablations_objective/`: Tests each term's contribution (Table I)
- Score column in `opt_summary_local.csv`
- Trace CSVs in `objective_surface_traces_representative_fullbudget_9leaf/`

---

## 5. Stochastic Hill Climbing (Algorithm 1)

### Plain Language
The optimizer works in two phases: First, it randomly samples many parameter combinations (like throwing darts at a dartboard). Then, it takes the best dart and makes small adjustments around it (like fine-tuning a dial). The random phase explores broadly; the refinement phase zooms in on promising regions.

### Technical Description
- **Phase 1 (Random)**: Sample $K=64$ candidates from Uniform($\Theta$), evaluate each, keep best.
- **Phase 2 (Refine)**: Take top-4 random candidates, apply $R=12$ perturbation steps per seed with cooling schedule $\sigma_j = \sigma_0 \cdot 0.85^j$.
- **Per image**: Total ~112 evaluations (64 random + 48 refine)
- **Gradient-free**: Required because the pipeline involves thresholding, connected-component filtering, and iterative snake evolution — all non-differentiable operations.

### Connected Outputs
- `ablations_search/`: Tests search strategy variants (Table II)
- `objective_surface_traces_representative_fullbudget_9leaf/`: Traces showing random vs refine phases
- Fig 12: Convergence trajectory showing phase transition

---

## 6. Trace-Based Objective Surface Analysis

### Plain Language
By logging every candidate the optimizer evaluates (not just the final answer), the paper reconstructs what the objective landscape looks like. For any pair of parameters, you can plot a 3D surface showing how the score changes. This reveals whether the landscape has sharp peaks (easy to optimize) or is flat (hard to improve upon).

### Technical Description
The candidate trace set $\mathcal{T}_i = \{(\theta_k, s_k, t_k) : k = 1, \ldots, K_i\}$ records all evaluations for image $i$. Projected pairwise surfaces are:
$$\mathcal{S}_{p,q}(I_i) = \{(\theta_k[p], \theta_k[q], s_k) : (\theta_k, s_k, t_k) \in \mathcal{T}_i\}$$

This is a scatter-surface in $\mathbb{R}^3$ showing objective geometry projected onto two parameter axes. The paper distinguishes this from post-hoc approximations (PCA/UMAP of parameter vectors) — these surfaces reflect the *true explored landscape*.

### Key Findings
- High-score leaves: sharp optima with distinct peaks, refine gain +5.77 avg
- Low-score leaves: flat surfaces, refine gain +0.56 avg
- Dominant pair: $T \times \beta$ (energy threshold × balloon force) in 7/9 cases
- Multi-basin hints in some high-score leaves

### Connected Outputs
- `objective_surface_traces_representative_fullbudget_9leaf/`: Definitive 9-leaf study
- Figs 9-11 (surface plots), Fig 12 (trajectory), Appendix B (additional trajectories)

---

## 7. Parameter-Cloud Geometry and Manifold Analysis

### Plain Language
The paper asks: when the optimizer finds solutions for different images, do those solutions cluster in some organized way, or are they scattered randomly? Using PCA (a dimension-reduction technique), they find that solutions live on a structured low-dimensional surface (manifold). Moreover, the direction of the shift from global (whole-leaf) to local (patch-level) parameters partially predicts whether local re-optimization helps or hurts.

### Technical Description
Given $N$ patches with global parameters $\{\theta_j^g\}$ and local parameters $\{\theta_j^\ell\}$:
- Parameter shift: $\delta_j = \theta_j^\ell - \theta_j^g \in \mathbb{R}^7$
- Centroid shift: $\Delta\bar\theta = \bar\theta^\ell - \bar\theta^g$, magnitude 1.616 standardized units
- PCA: 2 PCs capture 46.5% of variance
- k-NN test: $k=7$ neighbors, agreement 0.534 for $\Delta D$ groups (vs 0.33 random)
- Patch type (interior/boundary): k-NN agreement 0.488 (near random 0.50)

### Solid-3D Geometry
- 2-sigma covariance ellipsoids fitted in PCA space
- Convex hulls for global/local clouds
- Positive-$\Delta D$ and negative-$\Delta D$ clouds show partial separation

### Literature Support
- Jolliffe & Cadima (2016) [jolliffe2016pca]: PCA review
- van der Maaten & Hinton (2008) [vandermaaten2008tsne]: t-SNE
- McInnes et al. (2018) [mcinnes2018umap]: UMAP

### Connected Outputs
- `parameter_shift_manifold_geometry/`: Comprehensive manifold analysis (30 plots, 3 reports)
- `parameter_shift_3d_analysis_solid/`: Solid-3D geometry (25 files)
- `optimization_manifold_and_solid3d_final/`: Merged final figures
- Figs 13-15, Appendix C

---

## 8. Patch Morphology Transfer Analysis

### Plain Language
The paper tests whether parameters optimized on a whole leaf still work when applied to small patches (512×512 crops). It finds that while overlap metrics (Dice) are reasonable (median 0.78), the patches systematically underestimate lesion area (0.026 vs 0.169). Local re-optimization helps with lesion count accuracy (35% error reduction) but not area.

### Technical Description
150 patches from 15 leaves (75 interior, 75 boundary). Two parameter sets tested:
- Global: $\theta^*$ from full-leaf optimization
- Local: $\theta_{\mathrm{local}}^*$ from per-patch re-optimization

Metrics: Dice, IoU, area fraction, connected-component count.

### Connected Outputs
- `patch_local_opt_v2.csv`: Raw patch-level results
- `patch_transfer_morphology_eval.csv`: Morphology metrics
- `patch_transfer_morphology_plots/`: Visualization (Figs 17-19)
- Table III in manuscript

---

## 9. Teacher-Student Distillation

### Plain Language
The physics optimizer is slow (~4 min/image). To make inference fast, the paper trains a small neural network (UNet student) to mimic the optimizer's outputs (teacher pseudo-labels). The student learns from 4,025 patches and can then segment new images in milliseconds.

### Technical Description
- Patchification: 512×512, stride 448, reflection padding, zero mask padding
- 4,599 patches → 4,025 after empty-mask subsampling (keep prob 0.35)
- Architecture: 3-level UNet (32-64-128 channels), 2 convs/block
- Loss: $\frac{1}{2}\mathrm{BCEWithLogits} + \frac{1}{2}\mathrm{DiceLoss}$
- Training: 40 epochs, Adam, lr $10^{-4}$, 85/15 split
- Inference: overlap-averaged stitching, threshold 0.30

### Connected Outputs
- `teacher_student_metrics.csv`, `teacher_student_metrics_roi_thr0p30.csv`
- `teacher_student_figures/`
- `threshold_sweep.csv`, `threshold_sweep_plot.png`
- `failures_dice_lt_0p3.csv`, `teacher_empty_cases.csv`
- `patches/` (training data)

---

## 10. Two-Stage Nested Lesion Inference (Algorithm 2)

### Plain Language
Some lesions have an outer halo (chlorosis) and an inner core (necrosis). The paper proposes a two-stage approach: first find the outer boundary, then search for the inner core within it. This produces nested masks enabling computation of biologically meaningful traits like necrosis ratio.

### Technical Description
- Stage A: Optimize on full image → $M_{\mathrm{outer}}$ (green overlay)
- Stage B: Restrict to $M_{\mathrm{outer}}$, increase threshold, re-optimize → $M_{\mathrm{inner}}$ (magenta overlay)
- Necrosis ratio: $|M_{\mathrm{inner}}| / (|M_{\mathrm{outer}}| + \varepsilon)$

### Connected Outputs
- Inner-mask definition is what causes teacher-empty disagreements (12 cases)
- The "teacher" in teacher-student evaluation uses the inner-mask definition
