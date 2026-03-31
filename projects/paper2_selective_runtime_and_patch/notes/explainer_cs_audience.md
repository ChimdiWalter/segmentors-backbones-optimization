# Paper 2 Deep Explainer — CS / ML Audience

**Paper:** *Black-Box Selective Semi-Amortized Inference for Runtime–Quality Tradeoffs in Physics-Based Lesion Phenotyping*
**Authors:** Chimdi Walter Ndubuisi, Toni Kazic
**Manuscript:** `manuscript/main_paper2.tex` → compiled PDF at `manuscript/main_paper2.pdf` / `pdf/paper2_final.pdf`

---

## 1. The Problem in One Sentence

Paper 1 produces high-quality per-image parameter vectors θ* for a physics PDE model — but costs ~260 seconds per leaf. Paper 2 asks: given a learned approximator `f_φ(x)` that predicts θ* cheaply, and an uncertainty signal `u(x)`, how should you allocate compute across images to navigate the runtime–quality Pareto frontier?

This is a **selective semi-amortized inference** problem: route cheap images to direct prediction, medium-confidence images to short local refinement, and hard images to the full physics optimizer.

---

## 2. Technical Setup

### 2.1 The Oracle
The **full oracle** `θ*_leaf(x)` is the output of the Paper 1 Navier-Stokes active contour optimizer (random-restart hill climbing) on full leaf `x`. It takes 260.49 s on average. The oracle produces a 6-dimensional parameter vector; evaluation is against normalized mean absolute error (nMAE) or raw MAE.

### 2.2 Inference Branches
Three branches with different cost–quality tradeoffs:

| Branch | Description | Runtime (s) |
|--------|-------------|------------|
| **Direct** | `f_φ(x)` — learned predictor on full-leaf input | 2.98 |
| **Refine (tiny/small/medium)** | Short perturbation search around `f_φ(x)` output | 11–29 |
| **Fallback** | Full physics optimizer from scratch | 260.49 |

### 2.3 Routing Policy
A threshold-based policy on leaf-level uncertainty `u(x)`:
```
π(u) = { direct    if u ≤ τ_lo
        { refine    if τ_lo < u ≤ τ_hi
        { fallback  if u > τ_hi
```
Deployment problem: find `(τ_lo, τ_hi)` such that expected quality is acceptable for the runtime budget.

### 2.4 Dual Oracle — The Key Complication for Patchwise Inference
A patchified branch processes patches `{p_k}` with the same predictor `f_φ`, producing per-patch predictions aggregated by operator `A`:
```
θ̂_patch(x) = A({ (θ̂_k, u_k) }_{k=1}^K )
```

But there are **two distinct oracle targets**:
- **Full-leaf oracle:** `θ*_leaf(x)` = optimizer run on the whole leaf
- **Aggregated patch oracle:** `θ̄*_patch` = aggregate of locally optimized patch parameters

Their discrepancy, the **oracle gap** `Δ_oracle = ||θ*_leaf − θ̄*_patch||`, is **0.327 nMAE** across the 35-leaf matched study. This is large enough that every patchwise result must be co-reported with this gap — patch inference is solving a related but not identical problem.

---

## 3. Experimental Substrates

Two distinct experimental substrates (not to be compared numerically without care):

| Substrate | Size | Primary metric | Use |
|-----------|------|---------------|-----|
| Matched 35-leaf comparison | 35 leaves | nMAE, Wilcoxon, bootstrap CI | Full-leaf vs patch comparison, oracle gap, routing frontier, patch ablation |
| Selective full study | 174 leaves (146 ID + 28 from `18.7` collection) | MAE, calibration, risk–coverage, R² | Threshold grid, uncertainty quality, learning curves, phenotype/morphology |

Source CSVs:
- Matched comparison: `experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/evaluation/E1_matched_comparison.csv`
- Routing tables: `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/tables/runtime_comparison.csv`
- Calibration bins: `..._v3/tables/calibration.csv`
- OOD robustness: `..._v3/tables/ood_robustness.csv`

---

## 4. Main Results: Full-Leaf vs Patchwise (35-leaf matched study)

**Table I (`tab:mainB`) — Matched comparison, lower nMAE is better:**

| Aggregation | Full-leaf nMAE | Patch nMAE | Oracle gap |
|------------|---------------|-----------|-----------|
| Mean | 1.154 | **0.906** | 0.327 |
| Uncertainty-weighted | 1.154 | 0.911 | 0.327 |
| Selective mixture | 1.154 | 0.921 | 0.327 |
| Robust median | 1.154 | 0.962 | 0.327 |
| Top-k confident | 1.154 | 1.037 | 0.327 |

**Statistical tests (Table II, `tab:statsB`):**
- Mean aggregation: mean improvement 0.248, 95% CI [0.097, 0.405], Wilcoxon p=0.003
- Uncertainty-weighted: improvement 0.243, p=0.005
- Robust median: improvement 0.192, p=0.049 (marginally significant)
- Top-k confident: improvement 0.117, CI includes 0, p=0.238 → **not significant**

**Reading:** Patchwise inference with mean/uncertainty-weighted aggregation significantly outperforms direct full-leaf prediction — but under a dual-oracle setting. The 0.327 oracle gap means a perfect patch predictor would still show worse-than-oracle performance when scored against the full-leaf target.

**Figures:** `figures/B_fig01_pipeline_schematic.pdf`, `B_fig02_dataset_split.pdf`, `B_fig03_main_comparison.pdf`, `B_fig05_oracle_gap.pdf`

---

## 5. Routing: Runtime–Quality Operating Points

**Table III (`tab:routingB`) — Three audited routing policies on 35-leaf matched study:**

| Policy (τ_lo, τ_hi) | Runtime (s) | nMAE | Fallback fraction | Speedup vs full |
|--------------------|------------|------|------------------|----------------|
| Quality-favoring (0.3, 0.7) | 74.34 | **0.556** | 0.257 | 3.50× |
| Balanced (0.3, 0.9) | 60.48 | 0.577 | 0.200 | 4.31× |
| Max-speedup audited (0.6, 0.9) | 55.77 | 0.819 | 0.200 | 4.67× |

**Reference modes (Table IV, `tab:runtimesB`):**
- Direct: 2.98 s (nMAE 1.154) — fastest, worst
- Full optimizer: 260.49 s (nMAE ≈0.026) — slowest, best

**Broader study (174 leaves):** Thresholded routing reduces mean MAE from 0.1033 (direct) to **0.0715** at best threshold pair (0.5, 0.7).

**Contribution framing:** The paper reports a Pareto *frontier* of operating points, not a single preferred policy. The runtime–quality tradeoff is made explicit and transparent.

**Figures:** `B_fig04_runtime_quality_pareto.pdf`, `B_fig38_gapfill_runtime_frontier.png`, `B_fig08_refinement_curves.pdf`, `B_fig19_selective_pareto_real.pdf`

---

## 6. Calibration and Risk–Coverage

**Calibration bins (Table V, `tab:calibB`) — selective full study:**

| Bin | n | Mean uncertainty | Mean error |
|-----|---|-----------------|-----------|
| 0 (lowest) | 35 | 0.0177 | 0.0082 |
| 1 | 35 | 0.0548 | 0.0266 |
| 2 | 34 | 0.1059 | 0.0624 |
| 3 | 35 | 0.1438 | 0.0975 |
| 4 (highest) | 35 | 0.1877 | 0.1157 |

**Key claim and limit:** Uncertainty monotonically increases with error — directionally correct and sufficient for routing decisions under Eq. (routing). Not strong enough for formal coverage guarantees or conformal prediction certificates. The paper is explicit about this.

**Figures:** `B_fig09_risk_coverage.pdf`, `B_fig10_calibration.pdf`, `B_fig27_selective_calibration.pdf`, `B_fig28_selective_risk_coverage.pdf`

---

## 7. Branch Anatomy: Patch Extraction, Aggregation, Heterogeneity, Parameter Difficulty

**Patch extraction ablation (Table VI, `tab:patchB`):**

| Config (size/stride) | Total patches | Mean patches/leaf | Mean heterogeneity |
|---------------------|-------------|------------------|-------------------|
| 224/224 (non-overlapping) | 13,192 | 75.82 | 1.299 |
| **224/112 (50% overlap)** | **47,916** | **275.38** | **1.903** |
| 336/168 | 19,228 | 110.51 | 1.291 |

The 224/112 setting gives highest heterogeneity (1.903) — captures more local variation but at 3.6× patch count of non-overlapping. This matters because heterogeneous leaves need multiple distinct local predictions; a single aggregated mean is then meaningfully better than a whole-leaf down-sampled prediction.

**Key parameter difficulty finding:** `energy_threshold` remains the hardest parameter to predict from the patch branch. This aligns with Paper 1's finding that T is the dominant parameter in the optimization landscape (highest R² in T-β surface regression).

**Direct model ablations (Table VII, `tab:ablationB`):**

| Variant | Test MAE |
|---------|---------|
| ResNet-18 ridge | 0.0965 |
| DINOv2 ridge | 0.1240 |
| kNN-10 | **0.0789** |
| MLP-small | 0.4129 |
| Best selective threshold (0.5, 0.7) | **0.0715** |

**Surprising finding:** kNN-10 (nearest-neighbor regression) outperforms ridge regression on ResNet features in this audited split. DINOv2 — despite being a stronger visual backbone — performs worse here. This is consistent with the paper's honest reporting: larger capacity doesn't always win on small scientific imaging datasets.

**Figures:** `B_fig11_aggregation_comparison.pdf`, `B_fig06_patch_ablation.pdf`, `B_fig12_heterogeneity.pdf`, `B_fig14_parameter_difficulty.pdf`

---

## 8. Morphology Fidelity and Phenotype Consistency

**Morphology fidelity (Table VIII, `tab:morphB`):**

| Refinement budget | Contour count error | Area-frac error | Mean score |
|-----------------|-------------------|----------------|-----------|
| Direct | 41.34 | 0.0233 | 17.111 |
| Tiny | 40.60 | 0.0255 | 17.791 |
| Small | 46.69 | 0.0282 | 17.854 |
| **Medium** | **39.46** | **0.0229** | **18.243** |

Medium refinement dominates on both error metrics and moves mean score closest to oracle (18.053). The non-monotone pattern (small refinement slightly worse than direct) is expected variance over 35 leaves, not a bug.

**Phenotype consistency (Table IX, `tab:phenoB`) — mean PCA distance in trait space:**

| Method | Mean PCA dist |
|--------|--------------|
| B0 mean (simple mean) | **0.245** |
| B0b kNN | 0.264 |
| B1 ridge ResNet | 0.260 |
| B1 ridge DINO | 0.380 |
| B2 MLP DINO | 0.444 |
| B3 ensemble | 0.279 |

Simple mean aggregation has the **best phenotype consistency** despite not being the lowest-MAE method. Complex models (DINOv2 MLP) are worst on phenotype structure. This is a key finding: parameter MAE and biological phenotype preservation are not the same objective.

**Figures:** `B_fig25_phenotype_pca.pdf`, `B_fig26_per_param_scatter.pdf`

---

## 9. Robustness and Cross-Collection

**OOD robustness (Table X, `tab:oodB`) — ID vs `18.7` collection:**

| Method | ID MAE | 18.7 MAE |
|--------|--------|---------|
| B0 mean | 0.0854 | 0.0765 |
| B0b kNN | 0.0387 | **0.0176** |
| B1 ridge ResNet | 0.0447 | **0.0183** |
| B1 ridge DINO | 0.0536 | 0.0258 |
| B2 MLP DINO | 0.1501 | 0.1220 |

**Careful interpretation:** Several methods improve on `18.7`. This is NOT clean domain generalization — the paper explicitly warns this. The `18.7` collection has different sample composition (not necessarily harder), so lower MAE there reflects composition effects, not genuine transfer. Validate policies on your actual target collection.

**Figures:** `B_fig07_backbone_comparison.pdf`, `B_fig23_ood_robustness.pdf`

---

## 10. Difficulty Stratification

**Stratification (Table XI, `tab:strataB`):**

| Condition | Best simple baseline MAE | ResNet ridge MAE |
|-----------|--------------------------|-----------------|
| Low score tier | 0.0925 | 0.1001 |
| High score tier | 0.0712 | 0.1024 |
| Extreme parameters | 0.0870 | 0.1016 |
| Typical parameters | **0.0644** | **0.0889** |

Extreme parameter regimes remain harder for the ResNet ridge model (0.1016 vs 0.0889 on typical). Simple baselines are more robust across strata. This argues for uncertainty-based routing rather than uniform direct prediction.

---

## 11. Selective Full Study: Learning Curves and Non-Monotonicity

The learning curves (Figure `B_fig22_learning_curves.pdf`) at 25%, 50%, 75%, 100% of training data are intentionally kept in the paper and are **non-monotone**. This is treated as evidence, not noise to hide — it indicates that:
- The problem remains sensitive to split composition
- Backbone mismatch with the optimizer's feature distribution affects generalization
- Target noise (the oracle's own variance) creates irreducible difficulty for amortization

The paper is explicit that non-monotone learning curves weaken any claim that adding data straightforwardly improves performance.

---

## 12. What the Evidence Does and Does Not Support

**Supported:**
- Patchwise inference can outperform direct full-leaf prediction (p<0.01 for mean aggregation)
- Selective routing provides a meaningful runtime–quality Pareto frontier vs. the two extremes
- Uncertainty is monotonically aligned with error → sufficient for routing thresholds
- Morphology and phenotype structure are better preserved by some methods than others

**Not supported:**
- Patchwise inference has solved the full-leaf oracle problem (oracle gap is 0.327 nMAE)
- Uncertainty is strong enough for formal coverage certificates or PAC guarantees
- Cross-collection improvements represent genuine domain generalization
- Any single method dominates across all metrics

---

## 13. Figure Inventory and Source Paths

All 39 figures for Paper 2 are in `figures/` (paths from `notes/figure_provenance.csv`):

| Figure group | Representative files | Source |
|-------------|---------------------|--------|
| Setup & main comparison | `B_fig01–05` | `Paper_5/manuscript_blackbox_runtime/figs_from_outputs/` |
| Runtime–quality frontiers | `B_fig04, B_fig08, B_fig19, B_fig38` | above + `gapfill/` |
| Calibration & risk–coverage | `B_fig09, B_fig10, B_fig27, B_fig28` | above |
| Branch anatomy | `B_fig06, B_fig11, B_fig12, B_fig14` | above |
| Selective study | `B_fig20, B_fig21, B_fig22, B_fig24` | above |
| Phenotype & per-param | `B_fig25, B_fig26` | above |
| Leaf comparison & routing | `B_fig17, B_fig34` | above |
| Failures | `B_fig18, B_fig35, B_fig36` | above |
| Robustness | `B_fig07, B_fig23` | above |
| Geometry (appendix) | `B_fig15, B_fig16, B_fig29–33, B_fig37` | above |
| Gapfill overlay | `B_fig39, B_figA1` | `gapfill/` |

---

## 14. How Experiments Were Run

- **Full-leaf predictor training:** feature extraction (ResNet-18 or DINOv2 embeddings) on full-leaf images + ridge/kNN/MLP head trained on `θ*` from Paper 1 optimizer
- **Patchwise predictor:** same `f_φ` applied to 224×224 patches; per-patch uncertainty from prediction variance or dropout
- **Refinement:** short perturbation search (same random-restart HC code as Paper 1, truncated to tiny/small/medium budgets) seeded from `f_φ(x)` output
- **Routing evaluation:** threshold grid over `(τ_lo, τ_hi)` pairs; nMAE/MAE/runtime recorded per policy
- **Calibration:** bin uncertainty predictions into 5 equal-size bins; compare mean uncertainty to mean observed error per bin
- **All outputs in:** `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/` and `integrated_full_vs_patch_leaves_18p7_v1/`
