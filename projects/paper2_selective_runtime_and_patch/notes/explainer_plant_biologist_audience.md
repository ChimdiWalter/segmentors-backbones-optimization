# Paper 2 Deep Explainer — Plant Biologist / Phenotyping Audience

**Paper:** *Black-Box Selective Semi-Amortized Inference for Runtime–Quality Tradeoffs in Physics-Based Lesion Phenotyping*
**Authors:** Chimdi Walter Ndubuisi, Toni Kazic
**Full manuscript PDF:** `pdf/paper2_final.pdf`

---

## 1. Why This Paper Exists: The Deployment Bottleneck

Paper 1 solved the hard problem of measuring lesions without manually drawn outlines. The physics-based optimizer it developed finds the right settings for each leaf automatically and produces reliable lesion measurements.

The catch: **it takes about 260 seconds (~4.3 minutes) per leaf.** For a small-scale experiment with 174 leaves, that is manageable (~12.5 hours total). For a breeding trial with 5,000–50,000 leaves (realistic for high-throughput phenotyping), it becomes a bottleneck of 6–144 days of compute time.

Paper 2 asks: **Can we get measurements that are almost as good, but much faster?** And more importantly: **How fast can we go before measurement accuracy degrades too much to be scientifically useful?**

---

## 2. The Core Idea: Three Speed Levels

The paper proposes routing each leaf image to one of three processing tiers:

| Tier | What it does | Time |
|------|-------------|------|
| **Direct prediction** | A neural network makes an instant prediction based on visual features extracted from the image | ~3 seconds |
| **Short refinement** | Take the network's prediction and run a brief search around it to improve accuracy | 11–29 seconds |
| **Full physics optimizer** | Run the complete Paper 1 optimizer from scratch (Paper 1's algorithm) | ~260 seconds |

The routing decision is based on a **confidence score (uncertainty)**. For a leaf where the network is very confident, use the instant prediction. For a confusing leaf, spend the extra time running the full optimizer.

This is analogous to a doctor's triage system: easy cases go to a nurse practitioner, moderate cases get a standard consult, and ambiguous cases get a specialist. The key is being able to tell which is which.

---

## 3. What Is Being Predicted?

The target is the **7-parameter physics vector (θ) for each leaf** — the same six physics parameters from Paper 1 (viscosity, tension, rigidity, inflation force, elastic coupling, energy threshold) plus a diffusion rate. These parameters are not biologically interpretable directly, but they determine the shape and extent of the lesion mask, which in turn determines:

- **Lesion area fraction** (disease severity)
- **Lesion count** (disease distribution pattern)
- **Necrosis ratio** (fraction of lesion that is dead tissue vs. yellowing)

So predicting θ accurately = producing the same lesion measurements as running the full optimizer.

---

## 4. The Patch vs. Full-Leaf Question

A leaf is ~3900 × 1340 pixels — too large to feed directly into a neural network without shrinking (and losing local detail). The paper compares two strategies:

### Strategy A: Full-leaf (downsampled)
Resize the whole leaf to a fixed size and let the network see the whole leaf at once. Fast and simple, but loses local texture details in fine chlorotic halos.

### Strategy B: Patchwise
Cut the leaf into overlapping 224×224 pixel windows, predict physics parameters for each patch independently, then average the predictions back into one leaf-level estimate.

**Main result (35-leaf comparison):**

| Strategy | Normalized Error (nMAE) | Lower is better |
|----------|------------------------|-----------------|
| Full-leaf prediction | 1.154 | — |
| Patch mean aggregation | **0.906** | ✓ Best |
| Patch uncertainty-weighted | 0.911 | ✓ Significant improvement |
| Patch robust median | 0.962 | ✓ Marginal improvement |
| Patch top-k confident | 1.037 | ✗ Not significant |

**Patchwise processing with mean aggregation outperforms whole-leaf downsampling** (statistically significant, p=0.003). The reason makes biological sense: local lesion patches preserve the color and texture information that distinguishes necrotic spots from chlorotic halos, which gets blurred when the whole leaf is shrunk.

### Important caveat: The "oracle gap"
The patch approach is compared against a subtly different reference target than the full-leaf approach. When a leaf is cut into patches and each patch is optimized separately, the averaged result is not identical to the optimizer run on the whole leaf. This difference — the **oracle gap** — is 0.327 nMAE (about 28% of the full-leaf prediction error). This means even a perfect patchwise predictor would look imperfect when scored against full-leaf measurements. The paper is careful to report this distinction throughout.

---

## 5. How Well Does Routing Work? (Practical Numbers)

The paper tests specific routing threshold pairs (τ_lo, τ_hi) and reports the runtime–quality tradeoff for each:

| Routing policy | Processing time | Error (nMAE) | Speedup over full optimizer |
|---------------|----------------|-------------|---------------------------|
| Direct prediction only | 2.98 s | 1.154 | 87× but very inaccurate |
| Quality-favoring routing | **74.34 s** | **0.556** | 3.5× |
| Balanced routing | 60.48 s | 0.577 | 4.3× |
| Speed-favoring routing | 55.77 s | 0.819 | 4.7× |
| Full physics optimizer | 260.49 s | ~0.026 | 1× (baseline) |

**In plain language:** The quality-favoring policy takes about 74 seconds per leaf and has an error of 0.556 on the normalized scale — compared to the full optimizer's 260 seconds and error ≈0. That is a 3.5× speedup at the cost of increased (but still useful) measurement uncertainty.

For a 5,000-leaf experiment, this would reduce compute from ~15 days to ~4.3 days at quality-favoring settings, or ~3.2 days at speed-favoring settings.

In the broader 174-leaf study, selective routing reduced mean absolute error from 0.1033 (direct prediction) to **0.0715** at the best settings — a 31% improvement with moderate compute savings.

---

## 6. How Trustworthy Is the Confidence Score?

For routing to work, the network's self-reported confidence must actually track its accuracy. The paper tests this by sorting all 174 leaves into 5 uncertainty bins:

| Bin (1=most confident, 5=least) | Mean uncertainty score | Mean actual error |
|---------------------------------|----------------------|------------------|
| 1 (most confident) | 0.018 | **0.008** |
| 2 | 0.055 | 0.027 |
| 3 | 0.106 | 0.062 |
| 4 | 0.144 | 0.098 |
| 5 (least confident) | 0.188 | **0.116** |

The uncertainty signal correctly orders leaves from easiest to hardest. This is **directionally reliable** — enough to support routing decisions. However, it is **not perfectly calibrated**, meaning you cannot make exact statistical guarantees about how many leaves in any uncertainty bin will exceed a given error threshold.

**Practical implication:** The routing policy will generally allocate extra compute to the right leaves, but some high-uncertainty leaves may not actually need the full optimizer, and some low-uncertainty leaves may get under-processed. For screening purposes this is acceptable; for precise trait measurements, the safest choice remains the full optimizer on a random sample + uncertainty-based overflow.

---

## 7. Are the Measurements Biologically Meaningful?

A critical test: does routing preserve the structure that matters for plant science, not just raw parameter accuracy?

### 7.1 Morphology Fidelity

| Processing mode | Lesion count error | Area fraction error |
|----------------|-------------------|--------------------|
| Direct | 41.34 | 0.0233 |
| Medium refinement | **39.46** | **0.0229** |

Medium refinement slightly improves both morphological traits compared to direct prediction. The area fraction error of ~0.023 means lesion measurements are off by about ±2.3% of leaf area on average.

### 7.2 Phenotype Consistency (PCA distance — does grouping of genotypes/treatments hold up?)

| Method | Mean PCA distance in trait space |
|--------|----------------------------------|
| Simple mean aggregation | **0.245** (best) |
| kNN regression | 0.264 |
| ResNet + ridge regression | 0.260 |
| DINOv2 + MLP | 0.444 (worst) |

**Key finding:** The simplest aggregation method (just average the patch predictions) produces the best phenotype consistency. Complex deep-learning models produce higher raw parameter accuracy in some settings but disturb the biological structure of the measurements — genotypes that should cluster together get scattered in phenotype space.

**Practical implication:** If you are comparing disease severity across genotypes or time points, a simple mean-aggregation patchwise predictor is more reliable than more elaborate models, even if headline numbers suggest otherwise.

---

## 8. What Are the Limitations? (Honest Assessment)

The paper is deliberately transparent about what the results do and do not show:

**What it shows:**
- Patchwise processing can outperform full-leaf-downsampling (statistically confirmed)
- Routing across speed tiers provides a real range of runtime–quality operating points
- Confidence scores are directionally useful for routing decisions
- Morphology and phenotype structure are preserved to a useful degree

**What it does NOT show:**
- Patchwise processing is as good as running the full optimizer per patch (the oracle gap of 0.327 is too large)
- The confidence signal is precise enough for formal statistical guarantees
- Results generalize automatically to new image collections without revalidation

**Failure cases:**
- Leaves with **very low uncertainty scores** that are still measured inaccurately — these are cases where the network is confidently wrong (rare but exist)
- Leaves with **extreme lesion severity** (>30% area coverage) or very unusual morphology — these remain harder across all methods
- The learning curves for adding training data are non-monotone — adding more labeled examples doesn't always improve accuracy, likely because the optimizer's own noise is a fundamental limit

---

## 9. The Second Leaf Collection (18.7 Dataset)

The paper tests models trained on the primary soybean/maize dataset on a second leaf image collection. Results are mixed: some simple models actually perform *better* on the second collection, but this is not because the models generalized better — it's because the second collection happens to contain leaves that are easier for those models. The paper is careful to say: the second collection is a different substrate, not a harder one. You must validate routing policies on your actual target collection, not assume transfer.

---

## 10. Summary: What This Means for Your Phenotyping Pipeline

| Question | Answer from this paper |
|----------|----------------------|
| How much faster can I go than the full optimizer? | 3.5–4.7× speedup with moderate quality loss |
| Does patchwise processing help? | Yes, for mean/uncertainty-weighted aggregation (p<0.01) |
| Can I trust the uncertainty score to route hard leaves? | For routing decisions, yes. For formal guarantees, no. |
| Will routing preserve genotype differences? | Yes, especially with simple aggregation |
| Does this work on other leaf collections? | Validate on your specific collection; don't assume transfer |
| What are the key failure modes? | Extreme parameters, faint halos, high severity leaves |

---

## 11. Where to Find Everything

**Paper PDF:** `pdf/paper2_final.pdf` or `manuscript/main_paper2.pdf`

**All figures (39 total):** `figures/` directory — files named `B_fig01` through `B_fig39`

Key figures for biologists:
- **`B_fig01_pipeline_schematic.pdf`** — diagram of the three-branch system
- **`B_fig03_main_comparison.pdf`** — bar chart comparing full-leaf vs patchwise methods
- **`B_fig04_runtime_quality_pareto.pdf`** — the runtime vs. accuracy tradeoff curve
- **`B_fig17_leaf_comparison.png`** — leaf-by-leaf scatter comparing both branches
- **`B_fig25_phenotype_pca.pdf`** — phenotype space structure by method
- **`B_fig34_representative_leaf_routing.pdf`** — what routing decisions look like on real leaves
- **`B_fig35_failure_case_montage.pdf`** — example failure cases with explanations

**Raw data tables:**
- Routing results: `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/tables/`
- Matched comparison: `experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/evaluation/`
