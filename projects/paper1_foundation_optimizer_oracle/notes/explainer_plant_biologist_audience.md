# Paper 1 Deep Explainer — Plant Biologist / Phenotyping Audience

**Paper:** *Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours for Unsupervised High-Throughput Phenotyping*
**Authors:** Chimdi Walter Ndubuisi, Toni Kazic
**Full manuscript PDF:** `pdf/paper1_final.pdf`

---

## 1. What Problem Does This Paper Solve?

You grow hundreds of soybean (and maize) plants, inoculate them with a fungal pathogen, and photograph the leaves. Every leaf needs to be measured: *How large are the lesions? How many are there? What fraction of the leaf is necrotic? What fraction is chlorotic (yellowing)?* Doing this by eye for 174 (or eventually thousands of) images is slow, subjective, and inconsistent.

The standard computer vision solution — training a deep-learning segmentation model — requires someone to manually draw precise outlines around every lesion in dozens of images first. That annotation effort can take months and requires an expert who can distinguish a faint chlorotic halo from a leaf vein shadow.

**This paper's solution:** Build a physics-based algorithm that can figure out the right settings for itself *per image*, without needing any human-drawn outlines. It produces masks automatically, then uses those automatic masks to teach a fast neural network.

---

## 2. The Dataset

- **174 images** of soybean and maize leaves from a greenhouse experiment
- Multiple genotypes inoculated with a necrotrophic fungal pathogen (causes both necrotic spots and chlorotic halos)
- Images stored as **16-bit TIFFs** (high dynamic range) at ~3900 × 1340 pixels
- Standardized diffuse lighting; white-balance normalized
- No pixel-level ground truth annotations for the full set; lesions range from tiny discrete necrotic spots to large coalescing regions covering >14% of the leaf

---

## 3. The Physics of the Segmentation Model (Plain Language)

Think of the algorithm as deploying an **elastic rubber band** around where it thinks the lesion is, then letting physics pull the band to fit the boundary exactly. The rubber band has three physical properties:

| Property | What it controls | Biological effect |
|----------|-----------------|-------------------|
| **Tension (α)** | How much the band resists stretching | High tension → band shrinks toward center; needed to avoid leaking into healthy tissue |
| **Rigidity (β)** | How much the band resists bending | High rigidity → smooth boundary; important for coalescing lesions |
| **Inflation force (γ)** | Outward pressure pushing the band outward | Higher inflation → band expands into faint chlorotic halos |

Before the rubber band is released, the algorithm pre-processes the image using a mathematical smoothing procedure (inspired by how viscous fluids flow — hence the Navier-Stokes reference). This produces an **energy map**: a grey-scale image where lesion boundaries glow bright and healthy tissue stays dark. The rubber band is attracted to the bright edges.

A seventh parameter, the **energy threshold (T)**, controls which edges are bright enough to count. A low threshold detects faint chlorotic halos; a high threshold detects only sharp necrotic borders.

---

## 4. Why Per-Image Optimization?

A single set of physical parameters does not work for all leaves. Consider:

- A chlorotic halo on a pale background needs a low energy threshold and high inflation
- A crisp necrotic spot on a dark leaf needs a high threshold and low inflation
- Coalescing lesions need high rigidity to prevent the band from fragmenting into many small blobs

The algorithm treats finding the right parameter set for each image as a **search problem**: try many combinations, score each one, keep the best.

### How the search works (without maths)
1. **Random phase:** Try 64 random combinations of the 6 parameters (on a slightly shrunk copy of the image to save time)
2. **Refinement phase:** Take the top 4 combinations and try small variations around each, gradually narrowing the search radius over 12 steps
3. Return the best-scoring parameter combination

The whole search takes about 4000–5000 seconds (~70–80 minutes) per image running on a CPU. A watchdog timer cuts it off and uses the best-so-far if it runs too long.

---

## 5. How Does the Algorithm Know a Segmentation Is Good Without Ground Truth?

This is the key innovation. The score that guides the search has four components:

### 5.1 Gradient Alignment
*"Are the mask edges sitting on real image edges?"*
It measures whether the boundary of the proposed lesion mask coincides with regions where the image brightness changes sharply. A mask that runs through the middle of a green leaf scores poorly; one that hugs a necrotic border scores well.

### 5.2 Color Separability (in Lab color space)
*"Is the inside of the mask a different color from the outside?"*
The algorithm converts the image to the CIE Lab color space (where the `a*` and `b*` channels capture redness/greenness and yellowness/blueness). A good lesion mask should have a distinctly different average color inside (brown/yellow necrosis or chlorosis) vs. outside (green healthy tissue).

### 5.3 Lesion Multiplicity Reward
*"Are there multiple distinct lesion blobs?"*
Fungal infections typically produce multiple spots. The score rewards masks with multiple connected components (lesion islands), but uses a log scale so it doesn't reward breaking one lesion into thousands of dust particles.

### 5.4 Penalty Terms
- Penalizes masks that are completely empty (no lesion found)
- Penalizes masks that cover more than 45% of the leaf (probably a segmentation failure)
- Penalizes tiny components below ~2600 pixels (dust, sensor noise, leaf hairs)

---

## 6. What the Ablation Studies Show

The paper ran the same 30 images through versions of the algorithm where one scoring component was removed at a time. Results (Table I in the paper):

| What was removed | Effect |
|-----------------|--------|
| Lesion count reward | Score collapses from 14.80 → 3.68 median; masks become trivial |
| Gradient alignment | Score drops from 14.80 → 9.71; boundaries become less precise |
| Color separability | Modest effect on score (14.80 → 14.74) but masks are larger/blurrier |
| Area penalty removed | Score stays high but mean lesion count jumps (131 → 131, stable) |
| Noise penalty removed | Lesion count inflates badly (134 → 188 median); speckle noise fragments masks |

**Biological interpretation:** The topology reward (lesion count) is the single most important component — it prevents the algorithm from giving up and producing empty masks when lesions are faint. Without it, the optimizer consistently finds a degenerate solution (no lesion = no measurement error).

---

## 7. Distilling the Optimizer into a Fast Neural Network

Running the full optimizer on every new image takes ~70 minutes. For a breeding trial with 10,000 leaves, that is impractical.

**Solution:** Train a neural network (UNet — a standard encoder-decoder architecture for image segmentation) on the optimizer's masks. The optimizer's outputs become "teacher labels"; the UNet is the "student." After training, the student can segment a new image in seconds.

Training details:
- The 174 full-resolution leaf images are cut into 512×512 pixel patches (4025 patches after removing mostly-empty ones)
- Training for 40 epochs on a standard GPU
- Patches are stitched back together after inference, with overlapping regions averaged

**Student accuracy (mean across all 174 images):**
- Mean Dice coefficient: 0.598 (range 0–0.84+; median 0.682)
- Mean IoU: 0.457

On images where the teacher found a non-empty mask (162 of 174):
- Mean Dice: 0.642; Median Dice: 0.686

**What Dice means for you:** A Dice of 0.682 means the student's mask and the teacher's mask overlap at about 68%. For high-throughput phenotyping where you need *relative* differences between genotypes, this level of agreement is useful. For precise absolute area measurements, the limitations below are important.

---

## 8. Two-Stage Inference: Necrotic Core + Chlorotic Halo

Many lesions have **two zones**:
- An outer chlorotic halo (yellowing around the dead tissue)
- An inner necrotic core (dead, dark tissue)

The paper describes how to run the algorithm twice:
1. First pass finds the outer halo boundary
2. Second pass restricts the search to *inside* the halo and finds the necrotic core using a tighter threshold

This produces nested masks rendered as:
- **Green overlay** = outer chlorotic halo
- **Magenta overlay** = inner necrotic core

The **necrosis ratio** = (necrotic core area) / (total lesion area) is then a quantitative trait you can use to rank genotypes for pathogen resistance, distinguish disease progression stages, or compare treatments.

---

## 9. Patch-Level Morphology: Important Caveats for Trait Extraction

The paper explicitly tests whether traits extracted from the algorithm are faithful at the patch (local crop) level. Key finding from Table III:

| Measurement | Algorithm output | True (teacher) |
|------------|-----------------|----------------|
| Lesion area fraction | 0.026 (2.6%) | 0.169 (16.9%) |
| Lesion count (components) | 39.4 (global params) | 23.9 |

**Why the area underestimate?** The algorithm uses a strict two-stage (inner/outer) mask definition. Peripheral necrotic tissue outside the inner core is not counted. This is a semantic choice, not a calibration failure.

**Why the count overestimate?** Cutting a full leaf into patches truncates lesion context at patch boundaries, causing some lesions to appear fragmented (one lesion split across two patches = two components). Local re-optimization reduces this by 35% (count error 19.1 → 12.4).

**Bottom line for phenotyping:** Use Dice as a sanity check, but extract *area fraction* and *lesion count* as the primary agronomic traits, and be aware they are measured relative to the inner-mask definition.

---

## 10. The Objective-Surface Analysis (What It Tells a Biologist)

The paper re-ran the optimizer on 9 representative leaves (3 high-scoring, 3 medium, 3 low) while logging every candidate evaluation. This allowed visualization of the "search landscape."

**Key finding with biological meaning:**
- For **high-scoring leaves** (clear necrotic lesions on good-contrast backgrounds): the landscape has a sharp peak — there is a clearly "correct" parameter setting, and the algorithm reliably finds it
- For **low-scoring leaves** (faint halos, complex venation, glare): the landscape is flat — many parameter settings give similarly mediocre results, and the algorithm can't do much better because the image itself lacks strong discriminating information

The dominant parameter pair driving the landscape shape is *energy threshold T* vs. *balloon force γ*. This makes biological sense: whether you detect a faint halo (low T + high γ) vs. a sharp necrotic spot (high T + low γ) is essentially a choice between two different biological phenotype definitions.

Figures are in `figures/9leaf/`: `surface_DSC_0198_...png` (high-score), `surface_DSC_0379_...png` (median), `surface_DSC_0261_...png` (low-score).

---

## 11. Summary of All Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Dataset size | 174 16-bit TIFF leaf images | `experiments/exp_v1/` |
| Typical resolution | ~3900 × 1340 px | Manuscript Section 5.1 |
| Optimizer runtime (median) | ~4047–5428 s (~70–90 min) per image | Table II; `figures/exp_v1/optimizer_elapsed_seconds_hist.png` |
| Heuristic objective score (median, full obj.) | 14.80–14.95 | Tables I & II |
| UNet student, mean Dice (all 174) | 0.598 ± 0.226 | Section 6.1 |
| UNet student, median Dice (teacher-nonempty) | 0.686 | Section 6.1 |
| Area underestimate (patch, inner mask) | 0.026 vs. 0.169 (teacher) | Table III |
| Count error reduction with local params | 35% (19.1 → 12.4) | Table III |
| PCA centroid shift (global→local params) | 1.616 std. units | Section 7.1 |
| k-NN ΔD agreement | 0.534 vs. 0.33 baseline | Section 7.4 |
| Disagreement cases (teacher-empty / student-not) | 12 / 174 | Section 6.1 |

---

## 12. How to Run the Code

The core pipeline lives in the parent `segmentors_backbones` repository. Key entry points:
- **Main optimizer run:** `dataset_segmentation/run.py` (or `run_1.py`, `run_2.py` for variant versions)
- **Dataset/patch pipeline:** `dataset_segmentation/dataset.py`
- **Evaluation:** `dataset_segmentation/eval.py`
- **Figure generation for 9-leaf trace analysis:** scripts under the project or in `dataset_segmentation/` using logged trace CSVs
- **Manifold/PCA figures:** `dataset_segmentation/Manifold_experiments/` and `dataset_segmentation/Manifold_clustering/`

Experiment outputs are in `experiments/exp_v1/outputs/`. All figures are pre-built in `figures/` subdirectories.
