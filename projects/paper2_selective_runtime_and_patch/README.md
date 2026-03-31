# Paper 2: Black-Box Selective Semi-Amortized Inference for Runtime–Quality Tradeoffs in Physics-Based Lesion Phenotyping

**Authors:** Chimdi Walter Ndubuisi and Toni Kazic

Addresses the deployment bottleneck of Paper 1: the physics optimizer costs ~260 s/leaf. This paper trains learned predictors that approximate the optimizer output and routes each leaf to one of three tiers (direct prediction, short refinement, full fallback) based on uncertainty. Achieves 3.5–4.7× speedup with meaningful quality retention.

---

## Key Results

| Metric | Value |
|--------|-------|
| Full-leaf direct prediction nMAE | 1.154 |
| Best patchwise nMAE (mean aggregation) | 0.906 |
| Oracle gap (full-leaf vs patch oracle) | 0.327 nMAE |
| Quality-favoring routing nMAE | 0.556 at 74.3 s (3.50× speedup) |
| Balanced routing nMAE | 0.577 at 60.5 s (4.31× speedup) |
| Max-speedup routing nMAE | 0.819 at 55.8 s (4.67× speedup) |
| Full optimizer baseline | ~0.026 nMAE at 260.5 s |
| Best selective MAE (full study, 174 leaves) | 0.0715 at threshold (0.5, 0.7) |

---

## Installation

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib Pillow \
            torch torchvision opencv-python tifffile pandas umap-learn tqdm
```

Or:

```bash
pip install -r requirements.txt
```

---

## Data

| Collection | Location | Description |
|-----------|----------|-------------|
| `leaves/` | `segmentors_backbones/leaves/` | 174 16-bit TIFF leaf images (primary, ID set) |
| `18.7/` | `segmentors_backbones/18.7/` | 28-image second collection used for robustness evaluation |

Both collections contain segmented `.tif` files (optimizer outputs). The optimizer CSV `opt_summary.csv` provides the ground-truth θ* targets for training predictors.

**Experimental substrates (do not mix their numbers):**

| Substrate | Size | Used for |
|-----------|------|---------|
| Matched 35-leaf comparison | 35 leaves | Full-leaf vs patchwise comparison, oracle gap, routing frontier |
| Selective full study | 174 leaves (146 ID + 28 from `18.7`) | Threshold grid, calibration, learning curves |

---

## Running the Experiments

### Prerequisites

Run Paper 1's optimizer first to generate `opt_summary.csv` (the per-leaf θ* targets):

```bash
python3 navier_optimize_robust_mostrecent.py \
    --input leaves/ --output navier_output/ \
    --mode fused --budget-random 64 --topk-refine 4 \
    --log opt_summary.csv
```

---

### Step 1 — Train a full-leaf physics parameter predictor

Trains a ResNet-18 feature extractor + ridge/MLP head to predict θ* from a full-leaf image.

```bash
python3 physics_predictor.py
```

Config at top of file: `CSV_PATH` (opt_summary.csv), `IMG_DIR` (leaves/), `EPOCHS=25`, `LR=1e-4`.
Saved model: `param_predictor_resnet.pth`.

For DINOv2 backbone:

```bash
python3 physics_predictor_dino.py
```

Saved model: `dino_param_predictor.pth`.

---

### Step 2 — Train a patchwise predictor

Patchifies leaves into 224×224 (or 336×336) windows and trains the same predictor on patch-level inputs.

```bash
python3 patchify_images.py \
    --img_dir   leaves/ \
    --out_dir   leaves_patches/ \
    --patch_size 224 \
    --stride     112
```

Then train on patches:

```bash
python3 physics_predictor.py  # set CSV_PATH to patch-level CSV, IMG_DIR to leaves_patches/
```

---

### Step 3 — Run selective semi-amortized inference

Routes each leaf to direct / refine / fallback based on uncertainty thresholds `(τ_lo, τ_hi)`.

```bash
python3 sam2_resnet_physics_predictor.py \
    --img_dir    leaves/ \
    --model      param_predictor_resnet.pth \
    --opt_csv    opt_summary.csv \
    --tau_lo     0.3 \
    --tau_hi     0.7 \
    --out_dir    selective_output/ \
    --refine_budget medium
```

`--refine_budget` options: `tiny` (~11 s), `small` (~18 s), `medium` (~29 s).

Reference runtimes per leaf:
- Direct: ~3 s
- Tiny refinement: ~11 s
- Small: ~18 s
- Medium: ~29 s
- Full optimizer: ~260 s

---

### Step 4 — Evaluate routing policies (threshold grid)

Sweeps over `(τ_lo, τ_hi)` pairs and records nMAE + runtime for each policy.

```bash
# Audited policies from the paper:
# Quality-favoring: --tau_lo 0.3 --tau_hi 0.7  →  nMAE 0.556, 74.3 s
# Balanced:         --tau_lo 0.3 --tau_hi 0.9  →  nMAE 0.577, 60.5 s
# Max-speedup:      --tau_lo 0.6 --tau_hi 0.9  →  nMAE 0.819, 55.8 s
```

Results land in `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/tables/runtime_comparison.csv`.

---

### Step 5 — Full-leaf vs patchwise matched comparison (35-leaf study)

```bash
python3 sam2_physics_predictor.py \
    --mode     compare \
    --img_dir  leaves/ \
    --model    param_predictor_resnet.pth \
    --opt_csv  opt_summary.csv \
    --out_dir  experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/
```

Outputs:
- `evaluation/E1_matched_comparison.csv` — nMAE for 5 aggregation methods
- `evaluation/E12_statistical_tests.csv` — Wilcoxon p-values and bootstrap CIs

Key numbers from the paper:

| Aggregation | nMAE |
|------------|------|
| Mean | **0.906** |
| Uncertainty-weighted | 0.911 |
| Selective mixture | 0.921 |
| Robust median | 0.962 |
| Top-k confident | 1.037 |
| Full-leaf direct | 1.154 |
| Oracle gap | 0.327 |

---

### Step 6 — Calibration and risk–coverage analysis

```bash
python3 postprocess_sweep.py \
    --selective_dir experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/ \
    --out_dir       figures/paper2/
```

Produces calibration bins (`calibration.csv`), risk–coverage curves, and score-recovery plots.

Calibration result (5 bins, 174 leaves):

| Bin | Mean uncertainty | Mean error |
|-----|-----------------|-----------|
| 0 (most confident) | 0.0177 | 0.0082 |
| 1 | 0.0548 | 0.0266 |
| 2 | 0.1059 | 0.0624 |
| 3 | 0.1438 | 0.0975 |
| 4 (least confident) | 0.1877 | 0.1157 |

---

### Step 7 — Patch extraction ablation

Tests three patch size/stride configurations:

```bash
# Config 1: 224/224 (non-overlapping) — 13,192 patches, heterogeneity 1.299
python3 patchify_images.py --patch_size 224 --stride 224 --img_dir leaves/ --out_dir patches_224_224/

# Config 2: 224/112 (50% overlap) — 47,916 patches, heterogeneity 1.903
python3 patchify_images.py --patch_size 224 --stride 112 --img_dir leaves/ --out_dir patches_224_112/

# Config 3: 336/168
python3 patchify_images.py --patch_size 336 --stride 168 --img_dir leaves/ --out_dir patches_336_168/
```

---

### Step 8 — Cross-collection robustness (18.7 dataset)

```bash
python3 physics_predictor_newimages.py \
    --model     param_predictor_resnet.pth \
    --img_dir   18.7/ \
    --out_csv   ood_robustness.csv
```

---

## Project Structure

```
paper2_selective_runtime_and_patch/
├── manuscript/
│   ├── main_paper2.tex               # Full manuscript source
│   └── main_paper2.pdf               # Compiled PDF
├── figures/                          # 39 figures (B_fig01–B_fig39)
│   ├── B_fig01_pipeline_schematic.pdf
│   ├── B_fig03_main_comparison.pdf
│   ├── B_fig04_runtime_quality_pareto.pdf
│   ├── B_fig17_leaf_comparison.png
│   ├── B_fig34_representative_leaf_routing.pdf
│   └── ...
├── notes/
│   ├── figure_provenance.csv         # Every figure: label, filename, source
│   ├── table_provenance.csv          # Every table: label, source CSV
│   ├── math_summary.txt              # All equations with variable definitions
│   └── explainer_cs_audience.{md,pdf}
│   └── explainer_plant_biologist_audience.{md,pdf}
├── bibliography/
│   └── refs.bib
└── pdf/
    └── paper2_final.pdf
```

Key scripts in `segmentors_backbones/` (parent directory):

| Script | Purpose |
|--------|---------|
| `physics_predictor.py` | Train ResNet-18 full-leaf θ predictor |
| `physics_predictor_dino.py` | Train DINOv2 full-leaf θ predictor |
| `physics_predictor_newimages.py` | Run predictor on new/OOD leaves |
| `physics_predictor_dino_newimages.py` | DINOv2 predictor on new leaves |
| `patchify_images.py` | Cut leaves into overlapping patches |
| `sam2_resnet_physics_predictor.py` | Selective routing (direct/refine/fallback) |
| `sam2_physics_predictor.py` | Full-leaf vs patchwise matched comparison |
| `unet_predictor_newimages.py` | UNet-based predictor on new leaves |
| `postprocess_sweep.py` | Calibration, risk–coverage, threshold grid |
| `score_unet_results.py` | Score UNet predictions against oracle |
| `check_leaf_params_transfer_to_patches.py` | Oracle gap analysis |

Experiment output directories:
- `experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/` — matched 35-leaf comparison
- `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/` — full 174-leaf routing study

---

## Relationship to Paper 1

Paper 1 builds the oracle (Navier-Stokes optimizer + UNet student). Paper 2 takes that oracle's outputs (`opt_summary.csv`, `navier_output/`) as inputs and trains predictors to approximate it faster. Run Paper 1 pipeline first.

---

## Reproducibility Notes

- All numbers come from two non-overlapping experimental substrates — do not compare nMAE from the 35-leaf study to MAE from the 174-leaf study directly
- The oracle gap (0.327 nMAE) is a fixed property of the dual-oracle setup, not a tunable parameter
- Learning curves are intentionally non-monotone (reported as evidence of target noise / split sensitivity)
- All OOD numbers from `18.7/` should be interpreted as a different deployment substrate, not a generalization benchmark
- Set `OMP_NUM_THREADS=1` on cluster to avoid BLAS oversubscription
