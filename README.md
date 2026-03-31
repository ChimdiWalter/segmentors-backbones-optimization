# Segmentors Backbones Optimization

**Authors:** Chimdi Walter Ndubuisi and Toni Kazic

Physics-based lesion segmentation and parameter prediction for high-throughput plant disease phenotyping. This repository contains the full codebase, manuscripts, figures, and analysis scripts for two papers on unsupervised leaf lesion segmentation from 16-bit TIFF imagery.

---

## Papers

| | Paper | Description |
|-|-------|-------------|
| **Paper 1** | [Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours for Unsupervised High-Throughput Phenotyping](projects/paper1_foundation_optimizer_oracle/README.md) | Per-image black-box optimization of a physics-based active contour model. No ground truth required. Distills results into a fast UNet student. |
| **Paper 2** | [Black-Box Selective Semi-Amortized Inference for Runtime–Quality Tradeoffs in Physics-Based Lesion Phenotyping](projects/paper2_selective_runtime_and_patch/README.md) | Learned predictors approximate the optimizer output at 3.5–4.7× speedup via uncertainty-guided routing (direct / refine / fallback). |

---

## Key Results

**Paper 1 — Physics Optimizer + UNet Distillation**

| Metric | Value |
|--------|-------|
| Dataset | 174 16-bit TIFF leaf images |
| Optimizer runtime (median) | ~67 min / leaf |
| UNet student median Dice | 0.686 |
| PCA variance in 2 components | 46.5% |
| k-NN ΔD agreement | 0.534 vs 0.33 baseline |

**Paper 2 — Selective Semi-Amortized Inference**

| Metric | Value |
|--------|-------|
| Best patchwise nMAE | 0.906 (mean aggregation) |
| Quality-favoring routing | 0.556 nMAE at 74 s (3.5× speedup) |
| Max-speedup routing | 0.819 nMAE at 56 s (4.7× speedup) |
| Full optimizer baseline | ~0.026 nMAE at 260 s |

---

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`, `Pillow`, `torch`, `torchvision`, `opencv-python`, `tifffile`, `fpdf2`, `pandas`, `umap-learn`, `tqdm`.

---

## Data

| Collection | Location | Description |
|-----------|----------|-------------|
| `leaves/` | `leaves/` | 174 high-resolution 16-bit TIFF soybean/maize leaf images (~3900×1340 px) |
| `18.7/` | `18.7/` | Second leaf collection (28 images) used for robustness evaluation in Paper 2 |

Images are lossless 16-bit TIFFs acquired under standardized diffuse lighting. Each image is a single leaf on a plain background.

---

## Quick Start

### Run the physics optimizer (generates teacher pseudo-labels)

```bash
python3 navier_optimize_robust_mostrecent.py \
    --input  leaves/ \
    --output navier_output/ \
    --mode   fused \
    --budget-random 64 \
    --topk-refine   4 \
    --refine-steps  12 \
    --downscale     0.75 \
    --workers       8 \
    --log opt_summary.csv
```

### Train the UNet student on optimizer pseudo-labels

```bash
python3 unet_train_patchify.py   # config at top of file
```

### Run UNet inference on new leaves

```bash
python3 infer_unet_patchify_cli.py \
    --model unet_lesion.pth \
    --input_dir leaves/ \
    --output_dir unet_output/ \
    --thresh 0.30 --apply_leaf_roi
```

### Train a fast physics parameter predictor (Paper 2)

```bash
python3 physics_predictor.py          # ResNet-18 backbone
python3 physics_predictor_dino.py     # DINOv2 backbone
```

### Run selective semi-amortized inference (Paper 2)

```bash
python3 sam2_resnet_physics_predictor.py \
    --img_dir leaves/ \
    --model param_predictor_resnet.pth \
    --opt_csv opt_summary.csv \
    --tau_lo 0.3 --tau_hi 0.7 \
    --refine_budget medium \
    --out_dir selective_output/
```

---

## Repository Structure

```
segmentors-backbones-optimization/
├── projects/
│   ├── paper1_foundation_optimizer_oracle/    # Paper 1 manuscript, figures, notes
│   │   ├── README.md                          # Full run instructions for Paper 1
│   │   ├── manuscript/main_paper1.{tex,pdf}
│   │   ├── figures/{exp_v1,9leaf,manifold,solid3d}/
│   │   └── notes/{figure_provenance,table_provenance,math_summary}.csv/txt
│   ├── paper2_selective_runtime_and_patch/    # Paper 2 manuscript, figures, notes
│   │   ├── README.md                          # Full run instructions for Paper 2
│   │   ├── manuscript/main_paper2.{tex,pdf}
│   │   ├── figures/B_fig01–B_fig39.{pdf,png}
│   │   └── notes/
│   └── generate_explainer_pdfs.py             # Generate audience explainer PDFs
├── navier_optimize_robust_mostrecent.py       # Main per-image physics optimizer
├── unet_train_patchify.py                     # Train UNet student
├── infer_unet_patchify_cli.py                 # UNet inference on new leaves
├── eval_teacher_student.py                    # Compare teacher vs student masks
├── physics_predictor.py                       # ResNet-18 parameter predictor
├── physics_predictor_dino.py                  # DINOv2 parameter predictor
├── sam2_resnet_physics_predictor.py           # Selective routing inference
├── sam2_physics_predictor.py                  # Full-leaf vs patchwise comparison
├── patchify_images.py                         # Cut leaves into patches
├── patch_local_optimize_compare.py            # Per-patch re-optimization
├── build_parameter_shift_manifold_geometry.py # PCA/manifold figures
├── build_solid_3d_and_manifold_plots_representative_plus.py  # 3D geometry figures
├── instrument_full_optimizer_traces.py        # 9-leaf trace logging
├── reconstruct_objective_surfaces_from_traces_representative_9leaf.py
├── auto_phenotype.py                          # End-to-end phenotyping pipeline
├── requirements.txt                           # Python dependencies
├── pyproject.toml                             # Package metadata
└── leaves/                                    # Leaf image dataset (not tracked)
```

---

## Experiment Outputs

| Directory | Contents |
|-----------|----------|
| `navier_output/` | Per-leaf overlay TIFFs + `opt_summary.csv` (θ* per leaf) |
| `unet_output/` | UNet mask predictions |
| `experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/` | Paper 2 matched comparison results |
| `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/` | Paper 2 routing study results |

---

## Citation

```bibtex
@article{ndubuisi2026paper1,
  title={Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours
         for Unsupervised High-Throughput Phenotyping},
  author={Ndubuisi, Chimdi Walter and Kazic, Toni},
  year={2026}
}

@article{ndubuisi2026paper2,
  title={Black-Box Selective Semi-Amortized Inference for Runtime--Quality
         Tradeoffs in Physics-Based Lesion Phenotyping},
  author={Ndubuisi, Chimdi Walter and Kazic, Toni},
  year={2026}
}
```
