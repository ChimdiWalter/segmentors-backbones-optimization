# Paper 1: Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours for Unsupervised High-Throughput Phenotyping

**Authors:** Chimdi Walter Ndubuisi and Toni Kazic

Fully automated, unsupervised segmentation of plant disease lesions from 16-bit TIFF leaf images. No pixel-level ground truth required. A physics-based active contour model is self-configured per image via black-box stochastic optimization, then distilled into a fast UNet student.

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 174 high-res 16-bit TIFF leaf images |
| Optimizer runtime (median) | ~4047 s per image (~67 min) |
| UNet student median Dice (teacher-nonempty) | 0.686 |
| UNet student mean Dice (all 174) | 0.598 ± 0.226 |
| PCA variance captured (2 PCs) | 46.5% |
| Global→local centroid shift | 1.616 standardized units |
| k-NN ΔD agreement | 0.534 vs 0.33 (random baseline) |
| Count error reduction (local re-opt) | 35% (19.1 → 12.4) |

---

## Installation

```bash
# Requires Python 3.10+
pip install numpy scipy scikit-image scikit-learn matplotlib Pillow \
            torch torchvision opencv-python tifffile fpdf2 pandas umap-learn tqdm
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

The code runs CPU-only (no GPU required for the optimizer). GPU recommended for UNet training.

---

## Data

| Collection | Location | Description |
|-----------|----------|-------------|
| `leaves/` | `segmentors_backbones/leaves/` | 174 16-bit TIFF soybean/maize leaf images (~3900×1340 px) |
| `18.7/` | `segmentors_backbones/18.7/` | Second leaf collection (28 images used in Paper 2) |

Images are stored as lossless 16-bit TIFFs to preserve the full dynamic range of chlorotic halos and necrotic boundaries. Each image is a single leaf on a plain background under standardized diffuse lighting.

---

## Running the Experiments

### Step 1 — Run the physics optimizer (teacher labels)

Runs Navier-Stokes active contour optimization on every leaf. Each image gets its own optimal parameter vector `θ* = (μ, λ, α, β, γ, T)`. Results are written to a CSV summary.

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
    --per-image-seconds 5400 \
    --log opt_summary.csv
```

Key flags:
- `--mode fused` — uses a fused grayscale+Lab channel (recommended); `bestch` picks the single best channel
- `--budget-random` — number of random parameter evaluations per image (default 64)
- `--topk-refine` — how many top random candidates enter local hill-climbing (default 4)
- `--downscale` — image scale factor during search (0.75 = 75% resolution, faster)
- `--per-image-seconds` — watchdog timeout per image in seconds

Output: `navier_output/<leaf_name>_overlay.tif` (green outer mask, magenta inner mask), `opt_summary.csv` (per-image θ*, scores, runtime).

---

### Step 2 — Summarize optimizer results

```bash
python3 summarize_optimizer.py --log opt_summary.csv --out_dir figures/exp_v1/
```

Produces: status bar chart, runtime histogram, objective score histogram.

---

### Step 3 — Train the UNet student

Patchifies teacher masks (512×512, stride 448) and trains a 3-level encoder-decoder.

```bash
python3 unet_train_patchify.py
```

Config is set at the top of the file (`PATCH_METADATA_CSV`, `IMG_DIR`, `EPOCHS=40`, `LR=1e-4`). Trained model saved as `unet_lesion.pth`.

---

### Step 4 — Run UNet inference on new leaves

```bash
python3 infer_unet_patchify_cli.py \
    --model      unet_lesion.pth \
    --input_dir  leaves/ \
    --output_dir unet_output/ \
    --patch_size 512 \
    --overlap    64 \
    --thresh     0.30 \
    --apply_leaf_roi
```

Stitches overlapping patch predictions back to full resolution via probability averaging.

---

### Step 5 — Evaluate teacher vs. student agreement

```bash
python3 eval_teacher_student.py \
    --opt_csv     opt_summary.csv \
    --student_dir unet_output/ \
    --out_csv     teacher_student_metrics.csv
```

Reports per-image Dice, IoU, and teacher-empty/student-nonempty disagreement counts.

---

### Step 6 — Trace-based objective surface analysis (9-leaf study)

Re-runs the optimizer on 9 selected leaves with full candidate logging.

```bash
python3 instrument_full_optimizer_traces.py \
    --input  leaves/ \
    --manifest nine_leaf_manifest.txt \
    --output trace_output/
```

Then reconstruct projected surfaces:

```bash
python3 reconstruct_objective_surfaces_from_traces_representative_9leaf.py \
    --trace_dir trace_output/ \
    --out_dir   figures/9leaf/
```

---

### Step 7 — Parameter-cloud geometry (PCA / manifold analysis)

```bash
python3 patch_local_optimize_compare.py \
    --opt_csv   opt_summary.csv \
    --img_dir   leaves/ \
    --out_dir   patch_local_output/

python3 build_parameter_shift_manifold_geometry.py \
    --local_dir patch_local_output/ \
    --out_dir   figures/manifold/

python3 build_solid_3d_and_manifold_plots_representative_plus.py \
    --local_dir patch_local_output/ \
    --out_dir   figures/solid3d/
```

---

### Step 8 — Patch morphology transfer evaluation

```bash
python3 patch_transfer_morphology_eval.py \
    --opt_csv   opt_summary.csv \
    --img_dir   leaves/ \
    --out_csv   patch_morphology_metrics.csv
```

Reports area fraction and lesion count fidelity for global vs. locally re-optimized parameters.

---

### Ablation Studies

```bash
# Objective function ablations (removes one term at a time)
python3 navier_optimize_robust_mostrecent.py \
    --input leaves_subset/ --output ablation_no_color/ \
    --ablate-color --log ablation_no_color.csv

# Search strategy ablations
python3 navier_optimize_robust_mostrecent.py \
    --input leaves_subset/ --output ablation_random_only/ \
    --no-refine --log ablation_random_only.csv
```

Run on the 30-image stratified subset (5 area-fraction bins × 6 images).

---

## Project Structure

```
paper1_foundation_optimizer_oracle/
├── manuscript/
│   ├── main_paper1.tex          # Full IEEE-format manuscript
│   └── main_paper1.pdf          # Compiled PDF
├── figures/
│   ├── exp_v1/                  # Optimizer diagnostics, UNet results (21 plots)
│   ├── 9leaf/                   # Trace surfaces and trajectories (54 plots)
│   ├── manifold/                # PCA, t-SNE, UMAP parameter plots (6 plots)
│   └── solid3d/                 # 3D covariance ellipsoids, convex hulls (4 plots)
├── notes/
│   ├── figure_provenance.csv    # Every figure: label, filename, source path
│   ├── table_provenance.csv     # Every table: label, source CSV, notes
│   ├── math_summary.txt         # All equations with variable definitions
│   └── explainer_cs_audience.{md,pdf}
│   └── explainer_plant_biologist_audience.{md,pdf}
├── bibliography/
│   └── refs.bib                 # 27 references
└── pdf/
    └── paper1_final.pdf
```

Key scripts in `segmentors_backbones/` (parent directory):

| Script | Purpose |
|--------|---------|
| `navier_optimize_robust_mostrecent.py` | Main per-image optimizer (Step 1) |
| `unet_train_patchify.py` | Train UNet student on teacher pseudo-labels (Step 3) |
| `infer_unet_patchify_cli.py` | Run trained UNet on new leaves (Step 4) |
| `eval_teacher_student.py` | Compare teacher and student masks (Step 5) |
| `instrument_full_optimizer_traces.py` | 9-leaf trace logging (Step 6) |
| `reconstruct_objective_surfaces_from_traces_representative_9leaf.py` | Objective surface plots |
| `build_parameter_shift_manifold_geometry.py` | PCA/manifold plots (Step 7) |
| `build_solid_3d_and_manifold_plots_representative_plus.py` | 3D covariance plots |
| `patch_local_optimize_compare.py` | Per-patch re-optimization (Step 7) |
| `patch_transfer_morphology_eval.py` | Morphology fidelity (Step 8) |
| `summarize_optimizer.py` | Diagnostic plots from opt_summary.csv (Step 2) |
| `auto_phenotype.py` | End-to-end phenotyping pipeline (optimizer → traits) |
| `analyze_optimization_surfaces.py` | Surface analysis figures |
| `analyze_parameter_shifts_3d.py` | 3D shift analysis |

---

## Reproducibility Notes

- Set `--seed 1337` (default) for deterministic random restart order
- Downscale `s=0.75` used for all production runs; `s=1.0` gives higher scores (~10%) at ~35% more runtime
- All figures in `figures/` are direct copies of analysis script outputs — no manual editing
- The 30-image ablation subset is area-fraction stratified (5 quantile bins × 6 images, fixed seed)
- Experiments run on a CPU cluster (hellbender); enforce `OMP_NUM_THREADS=1` to avoid oversubscription
