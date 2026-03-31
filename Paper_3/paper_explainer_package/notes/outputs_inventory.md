# Experiment Outputs Inventory — experiments/exp_v1/outputs/

**Date**: 2026-03-13
**Base path**: `experiments/exp_v1/outputs/`

---

## 1. Production Optimizer Run

### opt_summary_local.csv
- **What**: Per-image optimization results for all 174 leaves
- **Why**: Core production run — the foundation of all downstream analyses
- **Script**: `navier_optimize_robust_mostrecent.py` (stochastic hill climbing optimizer)
- **Outputs**: 174 rows with filename, status, score, n_contours, area_px, 7 parameters, paths to mask16/overlay16, elapsed_seconds
- **Paper role**: Main paper — referenced throughout results
- **Key result**: 100% OK, mean score 21.10, median runtime 242s
- **Provenance**: FINAL — production run

---

## 2. Ablation Studies

### ablations_objective/ (6 subdirectories)
- **What**: Objective function ablations — each directory removes one term
- **Subdirs**: full, no_area, no_color_dist, no_contour, no_grad_align, no_small_base
- **Why**: Justify each term in Eq. 9 (the heuristic objective)
- **Script**: `navier_optimize_robust_mostrecent.py` with modified weights
- **Paper role**: Main paper — Table I
- **Key result**: Removing contour-count term drops median score from 14.80 to 3.68; removing noise penalty increases n_c to 188 (fragmentation)
- **Provenance**: FINAL

### ablations_search/ (5 subdirectories)
- **What**: Search strategy ablations — each directory modifies one search knob
- **Subdirs**: default, fast, low_budget, no_downscale, random_only
- **Why**: Justify the two-phase search strategy
- **Script**: Same optimizer with modified search parameters
- **Paper role**: Main paper — Table II
- **Key result**: No-downscale achieves highest score (16.58) but at 34% higher runtime; random-only is competitive (14.88 vs 14.95)
- **Provenance**: FINAL

---

## 3. Trace-Based Objective Surface Studies

### objective_surface_traces_fullstudy/
- **What**: Full-population trace-based surface reconstruction (all 174 leaves)
- **Why**: Characterize objective landscape geometry across entire dataset
- **Contents**: candidate_traces/, surface_plots/, trajectory_plots/, merged CSV, reports
- **Paper role**: Appendix/diagnostic — too many plots for main text
- **Key result**: Comprehensive landscape characterization; superseded by 9-leaf representative for main paper
- **Provenance**: FINAL but superseded for main paper use

### objective_surface_traces_representative_fullbudget/
- **What**: Earlier representative trace study (subset of leaves)
- **Why**: Pilot for the representative analysis approach
- **Paper role**: SUPERSEDED by 9-leaf study
- **Provenance**: SUPERSEDED

### objective_surface_traces_representative_fullbudget_9leaf/
- **What**: Definitive 9-leaf representative trace study
- **Selection**: 3 high (DSC_0163, DSC_0166, DSC_0198), 3 median (DSC_0274, DSC_0379, DSC_0205), 3 low (DSC_0199, DSC_0185, DSC_0261)
- **Why**: Characterize score landscape geometry across the score distribution
- **Script**: `instrument_full_optimizer_traces_representative_9leaf.py` + `reconstruct_objective_surfaces_from_traces_representative_9leaf.py`
- **Source run**: SLURM job 12783426 — dedicated rerun with IDENTICAL settings as production
- **Trace provenance**: ALL 9 traces from dedicated single rerun — NO fallback, NO reused traces
- **Contents**: 9 candidate trace CSVs (833 total rows), 27+ surface plots, 54+ trajectory plots, surface_report.pdf/.txt
- **Paper role**: Main paper — Figs 9-12 and Section 8.4
- **Key results**:
  - High-score leaves: sharp landscapes, refine gain +5.77 avg
  - Low-score leaves: flat landscapes, refine gain +0.56 avg
  - Dominant parameter pair: T × β (7/9 cases)
- **Provenance**: FINAL — homogeneous dedicated rerun

### objective_surface_traces_representative_fullbudget_plus/
- **What**: Extended representative study beyond the core 9 leaves
- **Why**: Broader population evidence for landscape geometry
- **Contents**: candidate_traces/, surface_plots/, trajectory_plots/, manifold_plots/ (EMPTY), solid_3d_plots/ (EMPTY)
- **Paper role**: Supporting/appendix if needed
- **Note**: manifold_plots/ and solid_3d_plots/ were created but never populated
- **Provenance**: PARTIAL — extended analysis not fully completed

---

## 4. Optimization Surface Analysis (Non-Trace-Based)

### optimization_surface_analysis/
- **What**: Score landscape cross-sections from production run (heatmaps + 3D surfaces)
- **Contents**: ~97 files — heatmaps and surfaces for high/low-score leaves and patches across 3 parameter pairs
- **Script**: `analyze_optimization_surfaces.py`
- **Paper role**: Appendix — useful for illustration but superseded by trace-based analysis
- **Provenance**: FINAL but superseded

### optimization_surface_analysis_codex/
- **What**: Codex-style alternate rendering of optimization surfaces
- **Contents**: ~30 files including HTML interactive versions
- **Paper role**: Diagnostic only
- **Provenance**: DIAGNOSTIC

---

## 5. Optimizer Summary Statistics

### optimizer_summary/
- **What**: Distribution summaries of the 174-leaf production run
- **Contents**: 17 files — histograms (score, runtime, area, n_contours, 7 parameters), status bar, top/bottom CSVs
- **Script**: `summarize_optimizer.py`
- **Paper role**: Main paper (Figs 1-3: status bar, runtime hist, score hist); Appendix (Fig A2: parameter histograms, Fig A1: area/n_c histograms)
- **Provenance**: FINAL

---

## 6. Parameter Shift / Manifold / Solid-3D Geometry

### parameter_shift_3d_analysis/
- **What**: 3D visualization of global→local parameter shifts
- **Contents**: 25 files — PCA 3D plots, delta pairplots, correlation heatmap, reports
- **Script**: `analyze_parameter_shifts_3d.py`
- **Paper role**: Appendix — superseded by manifold geometry and solid-3D
- **Provenance**: FINAL but largely superseded

### parameter_shift_3d_analysis_solid/
- **What**: Volumetric geometry of parameter clouds (convex hulls, ellipsoids, density isosurfaces)
- **Contents**: 25 files — hulls, ellipsoids, positive/negative dD surfaces
- **Key metrics**: Centroid shift = 1.616 std units; 73 positive-dD, 48 negative-dD, 29 neutral patches
- **Script**: `analyze_parameter_shifts_solid_3d.py`
- **Paper role**: Main paper (Fig 15: covariance ellipsoids); Appendix (Fig C1: hulls + dD surfaces)
- **Provenance**: FINAL

### parameter_shift_3d_analysis_solid_codex/
- **What**: Codex-style duplicate of solid 3D analysis
- **Contents**: 22 files
- **Paper role**: Diagnostic only
- **Provenance**: DIAGNOSTIC

### parameter_shift_manifold_geometry/
- **What**: Comprehensive manifold analysis of 7D parameter space
- **Contents**: plots/ (30 files), reports/ (3 files), data/ (4 CSVs)
- **Key metrics**: 2 PCs capture 46.5% variance; k-NN dD agreement = 0.534 (vs 0.33 random); patch-type k-NN = 0.488 (near random)
- **Script**: `build_parameter_shift_manifold_geometry.py`
- **Paper role**: Main paper (Figs 13-14: PCA, mean delta); Appendix (Fig C2: t-SNE, centroid groups)
- **Provenance**: FINAL

### optimization_manifold_and_solid3d_final/
- **What**: Merged manifold + solid-3D analysis with final figures
- **Contents**: figures/ (19 files), merged_data/, reports/, tables/
- **Paper role**: Provides the final merged figures used in the manuscript
- **Provenance**: FINAL

---

## 7. Patch Transfer Morphology

### patch_transfer_morphology_plots/
- **What**: Morphology transfer analysis plots (area error, count error, scatter, montages)
- **Contents**: 7 files
- **Paper role**: Main paper (Figs 17-19: histograms + scatter)
- **Provenance**: FINAL

### patch_transfer_morphology_montages/
- **What**: Visual montages of patch transfer results
- **Contents**: analysis/ + montages/ subdirectories
- **Paper role**: Supporting — visual verification
- **Provenance**: FINAL

### patch_transfer_morphology_filled/
- **What**: Filled-morphology variant of patch transfer analysis
- **Contents**: montages/, plots/, eval CSV, summary TXT
- **Paper role**: Diagnostic — alternative visualization
- **Provenance**: FINAL

---

## 8. Teacher-Student Distillation

### teacher_student_figures/
- **What**: Teacher-student comparison montages and plots
- **Contents**: montage_teacher_student.png, examples/, plots/, summary.txt
- **Paper role**: Main paper (Figs 5-6: montage + disagreement); Fig 16 (qualitative grid)
- **Provenance**: FINAL

### unet_infer_on_leaves/ (if exists)
- **What**: UNet student inference outputs (full-leaf stitched masks)
- **Paper role**: Main paper — provides student predictions for teacher-student comparison
- **Provenance**: FINAL

### unet_infer_on_leaves_roi_thr0p30/ (if exists)
- **What**: ROI-gated UNet inference at threshold 0.30
- **Paper role**: Main paper — ROI-gated comparison
- **Provenance**: FINAL

---

## 9. Patches (Training Data)

### patches/
- **What**: Patch-based training data for UNet student
- **Contents**: images/, masks/ — 512×512 patches from 174 leaves
- **Paper role**: Infrastructure — training data generation
- **Provenance**: FINAL

---

## 10. Param Predictor

### param_predictor/
- **What**: Empty directory — placeholder for amortized parameter prediction
- **Paper role**: Future work
- **Provenance**: NOT STARTED

---

## 11. Experiment Summary Master

### experiment_summary_master/
- **What**: Prior attempt at experiment summarization
- **Contents**: .md, .tex files, figs/, notes/, pdf/, tables/
- **Paper role**: Reference/diagnostic
- **Provenance**: PARTIAL — prior summary attempt

---

## 12. Standalone CSV/TXT/PNG Files

### opt_summary_local.csv
- Production optimizer results (174 leaves) — MAIN DATA SOURCE

### patch_local_opt_v2.csv
- Per-patch local optimization results (150 patches, 15 leaves) — Main paper Table III

### patch_transfer_morphology_eval.csv
- Morphology-focused evaluation of patch transfer — Main paper Table III

### patch_transfer_morphology_summary.txt
- Summary statistics for morphology transfer — Reference

### threshold_sweep.csv
- Threshold sweep on stitched probability maps — Main paper Fig 4

### teacher_student_metrics.csv
- Baseline teacher-student Dice/IoU (174 leaves) — Main paper Section 8.1

### teacher_student_metrics_roi_thr0p30.csv
- ROI-gated teacher-student metrics — Main paper Section 8.1

### failures_dice_lt_0p3.csv
- 18 failure cases (Dice < 0.3) — Discussion

### teacher_empty_cases.csv
- 12 teacher-empty/student-nonempty disagreements — Discussion

### postprocess_sweep.csv
- Post-processing parameter sweep — Diagnostic

### paper_disagreement_montage_roi_thr0p30.png
- Visual montage of disagreement cases — Main paper Fig 6

### threshold_sweep_plot.png
- Threshold sweep visualization — Main paper Fig 4

### representative_trace_audit.txt
- Audit of 9-leaf trace study provenance — Documentation

### representative_trace_provenance.csv
- Per-case provenance table for 9-leaf study — Documentation

### outputs_audit_resume_status.txt
- Resume status for prior audit session — Documentation

### outputs_inventory_detailed.md / outputs_inventory_summary.txt / outputs_inventory_table.csv
- Prior inventory attempts — Documentation

### Montage PNGs (montage_patch_opt_*.png, montage_patch_opt_v2_*.png)
- Per-leaf patch optimization visual montages (18 files) — Diagnostic

### _make_analysis_pdfs.py
- Script for generating analysis PDF reports — Infrastructure
