# Paper 2: Black-Box Selective Semi-Amortized Inference for Runtime-Quality Tradeoffs in Physics-Based Lesion Phenotyping

**Authors:** Chimdi Walter Ndubuisi and Toni Kazic

## Summary

This paper addresses the deployment problem for physics-based lesion phenotyping: a full Navier-Stokes active contour optimizer produces high-quality segmentations but costs ~260s per leaf. We present a selective semi-amortized inference framework with three actions (direct prediction, short refinement, full fallback) governed by uncertainty-guided routing. On a matched 35-leaf comparison, patchwise aggregation achieves nMAE 0.906 vs full-leaf 1.154 (21.5% improvement, p=0.003), bounded by an oracle gap of 0.327 nMAE. Selective routing achieves 3.5-4.7x speedup over the full optimizer.

## Directory Structure

```
paper2_selective_runtime_and_patch/
  manuscript/         LaTeX source (main_paper2.tex)
  figures/            All figure files (B_fig*.pdf/png)
  tables/             (data tables if extracted)
  scripts/            (analysis scripts if needed)
  notes/              Provenance, plans, math summary
  bibliography/       refs.bib
  pdf/                paper2_final.pdf
  README.md           This file
```

## How to Compile

```bash
cd manuscript
pdflatex main_paper2
bibtex main_paper2
pdflatex main_paper2
pdflatex main_paper2
```

## Evidence Sources

- `experiments/exp_v1/outputs/integrated_full_vs_patch_leaves_18p7_v1/` — Matched comparison
- `experiments/exp_v1/outputs/selective_semiamortized_fullstudy174_v3/` — Routing study
- `experiments/exp_v1/outputs/selective_semiamortized_patchstudy_leaves_18p7_v1/` — Patch study
- `experiments/exp_v1/outputs/dual_paper_build_codex/` — Synthesis assets

## Relationship to Paper 1

Paper 1 covers the optimizer design, objective function, search strategy, ablations, trace-based surface analysis, and parameter-cloud geometry. Paper 2 covers deployment: selective routing, runtime-quality tradeoffs, full-leaf vs patchwise comparison, calibration, and failure analysis. The two papers share the same optimizer and dataset but address distinct scientific questions.

## Key Results

| Metric | Value |
|--------|-------|
| Full-leaf nMAE | 1.154 |
| Best patchwise nMAE | 0.906 (mean aggregation) |
| Oracle gap | 0.327 nMAE |
| Quality-favoring routing | 0.556 nMAE at 74.34s (3.50x speedup) |
| Balanced routing | 0.577 nMAE at 60.48s (4.31x speedup) |
| Max-speedup routing | 0.819 nMAE at 55.77s (4.67x speedup) |
