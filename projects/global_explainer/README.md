# Global Explainer: Dual-Paper Lesion Phenotyping Project

## Purpose

This package documents and explains the full body of work behind two journal-ready manuscripts on physics-based lesion phenotyping:

1. **Paper 1** — Gradient-Free Stochastic Optimization of Navier-Stokes Active Contours
2. **Paper 2** — Black-Box Selective Semi-Amortized Inference for Runtime-Quality Tradeoffs

## Directory Structure

```
global_explainer/
  tex/                  LaTeX source for explainer document
  pdf/                  Compiled PDFs (explainer + executive summary)
  figures/              (shared figures if needed)
  tables/               (shared tables if needed)
  notes/                Comprehensive audit and provenance files
    repository_audit.txt
    manuscript_line_inventory.csv
    output_family_inventory.csv
    dual_paper_evidence_matrix.csv
    dual_paper_gap_analysis.txt
    dual_paper_distinction.txt
    reference_master_audit.csv
    reference_quarantine.txt
    reference_usage_by_paper.csv
    figure_and_table_guide.csv
    claim_to_output_map.csv
    gapfill_report.txt
  gapfill/              New analyses (minimal)
    paper1_gapfill/
    paper2_gapfill/
    shared_gapfill/
  README.md             This file
```

## How to Compile

```bash
cd tex
pdflatex global_explainer
pdflatex global_explainer
```

## Key Files

- `pdf/global_explainer.pdf` — Full project explainer (6 pages)
- `pdf/global_executive_summary.pdf` — Executive summary
- `notes/dual_paper_distinction.txt` — How the two papers differ
- `notes/claim_to_output_map.csv` — Every claim mapped to evidence
