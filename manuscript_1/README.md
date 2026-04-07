# manuscript_1 — Paper 1: Gradient-Free Stochastic Optimization of Navier–Stokes Active Contours

Self-contained snapshot of Paper 1 (two versions) with PDFs, LaTeX sources,
all referenced figures, and `.org` outlines that link to the figures with
relative paths.

```
manuscript_1/
├── paper1_v3_article_final.pdf       ← latest compiled PDF (v3)
├── paper1_v2_article_final.pdf       ← previous compiled PDF (v2)
├── v3/
│   ├── main_paper1_v3_article.tex
│   ├── main_paper1_v3_article.org    ← outline + relative figure links
│   ├── IEEEtran.cls / IEEEtran.bst
│   └── figures/                      ← every figure referenced by the v3 tex
└── v2/
    ├── main_paper1_v2_article.tex
    ├── main_paper1_v2_article.org
    ├── IEEEtran.cls / IEEEtran.bst
    └── figures/
```

The `.org` files mirror the LaTeX section structure and embed each figure as
`[[file:figures/<name>]]` so they render in Emacs / Doom / VS Code Org.
