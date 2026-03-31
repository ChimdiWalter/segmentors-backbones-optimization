#!/usr/bin/env python3
"""Summarize ablation opt_summary.csv files and generate LaTeX tables.

For each opt_summary.csv, computes one summary row:
  ok%, nonempty%, median_score, median_area_px, median_n_contours, median_elapsed_seconds

Writes:
  Paper_3/tables/tableI_objective_ablations.tex
  Paper_3/tables/tableII_search_ablations.tex

Then patches paper3_exp_v1.tex to replace the template table environments.
"""

import csv, os, sys, re, statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent  # segmentors_backbones/
OBJ_DIR    = BASE / "experiments" / "exp_v1" / "outputs" / "ablations_objective"
SEARCH_DIR = BASE / "experiments" / "exp_v1" / "outputs" / "ablations_search"
TABLE_DIR  = BASE / "Paper_3" / "tables"
PAPER_TEX  = BASE / "Paper_3" / "paper3_exp_v1.tex"

# variant name -> (display name, weight zeroed)
OBJ_VARIANTS = [
    ("full",            "Full objective (ours)"),
    ("no_color_dist",   r"w/o color distance ($W_{\mathrm{color}}{=}0$)"),
    ("no_grad_align",   r"w/o gradient align ($W_{\mathrm{grad}}{=}0$)"),
    ("no_contour",      r"w/o contour count ($W_{\mathrm{topo}}{=}0$)"),
    ("no_area",         r"w/o area penalty ($W_{\mathrm{area}}{=}0$)"),
    ("no_small_base",   r"w/o noise penalty ($W_{\mathrm{small}}{=}0$)"),
]

SEARCH_VARIANTS = [
    ("default",      "Full pipeline (ours)"),
    ("random_only",  "Random-only (no refine)"),
    ("low_budget",   "Low budget ($K{=}16$)"),
    ("no_downscale", "No downscale ($s{=}1.0$)"),
    ("fast",         "Fast (60 s timeout)"),
]


def summarize_csv(csv_path):
    """Return dict with summary stats, or None if file missing."""
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    total = len(rows)
    # Both "ok" and "timeout" produce valid results (timeout just means
    # the per-image-seconds budget was exceeded, but a result was still written)
    valid_statuses = {"ok", "timeout"}
    ok_rows = [r for r in rows if r.get("status", "").strip() in valid_statuses]
    ok_pct = 100.0 * len(ok_rows) / total if total else 0.0

    nonempty = [r for r in ok_rows if int(float(r.get("area_px", 0))) > 0]
    nonempty_pct = 100.0 * len(nonempty) / total if total else 0.0

    def med(key, subset=None):
        src = subset if subset is not None else ok_rows
        vals = []
        for r in src:
            v = r.get(key)
            if v is not None and v != "":
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return statistics.median(vals) if vals else 0.0

    return {
        "ok_pct":          ok_pct,
        "nonempty_pct":    nonempty_pct,
        "median_score":    med("score"),
        "median_area_px":  med("area_px"),
        "median_n_contours": med("n_contours"),
        "median_elapsed":  med("elapsed_seconds"),
    }


def fmt_pct(v):
    return f"{v:.0f}\\%"

def fmt_score(v):
    return f"{v:.2f}"

def fmt_int(v):
    return f"{int(v)}"

def fmt_sec(v):
    return f"{v:.1f}"


def build_obj_table():
    """Generate LaTeX for Table I: Objective Ablations."""
    summaries = []
    for dirname, label in OBJ_VARIANTS:
        csv_path = OBJ_DIR / dirname / "opt_summary.csv"
        s = summarize_csv(csv_path)
        summaries.append((label, s))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Objective ablations on the 30-image stratified subset. Each row removes one term from the composite objective (Eq.~\ref{eq:objective}).}")
    lines.append(r"\label{tab:abl_obj}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & OK\% $\uparrow$ & Non-empty\% $\uparrow$ & Med.\ score $\uparrow$ & Med.\ area & Med.\ $n_c$ & Med.\ time (s) $\downarrow$ \\")
    lines.append(r"\midrule")
    for label, s in summaries:
        if s is None:
            lines.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
        else:
            lines.append(
                f"{label} & {fmt_pct(s['ok_pct'])} & {fmt_pct(s['nonempty_pct'])} "
                f"& {fmt_score(s['median_score'])} & {fmt_int(s['median_area_px'])} "
                f"& {fmt_int(s['median_n_contours'])} & {fmt_sec(s['median_elapsed'])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_search_table():
    """Generate LaTeX for Table II: Search Ablations."""
    summaries = []
    for dirname, label in SEARCH_VARIANTS:
        csv_path = SEARCH_DIR / dirname / "opt_summary.csv"
        s = summarize_csv(csv_path)
        summaries.append((label, s))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Search-strategy ablations on the 30-image stratified subset. Each row changes one knob of the stochastic search while keeping the full objective.}")
    lines.append(r"\label{tab:abl_search}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Setting & OK\% $\uparrow$ & Non-empty\% $\uparrow$ & Med.\ score $\uparrow$ & Med.\ area & Med.\ $n_c$ & Med.\ time (s) $\downarrow$ \\")
    lines.append(r"\midrule")
    for label, s in summaries:
        if s is None:
            lines.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
        else:
            lines.append(
                f"{label} & {fmt_pct(s['ok_pct'])} & {fmt_pct(s['nonempty_pct'])} "
                f"& {fmt_score(s['median_score'])} & {fmt_int(s['median_area_px'])} "
                f"& {fmt_int(s['median_n_contours'])} & {fmt_sec(s['median_elapsed'])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def replace_table_env(tex, label, new_table):
    """Replace the \\begin{table}...\\end{table} environment containing \\label{<label>}."""
    # Find \label{<label>} first, then walk backwards to \begin{table} and forwards to \end{table}
    label_pattern = r"\\label\{" + re.escape(label) + r"\}"
    label_match = re.search(label_pattern, tex)
    if not label_match:
        print(f"  WARNING: could not find label {label}")
        return tex

    # Find the nearest \begin{table} before the label
    begin_pattern = r"\\begin\{table\}"
    begin_pos = None
    for m in re.finditer(begin_pattern, tex[:label_match.start()]):
        begin_pos = m.start()  # keep the last one (closest to label)

    if begin_pos is None:
        print(f"  WARNING: could not find \\begin{{table}} before label {label}")
        return tex

    # Find the first \end{table} after the label
    end_pattern = r"\\end\{table\}"
    end_match = re.search(end_pattern, tex[label_match.end():])
    if not end_match:
        print(f"  WARNING: could not find \\end{{table}} after label {label}")
        return tex

    end_pos = label_match.end() + end_match.end()
    tex = tex[:begin_pos] + new_table + tex[end_pos:]
    print(f"  Replaced table with label {label}")
    return tex


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate tables ---
    table_obj = build_obj_table()
    table_search = build_search_table()

    obj_path = TABLE_DIR / "tableI_objective_ablations.tex"
    search_path = TABLE_DIR / "tableII_search_ablations.tex"

    obj_path.write_text(table_obj + "\n")
    print(f"Wrote {obj_path}")
    search_path.write_text(table_search + "\n")
    print(f"Wrote {search_path}")

    # --- Patch paper ---
    tex = PAPER_TEX.read_text()
    tex = replace_table_env(tex, "tab:abl_obj", table_obj)
    tex = replace_table_env(tex, "tab:abl_search", table_search)
    PAPER_TEX.write_text(tex)
    print(f"Patched {PAPER_TEX}")


if __name__ == "__main__":
    main()
