#!/usr/bin/env python3.11
"""Generate PDFs for all four paper explainers using fpdf2.

Usage (from repo root):
    source /cluster/VAST/kazict-lab/e/lesion_phes/lesenv/bin/activate
    python3.11 segmentors_backbones/projects/generate_explainer_pdfs.py
"""

import re
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF


EXPLAINERS = [
    (
        "paper1_foundation_optimizer_oracle/notes/explainer_cs_audience.md",
        "paper1_foundation_optimizer_oracle/notes/explainer_cs_audience.pdf",
    ),
    (
        "paper1_foundation_optimizer_oracle/notes/explainer_plant_biologist_audience.md",
        "paper1_foundation_optimizer_oracle/notes/explainer_plant_biologist_audience.pdf",
    ),
    (
        "paper2_selective_runtime_and_patch/notes/explainer_cs_audience.md",
        "paper2_selective_runtime_and_patch/notes/explainer_cs_audience.pdf",
    ),
    (
        "paper2_selective_runtime_and_patch/notes/explainer_plant_biologist_audience.md",
        "paper2_selective_runtime_and_patch/notes/explainer_plant_biologist_audience.pdf",
    ),
]

BASE = Path(__file__).parent  # segmentors_backbones/projects/


def clean(text: str) -> str:
    """Strip/replace characters fpdf latin-1 cannot handle."""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2026": "...",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
        "\u03b4": "delta", "\u03b7": "eta", "\u03b8": "theta",
        "\u03bb": "lambda", "\u03bc": "mu", "\u03c3": "sigma",
        "\u03c6": "phi", "\u03c7": "chi", "\u03c8": "psi",
        "\u03c9": "omega", "\u03a3": "Sigma", "\u0398": "Theta",
        "\u222b": "integral", "\u2202": "d",
        "\u2208": "in", "\u2265": ">=", "\u2264": "<=",
        "\u00d7": "x", "\u00b1": "+/-", "\u00b2": "^2",
        "\u2192": "->", "\u2190": "<-", "\u21d2": "=>",
        "\u2713": "OK", "\u2717": "X", "\u2022": "-",
        "\u03b5": "epsilon", "\u03a9": "Omega",
        "\u2211": "sum", "\u220f": "prod",
        "\u221e": "inf", "\u2207": "grad",
        "\u03a3": "Sigma", "\u03a6": "Phi",
        "\u2248": "~=", "\u2260": "!=",
        "\u00e9": "e", "\u00e8": "e", "\u00ea": "e",
        "\u00e0": "a", "\u00e2": "a",
        "\u2081": "_1", "\u2082": "_2",
        "\u2080": "_0",
    }
    for u, a in replacements.items():
        text = text.replace(u, a)
    # strip any remaining non-latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ExplainerPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, clean(self.doc_title), align="L")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def render_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Extract title from first H1
    title = md_path.stem.replace("_", " ").title()
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    pdf = ExplainerPDF(title=title)
    pdf.add_page()
    pdf.set_left_margin(18)
    pdf.set_right_margin(18)

    i = 0
    in_table = False
    table_rows: list[list[str]] = []
    in_code = False
    code_lines: list[str] = []

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            in_table = False
            return
        # filter separator rows
        data_rows = [r for r in table_rows if not all(
            set(c.strip()).issubset({"-", ":", " ", ""}) for c in r
        )]
        if not data_rows:
            in_table = False
            table_rows = []
            return

        col_n = max(len(r) for r in data_rows)
        # estimate col widths
        usable = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = usable / col_n

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(230, 230, 245)
        for ci, cell in enumerate(data_rows[0]):
            pdf.cell(col_w, 6, clean(cell.strip())[:50], border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(245, 245, 250)
        for ri, row in enumerate(data_rows[1:]):
            fill = (ri % 2 == 0)
            for ci in range(col_n):
                cell_text = row[ci].strip() if ci < len(row) else ""
                pdf.cell(col_w, 5, clean(cell_text)[:50], border=1, fill=fill)
            pdf.ln()
        pdf.ln(2)
        table_rows = []
        in_table = False

    def flush_code():
        nonlocal code_lines, in_code
        if not code_lines:
            in_code = False
            return
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Courier", "", 8)
        for cl in code_lines:
            txt = clean(cl)
            if txt:
                pdf.multi_cell(0, 4.5, txt, fill=True, border=0)
            else:
                pdf.ln(2)
        pdf.ln(2)
        code_lines = []
        in_code = False

    while i < len(lines):
        line = lines[i]

        # --- code fence ---
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
            else:
                if in_table:
                    flush_table()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # --- table row ---
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
            cells = [c for c in line.split("|") if c != ""]
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            flush_table()

        stripped = line.strip()

        # blank line
        if stripped == "":
            pdf.ln(2)
            i += 1
            continue

        # headings
        if stripped.startswith("#### "):
            pdf.set_font("Helvetica", "BI", 10)
            pdf.set_text_color(60, 60, 130)
            pdf.multi_cell(0, 6, clean(stripped[5:]))
            pdf.set_text_color(0, 0, 0)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 40, 110)
            pdf.multi_cell(0, 7, clean(stripped[4:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 20, 90)
            pdf.set_fill_color(235, 235, 250)
            pdf.multi_cell(0, 8, clean(stripped[3:]), fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(10, 10, 70)
            pdf.multi_cell(0, 10, clean(stripped[2:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
        # horizontal rule
        elif stripped.startswith("---") and len(set(stripped)) == 1:
            pdf.set_draw_color(180, 180, 200)
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
        # bullet list
        elif stripped.startswith("- ") or stripped.startswith("* "):
            # remove markdown bold/italic/code markers for display
            content = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped[2:])
            content = re.sub(r"\*(.+?)\*", r"\1", content)
            content = re.sub(r"`(.+?)`", r"\1", content)
            content = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", content)
            pdf.set_font("Helvetica", "", 9.5)
            # indent bullet
            x0 = pdf.get_x()
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(0, 5, "  - " + clean(content))
            pdf.set_x(x0)
        # numbered list
        elif re.match(r"^\d+\. ", stripped):
            content = re.sub(r"^\d+\. ", "", stripped)
            content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
            content = re.sub(r"`(.+?)`", r"\1", content)
            content = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", content)
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(0, 5, "  " + clean(content))
        # regular paragraph
        else:
            content = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            content = re.sub(r"\*(.+?)\*", r"\1", content)
            content = re.sub(r"`(.+?)`", r"\1", content)
            content = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", content)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.5, clean(content))
            pdf.ln(1)

        i += 1

    if in_table:
        flush_table()
    if in_code:
        flush_code()

    pdf.output(str(pdf_path))
    print(f"  Written: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    print("Generating explainer PDFs...")
    for md_rel, pdf_rel in EXPLAINERS:
        md_path = BASE / md_rel
        pdf_path = BASE / pdf_rel
        if not md_path.exists():
            print(f"  MISSING: {md_path}")
            continue
        print(f"\nProcessing: {md_path.name}")
        try:
            render_markdown_to_pdf(md_path, pdf_path)
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\nDone.")
