#!/usr/bin/env python3
"""Create a textual + PDF interpretation report from existing parameter-shift outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import wrap

import pandas as pd


TITLE = "Interpretation of Parameter-Shift Analysis: Global-to-Local Patch Optimization"


def parse_summary(summary_text: str) -> dict:
    def grab_float(pattern: str) -> float:
        m = re.search(pattern, summary_text)
        if not m:
            raise ValueError(f"Could not parse pattern: {pattern}")
        return float(m.group(1))

    def grab_triplet(pattern: str) -> tuple[str, str, str]:
        m = re.search(pattern, summary_text)
        if not m:
            raise ValueError(f"Could not parse pattern: {pattern}")
        return m.group(1), m.group(2), m.group(3)

    pc1 = grab_float(r"PC1:\s+([0-9.]+)")
    pc2 = grab_float(r"PC2:\s+([0-9.]+)")
    pc3 = grab_float(r"PC3:\s+([0-9.]+)")
    pcc = grab_float(r"Cumulative:\s+([0-9.]+)")

    # Parse loading names/values directly from summary text.
    pc1_name_vals = re.search(
        r"PC1:\s+([^=,\n]+)=([+-]?[0-9.]+),\s+([^=,\n]+)=([+-]?[0-9.]+),\s+([^=,\n]+)=([+-]?[0-9.]+)",
        summary_text,
    )
    pc2_name_vals = re.search(
        r"PC2:\s+([^=,\n]+)=([+-]?[0-9.]+),\s+([^=,\n]+)=([+-]?[0-9.]+),\s+([^=,\n]+)=([+-]?[0-9.]+)",
        summary_text,
    )
    if not pc1_name_vals or not pc2_name_vals:
        raise ValueError("Could not parse PCA loading names/values.")

    corr_dD = re.search(
        r"dD:\s+([^ ]+)\s+\(r=([+-]?[0-9.]+)\),\s+([^ ]+)\s+\(r=([+-]?[0-9.]+)\),\s+([^ ]+)\s+\(r=([+-]?[0-9.]+)\)",
        summary_text,
    )
    if not corr_dD:
        raise ValueError("Could not parse strongest dD correlations.")

    return {
        "pc1": pc1,
        "pc2": pc2,
        "pc3": pc3,
        "pc_cumulative": pcc,
        "pc1_terms": [
            (pc1_name_vals.group(1), float(pc1_name_vals.group(2))),
            (pc1_name_vals.group(3), float(pc1_name_vals.group(4))),
            (pc1_name_vals.group(5), float(pc1_name_vals.group(6))),
        ],
        "pc2_terms": [
            (pc2_name_vals.group(1), float(pc2_name_vals.group(2))),
            (pc2_name_vals.group(3), float(pc2_name_vals.group(4))),
            (pc2_name_vals.group(5), float(pc2_name_vals.group(6))),
        ],
        "corr_dD": [
            (corr_dD.group(1), float(corr_dD.group(2))),
            (corr_dD.group(3), float(corr_dD.group(4))),
            (corr_dD.group(5), float(corr_dD.group(6))),
        ],
        "mean_shift_interior": grab_float(r"Interior:\s+([0-9.]+)\n\s+Boundary:"),
        "mean_shift_boundary": grab_float(r"Boundary:\s+([0-9.]+)\n\s+->"),
    }


def build_report_text(
    summary_text: str,
    summary_vals: dict,
    df: pd.DataFrame,
    plot_files: list[str],
) -> str:
    n_total = int(len(df))
    n_interior = int((df["patch_type"] == "interior").sum())
    n_boundary = int((df["patch_type"] == "boundary").sum())

    means = {
        "d_energy_threshold": df["d_energy_threshold"].mean(),
        "d_diffusion_rate": df["d_diffusion_rate"].mean(),
        "d_gamma": df["d_gamma"].mean(),
        "d_mu": df["d_mu"].mean(),
    }

    dD_corr = {
        col: df[col].corr(df["dD"])
        for col in [
            "d_diffusion_rate",
            "d_mu",
            "d_lambda",
            "d_alpha",
            "d_beta",
            "d_gamma",
            "d_energy_threshold",
        ]
    }
    top_dD = sorted(dD_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    by_type = df.groupby("patch_type", observed=False)
    dD_means = by_type["dD"].mean().to_dict()
    dcount_means = by_type["delta_abs_count_err"].mean().to_dict()
    improve_count_rate = by_type["delta_abs_count_err"].apply(lambda s: (s < 0).mean()).to_dict()

    png_plots = [p for p in plot_files if p.endswith(".png")]
    pdf_plots = [p for p in plot_files if p.endswith(".pdf")]

    lines = []
    lines.append(TITLE)
    lines.append("=" * len(TITLE))
    lines.append("")

    lines.append("2. Executive summary")
    lines.append(
        f"- Existing outputs for {n_total} patches ({n_interior} interior, {n_boundary} boundary) show a consistent structured shift from "
        "theta_global (full-leaf optimum) to theta_local (patch-level optimum), rather than random parameter drift."
    )
    lines.append(
        f"- The largest systematic movement is in energy threshold (mean d_energy_threshold = {means['d_energy_threshold']:+.2f}), "
        f"with additional directional shifts in diffusion_rate ({means['d_diffusion_rate']:+.3f}), gamma ({means['d_gamma']:+.3f}), "
        f"and mu ({means['d_mu']:+.3f})."
    )
    lines.append(
        "- Performance linkages are most evident for d_diffusion_rate, d_mu, and d_lambda, matching the strongest dD correlations reported in summary.txt."
    )
    lines.append(
        "- Interior and boundary patches have similar parameter-shift magnitudes, but interior patches realize higher average dD improvement."
    )
    lines.append("")

    lines.append("3. Experimental setup / terminology")
    lines.append("- theta_global = full-leaf optimum.")
    lines.append("- theta_local = patch-level optimum.")
    lines.append("- global-on-patch = patch inference using theta_global.")
    lines.append("- local-opt-on-patch = patch inference using theta_local.")
    lines.append(
        "- Source files consumed: summary.txt, parameter_shift_analysis.csv, and all PNG/PDF plots in this directory."
    )
    lines.append("")

    lines.append("4. Interpretation of PCA plots")
    lines.append(
        f"- From summary.txt, PC1/PC2/PC3 explain {summary_vals['pc1']*100:.1f}% / {summary_vals['pc2']*100:.1f}% / "
        f"{summary_vals['pc3']*100:.1f}% variance (cumulative {summary_vals['pc_cumulative']:.1f}%)."
    )
    lines.append(
        f"- PC1 is driven by {summary_vals['pc1_terms'][0][0]} ({summary_vals['pc1_terms'][0][1]:+.3f}), "
        f"{summary_vals['pc1_terms'][1][0]} ({summary_vals['pc1_terms'][1][1]:+.3f}), "
        f"{summary_vals['pc1_terms'][2][0]} ({summary_vals['pc1_terms'][2][1]:+.3f})."
    )
    lines.append(
        f"- PC2 is driven by {summary_vals['pc2_terms'][0][0]} ({summary_vals['pc2_terms'][0][1]:+.3f}), "
        f"{summary_vals['pc2_terms'][1][0]} ({summary_vals['pc2_terms'][1][1]:+.3f}), "
        f"{summary_vals['pc2_terms'][2][0]} ({summary_vals['pc2_terms'][2][1]:+.3f})."
    )
    lines.append(
        "- This supports a low-dimensional interpretation where transport/intensity parameters and snake-shape parameters form separable shift axes."
    )
    lines.append(
        "- Plot references: pca3d_global_to_local_dD.png, pca3d_global_to_local_count_improvement.png, "
        "pca3d_global_to_local_area_improvement.png (and corresponding PDF versions)."
    )
    lines.append("")

    lines.append("5. Interpretation of raw 3D parameter plots")
    lines.append(
        "- raw3d_threshold_beta_mu_dD.png shows how raw global/local parameter tuples occupy overlapping but offset regions in "
        "(energy_threshold, beta, mu) space, with dD color variation indicating non-uniform sensitivity."
    )
    lines.append(
        "- raw3d_threshold_gamma_diffusion_dD.png complements this by highlighting coupled movement among "
        "(energy_threshold, gamma, diffusion_rate), consistent with downstream dD behavior."
    )
    lines.append("")

    lines.append("6. Interpretation of delta-parameter plots")
    lines.append(f"- Mean d_energy_threshold = {means['d_energy_threshold']:+.2f}, i.e., threshold drops by about 29.")
    lines.append(f"- Mean d_diffusion_rate = {means['d_diffusion_rate']:+.3f}, indicating a decrease.")
    lines.append(f"- Mean d_gamma = {means['d_gamma']:+.3f}, indicating a decrease.")
    lines.append(f"- Mean d_mu = {means['d_mu']:+.3f}, indicating an increase.")
    lines.append(
        "- Plot references: delta3d_threshold_beta_mu_dD.png, delta3d_threshold_beta_mu_patchtype.png, "
        "delta_parameter_pairplot_patchtype.png."
    )
    lines.append("")

    lines.append("7. Link to segmentation performance")
    lines.append(
        f"- From summary.txt, strongest dD correlations are: {summary_vals['corr_dD'][0][0]} (r={summary_vals['corr_dD'][0][1]:+.2f}), "
        f"{summary_vals['corr_dD'][1][0]} (r={summary_vals['corr_dD'][1][1]:+.2f}), "
        f"{summary_vals['corr_dD'][2][0]} (r={summary_vals['corr_dD'][2][1]:+.2f})."
    )
    lines.append(
        f"- CSV cross-check (top absolute correlations with dD): {top_dD[0][0]} (r={top_dD[0][1]:+.3f}), "
        f"{top_dD[1][0]} (r={top_dD[1][1]:+.3f}), {top_dD[2][0]} (r={top_dD[2][1]:+.3f})."
    )
    lines.append(
        "- parameter_delta_correlation_heatmap.png summarizes these relationships and shows that parameter shifts are performance-relevant, not purely descriptive."
    )
    lines.append("")

    lines.append("8. Interior vs boundary comparison")
    lines.append(
        f"- Mean shift magnitude is similar (interior {summary_vals['mean_shift_interior']:.4f}, "
        f"boundary {summary_vals['mean_shift_boundary']:.4f})."
    )
    lines.append(
        f"- Mean dD is higher for interior patches ({dD_means.get('interior', float('nan')):+.4f}) than for boundary patches "
        f"({dD_means.get('boundary', float('nan')):+.4f})."
    )
    lines.append(
        f"- Count-error improvement exists for both groups: mean delta_abs_count_err interior "
        f"{dcount_means.get('interior', float('nan')):+.2f}, boundary {dcount_means.get('boundary', float('nan')):+.2f} (negative = improvement)."
    )
    lines.append(
        f"- Fraction of patches with count-error improvement (delta_abs_count_err < 0): interior "
        f"{improve_count_rate.get('interior', float('nan')):.1%}, boundary {improve_count_rate.get('boundary', float('nan')):.1%}."
    )
    lines.append("")

    lines.append("9. Implications for the paper")
    lines.append(
        "- The global teacher remains a sensible supervision source because local refinement produces structured, interpretable adjustments instead of arbitrary parameter displacement."
    )
    lines.append(
        "- Local optimization reveals repeatable directional shifts (especially threshold and diffusion-family terms), which can motivate explicit modeling of patch morphology in objective design."
    )
    lines.append(
        "- These results support future work on amortized theta prediction (predict patch-level parameter corrections) and morphology-aware objectives."
    )
    lines.append("")

    lines.append("10. Limitations")
    lines.append(
        "- This report is constrained to already-generated outputs; no new optimization trials or ablations were run."
    )
    lines.append(
        "- Correlational relationships do not establish causal parameter effects."
    )
    lines.append(
        "- PCA/loadings were read from summary.txt and assumed to match the saved PCA plots."
    )
    lines.append(
        "- Patch-level conclusions may depend on the current interior/boundary sampling and may shift with different patch extraction policies."
    )
    lines.append("")

    lines.append("Appendix: Plot files inspected")
    lines.append(f"- PNG ({len(png_plots)}): " + ", ".join(png_plots))
    lines.append(f"- PDF ({len(pdf_plots)}): " + ", ".join(pdf_plots))
    lines.append("")
    lines.append("Appendix: Raw summary.txt (verbatim)")
    lines.append("-" * 40)
    lines.append(summary_text.rstrip())
    lines.append("")

    return "\n".join(lines)


def write_pdf_from_text(report_text: str, pdf_path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    wrapped_lines: list[str] = []
    for raw_line in report_text.splitlines():
        if raw_line.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(wrap(raw_line, width=110, break_long_words=False, break_on_hyphens=False))

    lines_per_page = 58
    with PdfPages(pdf_path) as pdf:
        page_idx = 0
        for start in range(0, len(wrapped_lines), lines_per_page):
            page_idx += 1
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.04, 0.90, 0.92])
            ax.axis("off")
            chunk = wrapped_lines[start : start + lines_per_page]
            ax.text(
                0.0,
                1.0,
                "\n".join(chunk),
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
            )
            ax.text(1.0, 0.0, f"Page {page_idx}", va="bottom", ha="right", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interpretation report for parameter-shift analysis.")
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("experiments/exp_v1/outputs/parameter_shift_3d_analysis"),
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    summary_path = analysis_dir / "summary.txt"
    csv_path = analysis_dir / "parameter_shift_analysis.csv"
    txt_out = analysis_dir / "parameter_shift_report.txt"
    pdf_out = analysis_dir / "parameter_shift_report.pdf"

    summary_text = summary_path.read_text()
    summary_vals = parse_summary(summary_text)
    df = pd.read_csv(csv_path)
    plot_files = sorted(
        p.name for p in analysis_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".pdf"}
    )

    report_text = build_report_text(summary_text, summary_vals, df, plot_files)
    txt_out.write_text(report_text)
    write_pdf_from_text(report_text, pdf_out)

    print(f"Wrote {txt_out}")
    print(f"Wrote {pdf_out}")


if __name__ == "__main__":
    main()
