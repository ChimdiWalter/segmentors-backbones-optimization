#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_mode_name(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    for pref in ("navier_output_", "navier_", "output_", "results_"):
        if base.startswith(pref):
            return base[len(pref):] or base
    return base or "unknown"

def load_results(dir_path: str) -> pd.DataFrame:
    csv_path = os.path.join(dir_path, "results_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing results_log.csv in {dir_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    needed = {"filename","mu","lambda","diffusion_rate","n_contours","total_area_px"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    df["mode"]  = infer_mode_name(dir_path)
    df["combo"] = list(zip(df["mu"], df["lambda"], df["diffusion_rate"]))
    return df

def parse_combo_list(s: str):
    out = []
    for chunk in s.split(";"):
        a = [float(x) for x in chunk.split(",")]
        if len(a) != 3:
            raise ValueError(f"Bad combo '{chunk}'. Use μ,λ,d triples, ';' separated.")
        out.append(tuple(a))
    return out

def combo_label(combo):
    mu, lam, dr = combo
    return f"μ={mu:g}, λ={lam:g}, d={dr:g}"

def distinct_colors(n):
    # tab20 gives 20 visually distinct hues; cycle if needed
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]

def plot_mode(df_mode, combos, colors, kind, out_dir):
    """kind: 'raw' or 'avg'"""
    mode = df_mode["mode"].iloc[0]

    if kind == "avg":
        plot_df = (df_mode.groupby(["combo","mu","lambda","diffusion_rate"], as_index=False)
                   .agg(n_contours=("n_contours","mean"),
                        total_area_px=("total_area_px","mean"),
                        n_runs=("filename","count")))
        title_suffix = "(avg over images)"
        fname_suffix = "avg"
    else:
        plot_df = df_mode.copy()
        title_suffix = "(raw)"
        fname_suffix = "raw"

    # n_contours vs total_area scatter, colored per combo (consistent across modes)
    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111)

    handles = []
    for i, combo in enumerate(combos):
        sub = plot_df[plot_df["combo"] == combo]
        if sub.empty:
            # nothing for this combo in this mode; just note it later
            continue
        size = 22 if kind == "raw" else (60 + 10*np.log1p(sub["n_runs"].iloc[0]))
        sc = ax.scatter(
            sub["n_contours"].values,
            sub["total_area_px"].values,
            s=size,
            c=[colors[combo]],
            edgecolors="k", linewidths=0.5, alpha=0.9,
            label=combo_label(combo)
        )

    ax.set_xlabel("n_contours")
    ax.set_ylabel("total_area_px")
    ax.set_title(f"{mode}: {title_suffix} — n_contours vs total_area")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(title="(μ, λ, d)", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{mode}__{fname_suffix}__ncontours_vs_area.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="2D plots of n_contours vs total_area from Navier logs.")
    ap.add_argument("dirs", nargs="+", help="One or more directories with results_log.csv")
    ap.add_argument("--out", default=None, help="Folder to save figures and CSV (default: first input dir)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--union", action="store_true", help="Use union of all combos across dirs (default).")
    grp.add_argument("--intersection", action="store_true", help="Use only combos present in ALL dirs.")
    ap.add_argument("--combos", type=str, default=None,
                    help="Explicit combo list 'μ,λ,d;μ,λ,d;...' to force same #points per mode (e.g., 8).")
    args = ap.parse_args()

    # Load data
    dfs = [load_results(d) for d in args.dirs]
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=["n_contours","total_area_px","mu","lambda","diffusion_rate"])

    # Decide combo set
    if args.combos:
        chosen = parse_combo_list(args.combos)
        chosen = [tuple(map(float, c)) for c in chosen]
        combos = chosen
        combo_set_name = f"user-specified ({len(combos)})"
    else:
        per_dir_sets = [set(df["combo"].unique()) for df in dfs]
        if args.intersection:
            combos = sorted(set.intersection(*per_dir_sets))
            combo_set_name = f"intersection ({len(combos)})"
        else:
            combos = sorted(set.union(*per_dir_sets))
            combo_set_name = f"union ({len(combos)})"

    if len(combos) == 0:
        raise SystemExit("No combos to plot (check your inputs / flags).")

    # Stable color mapping across modes
    palette = distinct_colors(len(combos))
    colors = {c: palette[i] for i, c in enumerate(combos)}

    out_dir = args.out or args.dirs[0]
    os.makedirs(out_dir, exist_ok=True)

    # Summary table (restricted to selected combos)
    filtered = data[data["combo"].isin(combos)].copy()
    summary = (filtered
        .groupby(["mode","mu","lambda","diffusion_rate"], as_index=False)
        .agg(
            runs=("filename","count"),
            mean_n_contours=("n_contours","mean"),
            median_n_contours=("n_contours","median"),
            mean_total_area=("total_area_px","mean"),
            median_total_area=("total_area_px","median"),
        )
        .sort_values(["mode","mean_n_contours"], ascending=[True, False])
    )
    summary_path = os.path.join(out_dir, "nv_summary_by_combo.csv")
    summary.to_csv(summary_path, index=False)

    # Per-mode plots
    saved = []
    missing_report = []
    for mode, dfm in filtered.groupby("mode"):
        # report missing combos (so you know why a mode has fewer than, say, 8 points)
        present = set(dfm["combo"].unique())
        missing = [c for c in combos if c not in present]
        if missing:
            missing_report.append((mode, missing))
        saved.append(plot_mode(dfm, combos, colors, "raw", out_dir))
        saved.append(plot_mode(dfm, combos, colors, "avg", out_dir))

    # Console report
    print(f"Combo set: {combo_set_name}")
    if missing_report:
        print("\nMissing combos by mode (consider rerunning those combos to get 8 vs 8 parity):")
        for mode, miss in missing_report:
            print(f"  {mode}: {', '.join(combo_label(c) for c in miss)}")
    else:
        print("\nAll selected combos present in all modes.")

    # Quick correlations
    for mode, dfm in filtered.groupby("mode"):
        corr = dfm["n_contours"].corr(dfm["total_area_px"])
        print(f"[{mode}] corr(n_contours, total_area_px) = {corr:.3f}")

    print("\nSaved files:")
    for p in saved:
        print("  ", p)
    print("  ", summary_path)

if __name__ == "__main__":
    main()
