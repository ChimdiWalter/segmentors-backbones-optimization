#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_IN  = "experiments/exp_v1/outputs/opt_summary_local.csv"
OUT_DIR = "experiments/exp_v1/outputs/optimizer_summary"

PARAM_COLS = ["mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold"]
META_COLS  = ["score","elapsed_seconds","n_contours","area_px","status","mode","filename"]

def ensure(p): os.makedirs(p, exist_ok=True)

def save_hist(series, title, xlabel, outpath, bins=30):
    plt.figure()
    plt.hist(series.dropna().to_numpy(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ensure(OUT_DIR)
    df = pd.read_csv(CSV_IN)
    df.columns = df.columns.str.strip()

    # Keep only columns we expect if present
    for c in META_COLS + PARAM_COLS:
        if c not in df.columns:
            pass

    # Status summary
    if "status" in df.columns:
        status_counts = df["status"].astype(str).str.strip().value_counts()
    else:
        status_counts = pd.Series({"(no status column)": len(df)})

    status_path = os.path.join(OUT_DIR, "status_counts.csv")
    status_counts.to_csv(status_path, header=["count"])
    print("Saved:", status_path)

    # If status exists, define ok subset
    ok = df
    if "status" in df.columns:
        ok = df[df["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    # Summary stats table
    summary = {}
    summary["n_total"] = len(df)
    summary["n_ok"] = len(ok)
    summary["ok_fraction"] = (len(ok) / max(1, len(df)))

    for col in ["score","elapsed_seconds","n_contours","area_px"]:
        if col in ok.columns:
            x = pd.to_numeric(ok[col], errors="coerce").dropna().to_numpy()
            if len(x):
                summary[f"{col}_mean"] = float(np.mean(x))
                summary[f"{col}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                summary[f"{col}_median"] = float(np.median(x))
                summary[f"{col}_p10"] = float(np.quantile(x, 0.10))
                summary[f"{col}_p90"] = float(np.quantile(x, 0.90))

    for col in PARAM_COLS:
        if col in ok.columns:
            x = pd.to_numeric(ok[col], errors="coerce").dropna().to_numpy()
            if len(x):
                summary[f"{col}_mean"] = float(np.mean(x))
                summary[f"{col}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                summary[f"{col}_median"] = float(np.median(x))

    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(OUT_DIR, "optimizer_summary_stats.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    # Plots (OK subset)
    if "score" in ok.columns:
        save_hist(pd.to_numeric(ok["score"], errors="coerce"), "Optimizer score distribution (ok)", "score",
                  os.path.join(OUT_DIR, "score_hist.png"))

    if "elapsed_seconds" in ok.columns:
        save_hist(pd.to_numeric(ok["elapsed_seconds"], errors="coerce"), "Runtime per image (seconds) (ok)", "elapsed_seconds",
                  os.path.join(OUT_DIR, "elapsed_seconds_hist.png"))

    if "n_contours" in ok.columns:
        save_hist(pd.to_numeric(ok["n_contours"], errors="coerce"), "Number of contours (components) (ok)", "n_contours",
                  os.path.join(OUT_DIR, "n_contours_hist.png"))

    if "area_px" in ok.columns:
        save_hist(pd.to_numeric(ok["area_px"], errors="coerce"), "Mask area in pixels (ok)", "area_px",
                  os.path.join(OUT_DIR, "area_px_hist.png"))

    # Parameter histograms
    for col in PARAM_COLS:
        if col in ok.columns:
            save_hist(pd.to_numeric(ok[col], errors="coerce"), f"Parameter distribution: {col} (ok)", col,
                      os.path.join(OUT_DIR, f"param_{col}_hist.png"))

    # Save top/bottom by score (if available)
    if "score" in ok.columns:
        ok2 = ok.copy()
        ok2["score"] = pd.to_numeric(ok2["score"], errors="coerce")
        ok2 = ok2.dropna(subset=["score"])
        top = ok2.sort_values("score", ascending=False).head(10)
        bot = ok2.sort_values("score", ascending=True).head(10)
        top.to_csv(os.path.join(OUT_DIR, "top10_by_score.csv"), index=False)
        bot.to_csv(os.path.join(OUT_DIR, "bottom10_by_score.csv"), index=False)
        print("Saved: top10_by_score.csv and bottom10_by_score.csv")

    # Quick status bar plot
    plt.figure()
    status_counts.plot(kind="bar")
    plt.title("Optimizer status counts")
    plt.xlabel("status")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "status_bar.png"), dpi=200)
    plt.close()
    print("Saved plots into:", OUT_DIR)

if __name__ == "__main__":
    main()