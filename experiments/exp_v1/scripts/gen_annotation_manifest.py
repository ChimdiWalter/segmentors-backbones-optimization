#!/usr/bin/env python3
"""Create annotation_subset_manifest.txt: N=60 images stratified by area_frac quantiles.

Reads param_morphology_metrics.csv, bins by area_frac into 5 quantile groups,
samples 12 per bin (SEED=42), writes manifest to Paper_3/tables/.
"""

import csv, random
from pathlib import Path

SEED = 42
N = 60
N_BINS = 5
SAMPLES_PER_BIN = N // N_BINS  # 12

BASE = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = BASE / "Paper_3" / "tables" / "param_morphology_metrics.csv"
OUT_PATH = BASE / "Paper_3" / "tables" / "annotation_subset_manifest.txt"


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Sort by area_frac
    rows.sort(key=lambda r: float(r["area_frac"]))

    # Bin into N_BINS quantile groups
    bin_size = len(rows) // N_BINS
    bins = []
    for i in range(N_BINS):
        start = i * bin_size
        end = start + bin_size if i < N_BINS - 1 else len(rows)
        bins.append(rows[start:end])

    rng = random.Random(SEED)
    selected = []
    for i, b in enumerate(bins):
        k = min(SAMPLES_PER_BIN, len(b))
        sampled = rng.sample(b, k)
        selected.extend(sampled)
        af_vals = [float(r["area_frac"]) for r in sampled]
        print(f"Bin {i+1}: {len(b)} images, sampled {k}, area_frac range [{min(af_vals):.4f}, {max(af_vals):.4f}]")

    # Write manifest
    with open(OUT_PATH, "w") as f:
        for r in selected:
            f.write(r["filename"] + "\n")

    print(f"\nWrote {OUT_PATH} ({len(selected)} images)")


if __name__ == "__main__":
    main()
