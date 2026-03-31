#!/usr/bin/env python3
"""Create a fixed 30-image stratified subset from opt_summary_local.csv.

Reads the CSV, filters status==ok, stratifies by score into quantile bins,
and samples proportionally to get exactly 30 images.
Writes: ablation_manifest.txt  (one filename per line)
Creates: ablation_input/       (symlinks to leaves/)
"""

import csv, os, sys, random, math
from pathlib import Path

SEED = 42
N_SAMPLE = 30
N_BINS = 5  # stratification bins by score

BASE = Path(__file__).resolve().parent.parent.parent.parent  # segmentors_backbones/
CSV_PATH = BASE / "experiments" / "exp_v1" / "outputs" / "opt_summary_local.csv"
MANIFEST = BASE / "experiments" / "exp_v1" / "configs" / "ablation_manifest.txt"
LINK_DIR = BASE / "experiments" / "exp_v1" / "configs" / "ablation_input"
LEAVES   = BASE / "leaves"


def main():
    # --- read CSV ---
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"].strip() == "ok":
                rows.append(r)
    print(f"Read {len(rows)} ok rows from {CSV_PATH}")

    if len(rows) < N_SAMPLE:
        sys.exit(f"ERROR: need {N_SAMPLE} ok rows, got {len(rows)}")

    # --- sort by score, assign bins ---
    rows.sort(key=lambda r: float(r["score"]))
    bin_size = len(rows) / N_BINS
    bins = [[] for _ in range(N_BINS)]
    for i, r in enumerate(rows):
        b = min(int(i / bin_size), N_BINS - 1)
        bins[b].append(r)

    print(f"Bin sizes: {[len(b) for b in bins]}")

    # --- stratified sample ---
    rng = random.Random(SEED)
    selected = []
    per_bin = N_SAMPLE // N_BINS  # 6 per bin
    remainder = N_SAMPLE - per_bin * N_BINS

    for i, b in enumerate(bins):
        n = per_bin + (1 if i < remainder else 0)
        n = min(n, len(b))
        chosen = rng.sample(b, n)
        selected.extend(chosen)

    # if we're still short (shouldn't happen), top up from pool
    pool = [r for r in rows if r not in selected]
    while len(selected) < N_SAMPLE:
        selected.append(pool.pop(rng.randrange(len(pool))))

    selected.sort(key=lambda r: r["filename"])
    filenames = [r["filename"] for r in selected]
    scores = [float(r["score"]) for r in selected]
    print(f"Selected {len(filenames)} images, score range: {min(scores):.2f} - {max(scores):.2f}")

    # --- write manifest ---
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as f:
        for fn in filenames:
            f.write(fn + "\n")
    print(f"Manifest: {MANIFEST}")

    # --- create symlink directory ---
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    created, skipped = 0, 0
    for fn in filenames:
        src = LEAVES / fn
        dst = LINK_DIR / fn
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.exists():
            dst.symlink_to(src)
            created += 1
        else:
            print(f"  WARNING: source not found: {src}")
            skipped += 1
    print(f"Symlinks: {created} created, {skipped} missing  ->  {LINK_DIR}")


if __name__ == "__main__":
    main()
