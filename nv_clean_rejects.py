#!/usr/bin/env python3
import argparse, os, sys, csv, glob, shutil
from datetime import datetime

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_reject_stems(reject_dir):
    if not os.path.isdir(reject_dir):
        raise FileNotFoundError(f"reject folder not found: {reject_dir}")
    rej = {stem(p) for p in glob.glob(os.path.join(reject_dir, "*")) if os.path.isfile(p)}
    return rej

def find_log_csvs(out_dir, recursive=True):
    if recursive:
        return glob.glob(os.path.join(out_dir, "**", "results_log.csv"), recursive=True)
    p = os.path.join(out_dir, "results_log.csv")
    return [p] if os.path.isfile(p) else []

def delete_file(path, dry_run=False):
    if not path: return False
    if not os.path.exists(path): return False
    if dry_run:
        print(f"[dry-run] rm {path}")
        return True
    try:
        os.remove(path)
        return True
    except Exception as e:
        print(f"[warn] could not delete {path}: {e}")
        return False

def backup_csv(csv_path):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"{csv_path}.bak.{ts}"
    shutil.copy2(csv_path, bak)
    print(f"[backup] {csv_path} -> {bak}")
    return bak

def process_csv(csv_path, reject_stems, dry_run=False, keep_backups=True):
    # Read all rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "filename" not in fieldnames:
        print(f"[skip] {csv_path} missing 'filename' column")
        return 0, 0

    # Prepare output
    keep_rows = []
    removed = 0
    deleted_files = 0

    # Optional columns that carry file paths
    overlay_col = next((c for c in fieldnames if c.lower() == "overlay16_path"), None)
    mask_col    = next((c for c in fieldnames if c.lower() == "mask16_path"), None)

    for row in rows:
        src_stem = stem(row["filename"])
        if src_stem in reject_stems:
            removed += 1
            # delete files if paths present
            if overlay_col and row.get(overlay_col):
                if delete_file(row[overlay_col], dry_run=dry_run):
                    deleted_files += 1
            if mask_col and row.get(mask_col):
                if delete_file(row[mask_col], dry_run=dry_run):
                    deleted_files += 1
        else:
            keep_rows.append(row)

    if removed == 0:
        print(f"[ok] {csv_path}: nothing to remove")
        return 0, 0

    print(f"[edit] {csv_path}: removing {removed} rows; deleting {deleted_files} files")
    if not dry_run:
        if keep_backups:
            backup_csv(csv_path)
        # write filtered CSV preserving column order
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(keep_rows)

    return removed, deleted_files

def main():
    ap = argparse.ArgumentParser(
        description="Remove Navier results for leaves found in reject_leaves/ and rewrite results_log.csv."
    )
    ap.add_argument("reject_leaves", help="Folder containing rejected leaf images (filenames define what to drop)")
    ap.add_argument("out_dirs", nargs="+", help="One or more output dirs (e.g., navier_output navier_output_2)")
    ap.add_argument("--no-recursive", action="store_true", help="Do not recurse for results_log.csv files")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be removed; do not delete or rewrite")
    ap.add_argument("--no-backup", action="store_true", help="Do not write CSV backups before editing")
    args = ap.parse_args()

    reject = load_reject_stems(args.reject_leaves)
    if not reject:
        print("[warn] reject_leaves is empty — nothing to do.")
        sys.exit(0)

    print(f"[info] reject stems: {len(reject)}")

    total_removed = 0
    total_deleted = 0
    for out_dir in args.out_dirs:
        if not os.path.isdir(out_dir):
            print(f"[skip] output dir not found: {out_dir}")
            continue
        csvs = find_log_csvs(out_dir, recursive=not args.no_recursive)
        if not csvs:
            print(f"[skip] no results_log.csv found under {out_dir}")
            continue
        for csv_path in csvs:
            removed, deleted = process_csv(
                csv_path, reject, dry_run=args.dry_run, keep_backups=not args.no_backup
            )
            total_removed += removed
            total_deleted += deleted

    print(f"\n[summary] rows removed: {total_removed} | files deleted: {total_deleted}")
    if args.dry_run:
        print("[summary] (dry-run) — no files were actually deleted and CSVs were not rewritten.")

if __name__ == "__main__":
    main()
