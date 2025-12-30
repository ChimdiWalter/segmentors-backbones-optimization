#!/usr/bin/env python3
"""
collect_leaf_tapes.py

Parse an Org-mode table listing NEF images and "use" flags, find matching TIFF
segment files under a leaf-tape root, and copy matches to a destination folder.

Example mapping:
  [[12r/aleph/17.8/DSC_2303.NEF]]
    → /deltos/c/maize/image_processing/data/leaf_tape_directory/12r/aleph/17.8/DSC_2303_segment_1.tif

If the expected subdir path doesn't exist, the script falls back to a recursive,
case-insensitive search for files named like: <basename>_segment_*.tif[f]
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ORG_LINK_RE = re.compile(
    r"""\|\s*\[\[([^\]]+)\]\]\s*\|\s*([^\|]+)\|""",
    re.IGNORECASE,
)

def parse_org_rows(org_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (relative_nef_path, use_value) from the Org table.
    Only parses lines with [[...]] links in the first column.
    """
    rows: List[Tuple[str, str]] = []
    with org_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ORG_LINK_RE.search(line)
            if not m:
                continue
            rel_path = m.group(1).strip()
            # Normalize whitespace & columns to extract "use"
            # After the first match, the rest of the row is between first and second pipe.
            # Our regex captured the inner link and then one column (use). Trim it:
            use_val = m.group(2).strip()
            # Collapse any spaces
            use_val = use_val.replace(" ", "")
            rows.append((rel_path, use_val))
    return rows

def wants_use(value: str) -> bool:
    """
    Keep only '1' (accept). Ignore '0' and '?'.
    """
    return value == "1"

def case_insensitive_glob(base: Path, pattern: str) -> List[Path]:
    """
    Perform a case-insensitive glob by filtering manually.
    pattern is a simple glob like '**/*segment_*.tif' (no regex chars).
    """
    # Python's glob is case-sensitive on Linux; we emulate CI matching
    # by walking and filtering with lower().
    # To keep it efficient, only use this when needed.
    parts = pattern.split("**")
    if len(parts) == 1:
        # No recursion — just listdir and match
        wanted = pattern.lower()
        parent = base
        results = []
        for p in parent.glob("*"):
            if fnmatch_ci(p.name, Path(pattern).name):
                results.append(p)
        return results
    else:
        # Recursive: we'll walk the tree and test fnmatch on the tail
        tail = parts[-1].lstrip("/\\")
        results = []
        for p in base.rglob("*"):
            if p.is_file() and fnmatch_ci(p.name, tail):
                results.append(p)
        return results

def fnmatch_ci(name: str, pat: str) -> bool:
    # Simple case-insensitive fnmatch for '*' and '?' wildcards
    # Use fnmatch with lowercased strings.
    import fnmatch as _fn
    return _fn.fnmatch(name.lower(), pat.lower())

def find_expected_candidates(
    leaf_tape_root: Path, subdir: Path, basename_no_ext: str
) -> List[Path]:
    """
    Look in the expected subdir first:
      leaf_tape_root / subdir / <basename>_segment_*.tif[f]
    If nothing found, do a recursive fallback search under leaf_tape_root for
    files with the same <basename>_segment_*.tif[f], case-insensitive.
    """
    expected_dir = (leaf_tape_root / subdir).resolve()
    patterns = [
        f"{basename_no_ext}_segment_*.tif",
        f"{basename_no_ext}_segment_*.tiff",
    ]
    hits: List[Path] = []

    if expected_dir.exists():
        for pat in patterns:
            hits += [p for p in expected_dir.glob(pat)]
        if hits:
            return sorted(hits)

        # Case-insensitive within the expected dir (rarely needed)
        for pat in patterns:
            hits += case_insensitive_glob(expected_dir, pat)
        if hits:
            return sorted(set(hits))

    # Fallback: recursive, case-insensitive search across the whole tree
    fallback_hits: List[Path] = []
    for pat in patterns:
        # We only compare the file name part in a CI way
        for p in leaf_tape_root.rglob("*"):
            if p.is_file() and fnmatch_ci(p.name, Path(pat).name):
                # And ensure the stem before _segment_ matches basename_no_ext (CI)
                stem = p.stem  # e.g., "DSC_0123_segment_1"
                if stem.lower().startswith((basename_no_ext.lower() + "_segment_")):
                    fallback_hits.append(p)

    return sorted(set(fallback_hits))

def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # Avoid overwriting by adding a numeric suffix if needed
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}__{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)
    return dst

def main():
    ap = argparse.ArgumentParser(
        description="Collect matching leaf-tape TIFFs for NEF images marked use==1 in an Org table."
    )
    ap.add_argument(
        "--org",
        required=True,
        type=Path,
        help="Path to possible_images.org (Org table).",
    )
    ap.add_argument(
        "--leaf-tape-root",
        required=True,
        type=Path,
        help="Root dir of leaf_tape_directory (e.g., /deltos/c/maize/image_processing/data/leaf_tape_directory).",
    )
    ap.add_argument(
        "--dest",
        required=True,
        type=Path,
        help="Destination directory to copy matching TIFFs into (replace your XXXXXXXXXX).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; just print what would happen and write the report.",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("collect_leaf_tapes_report.csv"),
        help="Where to write the CSV report (default: ./collect_leaf_tapes_report.csv).",
    )

    args = ap.parse_args()

    rows = parse_org_rows(args.org)
    if not rows:
        print(f"[!] No rows parsed from {args.org}")
        return

    selected = [(rel, use) for (rel, use) in rows if wants_use(use)]
    print(f"Parsed {len(rows)} rows; {len(selected)} with use==1.")

    # Each rel path looks like: '16r/gimmel/26.7/DSC_0090.NEF'
    # We expect TIFFs under leaf_tape_root / '16r/gimmel/26.7' / 'DSC_0090_segment_*.tif[f]'
    results = []
    copied_count = 0
    missing_count = 0
    multi_count = 0

    for rel, use in selected:
        rel_path = Path(rel)
        subdir = rel_path.parent  # e.g., 16r/gimmel/26.7
        nef_name = rel_path.name  # e.g., DSC_0090.NEF
        basename = Path(nef_name).stem  # 'DSC_0090'
        basename_no_ext = basename

        candidates = find_expected_candidates(args.leaf_tape_root, subdir, basename_no_ext)

        status = "FOUND_NONE"
        copied_files: List[str] = []
        note = ""

        if not candidates:
            status = "FOUND_NONE"
            missing_count += 1
        else:
            status = "FOUND_ONE" if len(candidates) == 1 else "FOUND_MULTI"
            if len(candidates) > 1:
                multi_count += 1

            for src in candidates:
                if args.dry_run:
                    copied_files.append(str(src))
                else:
                    dst = safe_copy(src, args.dest)
                    copied_files.append(str(dst))
                    copied_count += 1

        results.append(
            {
                "org_rel_nef": str(rel_path),
                "use": use,
                "expected_subdir": str(subdir),
                "basename": basename_no_ext,
                "found_count": len(candidates),
                "status": status,
                "copied_or_sources": ";".join(copied_files),
            }
        )

    # Write report
    fieldnames = [
        "org_rel_nef",
        "use",
        "expected_subdir",
        "basename",
        "found_count",
        "status",
        "copied_or_sources",
    ]
    with args.report.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    print("\n=== Summary ===")
    print(f"Org rows parsed:         {len(rows)}")
    print(f"Rows with use==1:        {len(selected)}")
    print(f"Missing matches:         {missing_count}")
    print(f"Multiple matches:        {multi_count}")
    if args.dry_run:
        print(f"Dry-run: no files copied. Would have copied total of {sum(r['found_count'] for r in results)} files.")
    else:
        print(f"Files copied:            {copied_count}")
    print(f"Report written to:       {args.report.resolve()}")

if __name__ == "__main__":
    main()
