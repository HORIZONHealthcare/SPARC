"""Build a single-column CSV manifest of TotalSegmentator CT paths.

Usage:
    python build_totalseg_manifest.py \
        --root /path/to/totalsegmentator_lite/Images_extracted/Images \
        --out  splits/totalseg_train.csv

The CSV has one column `ct_path` (absolute paths to .nii.gz files), which is
what `src/datasets/drr_dataset.py:DRRDataset` consumes.
"""

from __future__ import annotations
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
from src.utils.project_paths import project_relative_path

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dir containing .nii.gz")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap to first N files (sorted) - useful for smoke")
    parser.add_argument("--recursive", action="store_true",
                        help="recurse into subdirs (CT-RATE is deeply nested)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = sorted(root.rglob("*.nii.gz") if args.recursive else root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"no .nii.gz files in {root}")
    if args.limit is not None:
        files = files[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ct_path": [project_relative_path(f) for f in files]}).to_csv(out, index=False)
    print(
        f"wrote {out} with {len(files)} entries "
        f"(first={files[0].name}, last={files[-1].name})"
    )


if __name__ == "__main__":
    main()
