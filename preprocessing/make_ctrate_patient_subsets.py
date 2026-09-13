"""Create deterministic, nested CT-RATE training subsets by patient.

The manifest may contain multiple volumes per patient. N always means patients;
all volumes belonging to a selected patient are retained.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PATTERN = re.compile(r"^(valid_\d+)(?:_|$)")


def patient_id(ct_path: str) -> str:
    stem = Path(ct_path).name.removesuffix(".zarr")
    match = PATTERN.match(stem)
    if not match:
        raise ValueError(f"cannot parse CT-RATE patient ID from {ct_path!r}")
    return match.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="splits/cls3039_train_zarr.csv")
    ap.add_argument("--sizes", type=int, nargs="+", default=(10, 50, 100))
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", default="splits")
    ap.add_argument(
        "--tag",
        default="",
        help="optional filename suffix, e.g. draw2 -> cls3039_train_pat10_draw2.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "ct_path" not in df.columns:
        raise SystemExit(f"{args.manifest} has no ct_path column")
    df = df.copy()
    df["_patient"] = df["ct_path"].astype(str).map(patient_id)
    patients = np.array(sorted(df["_patient"].unique()))
    largest = max(args.sizes)
    if largest > len(patients):
        raise SystemExit(f"requested N={largest}, but train split has {len(patients)} patients")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous: set[str] = set()
    for n in sorted(set(args.sizes)):
        selected = set(patients[:n].tolist())
        if not previous.issubset(selected):
            raise AssertionError("subsets are not nested")
        subset = df[df["_patient"].isin(selected)].drop(columns="_patient")
        suffix = f"_{args.tag}" if args.tag else ""
        path = out_dir / f"cls3039_train_pat{n}{suffix}.csv"
        subset.to_csv(path, index=False)
        print(f"{path}: patients={n} volumes={len(subset)} seed={args.seed}")
        previous = selected


if __name__ == "__main__":
    main()
