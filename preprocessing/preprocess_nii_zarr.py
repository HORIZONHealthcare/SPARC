"""Manifest-driven NIfTI (.nii/.nii.gz) -> per-volume zarr in the CT-RATE/TotalSeg schema.

Standard NIfTI store real HU + real voxel spacing in the header (unlike CT-RATE's raw
values), so we read HU as-is via SimpleITK (which reads .nii.gz natively), take spacing
from the header, clip HU, compute the body bbox, and write the same attrs the recon
reader expects (spacing/bbox_lo/bbox_hi/shape/volume_name/done). Same process_one as
preprocess_mha_zarr.py; only the input is a --manifest CSV (ct_path column) so we can
control exactly which files (nested MSD tasks, AMOS CT-only, AbdomenCT-1K parts).

  python preprocessing/preprocess_nii_zarr.py --manifest splits/msd_full_nii.csv \
      --out-dir <zarr_dir> --out-manifest splits/msd_full_all_zarr.csv --workers 32
"""
from __future__ import annotations
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
from src.utils.project_paths import project_relative_path, resolve_project_path
import argparse, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zarr
from numcodecs import Blosc

HU_LO, HU_HI = -1024.0, 3071.0   # faithful range, same as dicom/mha/ts preprocessors
BODY_THR = -500.0
CHUNK = (64, 64, 64)


def _bbox_from_mask(mask):
    lo, hi = [], []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.nonzero(mask.any(axis=other))[0]
        if idx.size == 0:
            return None
        lo.append(int(idx[0])); hi.append(int(idx[-1]))
    return lo, hi


def _stem(p: str) -> str:
    s = Path(p).name
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return Path(p).stem


def process_one(args):
    nii_path, out_path = args
    out = Path(out_path)
    try:
        if out.exists() and zarr.open(str(out), mode="r").attrs.get("done", False):
            return ("skip", nii_path)
    except Exception:
        pass
    try:  # one bad file must never crash the pool
        img = sitk.ReadImage(nii_path)                          # reads .nii.gz natively
        arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (z, y, x)
        sp = img.GetSpacing()                                   # (x, y, z) mm from header
        hu = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))   # -> (x, y, z) axial-last
        zooms = [float(sp[0]), float(sp[1]), float(sp[2])]
        np.clip(hu, HU_LO, HU_HI, out=hu)
        hu = hu.astype(np.int16)
        bbox = _bbox_from_mask(hu > BODY_THR)
        lo, hi = bbox if bbox is not None else ([0, 0, 0], [s - 1 for s in hu.shape])
        out.parent.mkdir(parents=True, exist_ok=True)
        comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        z = zarr.open(str(out), mode="w", shape=hu.shape, chunks=CHUNK, dtype="i2", compressor=comp)
        z[:] = hu
        z.attrs["spacing"] = zooms
        z.attrs["bbox_lo"] = [int(x) for x in lo]
        z.attrs["bbox_hi"] = [int(x) for x in hi]
        z.attrs["shape"] = [int(x) for x in hu.shape]
        z.attrs["volume_name"] = _stem(nii_path)
        z.attrs["done"] = True
    except Exception as e:  # noqa: BLE001
        return (f"err:{type(e).__name__}:{str(e)[:60]}", nii_path)
    return ("ok", nii_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with ct_path column (nii/.nii.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    paths = [
        resolve_project_path(p)
        for p in pd.read_csv(a.manifest)["ct_path"].astype(str)
    ]
    out_dir = Path(a.out_dir)
    # collision guard: some datasets repeat stems across subdirs -> prefix would be needed,
    # but MSD (liver_/lung_/pancreas_/hepaticvessel_/colon_/spleen_), AbdomenCT-1K (Case_),
    # AMOS (amos_) stems are globally unique, so bare stem is safe. Assert uniqueness.
    jobs = [(p, str(out_dir / f"{_stem(p)}.zarr")) for p in paths]
    stems = [Path(o).name for _, o in jobs]
    assert len(set(stems)) == len(stems), f"stem collision: {len(stems)-len(set(stems))} dup(s)"
    print(f"{len(jobs)} volumes, {a.workers} workers", flush=True)
    ok = skip = err = 0
    valid = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs)):
            st, _ = fu.result()
            if st in ("ok", "skip"):
                ok += st == "ok"; skip += st == "skip"; valid.append(futs[fu][1])
            else:
                err += 1
                if err <= 30:
                    print("  ", st, futs[fu][0], flush=True)
            if (i + 1) % 300 == 0:
                print(f"[{i+1}/{len(jobs)}] ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE ok={ok} skip={skip} err={err}", flush=True)
    Path(a.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_manifest, "w") as f:
        w = csv.writer(f); w.writerow(["ct_path"])
        for p in sorted(valid):
            w.writerow([project_relative_path(p)])
    print(f"wrote {a.out_manifest} ({len(valid)} paths)", flush=True)


if __name__ == "__main__":
    main()
