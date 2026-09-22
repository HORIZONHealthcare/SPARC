"""Per-study DICOM (e.g. CQ500 head CT, JPEG-Lossless) -> per-study zarr in the
CT-RATE/TotalSegmentator zarr format.

Each study dir may contain several series subdirs; we pick the leaf series with the
MOST slices (the high-res axial), read it via SimpleITK (which decodes compressed
transfer syntaxes via bundled GDCM and applies rescale slope/intercept -> real HU),
reorient sitk's (z,y,x) to (x,y,z) axial-last to match the nibabel-based zarrs, clip
HU, compute the body bbox, and write the SAME attrs schema the recon pipeline reads
(spacing / bbox_lo / bbox_hi / shape / volume_name / done).

  python preprocessing/preprocess_dicom_zarr.py --root <root with study dirs> \
      --study-glob 'CQ500CT*' --out-dir $DATA_ROOT/processed/cq500 \
      --out-manifest cq500_all.csv --workers 16
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
import csv
import fnmatch
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from numcodecs import Blosc

HU_LO, HU_HI = -1024.0, 3071.0   # faithful range, same as preprocess_ts_zarr
BODY_THR = -500.0
CHUNK = (64, 64, 64)
MIN_SLICES = 20                  # skip scouts/localizers


def _bbox_from_mask(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    lo, hi = [], []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.nonzero(mask.any(axis=other))[0]
        if idx.size == 0:
            return None
        lo.append(int(idx[0])); hi.append(int(idx[-1]))
    return lo, hi


def _best_series_dir(study_dir: str) -> tuple[str | None, int]:
    best, bn = None, 0
    for r, _, files in os.walk(study_dir):
        n = sum(1 for x in files if x.lower().endswith(".dcm"))
        if n > bn:
            bn, best = n, r
    return best, bn


def process_one(args: tuple[str, str]) -> tuple[str, str]:
    study_dir, out_path = args
    out = Path(out_path)
    try:
        if out.exists() and zarr.open(str(out), mode="r").attrs.get("done", False):
            return ("skip", study_dir)
    except Exception:
        pass
    sdir, n = _best_series_dir(study_dir)
    if not sdir or n < MIN_SLICES:
        return (f"noseries:{n}", study_dir)
    try:
        rdr = sitk.ImageSeriesReader()
        rdr.SetFileNames(rdr.GetGDCMSeriesFileNames(sdir))
        img = rdr.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (z, y, x)
        sp = img.GetSpacing()                                  # (x, y, z) mm
    except Exception as e:  # noqa: BLE001
        return (f"readerr:{type(e).__name__}", study_dir)

    hu = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))    # -> (x, y, z) axial-last
    zooms = [float(sp[0]), float(sp[1]), float(sp[2])]
    np.clip(hu, HU_LO, HU_HI, out=hu)
    hu = hu.astype(np.int16)
    bbox = _bbox_from_mask(hu > BODY_THR)
    lo, hi = bbox if bbox is not None else ([0, 0, 0], [s - 1 for s in hu.shape])

    out.parent.mkdir(parents=True, exist_ok=True)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    z = zarr.open(str(out), mode="w", shape=hu.shape, chunks=CHUNK, dtype="i2",
                  compressor=comp)
    z[:] = hu
    z.attrs["spacing"] = zooms
    z.attrs["bbox_lo"] = [int(x) for x in lo]
    z.attrs["bbox_hi"] = [int(x) for x in hi]
    z.attrs["shape"] = [int(x) for x in hu.shape]
    z.attrs["volume_name"] = Path(study_dir).name.split()[0]
    z.attrs["done"] = True
    return ("ok", study_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--study-glob", default="CQ500CT*")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    studies = []
    for r, dirs, _ in os.walk(a.root):
        for dd in dirs:
            if fnmatch.fnmatch(dd, a.study_glob):
                studies.append(os.path.join(r, dd))
    studies = sorted(set(studies))
    # dedup by output zarr name: some datasets (CQ500: 18 cases) have duplicate study
    # folders that map to the same name -> avoids two workers writing the same zarr
    # concurrently (the .partial race) AND same-patient duplication/leakage. Keep first.
    seen, uniq = set(), []
    for s in studies:
        nm = Path(s).name.split()[0]
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append(s)
    if len(uniq) != len(studies):
        print(f"deduped {len(studies)} -> {len(uniq)} unique study names", flush=True)
    studies = uniq
    if a.limit:
        studies = studies[: a.limit]

    out_dir = Path(a.out_dir)
    jobs = [(s, str(out_dir / f"{Path(s).name.split()[0]}.zarr")) for s in studies]
    print(f"{len(jobs)} studies, {a.workers} workers", flush=True)

    ok = skip = err = 0
    valid: list[str] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_one, j): j for j in jobs}
        for fu in as_completed(futs):
            st, _ = fu.result()
            if st in ("ok", "skip"):
                ok += st == "ok"; skip += st == "skip"; valid.append(futs[fu][1])
            else:
                err += 1
                print("  ", st, flush=True)
    print(f"DONE ok={ok} skip={skip} err={err}", flush=True)

    Path(a.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_manifest, "w") as f:
        w = csv.writer(f)
        w.writerow(["ct_path"])
        for p in sorted(valid):
            w.writerow([project_relative_path(p)])
    print(f"wrote {a.out_manifest} ({len(valid)} zarr paths)", flush=True)


if __name__ == "__main__":
    main()
