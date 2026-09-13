"""TotalSegmentator CT NIfTI -> per-volume zarr in the CT-RATE zarr format.

Unlike CT-RATE (raw stored values + identity affine, needing a metadata CSV),
TotalSegmentator NIfTI are PROPER medical NIfTI: nib.get_fdata() already returns
real Hounsfield units, and the header zooms give real mm spacing. So no metadata
CSV is needed. We just clip HU, compute the body bbox, and write the SAME attrs
schema the recon pipeline reads (spacing / bbox_lo / bbox_hi / shape / done).

  python preprocessing/preprocess_ts_zarr.py \
      --manifest splits/ts_bench_nii.csv --out-dir <zarr_dir> \
      --out-manifest splits/ts_bench_zarr2.csv --workers 16
"""

from __future__ import annotations
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
from src.utils.project_paths import project_relative_path, resolve_project_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from numcodecs import Blosc

HU_LO, HU_HI = -1024.0, 3071.0   # same faithful range as preprocess_ctrate_zarr
BODY_THR = -500.0
CHUNK = (64, 64, 64)


def _bbox_from_mask(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    lo, hi = [], []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.nonzero(mask.any(axis=other))[0]
        if idx.size == 0:
            return None
        lo.append(int(idx[0])); hi.append(int(idx[-1]))
    return lo, hi


def process_one(args: tuple[str, str]) -> tuple[str, str]:
    ct_path, out_path = args
    out = Path(out_path)
    try:
        if out.exists() and zarr.open(str(out), mode="r").attrs.get("done", False):
            return ("skip", ct_path)
    except Exception:
        pass
    try:
        img = nib.load(ct_path)
        hu = img.get_fdata(dtype=np.float32)            # already real HU
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as e:  # noqa: BLE001
        return (f"readerr:{type(e).__name__}", ct_path)

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
    z.attrs["volume_name"] = Path(ct_path).parent.name   # e.g. s0002
    z.attrs["done"] = True
    return ("ok", ct_path)


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with image_path (NIfTI)")
    ap.add_argument("--col", default="image_path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--name-mode", choices=["parent", "stem"], default="parent",
                    help="zarr name from parent dir (TS: s0000/ct.nii.gz) or file stem "
                         "(MSD: imagesTr/liver_0.nii.gz -> liver_0)")
    args = ap.parse_args()

    def _zarr_name(p: str) -> str:
        if args.name_mode == "stem":
            return Path(p).name.split(".")[0]
        return Path(p).parent.name

    df = pd.read_csv(args.manifest)
    paths = [resolve_project_path(p) for p in df[args.col].astype(str)]
    if args.limit:
        paths = paths[: args.limit]
    out_dir = Path(resolve_project_path(args.out_dir))
    jobs = [(p, str(out_dir / f"{_zarr_name(p)}.zarr")) for p in paths]
    print(f"{len(jobs)} volumes, {args.workers} workers", flush=True)

    ok = skip = err = 0
    valid: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs)):
            status, _ = f.result()
            _, out_path = futs[f]
            if status in ("ok", "skip"):
                valid.append(out_path); ok += status == "ok"; skip += status == "skip"
            else:
                err += 1
                if err <= 20:
                    print(f"  ERR {status}  {futs[f][0]}", flush=True)
    print(f"DONE ok={ok} skip={skip} err={err}", flush=True)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ct_path": sorted(project_relative_path(p) for p in valid)}).to_csv(
        args.out_manifest, index=False
    )
    print(f"wrote {args.out_manifest} ({len(valid)} zarr paths)", flush=True)


if __name__ == "__main__":
    main()
