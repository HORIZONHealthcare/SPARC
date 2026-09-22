"""CT-RATE NIfTI -> per-volume zarr (correct HU + real spacing + body bbox).

WHY: CT-RATE NIfTI store RAW stored values (e.g. 0..16270), NOT Hounsfield.
The real HU = stored * RescaleSlope + RescaleIntercept (from train_metadata.csv or validation_metadata.csv;
typically slope=1, intercept=-8192). The NIfTI affine is identity, so the real
voxel spacing is also lost in the file and must come from XYSpacing + ZSpacing in
the metadata. Training directly on the raw values (the old pipeline) clamps all
soft tissue to a constant -> the encoder learns nothing. This script fixes that:

  HU = stored * slope + intercept  ->  clip [-1024, 3071]  ->  int16

stored as a chunked zarr for fast random-crop reads, with attrs:
  spacing  = [sx, sy, sz] mm     (real, from metadata)
  bbox_lo  = [i, j, k]           (body bbox, HU > -500; for foreground crops)
  bbox_hi  = [i, j, k]
  shape    = [D, H, W]
  done     = True                (resume guard)
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
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

# Store the FULL diagnostic CT range so the corpus is faithful + reusable for
# any CT foundation-model project (bone/contrast up to ~3071 HU preserved).
# Models clip to their own window when loading (SPARC uses [-1024, 1024], set by
# hu_min/hu_max in each config).
HU_LO, HU_HI = -1024.0, 3071.0
BODY_THR = -500.0
CHUNK = (64, 64, 64)

_META: dict | None = None


def parse_xy(s: str) -> tuple[float, float]:
    """'[0.82, 0.82]' -> (0.82, 0.82)."""
    v = ast.literal_eval(s)
    return float(v[0]), float(v[1])


def load_meta(path: str) -> dict:
    df = pd.read_csv(
        path,
        usecols=["VolumeName", "XYSpacing", "ZSpacing", "RescaleSlope", "RescaleIntercept"],
    )
    m: dict = {}
    for _, r in df.iterrows():
        sx, sy = parse_xy(r["XYSpacing"])
        m[str(r["VolumeName"])] = dict(
            slope=float(r["RescaleSlope"]),
            intercept=float(r["RescaleIntercept"]),
            sx=sx, sy=sy, sz=float(r["ZSpacing"]),
        )
    return m


def _init(meta_path: str) -> None:
    global _META
    _META = load_meta(meta_path)


def _bbox_from_mask(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    """Body bbox via 1D axis projections (memory-cheap vs np.argwhere)."""
    lo, hi = [], []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        proj = mask.any(axis=other)
        idx = np.nonzero(proj)[0]
        if idx.size == 0:
            return None
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]))
    return lo, hi


def process_one(args: tuple[str, str]) -> tuple[str, str]:
    ct_path, out_path = args
    out = Path(out_path)
    try:
        if out.exists():
            z = zarr.open(str(out), mode="r")
            if z.attrs.get("done", False):
                return ("skip", ct_path)
    except Exception:
        pass

    assert _META is not None
    name = Path(ct_path).name
    md = _META.get(name)
    if md is None:
        return ("nometa", ct_path)

    try:
        stored = nib.load(ct_path).get_fdata(dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        return (f"readerr:{type(e).__name__}", ct_path)

    # In-place HU rescale + clip to keep peak memory ~2x the volume (many
    # workers run in parallel; np.argwhere on a full mask would spike to ~1GB).
    stored *= md["slope"]
    stored += md["intercept"]
    np.clip(stored, HU_LO, HU_HI, out=stored)
    hu = stored.astype(np.int16)
    del stored

    bbox = _bbox_from_mask(hu > BODY_THR)
    if bbox is None:
        lo, hi = [0, 0, 0], [s - 1 for s in hu.shape]
    else:
        lo, hi = bbox

    out.parent.mkdir(parents=True, exist_ok=True)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    z = zarr.open(str(out), mode="w", shape=hu.shape, chunks=CHUNK,
                  dtype="i2", compressor=comp)
    z[:] = hu
    z.attrs["spacing"] = [md["sx"], md["sy"], md["sz"]]
    z.attrs["bbox_lo"] = [int(x) for x in lo]
    z.attrs["bbox_hi"] = [int(x) for x in hi]
    z.attrs["shape"] = [int(x) for x in hu.shape]
    z.attrs["volume_name"] = name
    z.attrs["done"] = True
    return ("ok", ct_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with ct_path (NIfTI)")
    ap.add_argument("--metadata", required=True, help="train_metadata.csv")
    ap.add_argument("--out-dir", required=True, help="zarr output dir")
    ap.add_argument("--out-manifest", required=True, help="CSV of zarr paths to write")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    paths = [
        resolve_project_path(p)
        for p in pd.read_csv(args.manifest)["ct_path"].astype(str)
    ]
    out_dir = Path(args.out_dir)
    jobs = []
    for p in paths:
        stem = Path(p).name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        jobs.append((p, str(out_dir / f"{stem}.zarr")))

    print(f"{len(jobs)} volumes, {args.workers} workers", flush=True)
    ok = skip = err = 0
    valid: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init, initargs=(args.metadata,)) as ex:
        futs = {ex.submit(process_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs)):
            status, _ = f.result()
            _, out_path = futs[f]
            if status in ("ok", "skip"):
                valid.append(out_path)
                ok += status == "ok"
                skip += status == "skip"
            else:
                err += 1
                if err <= 20:
                    print(f"  ERR {status}  {futs[f][0]}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"[{i+1}/{len(jobs)}] ok={ok} skip={skip} err={err}", flush=True)

    print(f"DONE ok={ok} skip={skip} err={err}", flush=True)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ct_path": sorted(project_relative_path(p) for p in valid)}).to_csv(
        args.out_manifest, index=False
    )
    print(f"wrote {args.out_manifest} ({len(valid)} zarr paths)", flush=True)


if __name__ == "__main__":
    main()
