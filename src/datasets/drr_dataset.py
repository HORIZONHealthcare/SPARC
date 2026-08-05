"""Dataset that yields paired (CT, sparse 2D X-ray bag, Plücker rays).

For each sample we:
1. Pick a CT volume (from a CSV manifest, or the bundled example).
2. Sample one of 5 view-geometry patterns + V views + SAD / SDD.
3. Render V DRRs with DiffDRR + collect per-pixel Plücker for each ray.

The CT is returned untouched (apart from normalisation) so the 3D target
encoder can ingest it independently. Pretraining loss is feature-level
(V-JEPA-style) so the dataset itself does not need to know what the encoder
will do with it.

Rendering happens in the worker process. DiffDRR is fast on GPU; if the
worker has no GPU, we fall back to CPU which is acceptable for smoke.
Production will move DRR to the main GPU and pre-batch geometries.

Eval mode: pass any combination of `fixed_v` / `fixed_pattern` / `fixed_sad_mm`
/ `fixed_sdd_ratio` / `fixed_elev_deg` + `deterministic=True` to lock the
geometry to a single configuration so PSNR is comparable across eval calls.
For the standard recon benchmark we use uniform_circular + V=8 + sad=1000 +
sdd_ratio=1.5 + elev=0 -> 100% reproducible per (CT, fixed config).
"""

from __future__ import annotations
from src.utils.project_paths import resolve_project_path

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.crop_resample import (
    crop_resample_cube, crop_resample_zarr, sample_target_spacing,
)
from src.utils.plucker import compute_plucker

PATTERNS = (
    "uniform_circular",
    "random_circular",
    "limited_angle",
    "bi_planar",
    "cluster",
)


@dataclass
class SampleGeometry:
    pattern: str
    v: int
    azimuths_rad: torch.Tensor
    elevations_rad: torch.Tensor
    sad_mm: float
    sdd_mm: float
    det_height_px: int
    det_width_px: int
    det_pixel_mm: float


def _sample_pattern(rng: random.Random) -> str:
    return rng.choice(PATTERNS)


def _sample_v(rng: random.Random, v_min: int, v_max: int) -> int:
    """Log-uniform view count, clipped to [v_min, v_max]."""
    log_v = rng.uniform(math.log(v_min), math.log(v_max))
    return max(v_min, min(v_max, int(round(math.exp(log_v)))))


def _sample_azimuths(pattern: str, v: int, rng: random.Random) -> list[float]:
    # SPARSE-VIEW CONVENTION: for line-integral DRR, theta and theta+180 are
    # mirror images of the same physical rays through the patient (same info,
    # cone-beam magnification differs slightly). So we span [0, 180) - same
    # angular density as 360 but 2x unique views. Matches clinical short-scan.
    if pattern == "uniform_circular":
        return [i * 180.0 / v for i in range(v)]
    if pattern == "random_circular":
        return sorted(rng.uniform(0, 180) for _ in range(v))
    if pattern == "limited_angle":
        start = rng.uniform(0, 180)
        width = rng.uniform(30, 120)
        return [start + i * width / max(v - 1, 1) for i in range(v)]
    if pattern == "bi_planar":
        v = min(v, 4)
        base = [0.0, 90.0, 45.0, 135.0][:v]   # already within [0, 180)
        return [b + rng.uniform(-10, 10) for b in base]
    if pattern == "cluster":
        n_clusters = rng.choice([2, 3])
        per = max(1, v // n_clusters)
        centers = sorted(rng.uniform(0, 180) for _ in range(n_clusters))
        azs = []
        for c in centers:
            azs.extend(c + rng.uniform(-15, 15) for _ in range(per))
        return azs[:v]
    raise ValueError(pattern)


def sample_geometry(
    rng: random.Random,
    ct_diameter_mm: float,
    v_min: int,
    v_max: int,
    det_px: int,
    fixed_pattern: str | None = None,
    fixed_v: int | None = None,
    fixed_sad_mm: float | None = None,
    fixed_sdd_ratio: float | None = None,
    fixed_elev_deg: float | None = None,
    fixed_det_pixel_mm: float | None = None,
) -> SampleGeometry:
    """Sample (or pin) one view geometry.

    Any `fixed_*` arg overrides the corresponding random draw, leaving other
    fields random. Pass all of them to get a 100% deterministic geometry per
    CT (used by eval / benchmark). `fixed_det_pixel_mm` pins the detector
    pixel size (e.g., C2RV uses 2.4mm) instead of the diameter-adaptive sizing.
    """
    pattern = fixed_pattern if fixed_pattern is not None else _sample_pattern(rng)
    if fixed_v is not None:
        v = fixed_v
    elif pattern == "bi_planar":
        v = rng.choice([2, 3, 4])
    else:
        v = _sample_v(rng, v_min, v_max)
    sad = fixed_sad_mm if fixed_sad_mm is not None else rng.uniform(600.0, 2000.0)
    sdd_ratio = (
        fixed_sdd_ratio if fixed_sdd_ratio is not None else rng.uniform(1.05, 2.5)
    )
    sdd = sad * sdd_ratio
    if fixed_det_pixel_mm is not None:
        det_pixel_mm = fixed_det_pixel_mm
    else:
        det_size_mm = ct_diameter_mm * sdd_ratio * 1.1
        det_pixel_mm = det_size_mm / det_px

    if fixed_pattern is not None and fixed_v is not None:
        # Deterministic azimuths when both fixed - no rng calls for the canonical
        # benchmark setup (uniform_circular + fixed V).
        if pattern == "uniform_circular":
            azs_deg = [i * 360.0 / v for i in range(v)]
        else:
            azs_deg = _sample_azimuths(pattern, v, rng)
    else:
        azs_deg = _sample_azimuths(pattern, v, rng)
    v = len(azs_deg)                # cluster pattern can return fewer

    azs_t = torch.tensor([a % 360 for a in azs_deg], dtype=torch.float32)
    elev_val = (
        fixed_elev_deg if fixed_elev_deg is not None else rng.uniform(-5.0, 5.0)
    )
    elev_t = torch.full((v,), float(elev_val), dtype=torch.float32)

    return SampleGeometry(
        pattern=pattern,
        v=v,
        azimuths_rad=azs_t,            # name kept for backward compat; now degrees
        elevations_rad=elev_t,
        sad_mm=sad,
        sdd_mm=sdd,
        det_height_px=det_px,
        det_width_px=det_px,
        det_pixel_mm=det_pixel_mm,
    )


class DRRDataset(Dataset):
    """Online DRR dataset (backed by nanodrr).

    Args:
        manifest: CSV path with column `ct_path` (one row per CT volume).
        v_min, v_max: view-count range (log-uniform). Ignored when fixed_v set.
        det_px: detector resolution in pixels (square).
        device: where to run nanodrr rendering. "cpu" works in workers.
        seed: base seed; each worker XORs in its own worker_id (unless
            deterministic=True).
        deterministic: when True, drop the worker_id XOR so all workers /
            eval calls draw the same sequence for the same idx. Use for eval.
        fixed_v / fixed_pattern / fixed_sad_mm / fixed_sdd_ratio /
            fixed_elev_deg: pin geometry to single values for benchmark eval.
    """

    def __init__(
        self,
        manifest: str | Path,
        v_min: int = 2,
        v_max: int = 16,
        det_px: int = 128,
        device: str = "cpu",
        seed: int = 0,
        deterministic: bool = False,
        fixed_v: int | None = None,
        fixed_pattern: str | None = None,
        fixed_sad_mm: float | None = None,
        fixed_sdd_ratio: float | None = None,
        fixed_elev_deg: float | None = None,
        fixed_det_pixel_mm: float | None = None,
        npy_spacing_mm: float = 1.0,
        crop_aug: bool = False,
        crop_out_size: int = 256,
        spacing_lo: float = 1.0,
        spacing_hi: float = 2.0,
        center_crop: bool = False,
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.det_px = det_px
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.fixed_v = fixed_v
        self.fixed_pattern = fixed_pattern
        self.fixed_sad_mm = fixed_sad_mm
        self.fixed_sdd_ratio = fixed_sdd_ratio
        self.fixed_elev_deg = fixed_elev_deg
        self.fixed_det_pixel_mm = fixed_det_pixel_mm
        self.npy_spacing_mm = npy_spacing_mm
        self.crop_aug = crop_aug
        self.crop_out_size = crop_out_size
        self.spacing_lo = spacing_lo
        self.spacing_hi = spacing_hi
        self.center_crop = center_crop
        df = pd.read_csv(manifest)
        assert "ct_path" in df.columns, "manifest must have ct_path column"
        self.ct_paths = [resolve_project_path(p) for p in df["ct_path"].astype(str)]

    def __len__(self) -> int:
        return len(self.ct_paths)

    def _load_raw(self, ct_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load raw CT HU (D, H, W) + voxel->world affine (4, 4).

        Dispatches by file extension:
          - .npy: uint8 HU-clipped [-1024,1024] -> recovered float HU, or
            float16/32 -> direct HU. Synthetic isotropic `npy_spacing_mm` affine.
          - .nii / .nii.gz: nibabel (preserves real affine + spacing).
        Keeps the worker light: IO only, no diameter / mask work.
        """
        if ct_path.endswith(".npy"):
            arr = np.load(ct_path)
            if arr.dtype == np.uint8:
                data = arr.astype(np.float32) * (2048.0 / 255.0) - 1024.0
            else:
                data = arr.astype(np.float32)        # float16/32 npy = direct HU
            s = self.npy_spacing_mm
            affine = np.diag([s, s, s, 1.0]).astype(np.float64)
        else:
            import nibabel as nib
            img = nib.load(ct_path)
            data = img.get_fdata(dtype=np.float32)         # (D, H, W) HU
            affine = np.asarray(img.affine, dtype=np.float64)
        return data, affine

    @staticmethod
    def _diameter_from(data: np.ndarray, affine: np.ndarray) -> float:
        """Patient bbox max single-axis extent (mm) -> detector sizing."""
        mask = data > -500.0
        if mask.any():
            nz = np.argwhere(mask)
            bbox_voxel = nz.max(axis=0) - nz.min(axis=0) + 1
        else:
            bbox_voxel = np.array(data.shape)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)    # (3,) mm
        return float((bbox_voxel * spacing).max())

    def _load_ct_and_diameter(self, ct_path: str, rng: random.Random):
        """Load CT HU + affine and compute patient diameter.

        With `crop_aug` (Stage 2 foundation pretrain) we crop a random
        off-centre native region and resample it to `crop_out_size`^3 at a
        random isotropic spacing in [spacing_lo, spacing_hi] mm -- the returned
        cube IS the resampled CT (DRR source + GT both derive from it). Without
        `crop_aug` (benchmark eval) the raw volume is returned untouched.
        """
        if self.crop_aug:
            if self.deterministic:
                np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
            else:
                np_rng = np.random.default_rng()
            s = sample_target_spacing(np_rng, self.spacing_lo, self.spacing_hi)
            if ct_path.endswith(".zarr"):
                cube, affine_iso = crop_resample_zarr(
                    ct_path, self.crop_out_size, s, np_rng,
                    center_crop=self.center_crop,
                )                                            # reads only the block
            else:
                data, affine = self._load_raw(ct_path)
                cube, affine_iso = crop_resample_cube(
                    torch.from_numpy(data), affine, self.crop_out_size, s, np_rng,
                    center_crop=self.center_crop,
                )                                            # (1, out, out, out)
            diameter_mm = self._diameter_from(cube[0].numpy(), affine_iso)
            return cube, torch.from_numpy(affine_iso), diameter_mm

        data, affine = self._load_raw(ct_path)
        diameter_mm = self._diameter_from(data, affine)
        ct_hu = torch.from_numpy(data).unsqueeze(0)          # (1, D, H, W) float32
        return ct_hu, torch.from_numpy(affine), diameter_mm

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        if self.deterministic:
            rng = random.Random(self.seed ^ idx)
        else:
            rng = random.Random(self.seed ^ idx ^ (worker_id << 16))

        ct_path = self.ct_paths[idx % len(self.ct_paths)]
        ct_hu, ct_affine, diameter = self._load_ct_and_diameter(ct_path, rng)
        geom = sample_geometry(
            rng, diameter, self.v_min, self.v_max, self.det_px,
            fixed_pattern=self.fixed_pattern,
            fixed_v=self.fixed_v,
            fixed_sad_mm=self.fixed_sad_mm,
            fixed_sdd_ratio=self.fixed_sdd_ratio,
            fixed_elev_deg=self.fixed_elev_deg,
            fixed_det_pixel_mm=self.fixed_det_pixel_mm,
        )

        # Rotations + source-position tensor (V, 3) each.
        rotations = torch.stack(
            [
                geom.azimuths_rad,
                geom.elevations_rad,
                torch.zeros(geom.v, dtype=torch.float32),
            ],
            dim=-1,
        )
        translations = torch.zeros(geom.v, 3, dtype=torch.float32)
        translations[:, 1] = geom.sad_mm

        return {
            "ct_hu": ct_hu,                            # (1, D, H, W) HU float32
            "ct_affine": ct_affine,                    # (4, 4) float64
            "rotations": rotations,                    # (V, 3) degrees
            "translations": translations,              # (V, 3) mm
            "sdd_mm": float(geom.sdd_mm),
            "sad_mm": float(geom.sad_mm),
            "det_pixel_mm": float(geom.det_pixel_mm),
            "det_h": int(geom.det_height_px),
            "det_w": int(geom.det_width_px),
            "pattern": geom.pattern,
            "v": int(geom.v),
            "ct_path": ct_path,
        }


def eval_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic eval.

    Seeds python random / numpy / torch per worker so any incidental randomness
    inside the worker process is reproducible. The DRRDataset itself already
    uses a per-idx seed (and ignores worker_id when `deterministic=True`).
    """
    import numpy as np
    seed = 0xC0DE + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
