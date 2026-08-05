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

import pandas as pd
import torch
from torch.utils.data import Dataset

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
) -> SampleGeometry:
    """Sample (or pin) one view geometry.

    Any `fixed_*` arg overrides the corresponding random draw, leaving other
    fields random. Pass all of them to get a 100% deterministic geometry per
    CT (used by eval / benchmark).
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
        df = pd.read_csv(manifest)
        assert "ct_path" in df.columns, "manifest must have ct_path column"
        self.ct_paths = [resolve_project_path(p) for p in df["ct_path"].astype(str)]

    def __len__(self) -> int:
        return len(self.ct_paths)

    def _load_ct_and_diameter(self, ct_path: str):
        """Load CT HU + affine and compute patient diameter.

        Dispatches by file extension:
          - .nii / .nii.gz: nibabel (preserves real affine)
          - .npy: assumes uint8 HU-clipped [-1024,1024] with identity 1mm
            affine (LIDC-style preprocessed). Spacing is unknown -> 1mm.

        Keeps worker process light: just IO + HU threshold for bbox.
        """
        import numpy as np
        if ct_path.endswith(".npy"):
            arr = np.load(ct_path)
            if arr.dtype == np.uint8:
                data = arr.astype(np.float32) * (2048.0 / 255.0) - 1024.0
            else:
                data = arr.astype(np.float32)
            affine = np.eye(4, dtype=np.float64)
        else:
            import nibabel as nib
            img = nib.load(ct_path)
            data = img.get_fdata(dtype=np.float32)         # (D, H, W) HU
            affine = np.asarray(img.affine, dtype=np.float64)

        # Patient bbox diagonal -> detector sizing (max axis extent).
        mask = data > -500.0
        if mask.any():
            nz = np.argwhere(mask)
            bbox_voxel = nz.max(axis=0) - nz.min(axis=0) + 1
        else:
            bbox_voxel = np.array(data.shape)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)    # (3,) mm
        extent_mm = bbox_voxel * spacing
        diameter_mm = float(extent_mm.max())

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
        ct_hu, ct_affine, diameter = self._load_ct_and_diameter(ct_path)
        geom = sample_geometry(
            rng, diameter, self.v_min, self.v_max, self.det_px,
            fixed_pattern=self.fixed_pattern,
            fixed_v=self.fixed_v,
            fixed_sad_mm=self.fixed_sad_mm,
            fixed_sdd_ratio=self.fixed_sdd_ratio,
            fixed_elev_deg=self.fixed_elev_deg,
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
