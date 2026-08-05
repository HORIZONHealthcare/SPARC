"""CT-only dataset for CT-MAE pretrain and downstream CT-input tasks.

Loads a NIfTI CT volume via nibabel, resamples to a fixed cube, and returns
HU tensor + affine. No DRR rendering (that's for DRRDataset).
"""

from __future__ import annotations
from src.utils.project_paths import resolve_project_path

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.datasets.crop_resample import (
    crop_resample_cube, crop_resample_zarr, sample_target_spacing,
)


def _load_ct_nibabel(ct_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load CT HU and affine via nibabel (handles non-orthonormal direction)."""
    import nibabel as nib
    img = nib.load(ct_path)
    data = img.get_fdata(dtype=np.float32)             # (D, H, W) HU
    affine = np.asarray(img.affine, dtype=np.float64)
    return torch.from_numpy(data), torch.from_numpy(affine)


def _resample_cube(ct: torch.Tensor, target_size: int) -> torch.Tensor:
    """Trilinear-resample (D, H, W) -> (1, target, target, target)."""
    if ct.dim() == 3:
        ct = ct.unsqueeze(0).unsqueeze(0)              # (1, 1, D, H, W)
    elif ct.dim() == 4:
        ct = ct.unsqueeze(0)
    out = F.interpolate(ct, size=(target_size,) * 3, mode="trilinear", align_corners=False)
    return out[0]                                       # (1, T, T, T)


class CTDataset(Dataset):
    """Yields `{ct_hu, ct_affine, ct_path}` after resampling to vol_size cube."""

    def __init__(
        self,
        manifest: str | Path,
        vol_size: int = 128,
        seed: int = 0,
        crop_aug: bool = False,
        crop_out_size: int = 256,
        spacing_lo: float = 1.0,
        spacing_hi: float = 2.0,
    ):
        super().__init__()
        df = pd.read_csv(manifest)
        assert "ct_path" in df.columns
        self.ct_paths = [resolve_project_path(p) for p in df["ct_path"].astype(str)]
        self.vol_size = vol_size
        self.seed = seed
        self.crop_aug = crop_aug
        self.crop_out_size = crop_out_size
        self.spacing_lo = spacing_lo
        self.spacing_hi = spacing_hi

    def __len__(self) -> int:
        return len(self.ct_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ct_path = self.ct_paths[idx % len(self.ct_paths)]
        if self.crop_aug:
            # Crop -> resample to crop_out_size^3 -> downsample to vol_size^3.
            # The two-step (native->256->128) matches Stage 2's coarse target
            # path exactly, so the frozen MAE sees the same 128^3 distribution.
            rng = np.random.default_rng()
            s = sample_target_spacing(rng, self.spacing_lo, self.spacing_hi)
            if ct_path.endswith(".zarr"):
                cube, affine_iso = crop_resample_zarr(
                    ct_path, self.crop_out_size, s, rng,
                )                                      # reads only the crop block
            else:
                hu, affine = _load_ct_nibabel(ct_path)
                cube, affine_iso = crop_resample_cube(
                    hu, affine.numpy(), self.crop_out_size, s, rng,
                )                                      # (1, out, out, out)
            hu = _resample_cube(cube, self.vol_size)   # (1, V, V, V)
            affine = torch.from_numpy(affine_iso)
        else:
            hu, affine = _load_ct_nibabel(ct_path)
            hu = _resample_cube(hu, self.vol_size)     # (1, V, V, V)
        return {"ct_hu": hu, "ct_affine": affine, "ct_path": ct_path}
