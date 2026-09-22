"""Crop-then-resample augmentation for anti-hallucination recon pretraining.

The model should learn the *recovery* skill (sparse-view -> dense volume), not
*memorise* a fixed anatomical region (memorisation -> hallucination, dangerous
in medicine). To force generalisation across physical scale and region we:

  1. pick a random isotropic target spacing in [spacing_lo, spacing_hi] mm,
  2. crop a native voxel block whose physical size is ``out_size * spacing``
     at a random (foreground-biased, off-centre) location,
  3. resample ONLY that block to ``out_size^3`` (crop-then-resample is far
     cheaper than resampling the whole volume).

The returned cube IS the resampled CT. Both the DRR source and the voxel GT are
derived from this single cube, so source-alignment is preserved by
construction. The affine is isotropic ``diag([s, s, s, 1])`` because the output
grid is regular and isotropic at the chosen spacing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

HU_AIR = -1024.0
BODY_HU_THRESHOLD = -500.0


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Per-array-axis voxel spacing (mm) = column norms of the 3x3 block."""
    return np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)


def _body_bbox(hu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive (lo, hi) voxel indices of the patient body bbox.

    Falls back to the full volume when no voxel clears the air threshold.
    """
    mask = hu > BODY_HU_THRESHOLD
    if not mask.any():
        return np.zeros(3, dtype=np.int64), np.asarray(hu.shape, dtype=np.int64) - 1
    nz = np.argwhere(mask)
    return nz.min(axis=0), nz.max(axis=0)


def crop_resample_cube(
    hu: torch.Tensor,
    affine: np.ndarray | torch.Tensor,
    out_size: int,
    target_spacing: float,
    rng: np.random.Generator,
    foreground_bias: bool = True,
    center_crop: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    """Crop a native region and resample to ``out_size^3`` at isotropic spacing.

    Args:
        hu: (D, H, W) or (1, D, H, W) float HU tensor.
        affine: (4, 4) voxel->world affine; only its column norms (spacing) are
            used to size the native crop.
        out_size: output cube side length in voxels.
        target_spacing: isotropic output spacing in mm.
        rng: numpy Generator for the crop centre (use a fresh one per call in
            training for max augmentation diversity).
        foreground_bias: place the crop centre inside the body bbox so the crop
            contains anatomy (off-centre, but not pure air).

    Returns:
        cube: (1, out_size, out_size, out_size) float32 HU.
        affine_iso: (4, 4) float64 diag([s, s, s, 1]).
    """
    if hu.dim() == 4:
        hu = hu[0]
    hu_np = hu.detach().cpu().numpy().astype(np.float32, copy=False)
    shape = np.asarray(hu_np.shape, dtype=np.int64)               # (D, H, W)
    sp = spacing_from_affine(affine if isinstance(affine, np.ndarray)
                             else affine.detach().cpu().numpy())

    fov_mm = out_size * float(target_spacing)                      # physical side
    crop_vox = np.maximum(np.round(fov_mm / sp).astype(np.int64), 1)

    if foreground_bias:
        lo, hi = _body_bbox(hu_np)
        if center_crop:
            ctr = (lo + hi) // 2                                   # body-bbox centre
        else:
            ctr = np.array(
                [int(rng.integers(lo[i], hi[i] + 1)) for i in range(3)], dtype=np.int64
            )
    else:
        ctr = np.array(
            [int(rng.integers(0, shape[i])) for i in range(3)], dtype=np.int64
        )
    origin = ctr - crop_vox // 2                                   # may be < 0 (pad)

    # Air-padded crop; copy the in-bounds overlap from the source volume.
    crop = np.full(tuple(int(c) for c in crop_vox), HU_AIR, dtype=np.float32)
    src_lo = np.maximum(origin, 0)
    src_hi = np.minimum(origin + crop_vox, shape)
    if np.all(src_hi > src_lo):
        dst_lo = src_lo - origin
        dst_hi = dst_lo + (src_hi - src_lo)
        crop[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = (
            hu_np[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
        )

    crop_t = torch.from_numpy(crop)[None, None]                    # (1,1,*crop_vox)
    cube = F.interpolate(
        crop_t, size=(out_size,) * 3, mode="trilinear", align_corners=False
    )[0]                                                            # (1, out, out, out)

    s = float(target_spacing)
    affine_iso = np.diag([s, s, s, 1.0]).astype(np.float64)
    return cube, affine_iso


def sample_target_spacing(
    rng: np.random.Generator, spacing_lo: float, spacing_hi: float
) -> float:
    """Uniform isotropic target spacing in [spacing_lo, spacing_hi] mm."""
    return float(rng.uniform(spacing_lo, spacing_hi))


def _pad_and_resample(
    crop: np.ndarray, out_size: int, target_spacing: float
) -> tuple[torch.Tensor, np.ndarray]:
    """Trilinear-resample a native crop block to out_size^3 + iso affine."""
    crop_t = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None, None]
    cube = F.interpolate(
        crop_t, size=(out_size,) * 3, mode="trilinear", align_corners=False
    )[0]
    s = float(target_spacing)
    affine_iso = np.diag([s, s, s, 1.0]).astype(np.float64)
    return cube, affine_iso


def crop_resample_zarr(
    zarr_path: str,
    out_size: int,
    target_spacing: float,
    rng: np.random.Generator,
    foreground_bias: bool = True,
    center_crop: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    """Like ``crop_resample_cube`` but reads ONLY the crop block from a zarr.

    The zarr (from ``preprocess_ctrate_zarr.py``) holds correct int16 HU and
    attrs ``spacing`` / ``bbox_lo`` / ``bbox_hi`` / ``shape``. We pick a
    foreground-biased crop centre from the stored bbox (no full-volume scan),
    read only the in-bounds block (chunked -> fast), air-pad the rest, and
    trilinear-resample to ``out_size^3`` at isotropic ``target_spacing``.
    """
    import zarr

    z = zarr.open(str(zarr_path), mode="r")
    shape = np.asarray(z.attrs["shape"], dtype=np.int64)
    sp = np.asarray(z.attrs["spacing"], dtype=np.float64)

    fov_mm = out_size * float(target_spacing)
    crop_vox = np.maximum(np.round(fov_mm / sp).astype(np.int64), 1)

    if foreground_bias:
        lo = np.asarray(z.attrs["bbox_lo"], dtype=np.int64)
        hi = np.asarray(z.attrs["bbox_hi"], dtype=np.int64)
        if center_crop:
            ctr = (lo + hi) // 2                                   # body-bbox centre
        else:
            ctr = np.array(
                [int(rng.integers(lo[i], hi[i] + 1)) for i in range(3)], dtype=np.int64
            )
    else:
        ctr = np.array(
            [int(rng.integers(0, shape[i])) for i in range(3)], dtype=np.int64
        )
    origin = ctr - crop_vox // 2

    crop = np.full(tuple(int(c) for c in crop_vox), HU_AIR, dtype=np.float32)
    src_lo = np.maximum(origin, 0)
    src_hi = np.minimum(origin + crop_vox, shape)
    if np.all(src_hi > src_lo):
        dst_lo = src_lo - origin
        dst_hi = dst_lo + (src_hi - src_lo)
        blk = z[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
        crop[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = blk
    return _pad_and_resample(crop, out_size, target_spacing)
