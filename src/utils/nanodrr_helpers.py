"""nanodrr wrapper that returns both the DRR and the per-pixel ray endpoints.

nanodrr's `render()` only returns the image. For SSL pretraining we also need
the per-pixel ray geometry (source + detector pixel positions in world coords)
to build Plücker embeddings consumed by the 2D context encoder.

This module computes the rays once (in camera coords), transforms them to
world coords via the inverse extrinsic, and reuses them as `src` / `tgt` for
the render call - no redundant work.
"""

from __future__ import annotations

import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.drr import render


def subject_from_tensor(
    ct_hu: torch.Tensor,
    ct_affine: torch.Tensor,
    device: str | torch.device = "cpu",
):
    """Build a nanodrr Subject from an in-memory HU tensor + affine.

    Bypasses NIfTI IO + SimpleITK entirely. Used when a worker has already
    loaded the CT and the main thread just needs to wrap it for rendering.

    Args:
        ct_hu: (1, D, H, W) or (D, H, W) HU float tensor.
        ct_affine: (4, 4) float64 affine matrix.
        device: target device for the Subject buffers.
    """
    import torchio as tio
    from nanodrr.data import Subject

    if ct_hu.dim() == 3:
        ct_hu = ct_hu.unsqueeze(0)
    affine_np = ct_affine.detach().cpu().numpy()
    scalar = tio.ScalarImage(tensor=ct_hu.detach().cpu(), affine=affine_np)
    return Subject.from_images(scalar).to(device)


def load_subject_from_npy(
    npy_path: str,
    device: str | torch.device = "cpu",
    hu_lo: float = -1024.0,
    hu_hi: float = 1024.0,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Load a LIDC/LUNA16-style uint8 .npy CT into a nanodrr Subject.

    The .npy is uint8 (0..255) representing HU clipped to [hu_lo, hu_hi].
    We invert the quantisation to recover float HU, then wrap with a
    synthetic identity-orientation affine using `spacing_mm` (default 1mm
    isotropic; LIDC actually has variable slice thickness but we don't
    have it stored - this is fine for architecture/training sanity).
    """
    import numpy as np
    import torchio as tio
    from nanodrr.data import Subject

    arr = np.load(npy_path)
    if arr.dtype == np.uint8:
        hu = arr.astype(np.float32) * ((hu_hi - hu_lo) / 255.0) + hu_lo
    else:
        hu = arr.astype(np.float32)
    tensor = torch.from_numpy(hu).unsqueeze(0)    # (1, D, H, W)
    sx, sy, sz = spacing_mm
    affine = np.array([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    scalar = tio.ScalarImage(tensor=tensor, affine=affine)
    return Subject.from_images(scalar).to(device)


def load_subject_nibabel(ct_path: str, device: str | torch.device = "cpu"):
    """Load a CT via nibabel and wrap into a nanodrr Subject.

    nanodrr's default `Subject.from_filepath` uses torchio's SimpleITK reader,
    which rejects NIfTI files whose direction cosines aren't strictly
    orthonormal (common in TotalSegmentator). nibabel accepts these files;
    we then construct a `torchio.ScalarImage` from the in-memory tensor +
    affine to skip SimpleITK entirely.
    """
    import nibabel as nib
    import numpy as np
    import torchio as tio
    from nanodrr.data import Subject

    img = nib.load(ct_path)
    data = img.get_fdata(dtype=np.float32)            # (D, H, W)
    affine = np.asarray(img.affine, dtype=np.float64)
    tensor = torch.from_numpy(data).unsqueeze(0)      # (1, D, H, W)
    scalar = tio.ScalarImage(tensor=tensor, affine=affine)
    return Subject.from_images(scalar).to(device)


def patient_diameter_mm(subject, hu_threshold: float = -500.0) -> float:
    """Detector-sizing diameter for a circular-orbit cone-beam.

    Returns the MAX single-axis extent of the patient bbox (in world coords).
    For a square detector this is the side length that covers any orbit angle,
    because a 2D projection of a 3D bbox is at most as large as the longest
    axis. Using the 3D diagonal instead would overcount by up to ~2x area.

    Args:
        subject: nanodrr Subject.
        hu_threshold: voxels at or below this HU are treated as air. -500 is
            a safe cut (air is around -1000, soft tissue around 0).
    """
    hu = subject._image_hu[0, 0]                       # (D, H, W)
    mask = hu > hu_threshold
    if not mask.any():
        shape = hu.shape
        bbox_voxel = torch.tensor(shape, dtype=torch.float32)
    else:
        nz = mask.nonzero()                            # (N, 3) voxel indices
        bbox_voxel = (nz.max(dim=0).values - nz.min(dim=0).values + 1).float()
    spacing = subject.voxel_to_world[:3, :3].norm(dim=0).cpu()   # (3,) mm
    extent_mm = bbox_voxel.cpu() * spacing             # (3,) mm
    return float(extent_mm.max().item())


def _make_tgt_camera(
    k_inv: torch.Tensor,        # (B, 3, 3)
    sdd: torch.Tensor,          # (B,)
    height: int,
    width: int,
) -> torch.Tensor:
    """Detector pixel positions in camera coords. Mirrors nanodrr.drr.renderer._make_tgt."""
    device, dtype = k_inv.device, k_inv.dtype
    n = height * width
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(n, 3)
    tgt = sdd[:, None, None] * torch.einsum("bij,nj->bni", k_inv, uv1)
    return tgt          # (B, H*W, 3) camera coords


def _transform_to_world(rt_inv: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply rt_inv (4x4) to a batch of 3D points (B, N, 3) -> (B, N, 3) world."""
    b, n, _ = pts.shape
    ones = torch.ones(b, n, 1, device=pts.device, dtype=pts.dtype)
    homog = torch.cat([pts, ones], dim=-1)                  # (B, N, 4)
    out = torch.einsum("bij,bnj->bni", rt_inv, homog)
    return out[..., :3]


def build_pose_tensors(
    rotations: torch.Tensor,
    translations: torch.Tensor,
    sdd: float,
    delx: float,
    height: int,
    width: int,
    dely: float | None = None,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str = "AP",
    isocenter: torch.Tensor | None = None,
) -> dict:
    """Build (k_inv, rt_inv, sdd_tensor) for nanodrr `render`."""
    device, dtype = rotations.device, rotations.dtype
    if dely is None:
        dely = delx
    k_inv = make_k_inv(
        sdd=sdd, delx=delx, dely=dely, x0=x0, y0=y0,
        height=height, width=width, dtype=dtype, device=device,
    )
    rt_inv = make_rt_inv(
        rotations, translations,
        orientation=orientation,
        isocenter=isocenter,
    )
    b = rotations.shape[0]
    sdd_t = torch.full((b,), float(sdd), device=device, dtype=dtype)
    return dict(k_inv=k_inv, rt_inv=rt_inv, sdd=sdd_t)


def pose_per_view(
    subject,
    rotations: torch.Tensor,        # (V, 3) Euler degrees
    translations: torch.Tensor,     # (V, 3) source pos mm
    sdd: float,
    delx: float,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-view (k_inv, rt_inv) for projection. Returns (V,3,3), (V,4,4).

    k_inv (intrinsics) is shared across views -> broadcast to V.
    """
    poses = build_pose_tensors(
        rotations=rotations,
        translations=translations,
        sdd=sdd,
        delx=delx,
        height=height,
        width=width,
        isocenter=subject.isocenter,
    )
    v = rotations.shape[0]
    k_inv = poses["k_inv"]
    if k_inv.shape[0] == 1:
        k_inv = k_inv.expand(v, -1, -1).contiguous()
    rt_inv = poses["rt_inv"]
    return k_inv, rt_inv


def _grid_to_world(subject, pts_grid: torch.Tensor) -> torch.Tensor:
    """pts_grid (N,3) in nanodrr [-1,1] grid coords -> world mm (N,3).

    nanodrr's world frame and grid coords are both in the SAME axis order
    consistent with PyTorch grid_sample on (B, C, D, H, W) tensors
    (last-dim order = x=W, y=H, z=D). Confirmed empirically: the
    world_to_grid matrix's diagonal is (2/W_len, 2/H_len, 2/D_len) which
    matches the W=512, H=512, D=324 lengths of our CTs.
    """
    from nanodrr.geometry import transform_point
    grid_to_world = torch.linalg.inv(subject.world_to_grid)
    return transform_point(grid_to_world[None], pts_grid[None])[0]


def subject_grid_to_world(subject) -> torch.Tensor:
    """Return the 4x4 matrix mapping nanodrr [-1,1] grid (= grid_sample
    (x=W, y=H, z=D) order) -> world mm. Used by ReconModule.infer_dense
    which queries a dense out_res^3 grid and multiplies by this matrix.
    """
    return torch.linalg.inv(subject.world_to_grid)


def sample_query_points(
    subject,
    n: int,
    hu_min: float,
    hu_max: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random query points for the fine branch.

    Returns (batched B=1):
        pts_grid:  (1, N, 3) [-1,1] nanodrr grid coords (for index_3d)
        pts_world: (1, N, 3) world mm (for projection)
        gt_hu:     (1, N) GT CT clipped to [hu_min,hu_max] scaled to [0,1]
    """
    from src.models.projection import index_3d
    pts_grid = torch.rand(n, 3, device=device, generator=generator) * 2.0 - 1.0
    hu_vol = subject._image_hu.to(device)
    gt = index_3d(hu_vol, pts_grid[None])[0, 0]
    gt = gt.clamp(hu_min, hu_max)
    gt = (gt - hu_min) / (hu_max - hu_min)
    pts_world = _grid_to_world(subject, pts_grid)
    return pts_grid[None], pts_world[None], gt[None]


def make_lr_grid(
    subject, g: int, device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regular g^3 grid. Returns (1,G3,3) grid coords + (1,G3,3) world."""
    lin = torch.linspace(-1, 1, g, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    world = _grid_to_world(subject, grid)
    return grid[None], world[None]


def make_patch_grid(
    subject, g: int, device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """G^3 patch-CENTRE grid in (d, h, w) raster order, matching a CTMAE's
    conv3d token order (flatten of (D,H,W)). Used so the foundation model's 3D
    feature volume F aligns voxel-for-voxel with the frozen CT-MAE tokens.

    Patch i = (d, h, w) raster (d slowest); its centre in [-1,1] grid_sample
    coords is (x=w, y=h, z=d) with c = (idx + 0.5)/G * 2 - 1.

    Returns (1, G^3, 3) grid coords [-1,1] + (1, G^3, 3) world mm.
    """
    lin = (torch.arange(g, device=device).float() + 0.5) / g * 2.0 - 1.0
    dd, hh, ww = torch.meshgrid(lin, lin, lin, indexing="ij")   # indexed [d,h,w]
    grid = torch.stack([ww, hh, dd], dim=-1).reshape(-1, 3)     # (x=w,y=h,z=d)
    world = _grid_to_world(subject, grid)
    return grid[None], world[None]


def render_with_rays(
    subject,                                # nanodrr.data.Subject
    rotations: torch.Tensor,                # (B, 3) Euler
    translations: torch.Tensor,             # (B, 3) source position
    sdd: float,
    delx: float,
    height: int,
    width: int,
    dely: float | None = None,
    orientation: str = "AP",
    n_samples: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render DRR(s) and return per-pixel ray endpoints in world coords.

    Returns:
        image:  (B, 1, H, W) - raw line-integral DRR.
        source: (B, H*W, 3)  - ray source positions in world coords.
        target: (B, H*W, 3)  - detector pixel positions in world coords.
    """
    isocenter = subject.isocenter
    poses = build_pose_tensors(
        rotations=rotations,
        translations=translations,
        sdd=sdd,
        delx=delx,
        height=height,
        width=width,
        dely=dely,
        orientation=orientation,
        isocenter=isocenter,
    )
    k_inv, rt_inv, sdd_t = poses["k_inv"], poses["rt_inv"], poses["sdd"]

    b = rotations.shape[0]
    src_cam = torch.zeros(b, 1, 3, device=k_inv.device, dtype=k_inv.dtype)
    tgt_cam = _make_tgt_camera(k_inv, sdd_t, height, width)

    src_world = _transform_to_world(rt_inv, src_cam).expand(-1, height * width, -1)
    tgt_world = _transform_to_world(rt_inv, tgt_cam)

    # Reuse camera-coord src/tgt for rendering (matches nanodrr's internal flow).
    image = render(
        subject=subject,
        k_inv=k_inv,
        rt_inv=rt_inv,
        sdd=sdd_t,
        height=height,
        width=width,
        n_samples=n_samples,
        src=src_cam.expand(-1, height * width, -1).contiguous(),
        tgt=tgt_cam,
    )
    return image, src_world.contiguous(), tgt_world.contiguous()
