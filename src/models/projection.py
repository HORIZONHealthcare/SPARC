"""Geometry: project 3D points to views + sample 2D/3D features.

DeepSparse-style explicit projection for the fine recon branch:
  - project_points: world 3D point -> normalised detector pixel coords
  - query_view_feats: bilinear-sample multi-view 2D feature maps at the
    projected points, fuse across views (max)
  - index_3d: trilinear-sample a 3D feature volume at normalised points

nanodrr produces `k_inv` (pixel->camera ray) and `rt_inv` (camera->world).
We invert them for the forward projection (world->pixel).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project_points(
    pts_world: torch.Tensor,   # (B, N, 3) world coords
    k_inv: torch.Tensor,       # (B, 3, 3) pixel->camera-ray (nanodrr)
    rt_inv: torch.Tensor,      # (B, 4, 4) camera->world (nanodrr)
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cone-beam pinhole projection -> normalised [-1, 1] grid_sample coords.

    Returns (B, N, 2) with (x, y) in [-1, 1] (grid_sample convention,
    last dim order = (x=width, y=height)).
    """
    b, n, _ = pts_world.shape
    rt = torch.linalg.inv(rt_inv)                        # world -> camera
    ones = torch.ones(b, n, 1, device=pts_world.device, dtype=pts_world.dtype)
    pw = torch.cat([pts_world, ones], dim=-1)            # (B, N, 4)
    p_cam = torch.einsum("bij,bnj->bni", rt, pw)[..., :3]

    k = torch.linalg.inv(k_inv)                          # camera -> pixel
    m = torch.einsum("bij,bnj->bni", k, p_cam)           # (B, N, 3)
    z = m[..., 2:3].clamp_min(eps)
    uv = m[..., :2] / z                                  # (B, N, 2) pixel coords

    # uv is already pixel-center (nanodrr's k_inv was constructed with
    # arange+0.5). For grid_sample align_corners=False, pixel-center u
    # maps to: 2*u/W - 1. Adding another +0.5 here would shift by half a
    # pixel (confirmed via projection consistency test).
    x = 2.0 * uv[..., 0] / width - 1.0
    y = 2.0 * uv[..., 1] / height - 1.0
    return torch.stack([x, y], dim=-1)                   # (B, N, 2)


def query_view_feats(
    feat_maps: torch.Tensor,   # (B, V, C, h, w)
    pts_proj: torch.Tensor,    # (B, V, N, 2) normalised [-1,1]
    fusion: str = "max",
) -> torch.Tensor:
    """Bilinear-sample each view's feature map at projected points, fuse.

    Returns (B, C, N).
    """
    b, v, c, h, w = feat_maps.shape
    n = pts_proj.shape[2]
    fm = feat_maps.reshape(b * v, c, h, w)
    grid = pts_proj.reshape(b * v, n, 1, 2)
    sampled = F.grid_sample(fm, grid, mode="bilinear",
                            align_corners=False, padding_mode="zeros")
    sampled = sampled.reshape(b, v, c, n)
    if fusion == "max":
        return sampled.max(dim=1).values
    if fusion == "mean":
        return sampled.mean(dim=1)
    if fusion == "sum":
        return sampled.sum(dim=1)
    raise ValueError(fusion)


def index_3d(
    vol: torch.Tensor,         # (B, C, D, H, W)
    pts_norm: torch.Tensor,    # (B, N, 3) normalised [-1,1], order (x,y,z)
    mode: str = "bilinear",    # "bilinear" for HU/features, "nearest" for labels
) -> torch.Tensor:
    """Sample a 3D volume at normalised points (trilinear by default).

    Returns (B, C, N). Pass mode="nearest" for integer label volumes (masks).
    """
    b, c = vol.shape[:2]
    n = pts_norm.shape[1]
    grid = pts_norm.reshape(b, n, 1, 1, 3)
    sampled = F.grid_sample(vol, grid, mode=mode,
                            align_corners=False, padding_mode="zeros")
    return sampled.reshape(b, c, n)
