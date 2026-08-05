"""ReconModule: DeepSparse-style fine branch as a standalone reusable unit.

Same architecture as the fine branch inside SparseViewJEPA (jepa.py) -
CNN pyramid + projection + 3D conv refine + point decoder - extracted so
Stage 3 (recon downstream) and dense inference (visualisation) can use it
without the coarse ViT / CT-MAE machinery.

Pretrained loader:
    load_recon_from_pretrain(module, ckpt_path) copies
        cnn_pyramid.*, conv3d_refine.*, point_decoder.*
    from a Stage 2 SparseViewJEPA ckpt into this module.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.cnn_pyramid import CNNPyramid2D
from src.models.jepa import Conv3DRefine
from src.models.point_decoder import PointDecoder
from src.models.projection import index_3d, project_points, query_view_feats


class ReconModule(nn.Module):
    """Sparse-view -> density at arbitrary 3D points (resolution-free)."""

    def __init__(
        self,
        cnn_pyramid: CNNPyramid2D,
        conv3d_refine: Conv3DRefine,
        point_decoder: PointDecoder,
        lr_grid_res: int = 16,
    ):
        super().__init__()
        self.cnn_pyramid = cnn_pyramid
        self.conv3d_refine = conv3d_refine
        self.point_decoder = point_decoder
        self.lr_grid_res = lr_grid_res

    @staticmethod
    def _project_all_views(pts_world, k_inv, rt_inv, det_h, det_w):
        v = k_inv.shape[1]
        outs = [project_points(pts_world, k_inv[:, vi], rt_inv[:, vi], det_h, det_w)
                for vi in range(v)]
        return torch.stack(outs, dim=1)                    # (B, V, N, 2)

    def _build_volume(self, pyr, lr_grid_world, k_inv, rt_inv, det_h, det_w):
        b = lr_grid_world.shape[0]
        g = self.lr_grid_res
        pp = self._project_all_views(lr_grid_world, k_inv, rt_inv, det_h, det_w)
        per_scale = [query_view_feats(f, pp, fusion="max") for f in pyr]
        cat = torch.cat(per_scale, dim=1)
        vol = cat.reshape(b, cat.shape[1], g, g, g)
        return self.conv3d_refine(vol)

    def encode_views(self, views: torch.Tensor) -> list[torch.Tensor]:
        """views (B, V, 1, H, W) -> list of (B, V, C_s, h_s, w_s)."""
        b, v, c, h, w = views.shape
        pyr = self.cnn_pyramid(views.reshape(b * v, c, h, w))
        return [f.reshape(b, v, *f.shape[1:]) for f in pyr]

    def forward(
        self,
        views: torch.Tensor,            # (B, V, 1, H, W)
        k_inv: torch.Tensor,            # (B, V, 3, 3)
        rt_inv: torch.Tensor,           # (B, V, 4, 4)
        det_h: int,
        det_w: int,
        q_pts_world: torch.Tensor,      # (B, N, 3) WORLD
        q_pts_norm: torch.Tensor,       # (B, N, 3) [-1,1] grid
        lr_grid_world: torch.Tensor,    # (B, G3, 3) WORLD
    ) -> torch.Tensor:
        """Predict density at the given query points.

        Returns (B, 1, N) - the raw point_decoder output. Caller applies
        any nonlinearity / loss against the (normalised) target.
        """
        pyr = self.encode_views(views)
        feat3d = self._build_volume(pyr, lr_grid_world, k_inv, rt_inv, det_h, det_w)
        f_3d = index_3d(feat3d, q_pts_norm)               # (B, Cr, N)
        pp = self._project_all_views(q_pts_world, k_inv, rt_inv, det_h, det_w)
        f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]
        pt_feat = torch.cat([f_3d] + f_2d, dim=1)
        return self.point_decoder(pt_feat)                # (B, 1, N)

    @torch.no_grad()
    def infer_dense(
        self,
        views: torch.Tensor,
        k_inv: torch.Tensor,
        rt_inv: torch.Tensor,
        det_h: int,
        det_w: int,
        lr_grid_world: torch.Tensor,
        grid_to_world: torch.Tensor,   # (4,4) maps norm [-1,1] grid -> world
        out_res: int = 256,
        chunk: int = 65536,
    ) -> torch.Tensor:
        """Dense recon: query an out_res^3 grid in chunks.

        Returns (1, 1, out_res, out_res, out_res) density volume.
        """
        device = views.device
        lin = torch.linspace(-1, 1, out_res, device=device)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        norm = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        ones = torch.ones(norm.shape[0], 1, device=device)
        homog = torch.cat([norm, ones], dim=-1)
        world = (grid_to_world @ homog.T).T[:, :3]

        pyr = self.encode_views(views)
        feat3d = self._build_volume(pyr, lr_grid_world, k_inv, rt_inv, det_h, det_w)

        out = []
        for s in range(0, norm.shape[0], chunk):
            qn = norm[s:s + chunk].unsqueeze(0)
            qw = world[s:s + chunk].unsqueeze(0)
            f_3d = index_3d(feat3d, qn)
            pp = self._project_all_views(qw, k_inv, rt_inv, det_h, det_w)
            f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]
            pt_feat = torch.cat([f_3d] + f_2d, dim=1)
            d = self.point_decoder(pt_feat).squeeze(1).squeeze(0)
            out.append(d)
        d_all = torch.cat(out, dim=0).reshape(out_res, out_res, out_res)
        return d_all.unsqueeze(0).unsqueeze(0)


def load_recon_from_pretrain(
    module: ReconModule, ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> int:
    """Copy cnn_pyramid / conv3d_refine / point_decoder weights from a
    Stage 2 SparseViewJEPA ckpt into the given ReconModule.

    Returns the iter the ckpt was saved at.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = state["model"]
    target = {}
    for k, v in src.items():
        for prefix in ("cnn_pyramid.", "conv3d_refine.", "point_decoder."):
            if k.startswith(prefix):
                target[k] = v
                break
    missing, unexpected = module.load_state_dict(target, strict=False)
    if missing or unexpected:
        print(f"WARN load_recon_from_pretrain missing={missing[:3]} unexpected={unexpected[:3]}")
    return int(state["iter"])
