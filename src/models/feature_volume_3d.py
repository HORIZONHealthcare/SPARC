"""3D feature volume F: lift per-view 2D features into a 3D grid, refine with
a 3D transformer (global attention).

This is the heart of the unified backbone. For each cell of a G^3 grid we
project its world centre into every view, bilinear-sample the multi-scale 2D
features and max-fuse across views (DeepSparse-style lifting). A 3D transformer
then mixes the G^3 = 4096 tokens (global context, feasible at G=16), producing a
single 3D feature volume F that BOTH heads consume (recon decodes F + projected
2D feats; semantic aligns F to the frozen CT-MAE tokens).

Grid token order is (d, h, w) raster, matching the CT-MAE patch grid so the
semantic head can align voxel-for-voxel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import TransformerBlock
from src.models.conv3d_refine import Conv3DRefine
from src.models.projection import project_points, query_view_feats


def _project_all_views(
    pts_world: torch.Tensor, k_inv: torch.Tensor, rt_inv: torch.Tensor,
    det_h: int, det_w: int,
) -> torch.Tensor:
    """pts_world (B,N,3); k_inv (B,V,3,3); rt_inv (B,V,4,4) -> (B,V,N,2)."""
    v = k_inv.shape[1]
    outs = [
        project_points(pts_world, k_inv[:, i], rt_inv[:, i], det_h, det_w)
        for i in range(v)
    ]
    return torch.stack(outs, dim=1)


class FeatureVolume3D(nn.Module):
    def __init__(
        self,
        pyramid_channels: list[int],
        grid_res: int = 16,
        dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        refine_type: str = "transformer",
    ):
        super().__init__()
        assert refine_type in ("transformer", "conv")
        self.grid_res = grid_res
        self.out_dim = dim
        self.refine_type = refine_type
        sum_ch = sum(pyramid_channels)
        if refine_type == "transformer":
            self.proj_in = nn.Linear(sum_ch, dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, grid_res ** 3, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                [TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(dim)
        else:
            self.conv = Conv3DRefine(in_ch=sum_ch, ch=dim, n_blocks=depth)

    def forward(
        self,
        pyr: list[torch.Tensor],     # each (B, V, C_s, h_s, w_s)
        grid_world: torch.Tensor,    # (B, G^3, 3) world, (d,h,w) raster
        k_inv: torch.Tensor,         # (B, V, 3, 3)
        rt_inv: torch.Tensor,        # (B, V, 4, 4)
        det_h: int,
        det_w: int,
    ) -> torch.Tensor:
        """Returns F: (B, dim, G, G, G)."""
        b = grid_world.shape[0]
        g = self.grid_res
        pp = _project_all_views(grid_world, k_inv, rt_inv, det_h, det_w)   # (B,V,G^3,2)
        per_scale = [query_view_feats(f, pp, fusion="max") for f in pyr]   # (B,C_s,G^3)
        cat = torch.cat(per_scale, dim=1)                                  # (B,sumC,G^3)

        if self.refine_type == "conv":
            vol = cat.reshape(b, cat.shape[1], g, g, g)
            return self.conv(vol)                                          # (B,dim,G,G,G)

        tok = self.proj_in(cat.transpose(1, 2))                            # (B,G^3,dim)
        tok = tok + self.pos_embed
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok.transpose(1, 2).reshape(b, self.out_dim, g, g, g)
