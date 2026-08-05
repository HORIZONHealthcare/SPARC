"""Per-view 2D X-ray encoder: ConvNeXt pyramid (dense) + transformer (global).

The unified foundation backbone needs 2D features that are (a) spatially DENSE
so 3D query points can be projected and bilinear-sampled precisely, and
(b) globally context-aware. A pure CNN gives (a) but not (b); a patch-ViT gives
(b) but is 16x too coarse for (a). So we use a hybrid: a ConvNeXt pyramid for
dense multi-scale features, with a transformer applied at the COARSEST scale
(few tokens -> cheap) that also injects the per-view Plucker geometry. The
transformer-refined coarse map is added back to the pyramid.

Output is a multi-scale pyramid (highest-res first), each (B, V, C_s, h_s, w_s),
consumed by FeatureVolume3D via projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_pyramid import CNNPyramid2D
from src.models.encoders import TransformerBlock
from src.utils.plucker import normalize_plucker, patch_pool_plucker


class XRayEncoder2D(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        in_ch: int = 1,
        base_ch: int = 64,
        n_stages: int = 3,
        ch_mult: int = 2,
        blocks_per_stage: int = 2,
        tf_dim: int = 512,
        tf_depth: int = 6,
        tf_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.pyramid = CNNPyramid2D(in_ch, base_ch, n_stages, ch_mult, blocks_per_stage)
        coarse_ch = self.pyramid.channels[-1]
        self.use_tf = tf_depth > 0                  # tf_depth=0 -> conv-only ablation
        if not self.use_tf:
            return
        self.pool_factor = 2 ** n_stages           # img_size / coarse grid
        assert img_size % self.pool_factor == 0
        self.coarse_grid = img_size // self.pool_factor
        n_coarse = self.coarse_grid ** 2

        self.to_tf = nn.Conv2d(coarse_ch, tf_dim, kernel_size=1)
        self.ray_embed = nn.Sequential(
            nn.Linear(6, tf_dim), nn.GELU(), nn.Linear(tf_dim, tf_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, n_coarse, tf_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(tf_dim, tf_heads, mlp_ratio) for _ in range(tf_depth)]
        )
        self.norm = nn.LayerNorm(tf_dim)
        self.from_tf = nn.Conv2d(tf_dim, coarse_ch, kernel_size=1)

    @property
    def channels(self) -> list[int]:
        return self.pyramid.channels

    def forward(
        self, views: torch.Tensor, plucker: torch.Tensor, sad_mm,
    ) -> list[torch.Tensor]:
        """views (B,V,1,H,W), plucker (B,V,H,W,6) -> list of (B,V,C_s,h_s,w_s)."""
        b, v, c, h, w = views.shape
        x = views.reshape(b * v, c, h, w)
        feats = self.pyramid(x)                          # each (B*V, C_s, h_s, w_s)

        if not self.use_tf:                              # conv-only ablation
            return [f.reshape(b, v, *f.shape[1:]) for f in feats]

        coarse = feats[-1]                               # (B*V, coarse_ch, hc, wc)
        hc, wc = coarse.shape[-2:]
        tok = self.to_tf(coarse).flatten(2).transpose(1, 2)   # (B*V, hc*wc, tf_dim)

        plk = normalize_plucker(plucker, sad_mm).reshape(b * v, h, w, 6)
        ray = patch_pool_plucker(plk, h // hc).reshape(b * v, hc * wc, 6)
        tok = tok + self.ray_embed(ray) + self.pos_embed

        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)

        refined = tok.transpose(1, 2).reshape(b * v, -1, hc, wc)
        feats[-1] = coarse + self.from_tf(refined)       # residual enrich

        return [f.reshape(b, v, *f.shape[1:]) for f in feats]
