"""Multi-scale CNN 2D encoder (MedNeXt-lite) for the fine recon branch.

Produces a feature pyramid: a list of feature maps at decreasing spatial
resolution. The fine branch projects 3D query points into each scale and
bilinear-samples (DeepSparse-style), so we want dense, near-image-resolution
features at the early scales - unlike the coarse ViT (patch=16) branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtBlock2D(nn.Module):
    """Depthwise 7x7 + pointwise expand/contract (ConvNeXt-style)."""

    def __init__(self, dim: int, exp_ratio: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * exp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * exp_ratio, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return r + x


class CNNPyramid2D(nn.Module):
    """Stem + N stages. Returns feature maps at each stage (highest-res first)."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 48,
        n_stages: int = 3,
        ch_mult: int = 2,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, base_ch),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        self.out_chs = [base_ch]
        ch = base_ch
        for _ in range(n_stages):
            out = ch * ch_mult
            blocks = [nn.Conv2d(ch, out, kernel_size=2, stride=2)]   # 2x down
            for _ in range(blocks_per_stage):
                blocks.append(ConvNeXtBlock2D(out))
            self.stages.append(nn.Sequential(*blocks))
            self.out_chs.append(out)
            ch = out

    @property
    def channels(self) -> list[int]:
        """Feature channels per scale (highest-res first)."""
        return self.out_chs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B*, in_ch, H, W) -> list of (B*, C_s, H_s, W_s).

        feats[0] = stem (full res); feats[i] = stage i (half res each).
        len = n_stages + 1.
        """
        feats = []
        x = self.stem(x)
        feats.append(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
