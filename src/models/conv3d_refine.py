"""3D convolutional refinement of the lifted feature volume.

Used by FeatureVolume3D: a 1x1 input projection followed by residual 3D conv
blocks, which add a local structural prior to the features sampled from the views.
"""

from __future__ import annotations

import torch.nn as nn


class _ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(1, ch)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(1, ch)
        self.act = nn.GELU()

    def forward(self, x):
        r = x
        x = self.act(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return self.act(r + x)


class Conv3DRefine(nn.Module):
    """1x1 in-proj + a few residual 3D conv blocks (structural prior)."""

    def __init__(self, in_ch: int, ch: int, n_blocks: int = 3):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, ch, 1)
        self.blocks = nn.Sequential(*[_ResBlock3D(ch) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(self.proj(x))
