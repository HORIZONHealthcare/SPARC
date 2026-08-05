"""Point decoder (INR head) - residual 1x1 Conv1d MLP.

Mirrors DeepSparse's PointDecoder: takes per-point features (concat of
3D trilinear-sampled + multi-scale 2D projected features) and maps each
point independently to a scalar density. Resolution-free: trains on a random
subset of points, infers a dense 256^3 / 512^3 grid in chunks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointDecoder(nn.Module):
    """Residual 1x1 conv MLP over points.

    Args:
        channels: [in_dim, h1, ..., out_dim]. out_dim=1 for density.
        residual: concat the input feature to every hidden layer.
        use_bn: BatchNorm1d between layers.
    """

    def __init__(self, channels: list[int], residual: bool = True, use_bn: bool = True):
        super().__init__()
        self.residual = residual
        self.mlps = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_c = channels[i]
            if i != 0 and residual:
                in_c = channels[i] + channels[0]
            mods: list[nn.Module] = [nn.Conv1d(in_c, channels[i + 1], kernel_size=1)]
            if i != len(channels) - 2:
                if use_bn:
                    mods.append(nn.BatchNorm1d(channels[i + 1]))
                mods.append(nn.LeakyReLU(inplace=True))
            self.mlps.append(nn.Sequential(*mods))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N) -> (B, out_dim, N)."""
        x0 = x
        h = x
        for i, m in enumerate(self.mlps):
            if i != 0 and self.residual:
                h = torch.cat([h, x0], dim=1)
            h = m(h)
        return h
