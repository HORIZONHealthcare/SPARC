"""Voxel decoder head: predicts CT voxels from predictor's 3D feature tokens.

Used as the "fine" head in the dual-head pretrain. A few refinement transformer
blocks + per-token linear projection to patch voxels (MAE-style decoder).

Loss is L1 against the (normalised) ground-truth CT volume.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import TransformerBlock


class VoxelDecoder(nn.Module):
    """Per-token MAE-style voxel decoder.

    Args:
        num_tokens: number of 3D query tokens (= grid_edge^3).
        in_dim: input feature dim (predictor output).
        embed_dim: internal decoder dim.
        depth: refinement transformer depth.
        num_heads: attention heads.
        patch_size: voxel patch edge (e.g. 16). Output volume edge =
            grid_edge * patch_size.
    """

    def __init__(
        self,
        num_tokens: int,
        in_dim: int = 1024,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
    ):
        super().__init__()
        g = round(num_tokens ** (1.0 / 3.0))
        assert g ** 3 == num_tokens, (
            f"num_tokens {num_tokens} must be a perfect cube for VoxelDecoder"
        )
        self.grid = g
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_voxels = patch_size ** 3

        self.proj_in = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, self.patch_voxels)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, N=g^3, p^3) -> (B, 1, g*p, g*p, g*p)."""
        b = patches.shape[0]
        g, p = self.grid, self.patch_size
        x = patches.reshape(b, g, g, g, p, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(b, 1, g * p, g * p, g * p)

    def forward(self, pred_feat: torch.Tensor) -> torch.Tensor:
        """Args:
            pred_feat: (B, N, in_dim) predictor output tokens.

        Returns:
            (B, 1, vol, vol, vol) voxel in normalised [0, 1] range.
        """
        x = self.proj_in(pred_feat) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patches = self.pred(x)
        voxels = self._unpatchify(patches)
        return torch.sigmoid(voxels)
