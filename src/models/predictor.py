"""Cross-attention predictor: 3D position queries -> 2D context tokens.

Per block: cross-attn (3D queries x 2D context) -> self-attn (over 3D
queries) -> MLP. This is the standard transformer-decoder layer without
causal masking and without a final lm-head.

After predictor.depth blocks we project from `pred_dim` to `target_dim` so
the predicted features sit in the same space as the EMA-updated target
encoder's outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    """One decoder block: cross-attn(Q=3D, KV=2D) + self-attn(Q) + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_kv_cross = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.cross_proj = nn.Linear(dim, dim, bias=True)

        self.norm_self = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.self_proj = nn.Linear(dim, dim, bias=True)

        hidden = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: Q from 3D queries, K/V from 2D context.
        q = self.q_proj(self.norm_q_cross(queries))
        kv = self.kv_proj(self.norm_kv_cross(context))
        k, v = kv.chunk(2, dim=-1)
        q_h, k_h, v_h = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        attn = F.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=self.dropout if self.training else 0.0
        )
        queries = queries + self.cross_proj(self._merge_heads(attn))

        # Self-attention over 3D queries.
        h = self.norm_self(queries)
        qkv = self.qkv(h)
        q2, k2, v2 = qkv.chunk(3, dim=-1)
        q2_h, k2_h, v2_h = self._split_heads(q2), self._split_heads(k2), self._split_heads(v2)
        attn2 = F.scaled_dot_product_attention(
            q2_h, k2_h, v2_h, dropout_p=self.dropout if self.training else 0.0
        )
        queries = queries + self.self_proj(self._merge_heads(attn2))

        # MLP.
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class CrossAttnPredictor(nn.Module):
    """3D queries cross-attend to 2D context.

    Args:
        num_queries: number of 3D query tokens (= target encoder's N_3d).
        ctx_dim: 2D context token dim (= sparse-view encoder embed_dim).
        target_dim: target encoder embed_dim (predictor output dim).
        pred_dim: internal predictor dim.
        depth, num_heads, mlp_ratio: transformer hyperparams.
    """

    def __init__(
        self,
        num_queries: int,
        ctx_dim: int,
        target_dim: int,
        pred_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.pred_dim = pred_dim

        self.query_pos = nn.Parameter(torch.zeros(1, num_queries, pred_dim))
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        self.ctx_proj = nn.Linear(ctx_dim, pred_dim)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(pred_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, target_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, V*N_2d, ctx_dim) tokens from sparse-view encoder.

        Returns:
            (B, num_queries, target_dim) predicted 3D features.
        """
        b = context.size(0)
        q = self.query_pos.expand(b, -1, -1)
        ctx = self.ctx_proj(context)
        for blk in self.blocks:
            q = blk(q, ctx)
        return self.out_proj(self.norm(q))
