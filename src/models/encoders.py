"""Two encoders: a 2D ViT for sparse X-ray views and a 3D ViT for CT volumes.

`SparseViewEncoder` is the context encoder. It takes a bag of V views and
their per-pixel Plücker rays, embeds each view into patch tokens, adds a
per-patch ray embedding, then runs a standard transformer over the
concatenation of all view tokens.

`CTTargetEncoder` is the target encoder. Plain 3D ViT: conv3d patch embed,
learnable or sin/cos 3D position embedding, transformer.

Both share a tiny TransformerBlock so the smoke test runs end to end without
depending on timm internals. We can swap to timm / V-JEPA blocks later
without changing the interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.plucker import normalize_plucker, patch_pool_plucker


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with SDPA self-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)            # each (B, H, N, Dh)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


def _sincos_pos_embed_3d(grid: tuple[int, int, int], dim: int) -> torch.Tensor:
    """Sin/cos 3D positional embedding -> (D*H*W, dim).

    Splits `dim` into 3 chunks (one per axis); each chunk uses standard
    sin/cos schedule.
    """
    if dim % 6 != 0:
        raise ValueError(f"dim {dim} must be divisible by 6 for sincos 3D embed")
    per_axis = dim // 3
    half = per_axis // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
    coords = [torch.arange(s).float() for s in grid]
    embs = []
    for c in coords:
        ang = c[:, None] * inv_freq[None, :]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        embs.append(emb)
    d_emb, h_emb, w_emb = embs
    D, H, W = grid
    pos = torch.cat([
        d_emb[:, None, None, :].expand(D, H, W, per_axis),
        h_emb[None, :, None, :].expand(D, H, W, per_axis),
        w_emb[None, None, :, :].expand(D, H, W, per_axis),
    ], dim=-1)
    return pos.reshape(D * H * W, dim)


class SparseViewEncoder(nn.Module):
    """2D ViT over a bag of sparse X-ray views with Plücker ray injection."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.tokens_per_view = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        self.ray_embed = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Spatial positional embedding shared across views; view identity is
        # carried by the ray embedding (different geometry -> different rays).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.tokens_per_view, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, views: torch.Tensor, plucker: torch.Tensor, sad_mm
    ) -> torch.Tensor:
        """
        Args:
            views:   (B, V, C, H, W)
            plucker: (B, V, H, W, 6)
            sad_mm:  scalar / (B,) / (B, V); used to normalise Plücker moment.

        Returns:
            (B, V * N_2d, embed_dim)
        """
        b, v, c, h, w = views.shape
        assert h == self.img_size and w == self.img_size, (
            f"views H,W={h},{w} but encoder expects {self.img_size}"
        )

        x = views.reshape(b * v, c, h, w)
        x = self.patch_embed(x)                          # (B*V, E, h/p, w/p)
        x = x.flatten(2).transpose(1, 2)                 # (B*V, N, E)
        x = x.reshape(b, v, self.tokens_per_view, self.embed_dim)

        plucker = normalize_plucker(plucker, sad_mm)     # (B, V, H, W, 6)
        plucker = plucker.reshape(b * v, h, w, 6)
        ray_patch = patch_pool_plucker(plucker, self.patch_size)
        ray_patch = ray_patch.reshape(b, v, self.tokens_per_view, 6)
        x = x + self.ray_embed(ray_patch)

        x = x + self.pos_embed                           # broadcasts across V
        x = x.reshape(b, v * self.tokens_per_view, self.embed_dim)

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class CTTargetEncoder(nn.Module):
    """3D ViT over a CT volume."""

    def __init__(
        self,
        vol_size: int = 64,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 132,        # divisible by 6 for sincos, by 4 for heads
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert vol_size % patch_size == 0
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid = (vol_size // patch_size,) * 3
        self.num_tokens = self.grid[0] * self.grid[1] * self.grid[2]

        self.patch_embed = nn.Conv3d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        if embed_dim % 6 == 0:
            pos = _sincos_pos_embed_3d(self.grid, embed_dim)
            self.register_buffer("pos_embed", pos.unsqueeze(0))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct: (B, 1, D, H, W) with D=H=W=vol_size

        Returns:
            (B, N_3d, embed_dim)
        """
        x = self.patch_embed(ct)                         # (B, E, d', h', w')
        x = x.flatten(2).transpose(1, 2)                 # (B, N, E)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
