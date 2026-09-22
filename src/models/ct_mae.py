"""3D Masked Autoencoder for CT volumes.

Pretrains a 3D ViT encoder we then plug in as the target encoder of the
cross-modal sparse-view pretrain (Head 1's smoothL1 target).

Follows He et al. (MAE) - high mask ratio (~0.75), asymmetric encoder /
decoder, reconstruction loss only on masked patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import TransformerBlock, _sincos_pos_embed_3d


class CTMAE(nn.Module):
    def __init__(
        self,
        vol_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dec_embed_dim: int = 512,
        dec_depth: int = 8,
        dec_num_heads: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        hu_min: float = -1024.0,
        hu_max: float = 1024.0,
    ):
        super().__init__()
        assert vol_size % patch_size == 0
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.grid = (vol_size // patch_size,) * 3
        self.num_patches = self.grid[0] ** 3
        self.patch_voxels = patch_size ** 3 * in_chans

        # Encoder
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if embed_dim % 6 == 0:
            enc_pos = _sincos_pos_embed_3d(self.grid, embed_dim)
            self.register_buffer("enc_pos", enc_pos.unsqueeze(0))
        else:
            self.enc_pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.enc_pos, std=0.02)
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.dec_proj = nn.Linear(embed_dim, dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if dec_embed_dim % 6 == 0:
            dec_pos = _sincos_pos_embed_3d(self.grid, dec_embed_dim)
            self.register_buffer("dec_pos", dec_pos.unsqueeze(0))
        else:
            self.dec_pos = nn.Parameter(torch.zeros(1, self.num_patches, dec_embed_dim))
            nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(dec_embed_dim, dec_num_heads, mlp_ratio) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_embed_dim)
        self.dec_pred = nn.Linear(dec_embed_dim, self.patch_voxels)

    def _patchify(self, ct: torch.Tensor) -> torch.Tensor:
        """(B,1,V,V,V) -> (B, N, p^3) raw voxel patches."""
        p = self.patch_size
        b, c, d, h, w = ct.shape
        nd, nh, nw = d // p, h // p, w // p
        x = ct.reshape(b, c, nd, p, nh, p, nw, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(b, nd * nh * nw, c * p * p * p)

    def _normalize_ct(self, ct: torch.Tensor) -> torch.Tensor:
        ct = ct.clamp(self.hu_min, self.hu_max)
        return (ct - self.hu_min) / (self.hu_max - self.hu_min)

    def _random_mask(self, b: int, n: int, device):
        """Standard MAE random masking. mask: 1 = masked, 0 = kept."""
        n_keep = int(n * (1 - self.mask_ratio))
        noise = torch.rand(b, n, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(b, n, device=device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return ids_keep, ids_restore, mask

    def encoder_forward(self, ct: torch.Tensor) -> torch.Tensor:
        """Full-volume encoder forward (no masking) - the frozen Stage-2 target encoder.

        Returns (B, N, embed_dim).
        """
        x = self._normalize_ct(ct)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.enc_pos
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_norm(x)

    def forward(self, ct: torch.Tensor) -> dict:
        """MAE training forward: mask, encode visible, decode all, loss on masked."""
        ct_norm = self._normalize_ct(ct)
        target = self._patchify(ct_norm)                    # (B, N, p^3)

        x = self.patch_embed(ct_norm).flatten(2).transpose(1, 2)
        x = x + self.enc_pos
        b, n, e = x.shape

        ids_keep, ids_restore, mask = self._random_mask(b, n, x.device)
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, e)
        )
        for blk in self.enc_blocks:
            x_visible = blk(x_visible)
        x_visible = self.enc_norm(x_visible)

        x_dec = self.dec_proj(x_visible)
        mask_tokens = self.mask_token.expand(b, n - x_dec.shape[1], -1)
        x_full = torch.cat([x_dec, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_embed_dim)
        )
        x_full = x_full + self.dec_pos
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        x_full = self.dec_norm(x_full)
        pred = self.dec_pred(x_full)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        return {"loss": loss, "pred": pred, "mask": mask}
