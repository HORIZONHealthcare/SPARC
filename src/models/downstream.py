"""Downstream head framework for the sparse-view foundation model.

Pattern:
    encoder = SparseViewEncoder(...)
    load_pretrained_encoder_state(encoder, pretrain_ckpt_path)
    head = ClsHead | ReconHead
    model = DownstreamWrapper(encoder, head, freeze_encoder=linear_probe)

Currently ships cls + recon. Seg / report can be added by writing a new
nn.Module head and plugging into DownstreamWrapper - no wrapper changes
needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import SparseViewEncoder
from src.models.predictor import CrossAttnPredictor
from src.models.voxel_decoder import VoxelDecoder


class ClsHead(nn.Module):
    """Mean-pool encoder tokens -> MLP -> logits."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(ctx_tokens.mean(dim=1))
        return self.mlp(x)


class ReconHead(nn.Module):
    """CrossAttn predictor + VoxelDecoder; mirrors pretrain's fine head."""

    def __init__(
        self,
        ctx_dim: int,
        num_voxel_tokens: int,
        pred_dim: int = 384,
        pred_depth: int = 12,
        pred_num_heads: int = 12,
        vox_dim: int = 384,
        vox_depth: int = 4,
        vox_num_heads: int = 8,
        patch_size: int = 16,
    ):
        super().__init__()
        self.predictor = CrossAttnPredictor(
            num_queries=num_voxel_tokens,
            ctx_dim=ctx_dim,
            target_dim=ctx_dim,
            pred_dim=pred_dim,
            depth=pred_depth,
            num_heads=pred_num_heads,
        )
        self.voxel_decoder = VoxelDecoder(
            num_tokens=num_voxel_tokens,
            in_dim=ctx_dim,
            embed_dim=vox_dim,
            depth=vox_depth,
            num_heads=vox_num_heads,
            patch_size=patch_size,
        )

    def forward(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        feat = self.predictor(ctx_tokens)
        return self.voxel_decoder(feat)


class DownstreamWrapper(nn.Module):
    """Encoder + task head. Loads pretrained encoder state externally."""

    def __init__(self, encoder: SparseViewEncoder, head: nn.Module, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def forward(self, views: torch.Tensor, plucker: torch.Tensor, sad_mm) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                ctx = self.encoder(views, plucker, sad_mm)
        else:
            ctx = self.encoder(views, plucker, sad_mm)
        return self.head(ctx)


def load_pretrained_encoder_state(
    encoder: SparseViewEncoder, ckpt_path: str, device: str | torch.device = "cpu",
) -> int:
    """Load `context_encoder.*` keys from a pretrain ckpt into `encoder`.

    Returns the iter the ckpt was saved at.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state["model"]
    prefix = "context_encoder."
    encoder_state = {
        k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        print(f"WARN load_pretrained_encoder_state: missing={missing} unexpected={unexpected}")
    return int(state["iter"])
