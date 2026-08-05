"""Unified sparse-view X-ray foundation model.

ONE backbone (XRayEncoder2D -> FeatureVolume3D) produces a single 3D feature
volume F from the sparse views. TWO lightweight heads, both consuming F, give
the pretraining signals:

  recon head    : PointDecoder(index_3d(F) + projected 2D feats) -> density,
                  voxel-supervised by the (resampled) CT.  [reconstruction]
  semantic head : a few transformer blocks + linear on F -> predicted CT-MAE
                  tokens, matched to a FROZEN CT-MAE (smooth_l1).  [JEPA]

F (and the backbone producing it) is THE transferable representation: downstream
recon / seg / cls all decode F. The heads are discardable pretraining signals,
not separate encoders -> this is a single foundation model, not two models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import TransformerBlock
from src.models.feature_volume_3d import _project_all_views
from src.models.point_decoder import PointDecoder
from src.models.projection import index_3d, query_view_feats


class SemanticHead(nn.Module):
    """F (B, Cf, G, G, G) -> predicted CT-MAE tokens (B, G^3, mae_dim)."""

    def __init__(self, in_dim: int, mae_dim: int, depth: int = 2,
                 num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(in_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.proj = nn.Linear(in_dim, mae_dim)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        tok = vol.flatten(2).transpose(1, 2)        # (B, G^3, Cf) (d,h,w raster)
        for blk in self.blocks:
            tok = blk(tok)
        return self.proj(tok)                        # (B, G^3, mae_dim)


class SparseViewFoundation(nn.Module):
    def __init__(
        self,
        encoder2d: nn.Module,        # XRayEncoder2D
        volume3d: nn.Module,         # FeatureVolume3D
        target_encoder: nn.Module,   # frozen CTMAE
        pointdec_mlp: list[int],
        semantic_depth: int = 2,
        semantic_heads: int = 8,
        recon_loss: str = "mse",
        air_weight: float = 1.0,
        air_thresh: float = 0.16,
        huber_delta: float = 0.05,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        freeze_target: bool = True,
        use_semantic: bool = True,
    ):
        super().__init__()
        assert recon_loss in ("mse", "l1", "huber")
        self.encoder2d = encoder2d
        self.volume3d = volume3d
        self.recon_loss = recon_loss
        self.air_weight = float(air_weight)
        self.air_thresh = float(air_thresh)
        self.huber_delta = float(huber_delta)
        self.coarse_weight = float(coarse_weight)
        self.fine_weight = float(fine_weight)
        self.freeze_target = bool(freeze_target)
        self.use_semantic = bool(use_semantic)        # False -> recon-only ablation

        f_dim = volume3d.out_dim
        sum_pyr = sum(encoder2d.channels)
        self.point_decoder = PointDecoder(
            channels=[f_dim + sum_pyr] + pointdec_mlp + [1], residual=True, use_bn=True,
        )
        if self.use_semantic:
            self.target_encoder = target_encoder
            mae_dim = getattr(target_encoder, "embed_dim")
            self.semantic_head = SemanticHead(
                in_dim=f_dim, mae_dim=mae_dim, depth=semantic_depth, num_heads=semantic_heads,
            )
            if self.freeze_target:
                for p in self.target_encoder.parameters():
                    p.requires_grad_(False)
        else:
            self.target_encoder = None
            self.semantic_head = None

    def backbone(self, views, plucker, sad_mm, grid_world, k_inv, rt_inv, det_h, det_w):
        """Sparse views -> (multi-scale 2D pyramid, 3D feature volume F)."""
        pyr = self.encoder2d(views, plucker, sad_mm)
        vol = self.volume3d(pyr, grid_world, k_inv, rt_inv, det_h, det_w)
        return pyr, vol

    def _target_tokens(self, ct: torch.Tensor) -> torch.Tensor:
        fn = getattr(self.target_encoder, "encoder_forward", self.target_encoder.forward)
        if self.freeze_target:
            with torch.no_grad():
                return fn(ct)
        return fn(ct)

    def forward(
        self,
        views: torch.Tensor,         # (B, V, 1, H, W)
        plucker: torch.Tensor,       # (B, V, H, W, 6)
        ct: torch.Tensor,            # (B, 1, vol, vol, vol) HU for the frozen MAE
        sad_mm,
        k_inv: torch.Tensor,         # (B, V, 3, 3)
        rt_inv: torch.Tensor,        # (B, V, 4, 4)
        det_h: int,
        det_w: int,
        grid_world: torch.Tensor,    # (B, G^3, 3) aligned patch-grid world (d,h,w)
        q_pts_world: torch.Tensor,   # (B, N, 3) recon query pts WORLD
        q_pts_norm: torch.Tensor,    # (B, N, 3) recon query pts [-1,1] grid
        q_gt_hu: torch.Tensor,       # (B, N) GT normalised HU
    ) -> dict:
        out: dict = {}
        pyr, vol = self.backbone(
            views, plucker, sad_mm, grid_world, k_inv, rt_inv, det_h, det_w,
        )

        # ---- recon head (dense, voxel-level) ----
        f_3d = index_3d(vol, q_pts_norm)                              # (B, Cf, N)
        pp = _project_all_views(q_pts_world, k_inv, rt_inv, det_h, det_w)
        f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]
        pt_feat = torch.cat([f_3d] + f_2d, dim=1)
        density = self.point_decoder(pt_feat).squeeze(1)             # (B, N)
        if self.recon_loss == "mse":
            err = (density - q_gt_hu) ** 2
        elif self.recon_loss == "huber":
            err = F.huber_loss(density, q_gt_hu, reduction="none", delta=self.huber_delta)
        else:  # l1
            err = (density - q_gt_hu).abs()
        if self.air_weight != 1.0:      # up-weight air/background points (normalized gt < air_thresh)
            w = torch.where(q_gt_hu < self.air_thresh, err.new_tensor(self.air_weight),
                            err.new_tensor(1.0))
            loss_fine = (w * err).sum() / w.sum()
        else:
            loss_fine = err.mean()

        # ---- semantic head (JEPA: predict frozen CT-MAE tokens) ----
        if self.use_semantic:
            pred_tok = self.semantic_head(vol)                       # (B, G^3, mae_dim)
            tgt_tok = self._target_tokens(ct).detach()               # (B, G^3, mae_dim)
            loss_coarse = F.smooth_l1_loss(pred_tok, tgt_tok)
        else:
            loss_coarse = density.new_zeros(())

        out["loss_fine"] = loss_fine
        out["loss_coarse"] = loss_coarse
        out["loss"] = self.coarse_weight * loss_coarse + self.fine_weight * loss_fine
        out["density"] = density
        return out

    @torch.no_grad()
    def infer_dense(
        self,
        views: torch.Tensor, plucker: torch.Tensor, sad_mm,
        grid_world: torch.Tensor, k_inv: torch.Tensor, rt_inv: torch.Tensor,
        det_h: int, det_w: int,
        q_world: torch.Tensor, q_norm: torch.Tensor,   # (1, M, 3) dense grid
        chunk: int = 65536,
    ) -> torch.Tensor:
        """Recon head on a dense query grid -> (1, M) density. For eval/seg/recon."""
        pyr, vol = self.backbone(
            views, plucker, sad_mm, grid_world, k_inv, rt_inv, det_h, det_w,
        )
        m = q_world.shape[1]
        outs = []
        for s in range(0, m, chunk):
            qw = q_world[:, s:s + chunk]
            qn = q_norm[:, s:s + chunk]
            f_3d = index_3d(vol, qn)
            pp = _project_all_views(qw, k_inv, rt_inv, det_h, det_w)
            f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]
            outs.append(self.point_decoder(torch.cat([f_3d] + f_2d, dim=1)).squeeze(1))
        return torch.cat(outs, dim=1)
