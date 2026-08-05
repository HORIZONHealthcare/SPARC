"""SparseViewJEPA: dual-branch foundation pretrain.

COARSE branch (semantic, for cls / report):
    ViT SparseViewEncoder -> CrossAttnPredictor -> 3D feature tokens
    smooth_l1 vs frozen CT-MAE encoder features.

FINE branch (recon, for visualisation / seg), DeepSparse-style:
    CNN feature pyramid over views
    -> project a low-res 3D grid into each scale (query_view_feats)
    -> reshape to 3D volume -> 3D conv refine  (structural prior)
    -> per query point: index_3d(refined vol) + multi-scale 2D projected
       features -> PointDecoder MLP -> density
    L1 vs GT HU at the (random) query points. Resolution-free.

Geometry (k_inv / rt_inv per view) comes from the dataset/render so the
fine branch uses explicit projection - consistent between pretrain and the
downstream recon (which just queries a denser point set with the same head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.projection import index_3d, project_points, query_view_feats


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


class SparseViewJEPA(nn.Module):
    def __init__(
        self,
        context_encoder: nn.Module,     # ViT SparseViewEncoder (coarse)
        target_encoder: nn.Module,      # frozen CT-MAE
        predictor: nn.Module,           # CrossAttnPredictor (coarse)
        cnn_pyramid: nn.Module,         # CNNPyramid2D (fine)
        point_decoder: nn.Module,       # PointDecoder (fine)
        conv3d_refine: nn.Module,       # Conv3DRefine (fine 3D prior)
        lr_grid_res: int = 16,
        feature_loss_type: str = "smooth_l1",
        fine_loss_type: str = "l1",
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        freeze_target: bool = True,
    ):
        super().__init__()
        assert feature_loss_type in ("smooth_l1", "l1", "l2")
        assert fine_loss_type in ("l1", "l2", "mse")
        self.fine_loss_type = fine_loss_type
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.cnn_pyramid = cnn_pyramid
        self.point_decoder = point_decoder
        self.conv3d_refine = conv3d_refine
        self.lr_grid_res = lr_grid_res
        self.feature_loss_type = feature_loss_type
        self.coarse_weight = float(coarse_weight)
        self.fine_weight = float(fine_weight)
        self.freeze_target = bool(freeze_target)
        if self.freeze_target:
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

    def _target_forward(self, ct):
        fn = getattr(self.target_encoder, "encoder_forward", self.target_encoder.forward)
        if self.freeze_target:
            with torch.no_grad():
                return fn(ct)
        return fn(ct)

    def _feat_loss(self, pred, target):
        if self.feature_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        if self.feature_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    @staticmethod
    def _project_all_views(pts_world, k_inv, rt_inv, det_h, det_w):
        """pts_world (B,N,3); k_inv (B,V,3,3); rt_inv (B,V,4,4) -> (B,V,N,2)."""
        v = k_inv.shape[1]
        outs = [
            project_points(pts_world, k_inv[:, vi], rt_inv[:, vi], det_h, det_w)
            for vi in range(v)
        ]
        return torch.stack(outs, dim=1)

    def _build_fine_volume(self, pyr, lr_grid_world, k_inv, rt_inv, det_h, det_w):
        b = lr_grid_world.shape[0]
        g = self.lr_grid_res
        pp = self._project_all_views(lr_grid_world, k_inv, rt_inv, det_h, det_w)
        per_scale = [query_view_feats(f, pp, fusion="max") for f in pyr]
        cat = torch.cat(per_scale, dim=1)                          # (B, sumC, G3)
        vol = cat.reshape(b, cat.shape[1], g, g, g)
        return self.conv3d_refine(vol)

    def forward(
        self,
        views: torch.Tensor,            # (B, V, 1, H, W)
        plucker: torch.Tensor,          # (B, V, H, W, 6)
        ct: torch.Tensor,               # (B, 1, D, H, W) HU
        sad_mm,
        k_inv: torch.Tensor,            # (B, V, 3, 3)
        rt_inv: torch.Tensor,           # (B, V, 4, 4)
        det_h: int,
        det_w: int,
        q_pts_world: torch.Tensor,      # (B, N, 3) query pts WORLD
        q_pts_norm: torch.Tensor,       # (B, N, 3) query pts [-1,1] CT grid
        q_gt_hu: torch.Tensor,          # (B, N) GT normalised HU
        lr_grid_world: torch.Tensor,    # (B, G3, 3) low-res grid WORLD
    ) -> dict:
        out = {}

        # ---- COARSE (semantic) ----
        ctx = self.context_encoder(views, plucker, sad_mm)
        pred_feat = self.predictor(ctx)
        tgt_feat = self._target_forward(ct).detach()
        loss_coarse = self._feat_loss(pred_feat, tgt_feat)
        out["loss_coarse"] = loss_coarse

        # ---- FINE (recon, DeepSparse-style) ----
        b, v = views.shape[:2]
        h, w = views.shape[-2:]
        x = views.reshape(b * v, 1, h, w)
        pyr = self.cnn_pyramid(x)
        pyr = [f.reshape(b, v, *f.shape[1:]) for f in pyr]         # (B,V,C,h,w)

        feat3d = self._build_fine_volume(pyr, lr_grid_world, k_inv, rt_inv, det_h, det_w)
        f_3d = index_3d(feat3d, q_pts_norm)                        # (B, Cr, N)

        pp = self._project_all_views(q_pts_world, k_inv, rt_inv, det_h, det_w)
        f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]

        pt_feat = torch.cat([f_3d] + f_2d, dim=1)                  # (B, Csum, N)
        density = self.point_decoder(pt_feat)                      # (B, 1, N)
        pred_density = density.squeeze(1)
        if self.fine_loss_type == "l1":
            loss_fine = F.l1_loss(pred_density, q_gt_hu)
        else:
            loss_fine = F.mse_loss(pred_density, q_gt_hu)
        out["loss_fine"] = loss_fine
        out["density"] = density

        out["loss"] = self.coarse_weight * loss_coarse + self.fine_weight * loss_fine
        return out
