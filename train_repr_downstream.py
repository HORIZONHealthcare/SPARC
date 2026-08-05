"""Downstream organ seg using OUR sparse-view foundation REPRESENTATION directly.

Instead of reconstructing CT then segmenting (recon -> U-Net), we tap the
foundation's per-point features the recon head consumes:
    pt_feat = cat([ index_3d(F, q) ] + [ query_view_feats(pyr, project(q)) ])
i.e. a coarse 3D feature + HIGH-RES 2D-projected features -> fine-grained, ideal
for pixel-wise tasks. A light per-point head maps pt_feat -> organ logits.

Two modes (cfg.mode):
  frozen   : backbone frozen (encode each CT once/epoch, many head steps) -> tests
             whether the learned representation ALONE is useful (linear-probe style).
  finetune : backbone + head trained end-to-end.

Compared against train_unet_seg.py (CT / recon -> 3D U-Net from scratch). The claim:
ours-representation >= CT->U-Net, i.e. the sparse-view representation is more useful
than directly using CT. Masks (build_ts_seg_masks.py) are on the recon/query grid.
Reported = FINAL model on the FULL test split, k-fold CV over all cases.

  python train_repr_downstream.py --config configs/repr_seg_ours_frozen.yaml --cv 5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from finetune_foundation_recon import build_warmstart, collate_first
from src.datasets.drr_dataset import DRRDataset, eval_worker_init_fn
from src.models.feature_volume_3d import _project_all_views
from src.models.projection import index_3d, query_view_feats
from src.utils.nanodrr_helpers import (make_lr_grid, make_patch_grid, pose_per_view,
                                       render_with_rays, subject_from_tensor)
from src.utils.plucker import compute_plucker

ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "aorta",
          "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
          "lung_middle_lobe_right", "lung_lower_lobe_right"]
N_CLS = len(ORGANS) + 1


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_ds(manifest, gd, seed):
    """Eval-style DRRDataset (deterministic, center crop, fixed spacing) using the
    geometry from a recon config's `data` section (gd)."""
    return DRRDataset(
        manifest=manifest, v_min=gd["v_min"], v_max=gd["v_max"], det_px=gd["det_px"],
        device="cpu", seed=seed, deterministic=True,
        fixed_v=gd["fixed_v"], fixed_pattern=gd["fixed_pattern"],
        fixed_sad_mm=gd["fixed_sad_mm"], fixed_sdd_ratio=gd["fixed_sdd_ratio"],
        fixed_elev_deg=gd["fixed_elev_deg"], fixed_det_pixel_mm=gd["fixed_det_pixel_mm"],
        npy_spacing_mm=gd.get("npy_spacing_mm", 1.0),
        crop_aug=True, crop_out_size=gd["crop_out_size"],
        spacing_lo=gd.get("eval_spacing", 1.6), spacing_hi=gd.get("eval_spacing", 1.6),
        center_crop=True,
    )


def render_inputs(sample, device, grid_res, out_res):
    """Replicate the recon export data-prep -> backbone inputs + dense query grid."""
    subject = subject_from_tensor(sample["ct_hu"], sample["ct_affine"], device=device)
    rot = sample["rotations"].to(device); trans = sample["translations"].to(device)
    det_h, det_w = sample["det_h"], sample["det_w"]
    image, src, tgt = render_with_rays(subject=subject, rotations=rot, translations=trans,
                                       sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"],
                                       height=det_h, width=det_w)
    plucker = compute_plucker(src, tgt).view(sample["v"], det_h, det_w, 6).unsqueeze(0)
    k_inv, rt_inv = pose_per_view(subject, rot, trans, sdd=sample["sdd_mm"],
                                  delx=sample["det_pixel_mm"], height=det_h, width=det_w)
    _, g_world = make_patch_grid(subject, grid_res, device)
    q_norm, q_world = make_lr_grid(subject, out_res, device)
    return dict(views=image.unsqueeze(0), plucker=plucker, sad_mm=sample["sad_mm"],
                g_world=g_world, k_inv=k_inv.unsqueeze(0), rt_inv=rt_inv.unsqueeze(0),
                det_h=det_h, det_w=det_w, q_world=q_world, q_norm=q_norm)


class SegHead(nn.Module):
    """Per-point MLP on pt_feat (B, C, N) -> (B, N_CLS, N)."""

    def __init__(self, in_dim, n_cls=N_CLS, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, n_cls, 1))

    def forward(self, pt_feat):                     # (B, C, N)
        return self.net(pt_feat)                    # (B, N_CLS, N)


def encode_backbone(raw, g):
    """Run the foundation backbone -> (pyr, vol)."""
    return raw.backbone(g["views"], g["plucker"], g["sad_mm"], g["g_world"],
                        g["k_inv"], g["rt_inv"], g["det_h"], g["det_w"])


def point_feats(pyr, vol, g, qn, qw):
    """pt_feat for query pts qn(1,N,3)/qw(1,N,3) given cached backbone outputs."""
    f_3d = index_3d(vol, qn)
    pp = _project_all_views(qw, g["k_inv"], g["rt_inv"], g["det_h"], g["det_w"])
    f_2d = [query_view_feats(f, pp, fusion="max") for f in pyr]
    return torch.cat([f_3d] + f_2d, dim=1)          # (1, C, N)


def dice_loss_pts(logits, target):
    """Soft Dice over points (logits (1,C,N), target (1,N)), excl background."""
    p = F.softmax(logits, dim=1)
    t = F.one_hot(target, N_CLS).permute(0, 2, 1).float()
    inter = (p * t).sum(dim=(0, 2)); union = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2))
    return 1.0 - ((2 * inter + 1.0) / (union + 1.0))[1:].mean()


@torch.no_grad()
def case_dice(logits_full, mask_full):
    """Per-organ Dice for one case. logits_full (N_CLS,N), mask_full (N,)."""
    pred = logits_full.argmax(dim=0)
    out = {}
    for c in range(1, N_CLS):
        g = (mask_full == c)
        if g.sum() == 0:
            continue
        p = (pred == c)
        out[c] = (2 * (p & g).sum().float() / (p.sum() + g.sum() + 1e-6)).item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cv", type=int, default=None)
    ap.add_argument("--out-res", type=int, default=None, help="override seg resolution")
    ap.add_argument("--mask-dir", default=None, help="override mask dir (e.g. masks128)")
    ap.add_argument("--smoke", action="store_true", help="2 folds, few cases/epochs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cv is not None:
        cfg["cv_folds"] = args.cv
    if args.out_res is not None:
        cfg["data"]["out_res"] = args.out_res
    if args.mask_dir is not None:
        cfg["data"]["mask_dir"] = args.mask_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]; o = cfg["optim"]
    mode = cfg.get("mode", "frozen")
    out_res = int(d.get("out_res", 64))
    n_query = int(d.get("n_query", 32768))
    chunk = int(d.get("chunk", 65536))
    cv = int(cfg.get("cv_folds", 5))
    steps_per_ct = int(o.get("steps_per_ct", 8))
    epochs = int(o.get("epochs", 30))
    lr = float(o["lr"]); wd = float(o.get("weight_decay", 1e-4))
    lr_head = float(o.get("lr_head", lr))
    lr_enc = float(o.get("lr_enc", o.get("lr_encoder", lr)))
    freeze_epochs = int(o.get("freeze_epochs", 0))
    warmup_lr = float(o.get("warmup_lr", lr_head))
    hidden = cfg.get("head", {}).get("hidden", 256)

    # foundation backbone (arch from init_ckpt; weights from load_ckpt = ours-ft)
    # from_scratch=True builds the SAME architecture with random weights: the
    # "no Stage-2 pretraining" arm of the ablation when probed directly, i.e.
    # without the in-domain reconstruction adaptation that would otherwise
    # train the encoder anyway and mask what pretraining actually contributes.
    raw, ckpt_cfg = build_warmstart(cfg["init_ckpt"], device, True,
                                    from_scratch=cfg.get("from_scratch", False))
    grid_res = ckpt_cfg["model"]["grid_res"]
    if cfg.get("load_ckpt"):
        ck = torch.load(cfg["load_ckpt"], map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"], strict=False)
    raw.to(device)
    # Every seed/fold must start from the exact same pretrained representation.
    # Keep this snapshot on CPU: the full backbone is too large to duplicate on GPU.
    raw_init_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
    ckpt_dir = Path(cfg.get("ckpt", {}).get("dir", f"outputs/{cfg['meta']['name']}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def reset_raw():
        raw.load_state_dict(raw_init_state)
        raw.requires_grad_(mode == "finetune")

    def save_repr(kind, index, head, val_metric=None, step=None, unit="seed"):
        payload = {
            "head": head.state_dict(), "config": cfg, "mode": mode,
            "init_ckpt": cfg["init_ckpt"], "load_ckpt": cfg.get("load_ckpt"),
            unit: int(index), "val_metric": val_metric, "step": step,
        }
        if mode == "finetune":
            payload["model"] = raw.state_dict()
        torch.save(payload, ckpt_dir / f"{kind}_{unit}{index}.pt")

    gd = load_config(cfg["geom_config"])["data"]
    mask_dir = Path(d["mask_dir"])
    manifest_mode = ("train_manifest" in d and "test_manifest" in d)

    # Pre-render + cache backbone INPUTS (views/plucker/geometry) + dense q grid +
    # mask per case (rendering is the expensive nanodrr step; do it once).
    def render_case(sample):
        """Render ONE sample -> (name, g_cpu, mask) exactly like a cache_cases entry,
        or None if the mask file is missing (same skip rule as cache_cases). Used for
        both caching (test/val) and on-the-fly TRAIN streaming."""
        name = Path(sample["ct_path"]).stem
        mf = mask_dir / f"{name}.npy"
        if not mf.exists():
            return None
        g = render_inputs(sample, device, grid_res, out_res)
        g = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in g.items()}
        mask = torch.from_numpy(np.load(mf).astype(np.int64)).reshape(-1)   # (out_res^3,)
        return (name, g, mask)

    def cache_cases(manifest):
        ds = build_ds(manifest, gd, cfg["meta"]["seed"])
        loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_first,
                            worker_init_fn=eval_worker_init_fn)
        cs = []
        for sample in loader:
            rc = render_case(sample)
            if rc is None:
                continue
            cs.append(rc)
            if args.smoke and len(cs) >= 12:
                break
        return cs

    if manifest_mode:
        # TRAIN is STREAMED (rendered on the fly each epoch), NOT cached, to avoid
        # holding ~1006 full-res cases in RAM (OOM). Build train_ds once; a fresh
        # shuffled DataLoader is created per epoch from the seed generator below.
        # Only test+val are cached (small; reused for eval + best-val).
        train_ds = build_ds(d["train_manifest"], gd, cfg["meta"]["seed"])
        n_train = len(train_ds)
        test_cases = cache_cases(d["test_manifest"])
        # Best-val selection: optional held-out val manifest (same caching mechanism).
        val_cases = cache_cases(d["val_manifest"]) if "val_manifest" in d else None
        cases = test_cases    # used only for pt_feat dim inference below
        print(f"[REPR] MANIFEST mode mode={mode} out_res={out_res} n_query={n_query} "
              f"n_train={n_train} n_test={len(test_cases)} "
              f"n_val={0 if val_cases is None else len(val_cases)} grid_res={grid_res} "
              f"organs={ORGANS}", flush=True)
    else:
        cases = cache_cases(d["manifest"])
        print(f"[REPR] mode={mode} out_res={out_res} n_query={n_query} cases={len(cases)} "
              f"grid_res={grid_res} organs={ORGANS}", flush=True)

    def to_dev(g):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}

    def val_mean_dice(cases):
        """Best-val metric: mean Dice over present organs across cached cases, using
        the CURRENT head/raw (late-bound each seed). Saves/restores head+raw training
        flags so training dynamics are unchanged."""
        head_was, raw_was = head.training, raw.training
        head.eval(); raw.eval()
        acc = {c: [] for c in range(1, N_CLS)}
        with torch.no_grad():
            for _, g_cpu, mask in cases:
                g = to_dev(g_cpu); mask = mask.to(device)
                pyr, vol = encode_backbone(raw, g)
                m = g["q_norm"].shape[1]; outs = []
                for s in range(0, m, chunk):
                    pf = point_feats(pyr, vol, g, g["q_norm"][:, s:s + chunk],
                                     g["q_world"][:, s:s + chunk])
                    outs.append(head(pf)[0])
                logits_full = torch.cat(outs, dim=1)
                for c, dsc in case_dice(logits_full, mask).items():
                    acc[c].append(dsc)
        head.train(head_was); raw.train(raw_was)
        means = [float(np.mean(v)) for v in acc.values() if v]
        return float(np.mean(means)) if means else float("nan")

    # infer pt_feat channel dim from one case
    g0 = to_dev(cases[0][1])
    with torch.no_grad():
        pyr0, vol0 = encode_backbone(raw, g0)
        c_dim = point_feats(pyr0, vol0, g0, g0["q_norm"][:, :16], g0["q_world"][:, :16]).shape[1]
    print(f"[REPR] pt_feat dim={c_dim}", flush=True)

    N = len(cases)
    folds = np.array_split(np.arange(N), cv if not args.smoke else 2)
    organ_acc = {c: [] for c in range(1, N_CLS)}
    n_folds = len(folds)
    ep = 2 if args.smoke else epochs

    # ---- MANIFEST MODE: fixed train/test split (bypasses k-fold CV) ----
    # Active only when BOTH train_manifest and test_manifest are present; otherwise
    # the existing k-fold CV path below runs unchanged. n_seeds reuses cv_folds and
    # the per-seed training is identical to a CV fold's training (same seeds/epochs/
    # steps/loss/optimizer/model), only the train and eval pools differ.
    if manifest_mode:
        n_seeds = 2 if args.smoke else cv
        organ_acc = {c: [] for c in range(1, N_CLS)}
        test_case_acc = {i: {c: [] for c in range(1, N_CLS)} for i in range(len(test_cases))}
        for si in range(n_seeds):
            torch.manual_seed(si)
            reset_raw()
            head = SegHead(c_dim, hidden=hidden).to(device)
            params = [{"params": head.parameters(), "lr": lr_head}]
            if mode == "finetune":
                params.append({"params": raw.parameters(), "lr": lr_enc})
            opt = torch.optim.AdamW(params, lr=lr_head, weight_decay=wd)
            gen = torch.Generator(); gen.manual_seed(si)
            best_val, has_best = -float("inf"), False

            for e in range(ep):
                encoder_train = mode == "finetune" and e >= freeze_epochs
                raw.requires_grad_(encoder_train)
                opt.param_groups[0]["lr"] = warmup_lr if not encoder_train else lr_head
                # Fresh shuffled stream over the train manifest each epoch. shuffle=True
                # + generator=gen makes RandomSampler draw torch.randperm(n, generator=gen)
                # once per epoch (matching the old per-epoch randperm) -> seeded/
                # reproducible, every train case seen once per epoch. Rendered on the fly
                # and discarded (no train caching).
                train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                                          generator=gen, num_workers=4,
                                          collate_fn=collate_first,
                                          worker_init_fn=eval_worker_init_fn)
                for sample in train_loader:
                    rc = render_case(sample)          # render on the fly, then discard
                    if rc is None:                    # skip missing-mask (as cache_cases)
                        continue
                    _, g_cpu, mask = rc
                    g = to_dev(g_cpu); mask = mask.to(device)
                    qn_full, qw_full = g["q_norm"], g["q_world"]
                    if mode == "frozen":
                        head.train()
                        with torch.no_grad():
                            pyr, vol = encode_backbone(raw, g)
                        for _ in range(steps_per_ct):
                            idx = torch.randint(0, qn_full.shape[1], (n_query,), device=device)
                            with torch.no_grad():
                                pf = point_feats(pyr, vol, g, qn_full[:, idx], qw_full[:, idx])
                            logits = head(pf); tgt = mask[idx][None]
                            loss = F.cross_entropy(logits, tgt) + dice_loss_pts(logits, tgt)
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    else:  # finetune: head warm-up, then gentle end-to-end adaptation
                        head.train(); raw.train(encoder_train)
                        pyr, vol = encode_backbone(raw, g)
                        idx = torch.randint(0, qn_full.shape[1], (n_query,), device=device)
                        pf = point_feats(pyr, vol, g, qn_full[:, idx], qw_full[:, idx])
                        logits = head(pf); tgt = mask[idx][None]
                        loss = F.cross_entropy(logits, tgt) + dice_loss_pts(logits, tgt)
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

                if val_cases is not None:             # end-of-epoch best-val snapshot
                    vmd = val_mean_dice(val_cases)
                    if vmd > best_val:
                        best_val = vmd
                        has_best = True
                        save_repr("best", si, head, best_val, e + 1)
                    print(f"[seed {si}] epoch={e + 1}/{ep} val_Dice={vmd:.4f} "
                          f"best={best_val:.4f}", flush=True)
                save_repr("latest", si, head,
                          None if val_cases is None else vmd, e + 1)

            if val_cases is not None and has_best:
                state = torch.load(ckpt_dir / f"best_seed{si}.pt", map_location=device,
                                   weights_only=False)
                head.load_state_dict(state["head"])
                if mode == "finetune":
                    raw.load_state_dict(state["model"])
            elif val_cases is None:
                best_val = float("nan")
                save_repr("best", si, head, best_val, ep)
            head.eval(); raw.eval()
            with torch.no_grad():
                for i in range(len(test_cases)):
                    _, g_cpu, mask = test_cases[i]
                    g = to_dev(g_cpu); mask = mask.to(device)
                    pyr, vol = encode_backbone(raw, g)
                    m = g["q_norm"].shape[1]; outs = []
                    for s in range(0, m, chunk):
                        pf = point_feats(pyr, vol, g, g["q_norm"][:, s:s + chunk],
                                         g["q_world"][:, s:s + chunk])
                        outs.append(head(pf)[0])                     # (N_CLS, chunk)
                    logits_full = torch.cat(outs, dim=1)             # (N_CLS, M)
                    for c, dsc in case_dice(logits_full, mask).items():
                        test_case_acc[i][c].append(dsc)
            if val_cases is not None:
                print(f"[seed {si}] best_val_Dice={best_val:.4f} eval on {len(test_cases)} test cases done", flush=True)
            else:
                print(f"[seed {si}] eval on {len(test_cases)} test cases done", flush=True)

        # per-case Dice averaged over seeds -> dump rows + overall aggregation
        rows = []
        for i in range(len(test_cases)):
            cid = test_cases[i][0]
            row = {"case_id": cid}
            present = []
            for c in range(1, N_CLS):
                vals = test_case_acc[i][c]
                if vals:
                    dval = float(np.mean(vals))
                    row[f"dice::{ORGANS[c - 1]}"] = dval
                    present.append(dval)
                    organ_acc[c].append(dval)
                else:
                    row[f"dice::{ORGANS[c - 1]}"] = float("nan")
            row["mean_dice"] = float(np.mean(present)) if present else float("nan")
            rows.append(row)

        cols = ["case_id"] + [f"dice::{o}" for o in ORGANS] + ["mean_dice"]
        pd.DataFrame(rows, columns=cols).to_csv(ckpt_dir / "dice_test.csv", index=False)

        organ_means = {ORGANS[c - 1]: float(np.mean(v)) for c, v in organ_acc.items() if v}
        overall = float(np.mean(list(organ_means.values()))) if organ_means else float("nan")
        print(f"[REPR SEG RESULT] {cfg['meta']['name']} mode={mode} test_mean_Dice={overall:.4f} "
              f"N={len(test_cases)} n_seeds={n_seeds} "
              f"per_organ={ {k: round(v, 4) for k, v in organ_means.items()} }", flush=True)
        print(f"[REPR] wrote per-case dice -> {ckpt_dir / 'dice_test.csv'} ({len(rows)} cases)", flush=True)
        sys.exit(0)

    for fi in range(n_folds):
        te_i = set(folds[fi].tolist())
        tr_i = [i for i in range(N) if i not in te_i]
        torch.manual_seed(fi)
        reset_raw()
        head = SegHead(c_dim, hidden=hidden).to(device)
        params = [{"params": head.parameters(), "lr": lr_head}]
        if mode == "finetune":
            params.append({"params": raw.parameters(), "lr": lr_enc})
        opt = torch.optim.AdamW(params, lr=lr_head, weight_decay=wd)
        gen = torch.Generator(); gen.manual_seed(fi)

        for e in range(ep):
            encoder_train = mode == "finetune" and e >= freeze_epochs
            raw.requires_grad_(encoder_train)
            opt.param_groups[0]["lr"] = warmup_lr if not encoder_train else lr_head
            order = torch.randperm(len(tr_i), generator=gen).tolist()
            for oi in order:
                _, g_cpu, mask = cases[tr_i[oi]]
                g = to_dev(g_cpu); mask = mask.to(device)
                qn_full, qw_full = g["q_norm"], g["q_world"]
                if mode == "frozen":
                    head.train()
                    with torch.no_grad():
                        pyr, vol = encode_backbone(raw, g)
                    for _ in range(steps_per_ct):
                        idx = torch.randint(0, qn_full.shape[1], (n_query,), device=device)
                        with torch.no_grad():
                            pf = point_feats(pyr, vol, g, qn_full[:, idx], qw_full[:, idx])
                        logits = head(pf); tgt = mask[idx][None]
                        loss = F.cross_entropy(logits, tgt) + dice_loss_pts(logits, tgt)
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                else:  # finetune: head warm-up, then gentle end-to-end adaptation
                    head.train(); raw.train(encoder_train)
                    pyr, vol = encode_backbone(raw, g)
                    idx = torch.randint(0, qn_full.shape[1], (n_query,), device=device)
                    pf = point_feats(pyr, vol, g, qn_full[:, idx], qw_full[:, idx])
                    logits = head(pf); tgt = mask[idx][None]
                    loss = F.cross_entropy(logits, tgt) + dice_loss_pts(logits, tgt)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        save_repr("latest", fi, head, step=ep, unit="fold")
        save_repr("best", fi, head, step=ep, unit="fold")
        head.eval(); raw.eval()
        with torch.no_grad():
            for i in folds[fi].tolist():
                _, g_cpu, mask = cases[i]
                g = to_dev(g_cpu); mask = mask.to(device)
                pyr, vol = encode_backbone(raw, g)
                m = g["q_norm"].shape[1]; outs = []
                for s in range(0, m, chunk):
                    pf = point_feats(pyr, vol, g, g["q_norm"][:, s:s + chunk],
                                     g["q_world"][:, s:s + chunk])
                    outs.append(head(pf)[0])                     # (N_CLS, chunk)
                logits_full = torch.cat(outs, dim=1)             # (N_CLS, M)
                for c, dsc in case_dice(logits_full, mask).items():
                    organ_acc[c].append(dsc)
        print(f"[fold {fi}] held-out {len(folds[fi])} done", flush=True)

    organ_means = {ORGANS[c - 1]: float(np.mean(v)) for c, v in organ_acc.items() if v}
    overall = float(np.mean(list(organ_means.values()))) if organ_means else float("nan")
    print(f"[REPR SEG RESULT] {cfg['meta']['name']} mode={mode} cv{n_folds}_mean_Dice={overall:.4f} "
          f"N={N} per_organ={ {k: round(v, 4) for k, v in organ_means.items()} }", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
