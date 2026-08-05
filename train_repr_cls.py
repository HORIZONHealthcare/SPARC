"""Protocol-B classification: OUR sparse-view foundation REPRESENTATION -> cls head.

The encoder-features tier of the downstream comparison (ours only). Instead of
reconstructing CT then classifying, we global-average-pool the foundation's 3D
feature volume F (the same backbone the recon/seg heads tap) -> a light cls head.
No reconstruction at inference.

Two modes (cfg.mode):
  frozen   : backbone frozen; pooled features cached once -> train head many steps (lp).
  finetune : backbone + head trained end-to-end (ft).

Reports PER-CLASS AUROC (18 abnormalities) + macro mean over seeds, full CT-RATE test.

  python train_repr_cls.py --config configs/reprcls_ours_frozen.yaml
"""

from __future__ import annotations
from src.utils.project_paths import resolve_project_path

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
from src.datasets.drr_dataset import eval_worker_init_fn
from train_repr_downstream import build_ds, render_inputs, encode_backbone

ABN_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class ClsHead(nn.Module):
    def __init__(self, in_dim, n_cls=len(ABN_COLS), hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


def per_class_auc(logits, targets):
    probs = torch.sigmoid(logits).cpu().numpy(); y = targets.cpu().numpy()
    aucs = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            aucs.append(float("nan")); continue
        order = np.argsort(probs[:, c]); ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        npos = yc.sum(); nneg = len(yc) - npos
        aucs.append((ranks[yc == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    valid = [a for a in aucs if not np.isnan(a)]
    return aucs, (float(np.mean(valid)) if valid else float("nan"))


def load_labels(csv):
    df = pd.read_csv(resolve_project_path(csv))
    df["key"] = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    return {r["key"]: np.array([r[c] for c in ABN_COLS], dtype=np.float32)
            for _, r in df.iterrows()}


def pooled_feat(raw, g):
    """Global-average-pool the foundation 3D feature volume -> (1, Cf)."""
    _, vol = encode_backbone(raw, g)            # vol: (1, Cf, G, G, G)
    return vol.mean(dim=(2, 3, 4))              # (1, Cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, o = cfg["data"], cfg["optim"]
    mode = cfg.get("mode", "frozen")
    n_seeds = int(cfg.get("n_seeds", 3))

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

    gd = load_config(cfg["geom_config"])["data"]
    lab = load_labels(d["label_csv"])

    def loader_for(manifest):
        ds = build_ds(manifest, gd, cfg["meta"]["seed"])
        return DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_first,
                          worker_init_fn=eval_worker_init_fn)

    def to_dev(g):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}

    @torch.no_grad()
    def cache_feats(manifest):
        """Stream the manifest, render+pool each CT, keep ONLY the small pooled feature
        (never store the full rendered inputs -> memory-safe)."""
        feats, ys, names = [], [], []
        for sample in loader_for(manifest):
            name = Path(sample["ct_path"]).stem
            if name not in lab:
                continue
            g = to_dev(render_inputs(sample, device, grid_res, out_res=16))
            feats.append(pooled_feat(raw, g).cpu())
            ys.append(torch.from_numpy(lab[name]))
            names.append(name)
        return torch.cat(feats), torch.stack(ys), names

    lr = float(o["lr"]); wd = float(o.get("weight_decay", 1e-4))
    hidden = cfg.get("head", {}).get("hidden", 512)
    iters = int(o.get("iters", 4000)); bs = int(d.get("batch_size", 16))
    epochs = int(o.get("epochs", 5))
    eval_every = int(o.get("eval_every", max(1, iters // 8)))  # best-val cadence (frozen)
    n_test = 0
    ckpt_dir = Path(cfg.get("ckpt", {}).get("dir", f"outputs/{cfg['meta']['name']}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_seed_checkpoint(kind, seed, head, raw_model=None, val_metric=None, stage=None):
        """Persist enough state to reproduce inference for one seed.

        Frozen probes only need the task head because the immutable backbone source is
        recorded in the payload. Finetune checkpoints include the adapted backbone.
        """
        payload = {
            "mode": mode,
            "seed": int(seed),
            "stage": stage,
            "val_metric": val_metric,
            "head": head.state_dict(),
            "backbone_ckpt": cfg.get("load_ckpt") or cfg.get("init_ckpt"),
            "config": cfg,
        }
        if raw_model is not None:
            payload["model"] = raw_model.state_dict()
        path = ckpt_dir / f"{kind}_seed{seed}.pt"
        torch.save(payload, path)
        return path

    if mode == "frozen":
        for p in raw.parameters():
            p.requires_grad_(False)
        Xtr, Ytr, _ = cache_feats(d["train_manifest"])
        Xte, Yte, te_names = cache_feats(d["test_manifest"])
        # Best-val selection: optional held-out val manifest, cached like train/test.
        Xva = Yva = None
        if d.get("val_manifest"):
            Xva, Yva, _ = cache_feats(d["val_manifest"])
            Xva = Xva.to(device)
        cdim = Xtr.shape[1]; n_test = Xte.shape[0]
        print(f"[REPR-CLS] mode=frozen train={Xtr.shape[0]} test={n_test} "
              f"val={0 if Xva is None else Xva.shape[0]} feat_dim={cdim}", flush=True)
        Xtr, Ytr, Xte = Xtr.to(device), Ytr.to(device), Xte.to(device)
        n = Xtr.shape[0]
        macros, pcs, seed_logits = [], [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            head = ClsHead(cdim, hidden=hidden).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
            gen = torch.Generator(device=device); gen.manual_seed(seed)
            head.train()
            best_val, best_state = -float("inf"), None
            last_val = None
            for it in range(iters):
                idx = torch.randint(0, n, (bs,), device=device, generator=gen)
                loss = F.binary_cross_entropy_with_logits(head(Xtr[idx]), Ytr[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if Xva is not None and (it + 1) % eval_every == 0:
                    head.eval()
                    with torch.no_grad():
                        vmacro = per_class_auc(head(Xva).cpu(), Yva)[1]
                    last_val = vmacro
                    head.train()
                    if vmacro > best_val:
                        best_val, best_state = vmacro, copy.deepcopy(head.state_dict())
                        save_seed_checkpoint("best", seed, head, val_metric=best_val,
                                             stage=f"lp_iter{it + 1}")
                    save_seed_checkpoint("latest", seed, head, val_metric=vmacro,
                                         stage=f"lp_iter{it + 1}")
            # Preserve the actual final optimizer state before restoring best-val.
            save_seed_checkpoint("latest", seed, head,
                                 val_metric=last_val,
                                 stage=f"lp_iter{iters}")
            if Xva is not None:
                head.eval()
                with torch.no_grad():
                    vmacro = per_class_auc(head(Xva).cpu(), Yva)[1]
                if vmacro > best_val:
                    best_val, best_state = vmacro, copy.deepcopy(head.state_dict())
                if best_state is not None:
                    head.load_state_dict(best_state)
            else:
                best_val, best_state = float("nan"), copy.deepcopy(head.state_dict())
            save_seed_checkpoint("best", seed, head, val_metric=best_val, stage="lp_best")
            head.eval()
            with torch.no_grad():
                logits_te = head(Xte).cpu()
            pc, macro = per_class_auc(logits_te, Yte)
            macros.append(macro); pcs.append(pc); seed_logits.append(logits_te)
            if Xva is not None:
                print(f"[seed {seed}] best_val_AUC={best_val:.4f} test_macro_AUC={macro:.4f}", flush=True)
            else:
                print(f"[seed {seed}] macro_AUC={macro:.4f}", flush=True)
    else:  # finetune: re-render per epoch (memory-safe), backprop through backbone
        with torch.no_grad():
            cdim = pooled_feat(raw, to_dev(render_inputs(next(iter(loader_for(d["test_manifest"]))),
                               device, grid_res, 16))).shape[1]
        print(f"[REPR-CLS] mode=finetune feat_dim={cdim}", flush=True)
        val_manifest = d.get("val_manifest")

        def stream_logits(manifest):
            """Render+encode+head over a manifest -> (logits, labels, names). Uses the
            CURRENT head/raw (late-bound each seed)."""
            L, Y, nm = [], [], []
            with torch.no_grad():
                for sample in loader_for(manifest):
                    name = Path(sample["ct_path"]).stem
                    if name not in lab:
                        continue
                    g = to_dev(render_inputs(sample, device, grid_res, 16))
                    L.append(head(pooled_feat(raw, g)).cpu()); Y.append(torch.from_numpy(lab[name]))
                    nm.append(name)
            return torch.cat(L), torch.stack(Y), nm

        macros, pcs, seed_logits = [], [], []
        # Stabilized finetune (was unstable: 2/3 seeds collapsed to ~random). Fixes:
        #  (1) reset backbone to pretrained EACH seed -- seeds shared a mutated `raw` before, so they
        #      were NOT independent; (2) discriminative LR: tiny lr_enc for the 136M backbone, normal
        #      lr_head for the head; (3) freeze backbone for the first freeze_epochs (head-first warmup);
        #  (4) grad accumulation + clipping to tame the batch=1 gradient noise.
        lr_head = float(o.get("lr_head", lr))
        lr_enc = float(o.get("lr_enc", lr * 0.05))
        freeze_epochs = int(o.get("freeze_epochs", 1))
        grad_clip = float(o.get("grad_clip", 1.0))
        accum = max(1, int(o.get("accum", 4)))
        lp_warmup_iters = int(o.get("lp_warmup_iters", 0))
        lp_lr = float(o.get("lp_lr", 1.0e-3))
        lp_eval_every = int(o.get("lp_eval_every", eval_every))
        raw_init = copy.deepcopy(raw.state_dict())
        print(f"[REPR-CLS finetune] lr_head={lr_head} lr_enc={lr_enc} freeze_epochs={freeze_epochs} "
              f"accum={accum} grad_clip={grad_clip} epochs={epochs} "
              f"lp_warmup_iters={lp_warmup_iters} lp_lr={lp_lr}", flush=True)

        # A fair LP->FT comparison first trains the head with exactly the cached-feature
        # recipe used by the linear probe. The previous FT warm-up saw only three streamed
        # epochs (~6k examples), versus 4k batches (~64k draws) for LP, so its head was
        # substantially under-trained before the 136M encoder was unfrozen.
        Xtr_lp = Ytr_lp = Xva_lp = Yva_lp = None
        if lp_warmup_iters > 0:
            Xtr_lp, Ytr_lp, _ = cache_feats(d["train_manifest"])
            Xtr_lp, Ytr_lp = Xtr_lp.to(device), Ytr_lp.to(device)
            if val_manifest is not None:
                Xva_lp, Yva_lp, _ = cache_feats(val_manifest)
                Xva_lp, Yva_lp = Xva_lp.to(device), Yva_lp.to(device)
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            raw.load_state_dict(raw_init)             # each seed starts from the SAME pretrained encoder
            head = ClsHead(cdim, hidden=hidden).to(device)
            best_val, best_state, best_stage = -float("inf"), None, None

            if lp_warmup_iters > 0:
                for p in raw.parameters():
                    p.requires_grad_(False)
                raw.eval(); head.train()
                lp_opt = torch.optim.AdamW(head.parameters(), lr=lp_lr, weight_decay=wd)
                lp_gen = torch.Generator(device=device); lp_gen.manual_seed(seed)
                lp_best, lp_best_state = -float("inf"), None
                for it in range(lp_warmup_iters):
                    idx = torch.randint(0, Xtr_lp.shape[0], (bs,), device=device, generator=lp_gen)
                    loss = F.binary_cross_entropy_with_logits(head(Xtr_lp[idx]), Ytr_lp[idx])
                    lp_opt.zero_grad(set_to_none=True); loss.backward(); lp_opt.step()
                    if Xva_lp is not None and (it + 1) % lp_eval_every == 0:
                        head.eval()
                        with torch.no_grad():
                            lp_val = per_class_auc(head(Xva_lp).cpu(), Yva_lp)[1]
                        head.train()
                        if lp_val > lp_best:
                            lp_best, lp_best_state = lp_val, copy.deepcopy(head.state_dict())
                if Xva_lp is not None:
                    head.eval()
                    with torch.no_grad():
                        lp_val = per_class_auc(head(Xva_lp).cpu(), Yva_lp)[1]
                    if lp_val > lp_best:
                        lp_best, lp_best_state = lp_val, copy.deepcopy(head.state_dict())
                    if lp_best_state is not None:
                        head.load_state_dict(lp_best_state)
                    best_val = lp_best
                    best_state = (copy.deepcopy(head.state_dict()), copy.deepcopy(raw.state_dict()))
                    best_stage = "lp_warmup"
                    save_seed_checkpoint("best", seed, head, raw, best_val, best_stage)
                print(f"[seed {seed}] LP warmup best_val_AUC={lp_best:.4f} "
                      f"iters={lp_warmup_iters}", flush=True)

            opt = torch.optim.AdamW([{"params": list(head.parameters()), "lr": lr_head},
                                     {"params": list(raw.parameters()), "lr": lr_enc}], weight_decay=wd)
            for ep in range(epochs):
                enc_on = ep >= freeze_epochs          # freeze backbone for the first freeze_epochs
                for p in raw.parameters():
                    p.requires_grad_(enc_on)
                head.train(); raw.train(enc_on)
                opt.zero_grad(set_to_none=True); k = 0
                for sample in loader_for(d["train_manifest"]):
                    name = Path(sample["ct_path"]).stem
                    if name not in lab:
                        continue
                    g = to_dev(render_inputs(sample, device, grid_res, 16))
                    feat = pooled_feat(raw, g)
                    y = torch.from_numpy(lab[name])[None].to(device)
                    loss = F.binary_cross_entropy_with_logits(head(feat), y) / accum
                    loss.backward(); k += 1
                    if k % accum == 0:
                        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(raw.parameters()), grad_clip)
                        opt.step(); opt.zero_grad(set_to_none=True)
                if k % accum != 0:                    # flush the last partial batch
                    torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(raw.parameters()), grad_clip)
                    opt.step(); opt.zero_grad(set_to_none=True)
                if val_manifest is not None:          # end-of-epoch val eval
                    head.eval(); raw.eval()
                    Lv, Yv, _ = stream_logits(val_manifest)
                    vmacro = per_class_auc(Lv, Yv)[1]
                    if vmacro > best_val:
                        best_val = vmacro
                        best_state = (copy.deepcopy(head.state_dict()),
                                      copy.deepcopy(raw.state_dict()))
                        best_stage = f"ft_epoch{ep + 1}"
                        save_seed_checkpoint("best", seed, head, raw, best_val, best_stage)
                    print(f"[seed {seed} ep {ep}] enc_on={enc_on} val_macro={vmacro:.4f} best={best_val:.4f}", flush=True)
                save_seed_checkpoint("latest", seed, head, raw,
                                     None if val_manifest is None else vmacro,
                                     stage=f"ft_epoch{ep + 1}")
            if val_manifest is not None and best_state is not None:
                head.load_state_dict(best_state[0]); raw.load_state_dict(best_state[1])
            elif best_state is None:
                best_val = float("nan")
            save_seed_checkpoint("best", seed, head, raw, best_val,
                                 best_stage or "ft_final")
            head.eval(); raw.eval()
            logits_te, Yte, te_names = stream_logits(d["test_manifest"])
            n_test = len(te_names)
            pc, macro = per_class_auc(logits_te, Yte)
            macros.append(macro); pcs.append(pc); seed_logits.append(logits_te)
            if val_manifest is not None:
                print(f"[seed {seed}] best_val_AUC={best_val:.4f} test_macro_AUC={macro:.4f}", flush=True)
            else:
                print(f"[seed {seed}] macro_AUC={macro:.4f}", flush=True)

    # Per-case predictions for bootstrapping: paired on VolumeName across methods.
    # y::<class> = ground truth (0/1); p::<class> = ensemble prob (mean sigmoid over seeds).
    probs = torch.stack([torch.sigmoid(lg) for lg in seed_logits]).mean(0).numpy()
    yt = Yte.cpu().numpy()
    pred_cols = {"VolumeName": te_names}
    for c, col in enumerate(ABN_COLS):
        pred_cols[f"y::{col}"] = yt[:, c].astype(int)
        pred_cols[f"p::{col}"] = probs[:, c]
    pred_path = ckpt_dir / "preds_test.csv"
    pd.DataFrame(pred_cols).to_csv(pred_path, index=False)
    print(f"[REPR-CLS] wrote per-case preds -> {pred_path} ({len(te_names)} cases)", flush=True)

    arr = np.array(macros); pc_mean = np.nanmean(np.array(pcs), axis=0)
    per_class_str = {ABN_COLS[i]: round(float(pc_mean[i]), 4) for i in range(len(ABN_COLS))}
    print(f"[REPR-CLS RESULT] {cfg['meta']['name']} mode={mode} "
          f"macro_AUC mean={arr.mean():.4f} std={arr.std():.4f} seeds={n_seeds} "
          f"N_test={n_test} per_class={per_class_str}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
