"""Config-driven training entry for SparseViewJEPA (single-GPU or DDP).

Single GPU:
    PYTHONPATH=. python train.py --config configs/smoke.yaml

Multi-GPU / multi-node (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*):
    torchrun --standalone --nproc_per_node=4 train.py --config <cfg>          # 1 node, 4 GPU
    srun torchrun --nnodes=$N --nproc_per_node=4 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER:29500 train.py --config <cfg>                 # N nodes

DDP notes:
  - Stage 2's coarse branch is V-JEPA-style. 3D-CT JEPA collapses under
    VJEPA-style high LR, so the config LR is used AS-IS (NO linear-scaling with
    world size). Per-rank batch is 1 (variable V); global batch = world_size.
  - Only rank 0 logs + checkpoints. `iter` is the per-rank global step.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.drr_dataset import DRRDataset
from src.models.ct_mae import CTMAE
from src.models.encoders import CTTargetEncoder, SparseViewEncoder
from src.models.cnn_pyramid import CNNPyramid2D
from src.models.jepa import Conv3DRefine, SparseViewJEPA
from src.models.point_decoder import PointDecoder
from src.models.predictor import CrossAttnPredictor
from src.utils.nanodrr_helpers import (
    make_lr_grid, pose_per_view, render_with_rays, sample_query_points,
    subject_from_tensor,
)
from src.utils.plucker import compute_plucker


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_ddp() -> tuple[bool, int, int, int]:
    """Init process group if launched under torchrun. Returns
    (is_ddp, rank, world_size, local_rank)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def collate_first(batch):
    assert len(batch) == 1, "batch_size > 1 not yet supported (variable V)"
    return batch[0]


def resize_ct(ct: torch.Tensor, target_size: int) -> torch.Tensor:
    if ct.dim() == 4:
        ct = ct.unsqueeze(0)
    elif ct.dim() != 5:
        raise ValueError(f"unexpected CT shape {ct.shape}")
    ct = F.interpolate(ct, size=(target_size,) * 3, mode="trilinear", align_corners=False)
    return ct.squeeze(0)


def normalize_ct(ct: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    ct = ct.clamp(hu_min, hu_max)
    return (ct - hu_min) / (hu_max - hu_min)


def render_sample_on_gpu(
    sample: dict, device: torch.device,
    n_query: int, lr_grid_res: int, hu_min: float, hu_max: float,
) -> dict:
    """Build Subject, render DRR + Plücker, sample fine-branch query points.

    Worker returned only CT tensor / affine / geometry params; rendering +
    projection happen here on GPU. Adds per-view k_inv/rt_inv and the
    random query points (+ low-res grid) for the fine recon branch.
    """
    subject = subject_from_tensor(sample["ct_hu"], sample["ct_affine"], device=device)
    rotations = sample["rotations"].to(device)
    translations = sample["translations"].to(device)
    det_h, det_w = sample["det_h"], sample["det_w"]

    image, source, target = render_with_rays(
        subject=subject, rotations=rotations, translations=translations,
        sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"],
        height=det_h, width=det_w,
    )
    plucker = compute_plucker(source, target).view(
        sample["v"], det_h, det_w, 6,
    )
    k_inv, rt_inv = pose_per_view(
        subject, rotations, translations,
        sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"],
        height=det_h, width=det_w,
    )                                                # (V,3,3), (V,4,4)
    q_grid, q_world, q_hu = sample_query_points(
        subject, n=n_query, hu_min=hu_min, hu_max=hu_max, device=device,
    )                                                # (1,N,3),(1,N,3),(1,N)
    lr_grid, lr_world = make_lr_grid(subject, lr_grid_res, device)

    return {
        "views": image,                              # (V,1,H,W)
        "plucker": plucker,                          # (V,H,W,6)
        "ct_hu": subject._image_hu.squeeze(0),       # (1,D,H,W) HU
        "sad_mm": sample["sad_mm"],
        "k_inv": k_inv, "rt_inv": rt_inv,
        "det_h": det_h, "det_w": det_w,
        "q_grid": q_grid, "q_world": q_world, "q_hu": q_hu,
        "lr_world": lr_world,
    }


def cosine_tau(it: int, total: int, tau_start: float, tau_end: float) -> float:
    if total <= 1:
        return tau_end
    progress = it / (total - 1)
    return tau_end - 0.5 * (tau_end - tau_start) * (1 + math.cos(math.pi * progress))


def warmup_cosine_lr(it: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup, then cosine decay to `lr_min`."""
    if it < warmup:
        return lr_max * (it + 1) / max(warmup, 1)
    progress = (it - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def build_target_encoder(cfg: dict, device: torch.device, verbose: bool = True) -> nn.Module:
    """Returns a frozen target encoder.

    If `target.ctmae_init_path` is set, loads a CT-MAE ckpt and returns the
    full CTMAE (its `encoder_forward` method is called by jepa.py).
    Otherwise returns a random-init CTTargetEncoder.
    """
    m = cfg["model"]
    target_cfg = cfg.get("target", {})
    ctmae_path = target_cfg.get("ctmae_init_path")
    if ctmae_path:
        state = torch.load(ctmae_path, map_location=device, weights_only=False)
        saved_cfg = state["config"]["model"]
        ctmae = CTMAE(
            vol_size=saved_cfg["vol_size"],
            patch_size=saved_cfg["patch_size"],
            in_chans=saved_cfg["in_chans"],
            embed_dim=saved_cfg["embed_dim"],
            depth=saved_cfg["depth"],
            num_heads=saved_cfg["num_heads"],
            mlp_ratio=saved_cfg["mlp_ratio"],
            dec_embed_dim=saved_cfg["dec_embed_dim"],
            dec_depth=saved_cfg["dec_depth"],
            dec_num_heads=saved_cfg["dec_num_heads"],
            mask_ratio=saved_cfg["mask_ratio"],
            norm_pix_loss=saved_cfg["norm_pix_loss"],
        ).to(device)
        ctmae.load_state_dict(state["model"])
        ctmae.eval()
        if verbose:
            print(f"loaded CT-MAE target encoder from {ctmae_path}  "
                  f"(iter={state['iter']}, embed_dim={saved_cfg['embed_dim']})")
        return ctmae
    return CTTargetEncoder(
        vol_size=m["vol_size"],
        patch_size=m["patch_size"],
        in_chans=m["in_chans"],
        embed_dim=m["embed_dim"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
    ).to(device)


def build_model(cfg: dict, device: torch.device, verbose: bool = True) -> SparseViewJEPA:
    m = cfg["model"]
    p = cfg["predictor"]
    loss_cfg = cfg["loss"]
    fine_cfg = cfg["fine"]

    # ---- coarse (semantic) ----
    ctx = SparseViewEncoder(
        img_size=m["img_size"], patch_size=m["patch_size"], in_chans=m["in_chans"],
        embed_dim=m["embed_dim"], depth=m["depth"], num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
    ).to(device)
    tgt = build_target_encoder(cfg, device, verbose=verbose)
    target_num_tokens = tgt.num_patches if isinstance(tgt, CTMAE) else tgt.num_tokens
    pred = CrossAttnPredictor(
        num_queries=target_num_tokens, ctx_dim=m["embed_dim"],
        target_dim=m["embed_dim"], pred_dim=p["pred_dim"], depth=p["depth"],
        num_heads=p["num_heads"], mlp_ratio=p["mlp_ratio"],
    ).to(device)

    # ---- fine (recon, DeepSparse-style) ----
    pyramid = CNNPyramid2D(
        in_ch=m["in_chans"], base_ch=fine_cfg["pyramid_base_ch"],
        n_stages=fine_cfg["pyramid_stages"], ch_mult=fine_cfg.get("pyramid_ch_mult", 2),
        blocks_per_stage=fine_cfg.get("pyramid_blocks", 2),
    ).to(device)
    sum_pyr_ch = sum(pyramid.channels)
    conv3d = Conv3DRefine(
        in_ch=sum_pyr_ch, ch=fine_cfg["conv3d_ch"],
        n_blocks=fine_cfg.get("conv3d_blocks", 3),
    ).to(device)
    pt_in = fine_cfg["conv3d_ch"] + sum_pyr_ch
    pdec = PointDecoder(
        channels=[pt_in] + fine_cfg["pointdec_mlp"] + [1],
        residual=True, use_bn=True,
    ).to(device)

    return SparseViewJEPA(
        context_encoder=ctx, target_encoder=tgt, predictor=pred,
        cnn_pyramid=pyramid, point_decoder=pdec, conv3d_refine=conv3d,
        lr_grid_res=fine_cfg["lr_grid_res"],
        feature_loss_type=loss_cfg.get("feature_type", "smooth_l1"),
        fine_loss_type=loss_cfg.get("fine_type", "l1"),
        coarse_weight=loss_cfg.get("coarse_weight", 1.0),
        fine_weight=loss_cfg.get("fine_weight", 1.0),
        freeze_target=True,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to yaml config")
    args = parser.parse_args()

    is_ddp, rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0

    cfg = load_config(args.config)
    torch.manual_seed(cfg["meta"]["seed"] + rank)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if is_main:
        print(f"loaded config: {args.config}")
        print(f"ddp={is_ddp} world_size={world_size} device={device} "
              f"torch={torch.__version__} cuda={torch.cuda.is_available()}")

    d = cfg["data"]
    dataset = DRRDataset(
        manifest=d["manifest"],
        v_min=d["v_min"],
        v_max=d["v_max"],
        det_px=d["det_px"],
        device="cpu",
        seed=cfg["meta"]["seed"],
        crop_aug=d.get("crop_aug", False),
        crop_out_size=d.get("crop_out_size", 256),
        spacing_lo=d.get("spacing_lo", 1.0),
        spacing_hi=d.get("spacing_hi", 2.0),
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                           shuffle=True, drop_last=True)
        if is_ddp else None
    )
    loader = DataLoader(dataset, batch_size=d["batch_size"],
                        num_workers=d["num_workers"],
                        shuffle=(sampler is None), sampler=sampler,
                        collate_fn=collate_first,
                        persistent_workers=d["num_workers"] > 0)
    if is_main:
        print(f"dataset: {len(dataset)} CTs, local_bs={d['batch_size']} "
              f"global_bs={d['batch_size'] * world_size}")

    model = build_model(cfg, device, verbose=is_main)
    raw_model = model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"model params: {n_params/1e6:.2f}M  (dual-branch: coarse ViT + fine CNN-pyramid)")

    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])

    ckpt_dir = Path(cfg["ckpt"]["dir"])
    save_every = cfg["ckpt"]["save_every_iters"]
    keep_latest = cfg["ckpt"].get("keep_latest", False)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_cfg = cfg.get("log", {})
    print_every = int(log_cfg.get("print_every", 1))
    csv_path = log_cfg.get("csv")
    csv_file = None
    if csv_path and is_main:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "a")
        if csv_path.stat().st_size == 0:
            csv_file.write("iter,loss,loss_coarse,loss_fine,lr,v,pattern,elapsed_s\n")
            csv_file.flush()

    total = cfg["optim"]["iters"]
    warmup = int(cfg["optim"].get("warmup_iters", 0))
    lr_max = float(cfg["optim"]["lr"])
    lr_min = float(cfg["optim"].get("min_lr", lr_max))

    # Resume: latest.pt (chain) else init_ckpt warm-start (Stage 2.5 seed).
    start_iter = 0
    latest_path = ckpt_dir / "latest.pt"
    init_ckpt_path = cfg["ckpt"].get("init_ckpt")
    if keep_latest and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_iter = int(state["iter"])
        if is_main:
            print(f"resumed from {latest_path}  start_iter={start_iter}")
    elif init_ckpt_path:
        state = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        msg = raw_model.load_state_dict(state["model"], strict=False)
        if is_main:
            print(f"warm-start from {init_ckpt_path}  iter=0  (opt reset)")
            if msg.missing_keys:
                print(f"  missing keys (first 5): {list(msg.missing_keys)[:5]}  total={len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  unexpected keys (first 5): {list(msg.unexpected_keys)[:5]}  total={len(msg.unexpected_keys)}")

    def save_ckpt(path: Path, it: int):
        if not is_main:
            return
        torch.save({"iter": it, "model": raw_model.state_dict(),
                    "opt": opt.state_dict(), "config": cfg}, path)

    losses = []
    t0 = time.perf_counter()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    loader_iter = iter(loader)

    for it in range(start_iter, total):
        try:
            sample = next(loader_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            sample = next(loader_iter)

        lr = warmup_cosine_lr(it, total, warmup, lr_max, lr_min)
        set_lr(opt, lr)

        r = render_sample_on_gpu(
            sample, device,
            n_query=cfg["fine"]["n_query"], lr_grid_res=cfg["fine"]["lr_grid_res"],
            hu_min=d["hu_min"], hu_max=d["hu_max"],
        )
        views   = r["views"].unsqueeze(0)            # (1,V,1,H,W)
        plucker = r["plucker"].unsqueeze(0)          # (1,V,H,W,6)
        ct      = resize_ct(r["ct_hu"], cfg["model"]["vol_size"]).unsqueeze(0)

        out = model(
            views=views, plucker=plucker, ct=ct, sad_mm=r["sad_mm"],
            k_inv=r["k_inv"].unsqueeze(0), rt_inv=r["rt_inv"].unsqueeze(0),
            det_h=r["det_h"], det_w=r["det_w"],
            q_pts_world=r["q_world"], q_pts_norm=r["q_grid"], q_gt_hu=r["q_hu"],
            lr_grid_world=r["lr_world"],
        )
        loss = out["loss"]
        loss_coarse = float(out["loss_coarse"])
        loss_fine = float(out["loss_fine"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss))
        dt = time.perf_counter() - t0

        if csv_file is not None:
            csv_file.write(
                f"{it+1},{float(loss):.6f},{loss_coarse:.6f},{loss_fine:.6f},"
                f"{lr:.2e},{sample['v']},{sample['pattern']},{dt:.2f}\n"
            )
            csv_file.flush()

        if is_main and ((it + 1) % print_every == 0 or it + 1 == total):
            print(
                f"iter {it+1}/{total}  loss={float(loss):.4f}  "
                f"coarse={loss_coarse:.4f}  fine={loss_fine:.4f}  lr={lr:.2e}  "
                f"V={sample['v']}  pattern={sample['pattern']}  elapsed={dt:.1f}s"
            )

        if keep_latest and ((it + 1) % print_every == 0 or it + 1 == total):
            save_ckpt(latest_path, it + 1)
        if save_every > 0 and (it + 1) % save_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter{it+1:06d}.pt", it + 1)
            if is_main:
                print(f"saved ckpt_iter{it+1:06d}.pt")

    if csv_file is not None:
        csv_file.close()

    if is_main:
        print(f"\nfinal_loss={losses[-1]:.4f}  iters_run={len(losses)}")
        if not all(torch.isfinite(torch.tensor(l)) for l in losses):
            print("FAIL: non-finite loss")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("DONE")
    sys.exit(0)


if __name__ == "__main__":
    main()
