"""Stage 2: unified sparse-view X-ray foundation model (single-GPU or DDP).

One backbone (XRayEncoder2D -> FeatureVolume3D -> F); two heads on F:
recon (voxel) + semantic (predict frozen CT-MAE 16^3 tokens). See
src/models/foundation.py. DDP / launch identical to train_ctmae.py.

  torchrun --standalone --nproc_per_node=4 train_foundation.py --config <cfg>
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
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# DataLoader workers pass 256^3 cubes (64MB) between processes. The default
# 'file_descriptor' strategy backs these with /dev/shm segments; at 16 ranks x
# 8 workers that exhausts the job's /dev/shm allotment and NCCL (which also needs
# /dev/shm) fails with "No space left on device". 'file_system' backs them with
# TMPDIR files (node-local /local, 512G) instead, leaving /dev/shm for NCCL.
mp.set_sharing_strategy("file_system")

from src.datasets.drr_dataset import DRRDataset
from src.models.ct_mae import CTMAE
from src.models.feature_volume_3d import FeatureVolume3D
from src.models.foundation import SparseViewFoundation
from src.models.xray_encoder_2d import XRayEncoder2D
from src.utils.nanodrr_helpers import (
    make_patch_grid, pose_per_view, render_with_rays, sample_query_points,
    subject_from_tensor,
)
from src.utils.plucker import compute_plucker
from src.utils.project_paths import expand_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return expand_env(yaml.safe_load(f))


def setup_ddp() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def collate_first(batch):
    assert len(batch) == 1, "batch_size > 1 not supported (variable V)"
    return batch[0]


def warmup_cosine_lr(it, total, warmup, lr_max, lr_min):
    if it < warmup:
        return lr_max * (it + 1) / max(warmup, 1)
    progress = (it - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(opt, lr):
    for pg in opt.param_groups:
        pg["lr"] = lr


def build_target_ctmae(cfg, device, verbose):
    path = cfg["target"]["ctmae_init_path"]
    state = torch.load(path, map_location=device, weights_only=False)
    sc = state["config"]["model"]
    mae = CTMAE(
        vol_size=sc["vol_size"], patch_size=sc["patch_size"], in_chans=sc["in_chans"],
        embed_dim=sc["embed_dim"], depth=sc["depth"], num_heads=sc["num_heads"],
        mlp_ratio=sc["mlp_ratio"], dec_embed_dim=sc["dec_embed_dim"],
        dec_depth=sc["dec_depth"], dec_num_heads=sc["dec_num_heads"],
        mask_ratio=sc["mask_ratio"], norm_pix_loss=sc["norm_pix_loss"],
    ).to(device)
    mae.load_state_dict(state["model"])
    mae.eval()
    if verbose:
        print(f"loaded frozen CT-MAE {path} (iter={state['iter']}, "
              f"vol={sc['vol_size']} patch={sc['patch_size']} grid={sc['vol_size']//sc['patch_size']}^3)")
    return mae, sc


def build_model(cfg, device, verbose=True):
    e, v, d = cfg["encoder2d"], cfg["volume3d"], cfg["data"]
    grid_res = cfg["model"]["grid_res"]
    enc = XRayEncoder2D(
        img_size=d["det_px"], in_ch=1, base_ch=e["base_ch"], n_stages=e["n_stages"],
        ch_mult=e["ch_mult"], blocks_per_stage=e["blocks_per_stage"],
        tf_dim=e["tf_dim"], tf_depth=e["tf_depth"], tf_heads=e["tf_heads"],
    ).to(device)
    vol = FeatureVolume3D(
        enc.channels, grid_res=grid_res, dim=v["dim"], depth=v["depth"],
        num_heads=v["num_heads"], refine_type=v.get("refine_type", "transformer"),
    ).to(device)
    use_semantic = cfg["loss"].get("use_semantic", True)
    tgt = None
    if use_semantic:
        tgt, sc = build_target_ctmae(cfg, device, verbose)
        mae_grid = sc["vol_size"] // sc["patch_size"]
        assert mae_grid == grid_res, f"MAE grid {mae_grid}^3 != model.grid_res {grid_res}"
        assert d["crop_out_size"] == sc["vol_size"], (
            f"crop_out_size {d['crop_out_size']} must equal MAE vol_size {sc['vol_size']}"
        )
    return SparseViewFoundation(
        encoder2d=enc, volume3d=vol, target_encoder=tgt,
        pointdec_mlp=cfg["fine"]["pointdec_mlp"],
        semantic_depth=cfg["semantic"]["depth"], semantic_heads=cfg["semantic"]["heads"],
        recon_loss=cfg["loss"].get("recon_loss", "mse"),
        coarse_weight=cfg["loss"].get("coarse_weight", 1.0),
        fine_weight=cfg["loss"].get("fine_weight", 1.0),
        freeze_target=True, use_semantic=use_semantic,
    ).to(device)


def render_sample_on_gpu(sample, device, n_query, grid_res, hu_min, hu_max):
    subject = subject_from_tensor(sample["ct_hu"], sample["ct_affine"], device=device)
    rotations = sample["rotations"].to(device)
    translations = sample["translations"].to(device)
    det_h, det_w = sample["det_h"], sample["det_w"]
    image, source, target = render_with_rays(
        subject=subject, rotations=rotations, translations=translations,
        sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"], height=det_h, width=det_w,
    )
    plucker = compute_plucker(source, target).view(sample["v"], det_h, det_w, 6)
    k_inv, rt_inv = pose_per_view(
        subject, rotations, translations, sdd=sample["sdd_mm"],
        delx=sample["det_pixel_mm"], height=det_h, width=det_w,
    )
    q_grid, q_world, q_hu = sample_query_points(
        subject, n=n_query, hu_min=hu_min, hu_max=hu_max, device=device,
    )
    _, g_world = make_patch_grid(subject, grid_res, device)
    return {
        "views": image, "plucker": plucker,
        "ct_hu": subject._image_hu.squeeze(0),       # (1, D, H, W) = 256^3
        "sad_mm": sample["sad_mm"], "k_inv": k_inv, "rt_inv": rt_inv,
        "det_h": det_h, "det_w": det_w,
        "q_grid": q_grid, "q_world": q_world, "q_hu": q_hu, "grid_world": g_world,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    is_ddp, rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0
    cfg = load_config(args.config)
    torch.manual_seed(cfg["meta"]["seed"] + rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if is_main:
        print(f"config={args.config} ddp={is_ddp} world={world_size} device={device}")

    d = cfg["data"]
    dataset = DRRDataset(
        manifest=d["manifest"], v_min=d["v_min"], v_max=d["v_max"], det_px=d["det_px"],
        device="cpu", seed=cfg["meta"]["seed"],
        crop_aug=True, crop_out_size=d["crop_out_size"],
        spacing_lo=d.get("spacing_lo", 1.0), spacing_hi=d.get("spacing_hi", 2.0),
        center_crop=d.get("center_crop", True),
    )
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True) if is_ddp else None)
    loader = DataLoader(dataset, batch_size=d["batch_size"], num_workers=d["num_workers"],
                        shuffle=(sampler is None), sampler=sampler, collate_fn=collate_first,
                        persistent_workers=d["num_workers"] > 0)
    if is_main:
        print(f"dataset: {len(dataset)} CTs, global_bs={d['batch_size']*world_size}")

    model = build_model(cfg, device, verbose=is_main)
    raw = model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        n = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"trainable params: {n/1e6:.2f}M")

    opt = torch.optim.AdamW(raw.parameters(), lr=cfg["optim"]["lr"],
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
            csv_file.write("iter,loss,loss_coarse,loss_fine,lr,v,elapsed_s\n")
            csv_file.flush()

    total = cfg["optim"]["iters"]
    warmup = int(cfg["optim"].get("warmup_iters", 0))
    lr_max = float(cfg["optim"]["lr"])
    lr_min = float(cfg["optim"].get("min_lr", lr_max))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))

    start_iter = 0
    latest = ckpt_dir / "latest.pt"
    if keep_latest and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_iter = int(state["iter"])
        if is_main:
            print(f"resumed {latest} start_iter={start_iter}")

    def save_ckpt(path, it):
        if is_main:
            torch.save({"iter": it, "model": raw.state_dict(),
                        "opt": opt.state_dict(), "config": cfg}, path)

    losses = []
    t0 = time.perf_counter()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    it_loader = iter(loader)

    for it in range(start_iter, total):
        try:
            sample = next(it_loader)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it_loader = iter(loader)
            sample = next(it_loader)

        lr = warmup_cosine_lr(it, total, warmup, lr_max, lr_min)
        set_lr(opt, lr)

        r = render_sample_on_gpu(sample, device, n_query=cfg["fine"]["n_query"],
                                 grid_res=cfg["model"]["grid_res"],
                                 hu_min=d["hu_min"], hu_max=d["hu_max"])
        out = model(
            views=r["views"].unsqueeze(0), plucker=r["plucker"].unsqueeze(0),
            ct=r["ct_hu"].unsqueeze(0), sad_mm=r["sad_mm"],
            k_inv=r["k_inv"].unsqueeze(0), rt_inv=r["rt_inv"].unsqueeze(0),
            det_h=r["det_h"], det_w=r["det_w"], grid_world=r["grid_world"],
            q_pts_world=r["q_world"], q_pts_norm=r["q_grid"], q_gt_hu=r["q_hu"],
        )
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw.parameters(), grad_clip)
        opt.step()

        losses.append(float(loss))
        dt = time.perf_counter() - t0
        if csv_file is not None:
            csv_file.write(f"{it+1},{float(loss):.6f},{float(out['loss_coarse']):.6f},"
                           f"{float(out['loss_fine']):.6f},{lr:.2e},{sample['v']},{dt:.2f}\n")
            csv_file.flush()
        if is_main and ((it + 1) % print_every == 0 or it + 1 == total):
            print(f"iter {it+1}/{total} loss={float(loss):.4f} "
                  f"coarse={float(out['loss_coarse']):.4f} fine={float(out['loss_fine']):.4f} "
                  f"lr={lr:.2e} V={sample['v']} elapsed={dt:.1f}s")
        if keep_latest and ((it + 1) % print_every == 0 or it + 1 == total):
            save_ckpt(latest, it + 1)
        if save_every > 0 and (it + 1) % save_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter{it+1:06d}.pt", it + 1)
            if is_main:
                print(f"saved ckpt_iter{it+1:06d}.pt")

    if csv_file is not None:
        csv_file.close()
    if is_main:
        print(f"final_loss={losses[-1]:.4f} iters_run={len(losses)}")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("DONE")
    sys.exit(0)


if __name__ == "__main__":
    main()
