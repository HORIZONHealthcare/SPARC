"""Config-driven training entry for CT-MAE (single-GPU or multi-GPU/node DDP).

Single GPU:
    PYTHONPATH=. python train_ctmae.py --config configs/ctmae_smoke.yaml

Multi-GPU / multi-node (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*):
    torchrun --standalone --nproc_per_node=4 train_ctmae.py --config <cfg>     # 1 node, 4 GPU
    srun torchrun --nnodes=$N --nproc_per_node=4 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER:29500 train_ctmae.py --config <cfg>            # N nodes

DDP notes:
  - LR is taken from the config AS-IS (no linear-scaling with world size). 3D-CT
    pretrain collapses under aggressive (VJEPA-style) LR; keep it small even as
    the global batch grows. Lengthen warmup / iters instead.
  - Only rank 0 logs + writes checkpoints. `iter` in the ckpt is the per-rank
    global step (consistent for the resume chain).
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
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.ct_dataset import CTDataset
from src.models.ct_mae import CTMAE
from src.utils.project_paths import expand_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return expand_env(yaml.safe_load(f))


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


def warmup_cosine_lr(it: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    if it < warmup:
        return lr_max * (it + 1) / max(warmup, 1)
    progress = (it - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(opt, lr):
    for pg in opt.param_groups:
        pg["lr"] = lr


def build_model(cfg: dict, device: torch.device) -> CTMAE:
    m = cfg["model"]
    d = cfg["data"]
    return CTMAE(
        vol_size=m["vol_size"],
        patch_size=m["patch_size"],
        in_chans=m["in_chans"],
        embed_dim=m["embed_dim"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dec_embed_dim=m["dec_embed_dim"],
        dec_depth=m["dec_depth"],
        dec_num_heads=m["dec_num_heads"],
        mask_ratio=m["mask_ratio"],
        norm_pix_loss=m["norm_pix_loss"],
        hu_min=d["hu_min"],
        hu_max=d["hu_max"],
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
    dataset = CTDataset(
        manifest=d["manifest"],
        vol_size=cfg["model"]["vol_size"],
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
    loader = DataLoader(
        dataset,
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=d["num_workers"] > 0,
    )
    if is_main:
        gb = d["batch_size"] * world_size
        print(f"dataset: {len(dataset)} CTs, local_bs={d['batch_size']} "
              f"global_bs={gb}")

    model = build_model(cfg, device)
    raw_model = model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(raw_model.parameters(),
                            lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

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
            csv_file.write("iter,loss,lr,elapsed_s\n")
            csv_file.flush()

    total = cfg["optim"]["iters"]
    warmup = int(cfg["optim"].get("warmup_iters", 0))
    lr_max = float(cfg["optim"]["lr"])
    lr_min = float(cfg["optim"].get("min_lr", lr_max))

    start_iter = 0
    latest_path = ckpt_dir / "latest.pt"
    if keep_latest and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_iter = int(state["iter"])
        if is_main:
            print(f"resumed from {latest_path}  start_iter={start_iter}")

    def save_ckpt(path: Path, it: int):
        if not is_main:
            return
        torch.save({"iter": it, "model": raw_model.state_dict(),
                    "opt": opt.state_dict(), "config": cfg}, path)

    t0 = time.perf_counter()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    loader_iter = iter(loader)
    losses = []

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

        ct = sample["ct_hu"].to(device, non_blocking=True)        # (B, 1, V, V, V)
        out = model(ct)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss))
        dt = time.perf_counter() - t0
        if csv_file is not None:
            csv_file.write(f"{it+1},{float(loss):.6f},{lr:.2e},{dt:.2f}\n")
            csv_file.flush()

        if is_main and ((it + 1) % print_every == 0 or it + 1 == total):
            print(f"iter {it+1}/{total}  loss={float(loss):.4f}  lr={lr:.2e}  elapsed={dt:.1f}s")

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
