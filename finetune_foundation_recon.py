"""Stage 3: recon finetune of a Stage-2 foundation backbone (single-GPU or DDP).

Warm-starts a SparseViewFoundation from a Stage-2 ckpt (recon path only;
use_semantic=False), then trains the recon objective (MSE) ReconModule-style.
Two protocols via cfg.mode:
  - linear_probe : freeze the backbone (encoder2d + volume3d), train only the
                   (reused) recon head -> measures the frozen representation.
  - finetune     : train everything; optional layer-wise LR decay (cfg.layer_decay
                   < 1.0) gives the backbone a smaller LR than the head.

Periodic held-out recon PSNR (dense 256^3, fixed geometry) is logged so we can
compare vs the from-scratch ReconModule (32.5 dB on LIDC). Runs on CT-RATE zarr
or LIDC .npy (DRRDataset dispatches by extension).

  torchrun --standalone --nproc_per_node=4 finetune_foundation_recon.py --config <cfg>
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.drr_dataset import DRRDataset, eval_worker_init_fn
from src.models.feature_volume_3d import FeatureVolume3D
from src.models.foundation import SparseViewFoundation
from src.models.projection import index_3d
from src.models.xray_encoder_2d import XRayEncoder2D
from src.utils.nanodrr_helpers import (
    _grid_to_world, make_lr_grid, make_patch_grid, pose_per_view, render_with_rays,
    sample_query_points, subject_from_tensor,
)
from src.utils.plucker import compute_plucker
from src.utils.project_paths import expand_env

mp.set_sharing_strategy("file_system")


def load_config(path):
    with open(path) as f:
        return expand_env(yaml.safe_load(f))


def setup_ddp():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), ws, int(os.environ.get("LOCAL_RANK", "0"))


def collate_first(batch):
    assert len(batch) == 1
    return batch[0]


def collate_list(batch):
    return batch


def cosine_factor(it, total, warmup, lr_min, lr_max):
    """Schedule as a multiplier in [lr_min/lr_max, 1] applied to each group's base LR."""
    if it < warmup:
        return (it + 1) / max(warmup, 1)
    p = (it - warmup) / max(total - warmup, 1)
    floor = lr_min / lr_max if lr_max > 0 else 0.0
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * p))


def update_early_stop_state(psnr, reference_psnr, bad_evals, min_delta):
    """Update patience against the last validation improvement of >= min_delta."""
    if not math.isfinite(reference_psnr) or psnr >= reference_psnr + min_delta:
        return psnr, 0
    return reference_psnr, bad_evals + 1


def recover_early_stop_state(log_path, min_delta):
    """Reconstruct early-stop state for checkpoints made before it was recorded."""
    reference_psnr = float("-inf")
    bad_evals = 0
    path = Path(log_path) if log_path else None
    if path is None or not path.is_file():
        return reference_psnr, bad_evals
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            psnr = float(row["psnr"])
            reference_psnr, bad_evals = update_early_stop_state(
                psnr, reference_psnr, bad_evals, min_delta
            )
    return reference_psnr, bad_evals


def write_early_stop_marker(path, *, iteration, epoch, best_psnr,
                            reference_psnr, bad_evals):
    """Atomically tell the recoverable Slurm chain not to resume this run."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        (
            f"iter={iteration}\n"
            f"epoch={epoch:.4f}\n"
            f"best_psnr={best_psnr:.6f}\n"
            f"reference_psnr={reference_psnr:.6f}\n"
            f"bad_evals={bad_evals}\n"
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_warmstart(init_ckpt, device, verbose, from_scratch=False):
    """Build SparseViewFoundation (recon path) from the Stage-2 ckpt's config.
    Loads backbone + recon-head weights, UNLESS from_scratch=True (build the
    IDENTICAL architecture but random-init — for the pretrain-vs-scratch ablation
    that isolates the foundation pretraining's contribution to recon).
    Returns (model, ckpt_cfg)."""
    state = torch.load(init_ckpt, map_location=device, weights_only=False)
    # A Stage-3 checkpoint stores its Stage-2 architecture config separately.
    # Supporting it here lets controlled continuation experiments start from a
    # completed per-dataset recon instead of silently rebuilding from Stage 2.
    cfg = state.get("ckpt_cfg", state["config"])
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
    model = SparseViewFoundation(
        encoder2d=enc, volume3d=vol, target_encoder=None,
        pointdec_mlp=cfg["fine"]["pointdec_mlp"],
        semantic_depth=cfg["semantic"]["depth"], semantic_heads=cfg["semantic"]["heads"],
        recon_loss=cfg["loss"].get("recon_loss", "mse"),
        air_weight=cfg["loss"].get("air_weight", 1.0),
        air_thresh=cfg["loss"].get("air_thresh", 0.16),
        huber_delta=cfg["loss"].get("huber_delta", 0.05),
        freeze_target=True,
        use_semantic=False,
    ).to(device)
    if from_scratch:
        if verbose:
            print(f"FROM-SCRATCH (arch from {init_ckpt}, random init) grid={grid_res}^3 "
                  f"refine={v.get('refine_type','transformer')} tf_depth={e['tf_depth']}")
        return model, cfg
    msg = model.load_state_dict(state["model"], strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith(("target_encoder", "semantic_head"))]
    assert not miss, f"missing recon-path keys on warm-start: {miss[:6]}"
    if verbose:
        print(f"warm-start {init_ckpt} iter={state['iter']} grid={grid_res}^3 "
              f"refine={v.get('refine_type','transformer')} tf_depth={e['tf_depth']}")
    return model, cfg


def resolve_ckpt(ckpt_dir, name):
    """--eval_ckpt may be a file path (a released checkpoint); a bare name is looked up in the run directory."""
    path = Path(name).expanduser()
    return path if path.is_file() else Path(ckpt_dir) / name


def load_recon_weights(model, path, device):
    """Load reconstruction weights for evaluation or export. strict=False tolerates only the modules
    the recon path never uses; any other missing or unexpected key is an error, because a silently
    skipped load would evaluate the warm-start weights instead."""
    ck = torch.load(path, map_location=device, weights_only=False)
    msg = model.load_state_dict(ck["model"], strict=False)
    unused = ("target_encoder", "semantic_head")
    missing = [k for k in msg.missing_keys if not k.startswith(unused)]
    unexpected = [k for k in msg.unexpected_keys if not k.startswith(unused)]
    if missing or unexpected:
        raise RuntimeError(f"{path} does not match the reconstruction model: "
                           f"missing {missing[:4]}, unexpected {unexpected[:4]}")
    return ck


def build_param_groups(model, base_lr, wd, mode, layer_decay):
    """linear_probe -> only point_decoder trainable. finetune -> all, with
    optional layer-wise LR decay (encoder2d < volume3d < point_decoder)."""
    if mode == "linear_probe":
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("point_decoder"))
        return [{"params": list(model.point_decoder.parameters()), "lr": base_lr, "weight_decay": wd}]

    for p in model.parameters():
        p.requires_grad_(True)
    if layer_decay >= 1.0:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr, "weight_decay": wd}]
    scale = {"encoder2d": layer_decay ** 2, "volume3d": layer_decay, "point_decoder": 1.0}
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups.setdefault(n.split(".", 1)[0], []).append(p)
    return [{"params": ps, "lr": base_lr * scale.get(k, 1.0), "weight_decay": wd}
            for k, ps in groups.items() if ps]


def sample_structured_query_patches(subject, patch_size, out_res, n_patches,
                                    hu_min, hu_max, device):
    """Sample regular local 3D query patches for differentiable structural loss.

    Half the patches are uniformly located; half are centred on a random body
    voxel (HU > -500) so the loss sees both background and anatomical structure.
    Returns flattened points in patch-major, z/y/x raster order.
    """
    if patch_size <= 0 or n_patches <= 0:
        return None, None, None
    if patch_size > out_res:
        raise ValueError(f"ssim_patch_size={patch_size} exceeds out_res={out_res}")
    hu = subject._image_hu.to(device)
    body = (hu[0, 0] > -500).nonzero(as_tuple=False)
    grids = []
    max_start = out_res - patch_size
    for pi in range(n_patches):
        if pi % 2 == 1 and body.numel() > 0:
            d, h, w = body[torch.randint(0, body.shape[0], (1,), device=device).item()]
            dd, hh, ww = hu.shape[-3:]
            center = torch.tensor([
                float(w) / max(ww - 1, 1),
                float(h) / max(hh - 1, 1),
                float(d) / max(dd - 1, 1),
            ], device=device) * (out_res - 1)
            start = (center.round().long() - patch_size // 2).clamp(0, max_start)
        else:
            start = torch.randint(0, max_start + 1, (3,), device=device)
        xs = torch.arange(start[0], start[0] + patch_size, device=device)
        ys = torch.arange(start[1], start[1] + patch_size, device=device)
        zs = torch.arange(start[2], start[2] + patch_size, device=device)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).float()
        grid = grid / max(out_res - 1, 1) * 2.0 - 1.0
        grids.append(grid.reshape(-1, 3))
    pts_grid = torch.cat(grids, dim=0)
    gt = index_3d(hu, pts_grid[None])[0, 0].clamp(hu_min, hu_max)
    gt = (gt - hu_min) / (hu_max - hu_min)
    pts_world = _grid_to_world(subject, pts_grid)
    return pts_grid[None], pts_world[None], gt[None]


def local_ssim_3d(pred, target, window=7):
    """Differentiable local 3D SSIM for tensors shaped (B,P,D,H,W)."""
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {target.shape}")
    if min(pred.shape[-3:]) < window:
        raise ValueError(f"SSIM window {window} exceeds patch shape {pred.shape[-3:]}")
    x = pred.reshape(-1, 1, *pred.shape[-3:])
    y = target.reshape_as(x)
    mu_x = F.avg_pool3d(x, window, stride=1)
    mu_y = F.avg_pool3d(y, window, stride=1)
    var_x = F.avg_pool3d(x * x, window, stride=1) - mu_x.square()
    var_y = F.avg_pool3d(y * y, window, stride=1) - mu_y.square()
    cov = F.avg_pool3d(x * y, window, stride=1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-8)
    return score.mean()


def render_sample(sample, device, n_query, grid_res, hu_min, hu_max,
                  structured_patch_size=0, structured_patches=0, out_res=256):
    subject = subject_from_tensor(sample["ct_hu"], sample["ct_affine"], device=device)
    rot = sample["rotations"].to(device)
    trans = sample["translations"].to(device)
    det_h, det_w = sample["det_h"], sample["det_w"]
    image, src, tgt = render_with_rays(
        subject=subject, rotations=rot, translations=trans,
        sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"], height=det_h, width=det_w,
    )
    plucker = compute_plucker(src, tgt).view(sample["v"], det_h, det_w, 6)
    k_inv, rt_inv = pose_per_view(subject, rot, trans, sdd=sample["sdd_mm"],
                                  delx=sample["det_pixel_mm"], height=det_h, width=det_w)
    q_grid, q_world, q_hu = sample_query_points(subject, n=n_query, hu_min=hu_min,
                                                hu_max=hu_max, device=device)
    p_grid, p_world, p_hu = sample_structured_query_patches(
        subject, structured_patch_size, out_res, structured_patches,
        hu_min, hu_max, device,
    )
    if p_grid is not None:
        q_grid = torch.cat([q_grid, p_grid], dim=1)
        q_world = torch.cat([q_world, p_world], dim=1)
        q_hu = torch.cat([q_hu, p_hu], dim=1)
    _, g_world = make_patch_grid(subject, grid_res, device)
    return dict(views=image, plucker=plucker, ct_hu=subject._image_hu.squeeze(0),
                sad_mm=sample["sad_mm"], k_inv=k_inv, rt_inv=rt_inv, det_h=det_h,
                det_w=det_w, q_grid=q_grid, q_world=q_world, q_hu=q_hu, grid_world=g_world,
                random_query_count=n_query, structured_patch_size=structured_patch_size,
                structured_patches=structured_patches)


def stack_rendered_batch(items):
    """Stack independently rendered fixed-geometry CTs for true batch training."""
    if not items:
        raise ValueError("empty rendered batch")
    for key in ("det_h", "det_w", "sad_mm"):
        reference = items[0][key]
        if any(item[key] != reference for item in items[1:]):
            raise ValueError(f"batch_size > 1 requires identical {key}")
    return dict(
        views=torch.stack([item["views"] for item in items], dim=0),
        plucker=torch.stack([item["plucker"] for item in items], dim=0),
        ct_hu=torch.stack([item["ct_hu"] for item in items], dim=0),
        sad_mm=items[0]["sad_mm"],
        k_inv=torch.stack([item["k_inv"] for item in items], dim=0),
        rt_inv=torch.stack([item["rt_inv"] for item in items], dim=0),
        det_h=items[0]["det_h"],
        det_w=items[0]["det_w"],
        q_grid=torch.cat([item["q_grid"] for item in items], dim=0),
        q_world=torch.cat([item["q_world"] for item in items], dim=0),
        q_hu=torch.cat([item["q_hu"] for item in items], dim=0),
        grid_world=torch.cat([item["grid_world"] for item in items], dim=0),
    )


def render_batch(samples, device, n_query, grid_res, hu_min, hu_max,
                 structured_patch_size=0, structured_patches=0, out_res=256):
    rendered = [
        render_sample(
            sample, device, n_query, grid_res, hu_min, hu_max,
            structured_patch_size=structured_patch_size,
            structured_patches=structured_patches, out_res=out_res,
        )
        for sample in samples
    ]
    return stack_rendered_batch(rendered)


@torch.no_grad()
def eval_psnr(raw_model, loader, device, grid_res, hu_min, hu_max, out_res, chunk, max_cts=None):
    """Mean PSNR over the loader; every volume unless max_cts is given."""
    raw_model.eval()
    psnrs = []
    for i, sample in enumerate(loader):
        if max_cts is not None and i >= max_cts:
            break
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
        pred = raw_model.infer_dense(views=image.unsqueeze(0), plucker=plucker,
                                     sad_mm=sample["sad_mm"], grid_world=g_world,
                                     k_inv=k_inv.unsqueeze(0), rt_inv=rt_inv.unsqueeze(0),
                                     det_h=det_h, det_w=det_w, q_world=q_world,
                                     q_norm=q_norm, chunk=chunk).reshape(out_res, out_res, out_res)
        gt = index_3d(subject._image_hu.to(device), q_norm)[0, 0]
        gt = ((gt.clamp(hu_min, hu_max) - hu_min) / (hu_max - hu_min)).reshape(out_res, out_res, out_res)
        psnrs.append((-10 * torch.log10(F.mse_loss(pred, gt).clamp_min(1e-10))).item())
    raw_model.train()
    return sum(psnrs) / len(psnrs) if psnrs else float("nan")


@torch.no_grad()
def eval_metrics_csv(raw_model, loader, device, grid_res, hu_min, hu_max,
                     out_res, chunk, output_csv, postprocess_floor_hu=None):
    """Stream exact formal int16-HU PSNR/3D-SSIM without storing dense exports."""
    import numpy as np
    from skimage.metrics import structural_similarity

    raw_model.eval()
    rows = []
    for i, sample in enumerate(loader):
        subject = subject_from_tensor(sample["ct_hu"], sample["ct_affine"], device=device)
        rot = sample["rotations"].to(device); trans = sample["translations"].to(device)
        det_h, det_w = sample["det_h"], sample["det_w"]
        image, src, tgt = render_with_rays(
            subject=subject, rotations=rot, translations=trans,
            sdd=sample["sdd_mm"], delx=sample["det_pixel_mm"],
            height=det_h, width=det_w,
        )
        plucker = compute_plucker(src, tgt).view(
            sample["v"], det_h, det_w, 6
        ).unsqueeze(0)
        k_inv, rt_inv = pose_per_view(
            subject, rot, trans, sdd=sample["sdd_mm"],
            delx=sample["det_pixel_mm"], height=det_h, width=det_w,
        )
        _, g_world = make_patch_grid(subject, grid_res, device)
        q_norm, q_world = make_lr_grid(subject, out_res, device)
        pred = raw_model.infer_dense(
            views=image.unsqueeze(0), plucker=plucker,
            sad_mm=sample["sad_mm"], grid_world=g_world,
            k_inv=k_inv.unsqueeze(0), rt_inv=rt_inv.unsqueeze(0),
            det_h=det_h, det_w=det_w, q_world=q_world, q_norm=q_norm,
            chunk=chunk,
        ).reshape(out_res, out_res, out_res)
        gt = index_3d(subject._image_hu.to(device), q_norm)[0, 0].reshape(
            out_res, out_res, out_res
        )
        recon_hu = (
            pred.clamp(0, 1) * (hu_max - hu_min) + hu_min
        ).cpu().numpy().astype(np.int16)
        gt_hu = gt.clamp(hu_min, hu_max).cpu().numpy().astype(np.int16)

        def metrics(reconstruction):
            rec = np.clip(
                reconstruction.astype(np.float32), hu_min, hu_max
            )
            target = np.clip(gt_hu.astype(np.float32), hu_min, hu_max)
            mse = float(np.mean((rec - target) ** 2) / ((hu_max - hu_min) ** 2))
            psnr = -10.0 * math.log10(max(mse, 1.0e-12))
            rec_norm = (rec - hu_min) / (hu_max - hu_min)
            target_norm = (target - hu_min) / (hu_max - hu_min)
            score = float(
                structural_similarity(target_norm, rec_norm, data_range=1.0)
            )
            return psnr, score

        raw_psnr, raw_ssim = metrics(recon_hu)
        row = {
            "ct_path": Path(sample["ct_path"]).stem,
            "psnr_db": f"{raw_psnr:.4f}",
            "ssim": f"{raw_ssim:.6f}",
        }
        if postprocess_floor_hu is not None:
            floored = recon_hu.copy()
            floored[floored < postprocess_floor_hu] = int(hu_min)
            floor_psnr, floor_ssim = metrics(floored)
            row["psnr_db_floor"] = f"{floor_psnr:.4f}"
            row["ssim_floor"] = f"{floor_ssim:.6f}"
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[METRICS] {i+1}/{len(loader.dataset)}", flush=True)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = ["ct_path", "psnr_db", "ssim"]
    if postprocess_floor_hu is not None:
        fieldnames += ["psnr_db_floor", "ssim_floor"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    raw_model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_only", action="store_true",
                    help="load a ckpt and evaluate on the FULL test manifest, then exit")
    ap.add_argument("--eval_manifest", default=None)
    ap.add_argument("--eval_max_cts", type=int, default=100000)
    ap.add_argument("--eval_ckpt", default="latest.pt",
                    help="checkpoint to evaluate or export: a file path, or a name in the run directory")
    ap.add_argument("--init_ckpt", default=None,
                    help="Stage-2 weights to build and warm-start from; overrides init_ckpt in the config")
    ap.add_argument("--metrics_csv", default=None,
                    help="stream formal per-case PSNR/SSIM CSV, then exit")
    ap.add_argument("--metrics_manifest", default=None)
    ap.add_argument("--postprocess_floor_hu", type=float, default=None)
    ap.add_argument("--export_dir", default=None,
                    help="if set: export dense recon volume (+ gt) per CT to this dir, then exit")
    ap.add_argument("--export_manifest", default=None)
    args = ap.parse_args()
    is_ddp, rank, world, local = setup_ddp()
    is_main = rank == 0
    cfg = load_config(args.config)
    if args.init_ckpt:
        cfg["init_ckpt"] = args.init_ckpt
    torch.manual_seed(cfg["meta"]["seed"] + rank)
    device = torch.device(f"cuda:{local}") if torch.cuda.is_available() else torch.device("cpu")
    d, o = cfg["data"], cfg["optim"]

    model, ckpt_cfg = build_warmstart(cfg["init_ckpt"], device, is_main,
                                      from_scratch=cfg.get("from_scratch", False))
    # Stage-3 loss settings belong to the finetune config, not the Stage-2
    # checkpoint config. Historical L1/Huber/air-weight variants were silently
    # ignored before this explicit override.
    loss_cfg = cfg.get("loss", {})
    if "recon_loss" in loss_cfg:
        if loss_cfg["recon_loss"] not in ("mse", "l1", "huber"):
            raise ValueError(f"unknown recon_loss={loss_cfg['recon_loss']}")
        model.recon_loss = loss_cfg["recon_loss"]
    for key in ("air_weight", "air_thresh", "huber_delta"):
        if key in loss_cfg:
            setattr(model, key, float(loss_cfg[key]))
    if is_main:
        print(f"loss recon={model.recon_loss} air_weight={model.air_weight} "
              f"air_thresh={model.air_thresh} huber_delta={model.huber_delta} "
              f"ssim_weight={loss_cfg.get('ssim_weight', 0.0)}")
    grid_res = ckpt_cfg["model"]["grid_res"]
    n_query = cfg["fine"].get("n_query", ckpt_cfg["fine"]["n_query"])
    ssim_weight = float(loss_cfg.get("ssim_weight", 0.0))
    ssim_patch_size = int(loss_cfg.get("ssim_patch_size", 0))
    ssim_patches = int(loss_cfg.get("ssim_patches", 0))
    ssim_window = int(loss_cfg.get("ssim_window", 7))
    mode = cfg["mode"]
    lr_max = float(o["lr"])
    groups = build_param_groups(model, lr_max, float(o["weight_decay"]),
                                mode, float(o.get("layer_decay", 1.0)))
    base_lrs = [g["lr"] for g in groups]
    if is_ddp:
        # PointDecoder uses BatchNorm1d; under DDP the default broadcast_buffers
        # reuses rank-0 running stats at eval (data-order dependent). Sync BN
        # stats across ranks so eval matches the full effective batch. (Milder
        # than the prior image-BN bug since BN1d here is over N=16384 points,
        # but removed for correctness + reproducibility.)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    raw = model
    if is_ddp:
        model = DDP(model, device_ids=[local], find_unused_parameters=False)
    if is_main:
        nt = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"mode={mode} layer_decay={o.get('layer_decay',1.0)} trainable={nt/1e6:.1f}M "
              f"group_lrs={[round(b, 7) for b in base_lrs]}")

    opt = torch.optim.AdamW(groups, lr=lr_max, weight_decay=float(o["weight_decay"]))

    def mk_ds(manifest, train):
        # train: configured spacing range + (optionally) random off-centre crop
        # for anti-overfit augmentation. eval: FIXED spacing + center crop so the
        # held-out PSNR stays reproducible / comparable across runs.
        if train:
            sp_lo, sp_hi = d["spacing_lo"], d["spacing_hi"]
            ctr = d.get("center_crop", True)
        else:
            sp_lo = sp_hi = d.get("eval_spacing", 1.6)
            ctr = True
        return DRRDataset(
            manifest=manifest, v_min=d["v_min"], v_max=d["v_max"], det_px=d["det_px"],
            device="cpu", seed=cfg["meta"]["seed"], deterministic=not train,
            fixed_v=d["fixed_v"], fixed_pattern=d["fixed_pattern"],
            fixed_sad_mm=d["fixed_sad_mm"], fixed_sdd_ratio=d["fixed_sdd_ratio"],
            fixed_elev_deg=d["fixed_elev_deg"], fixed_det_pixel_mm=d["fixed_det_pixel_mm"],
            npy_spacing_mm=d.get("npy_spacing_mm", 1.0),
            crop_aug=True, crop_out_size=d["crop_out_size"],
            spacing_lo=sp_lo, spacing_hi=sp_hi, center_crop=ctr,
        )

    train_ds = mk_ds(d["train_manifest"], train=True)
    batch_size = int(d.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                  drop_last=True) if is_ddp else None)
    loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=d["num_workers"],
        shuffle=(sampler is None), sampler=sampler, collate_fn=collate_list,
        persistent_workers=d["num_workers"] > 0, drop_last=True,
    )
    ckpt_dir = Path(cfg["ckpt"]["dir"])
    ev = cfg["eval"]
    if args.metrics_csv:
        metrics_manifest = args.metrics_manifest or args.eval_manifest or d["val_manifest"]
        metrics_ds = mk_ds(metrics_manifest, train=False)
        metrics_loader = DataLoader(
            metrics_ds, batch_size=1, num_workers=4,
            collate_fn=collate_first, worker_init_fn=eval_worker_init_fn,
        )
        ck = load_recon_weights(raw, resolve_ckpt(ckpt_dir, args.eval_ckpt), device)
        if is_main:
            print(
                f"[METRICS] ckpt={args.eval_ckpt} iter={ck.get('iter')} "
                f"manifest={metrics_manifest} N={len(metrics_ds)} "
                f"out_res={ev['out_res']} -> {args.metrics_csv}"
            )
        rows = eval_metrics_csv(
            raw, metrics_loader, device, grid_res, d["hu_min"], d["hu_max"],
            ev["out_res"], ev["chunk"], args.metrics_csv,
            postprocess_floor_hu=args.postprocess_floor_hu,
        )
        if is_main:
            mean_psnr = sum(float(row["psnr_db"]) for row in rows) / len(rows)
            mean_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
            print(
                f"[METRICS RESULT] {cfg['meta']['name']} "
                f"PSNR={mean_psnr:.4f} SSIM={mean_ssim:.6f} N={len(rows)}"
            )
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(0)
    if args.eval_only:
        eval_manifest = args.eval_manifest or d["val_manifest"]
        eval_ds = mk_ds(eval_manifest, train=False)
        eval_loader = DataLoader(eval_ds, batch_size=1, num_workers=4, collate_fn=collate_first,
                                 worker_init_fn=eval_worker_init_fn)
        ck = load_recon_weights(raw, resolve_ckpt(ckpt_dir, args.eval_ckpt), device); raw.eval()
        n_eval = min(args.eval_max_cts, len(eval_ds))
        if is_main:
            print(f"[EVAL_ONLY] ckpt={args.eval_ckpt} iter={ck.get('iter')} mode={mode} "
                  f"manifest={eval_manifest} N={n_eval}/{len(eval_ds)} "
                  f"out_res={ev['out_res']} det_px={d['det_px']}")
        psnr = eval_psnr(raw, eval_loader, device, grid_res, d["hu_min"], d["hu_max"],
                         ev["out_res"], ev["chunk"], args.eval_max_cts)
        if is_main:
            print(f"[EVAL_ONLY RESULT] {cfg['meta']['name']}  test_PSNR={psnr:.4f} dB  (N={n_eval})")
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(0)

    if args.export_dir:
        import numpy as np
        exp_manifest = args.export_manifest or d["val_manifest"]
        exp_ds = mk_ds(exp_manifest, train=False)
        exp_loader = DataLoader(exp_ds, batch_size=1, num_workers=4, collate_fn=collate_first,
                                worker_init_fn=eval_worker_init_fn)
        ck = load_recon_weights(raw, resolve_ckpt(ckpt_dir, args.eval_ckpt), device); raw.eval()
        out_dir = Path(args.export_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_res, chunk = ev["out_res"], ev["chunk"]
        hu_min, hu_max = d["hu_min"], d["hu_max"]
        print(f"[EXPORT] {cfg['meta']['name']} -> {out_dir} N={len(exp_ds)} out_res={out_res} (recon+gt int16 HU)")
        for i, sample in enumerate(exp_loader):
            vname = Path(sample["ct_path"]).stem
            rpath = out_dir / f"{vname}.npy"
            if rpath.exists():
                continue
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
            pred = raw.infer_dense(views=image.unsqueeze(0), plucker=plucker, sad_mm=sample["sad_mm"],
                                   grid_world=g_world, k_inv=k_inv.unsqueeze(0), rt_inv=rt_inv.unsqueeze(0),
                                   det_h=det_h, det_w=det_w, q_world=q_world, q_norm=q_norm,
                                   chunk=chunk).reshape(out_res, out_res, out_res)
            # recon is in [0,1] normalised HU -> back to HU int16
            recon_hu = (pred.clamp(0, 1) * (hu_max - hu_min) + hu_min).cpu().numpy().astype(np.int16)
            np.save(rpath, recon_hu)
            if not (out_dir / f"{vname}_gt.npy").exists():
                gt = index_3d(subject._image_hu.to(device), q_norm)[0, 0].reshape(out_res, out_res, out_res)
                gt_hu = gt.clamp(hu_min, hu_max).cpu().numpy().astype(np.int16)
                np.save(out_dir / f"{vname}_gt.npy", gt_hu)
            if (i + 1) % 25 == 0:
                print(f"[EXPORT] {i+1}/{len(exp_ds)} {vname}", flush=True)
        print(f"[EXPORT DONE] {cfg['meta']['name']} -> {out_dir}")
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(0)

    eval_ds = mk_ds(d["val_manifest"], train=False)
    eval_loader = DataLoader(eval_ds, batch_size=1, num_workers=2, collate_fn=collate_first,
                             worker_init_fn=eval_worker_init_fn)

    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = None
    if is_main and cfg.get("log", {}).get("csv"):
        Path(cfg["log"]["csv"]).parent.mkdir(parents=True, exist_ok=True)
        log_csv = open(Path(cfg["log"]["csv"]), "a")
        if log_csv.tell() == 0:
            log_csv.write("iter,epoch,loss,point_loss,ssim_loss,lr,psnr,elapsed_s\n")

    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise ValueError(
            f"empty train loader: N={len(train_ds)} world={world} batch_size={batch_size}"
        )
    if "epochs" in o:
        total = int(o["epochs"]) * steps_per_epoch
        if "iters" in o and int(o["iters"]) != total:
            raise ValueError(
                f"optim.iters={o['iters']} but epochs*steps_per_epoch={total}; "
                "keep them identical so Slurm completion guards remain correct"
            )
    else:
        total = int(o["iters"])
    warm = (
        int(o["warmup_epochs"]) * steps_per_epoch
        if "warmup_epochs" in o else int(o.get("warmup_iters", 0))
    )
    eval_every = (
        int(ev["eval_every_epochs"]) * steps_per_epoch
        if "eval_every_epochs" in ev else int(ev["eval_every"])
    )
    lr_min = float(o.get("min_lr", lr_max))
    grad_clip = float(o.get("grad_clip", 1.0))
    global_batch = batch_size * world
    if is_main:
        print(
            f"train N={len(train_ds)} local_batch={batch_size} world={world} "
            f"global_batch={global_batch} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total} epochs={total / steps_per_epoch:.1f}"
        )

    start_iter = 0
    best_psnr = float("-inf")
    early_cfg = cfg.get("early_stop", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_min_epochs = int(early_cfg.get("min_epochs", 0))
    early_patience = int(early_cfg.get("patience", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    if early_enabled:
        if early_patience < 1:
            raise ValueError("early_stop.patience must be >= 1 when enabled")
        if early_min_epochs < 0:
            raise ValueError("early_stop.min_epochs must be >= 0")
        if early_min_delta < 0:
            raise ValueError("early_stop.min_delta must be >= 0")
    early_reference_psnr = float("-inf")
    early_bad_evals = 0
    resume_path = ckpt_dir / "latest.pt"
    if cfg.get("ckpt", {}).get("resume", False) and resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        raw.load_state_dict(resume["model"])
        if "optimizer" not in resume:
            raise RuntimeError(f"cannot exactly resume {resume_path}: optimizer state is missing")
        opt.load_state_dict(resume["optimizer"])
        start_iter = int(resume["iter"])
        best_psnr = float(resume.get("best_psnr", best_psnr))
        saved_early = resume.get("early_stop")
        if early_enabled and saved_early:
            early_reference_psnr = float(
                saved_early.get("reference_psnr", early_reference_psnr)
            )
            early_bad_evals = int(saved_early.get("bad_evals", 0))
        elif early_enabled:
            early_reference_psnr, early_bad_evals = recover_early_stop_state(
                cfg.get("log", {}).get("csv"), early_min_delta
            )
        if is_main:
            print(
                f"[resume] {resume_path} iter={start_iter}/{total} "
                f"epoch={start_iter / steps_per_epoch:.1f} best_psnr={best_psnr:.4f}"
            )
            if early_enabled:
                print(
                    "[early-stop resume] "
                    f"reference={early_reference_psnr:.4f} "
                    f"bad_evals={early_bad_evals}/{early_patience}"
                )
    if start_iter >= total:
        if is_main:
            print(f"DONE already complete iter={start_iter}/{total}")
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    t0 = time.perf_counter()
    ep = start_iter // steps_per_epoch
    save_every_epochs = int(cfg.get("ckpt", {}).get("save_every_epochs", 0))
    save_every = save_every_epochs * steps_per_epoch
    early_stop_marker = ckpt_dir / "EARLY_STOPPED"

    def checkpoint_state(iteration, epoch, current_best):
        return {
            "iter": iteration,
            "epoch": epoch,
            "model": raw.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg,
            "ckpt_cfg": ckpt_cfg,
            "best_psnr": current_best,
            "world_size": world,
            "local_batch_size": batch_size,
            "global_batch_size": global_batch,
            "steps_per_epoch": steps_per_epoch,
            "early_stop": {
                "enabled": early_enabled,
                "min_epochs": early_min_epochs,
                "patience": early_patience,
                "min_delta": early_min_delta,
                "reference_psnr": early_reference_psnr,
                "bad_evals": early_bad_evals,
            },
        }

    def save_checkpoint(state, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    if sampler is not None:
        sampler.set_epoch(ep)
    it_loader = iter(loader)

    for it in range(start_iter, total):
        stop_training = False
        try:
            samples = next(it_loader)
        except StopIteration:
            ep += 1
            if sampler is not None:
                sampler.set_epoch(ep)
            it_loader = iter(loader)
            samples = next(it_loader)
        fac = cosine_factor(it, total, warm, lr_min, lr_max)
        for g, blr in zip(opt.param_groups, base_lrs):
            g["lr"] = blr * fac
        r = render_batch(
            samples, device, n_query, grid_res, d["hu_min"], d["hu_max"],
            structured_patch_size=ssim_patch_size,
            structured_patches=ssim_patches,
            out_res=ev["out_res"],
        )
        out = model(views=r["views"], plucker=r["plucker"],
                    ct=r["ct_hu"], sad_mm=r["sad_mm"],
                    k_inv=r["k_inv"], rt_inv=r["rt_inv"],
                    det_h=r["det_h"], det_w=r["det_w"], grid_world=r["grid_world"],
                    q_pts_world=r["q_world"], q_pts_norm=r["q_grid"], q_gt_hu=r["q_hu"])
        point_loss = out["loss_fine"]
        ssim_loss = point_loss.new_zeros(())
        if ssim_patch_size > 0 and ssim_patches > 0:
            n_struct = ssim_patches * (ssim_patch_size ** 3)
            pred_patch = out["density"][:, -n_struct:].reshape(
                len(samples), ssim_patches, ssim_patch_size, ssim_patch_size, ssim_patch_size,
            )
            gt_patch = r["q_hu"][:, -n_struct:].reshape_as(pred_patch)
            ssim_loss = 1.0 - local_ssim_3d(pred_patch, gt_patch, window=ssim_window)
        loss = point_loss + ssim_weight * ssim_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], grad_clip)
        opt.step()
        dt = time.perf_counter() - t0
        if is_main and (it + 1) % cfg["log"]["print_every"] == 0:
            print(f"iter {it+1}/{total} loss={float(loss):.5f} "
                  f"point={float(point_loss):.5f} ssim={float(ssim_loss):.5f} "
                  f"lr={base_lrs[-1]*fac:.2e} epoch={(it+1)/steps_per_epoch:.1f} "
                  f"elapsed={dt:.0f}s")
        do_eval = (it + 1) % eval_every == 0 or it + 1 == total
        if (
            is_main and save_every > 0 and (it + 1) % save_every == 0
            and not do_eval
        ):
            epoch = (it + 1) / steps_per_epoch
            state = checkpoint_state(it + 1, epoch, best_psnr)
            save_checkpoint(state, ckpt_dir / "latest.pt")
            print(f"[checkpoint] epoch={epoch:.1f} iter={it+1}/{total}")
        if do_eval:
            psnr = eval_psnr(raw, eval_loader, device, grid_res, d["hu_min"], d["hu_max"],
                             ev["out_res"], ev["chunk"], ev.get("max_cts"))   # whole validation split
            if is_main:
                epoch = (it + 1) / steps_per_epoch
                print(f"[eval iter {it+1} epoch {epoch:.1f}] PSNR={psnr:.2f} dB "
                      f"(N={len(eval_loader.dataset)} validation volumes)")
                if log_csv:
                    log_csv.write(f"{it+1},{epoch:.4f},{float(loss):.5f},{float(point_loss):.5f},"
                                  f"{float(ssim_loss):.5f},{base_lrs[-1]*fac:.2e},"
                                  f"{psnr:.4f},{dt:.0f}\n")
                    log_csv.flush()
                is_best = psnr > best_psnr
                if is_best:
                    best_psnr = psnr
                if early_enabled:
                    early_reference_psnr, early_bad_evals = update_early_stop_state(
                        psnr,
                        early_reference_psnr,
                        early_bad_evals,
                        early_min_delta,
                    )
                    print(
                        "[early-stop] "
                        f"epoch={epoch:.1f} reference={early_reference_psnr:.4f} "
                        f"bad_evals={early_bad_evals}/{early_patience} "
                        f"min_epoch={early_min_epochs}"
                    )
                state = checkpoint_state(it + 1, epoch, best_psnr)
                save_checkpoint(state, ckpt_dir / "latest.pt")
                if is_best:
                    save_checkpoint(state, ckpt_dir / "best.pt")
                    print(f"[best] epoch={epoch:.1f} PSNR={best_psnr:.4f} dB")
                should_stop = (
                    early_enabled
                    and epoch >= early_min_epochs
                    and early_bad_evals >= early_patience
                )
                if should_stop:
                    write_early_stop_marker(
                        early_stop_marker,
                        iteration=it + 1,
                        epoch=epoch,
                        best_psnr=best_psnr,
                        reference_psnr=early_reference_psnr,
                        bad_evals=early_bad_evals,
                    )
                    print(
                        "[early-stop] STOP "
                        f"epoch={epoch:.1f} best_psnr={best_psnr:.4f} "
                        f"after {early_bad_evals} non-improving evaluations"
                    )
                    stop_training = True
            if is_ddp:
                stop_tensor = torch.tensor(
                    [int(stop_training)], device=device, dtype=torch.int32
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            if stop_training:
                break

    if is_ddp:
        dist.barrier(); dist.destroy_process_group()
    if is_main:
        print("DONE")
    sys.exit(0)


if __name__ == "__main__":
    main()
