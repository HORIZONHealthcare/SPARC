"""Shared convergence machinery for the reconstruction trainers.

Ported verbatim (semantics-wise) from ``finetune_foundation_recon.py`` so that
the baselines can run the SAME protocol as SPARC: an epoch-denominated budget,
validation-selected ``best.pt``, patience-based early stopping, and exact
resume (model + optimizer) across Slurm walltime segments.

Every helper is opt-in. A config without ``optim.epochs`` and without
``early_stop`` keeps its original iteration-driven behaviour, so existing
baseline configs reproduce bit-for-bit apart from the extra checkpoint keys.

Why exact resume matters: a 400-epoch baseline run is tens of GPU-hours and the
standard partition caps a segment at 8h, so a run is chained across many
segments. Restoring the model but not the optimizer silently restarts Adam's
moments at every boundary, which is a warm restart, not a resumption.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import torch

__all__ = [
    "resolve_budget",
    "update_early_stop_state",
    "recover_early_stop_state",
    "write_early_stop_marker",
    "atomic_save",
    "load_resume",
    "EarlyStopConfig",
    "case_metrics",
    "save_interval",
    "write_metrics_csv",
]


# --------------------------------------------------------------------------- test metrics

def case_metrics(recon_hu, gt_hu, hu_min, hu_max):
    """Per-case PSNR/3D-SSIM on int16 HU, identical in definition to SPARC's
    ``eval_metrics_csv`` so baseline and SPARC numbers are directly comparable:
    both clip to [hu_min, hu_max], normalise to [0,1] and use data_range=1.0.
    """
    import numpy as np
    from skimage.metrics import structural_similarity

    rec = np.clip(recon_hu.astype(np.float32), hu_min, hu_max)
    tgt = np.clip(gt_hu.astype(np.float32), hu_min, hu_max)
    mse = float(np.mean((rec - tgt) ** 2) / ((hu_max - hu_min) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1.0e-12))
    rec_n = (rec - hu_min) / (hu_max - hu_min)
    tgt_n = (tgt - hu_min) / (hu_max - hu_min)
    ssim = float(structural_similarity(tgt_n, rec_n, data_range=1.0))
    return psnr, ssim


def write_metrics_csv(rows, path, run_name=""):
    """Write per-case rows and print the aggregate line the chain greps for."""
    import statistics

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["ct_path", "psnr_db", "ssim"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if rows:
        p = statistics.mean(float(r["psnr_db"]) for r in rows)
        s = statistics.mean(float(r["ssim"]) for r in rows)
        print(f"[METRICS RESULT] {run_name} PSNR={p:.4f} SSIM={s:.6f} N={len(rows)}",
              flush=True)
    return path


# --------------------------------------------------------------------------- budget

def resolve_budget(cfg, n_train, batch_size=1):
    """Return (total_iters, steps_per_epoch, epochs).

    ``optim.epochs`` wins when present; otherwise fall back to ``optim.iters``
    and merely report the implied epoch count.
    """
    o = cfg["optim"]
    steps_per_epoch = max(1, n_train // max(1, batch_size))
    epochs = o.get("epochs")
    if epochs:
        total = int(epochs) * steps_per_epoch
    else:
        total = int(o["iters"])
        epochs = total / steps_per_epoch
    return total, steps_per_epoch, epochs


def warmup_iters(cfg, steps_per_epoch):
    """Warmup in iterations, from ``optim.warmup_epochs`` or ``optim.warmup_iters``."""
    o = cfg["optim"]
    if o.get("warmup_epochs"):
        return int(o["warmup_epochs"]) * steps_per_epoch
    return int(o.get("warmup_iters", 0))


def save_interval(cfg, steps_per_epoch, default_epochs=5):
    """Checkpoint cadence in iterations, INDEPENDENT of the validation cadence.

    Tying latest.pt to the eval block is a trap on slow runs: if a walltime
    segment ends before the first validation, nothing is ever written, the chain
    resumes from scratch and the run makes no progress no matter how many
    segments it burns. Save far more often than we validate.
    """
    ck = cfg.get("ckpt", {}) or {}
    if ck.get("save_every_iters"):
        return int(ck["save_every_iters"])
    return int(ck.get("save_every_epochs", default_epochs)) * steps_per_epoch


def eval_interval(cfg, steps_per_epoch, default=4000):
    """Validation cadence in iterations.

    Prefers ``eval.eval_every_epochs`` (the unified protocol), then falls back to
    whichever iteration-based key the trainer already used: SVCT keeps
    ``optim.eval_every`` while the others use ``eval.eval_every``.
    """
    ev = cfg.get("eval", {}) or {}
    o = cfg.get("optim", {}) or {}
    if ev.get("eval_every_epochs"):
        return int(ev["eval_every_epochs"]) * steps_per_epoch
    if o.get("eval_every_epochs"):
        return int(o["eval_every_epochs"]) * steps_per_epoch
    if ev.get("eval_every"):
        return int(ev["eval_every"])
    if o.get("eval_every"):
        return int(o["eval_every"])
    return int(default)


# --------------------------------------------------------------------------- early stop

class EarlyStopConfig:
    """Parsed ``early_stop`` block; ``enabled`` is False when the block is absent."""

    def __init__(self, cfg):
        e = cfg.get("early_stop", {}) or {}
        self.enabled = bool(e.get("enabled", False))
        self.min_epochs = int(e.get("min_epochs", 0))
        self.patience = int(e.get("patience", 0))
        self.min_delta = float(e.get("min_delta", 0.0))
        if self.enabled:
            if self.patience < 1:
                raise ValueError("early_stop.patience must be >= 1 when enabled")
            if self.min_epochs < 0:
                raise ValueError("early_stop.min_epochs must be >= 0")
            if self.min_delta < 0:
                raise ValueError("early_stop.min_delta must be >= 0")

    def as_dict(self, reference_psnr, bad_evals):
        return {
            "enabled": self.enabled,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "reference_psnr": reference_psnr,
            "bad_evals": bad_evals,
        }


def update_early_stop_state(psnr, reference_psnr, bad_evals, min_delta):
    """Update patience against the last validation improvement of >= min_delta."""
    if not math.isfinite(reference_psnr) or psnr >= reference_psnr + min_delta:
        return psnr, 0
    return reference_psnr, bad_evals + 1


def recover_early_stop_state(log_path, min_delta):
    """Reconstruct early-stop state from log.csv for checkpoints predating it."""
    reference_psnr = float("-inf")
    bad_evals = 0
    path = Path(log_path) if log_path else None
    if path is None or not path.is_file():
        return reference_psnr, bad_evals
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                psnr = float(row["psnr"])
            except (KeyError, TypeError, ValueError):
                continue
            reference_psnr, bad_evals = update_early_stop_state(
                psnr, reference_psnr, bad_evals, min_delta
            )
    return reference_psnr, bad_evals


def write_early_stop_marker(path, *, iteration, epoch, best_psnr,
                            reference_psnr, bad_evals):
    """Atomically tell the recoverable Slurm chain not to resume this run."""
    path = Path(path)
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


# --------------------------------------------------------------------------- checkpoints

def atomic_save(state, path):
    """Write a checkpoint that a walltime kill cannot leave half-written."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(ckpt_dir, model, opt, cfg, early: EarlyStopConfig, device,
                best_psnr=float("-inf"), verbose=True):
    """Restore model + optimizer + counters from ``latest.pt``.

    Returns (start_iter, best_psnr, reference_psnr, bad_evals). Mirrors
    ``finetune_foundation_recon.py``: a checkpoint without optimizer state is a
    hard error rather than a silent warm restart.
    """
    reference_psnr, bad_evals = float("-inf"), 0
    resume_path = Path(ckpt_dir) / "latest.pt"
    if not (cfg.get("ckpt", {}).get("resume", False) and resume_path.exists()):
        if early.enabled:
            reference_psnr, bad_evals = recover_early_stop_state(
                cfg.get("log", {}).get("csv"), early.min_delta
            )
        return 0, best_psnr, reference_psnr, bad_evals

    resume = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(resume["model"])
    if "optimizer" not in resume:
        raise RuntimeError(
            f"cannot exactly resume {resume_path}: optimizer state is missing"
        )
    opt.load_state_dict(resume["optimizer"])
    start_iter = int(resume["iter"])
    best_psnr = float(resume.get("best_psnr", best_psnr))

    saved_early = resume.get("early_stop")
    if early.enabled and saved_early:
        reference_psnr = float(saved_early.get("reference_psnr", float("-inf")))
        bad_evals = int(saved_early.get("bad_evals", 0))
    elif early.enabled:
        reference_psnr, bad_evals = recover_early_stop_state(
            cfg.get("log", {}).get("csv"), early.min_delta
        )

    if verbose:
        print(f"[resume] {resume_path} iter={start_iter} best_psnr={best_psnr:.4f}",
              flush=True)
        if early.enabled:
            print(f"[early-stop resume] reference={reference_psnr:.4f} "
                  f"bad_evals={bad_evals}/{early.patience}", flush=True)
    return start_iter, best_psnr, reference_psnr, bad_evals
