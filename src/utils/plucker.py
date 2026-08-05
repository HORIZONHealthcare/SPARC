"""Plücker line coordinates for X-ray rays.

A ray is parameterised as `(direction, moment)`:
    direction = (target - source) / |target - source|
    moment    = source x direction

Plücker is convenient because the moment uniquely identifies the ray
independent of where along the ray we pick the basepoint. The concatenated
6-D vector is what we feed into the 2D context encoder per detector patch.

We normalise the moment by SAD so its magnitude is O(1) — without this it
scales with world-coordinate units of source position and the MLP overfits
to one component.
"""

from __future__ import annotations

import torch


EPS = 1e-9


def compute_plucker(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel Plücker (direction, moment).

    Args:
        source: (..., 3) ray source positions in world coords.
        target: (..., 3) ray endpoints (detector pixel positions).

    Returns:
        (..., 6) Plücker = [unit_direction, moment].
    """
    direction = target - source
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
    moment = torch.cross(source, direction, dim=-1)
    return torch.cat([direction, moment], dim=-1)


def normalize_plucker(plucker: torch.Tensor, sad_mm) -> torch.Tensor:
    """Scale the moment component by 1 / SAD so it becomes O(1).

    Args:
        plucker: (..., 6) last dim = [dir, moment].
        sad_mm: scalar or tensor broadcastable to all but the last axis.

    Returns:
        (..., 6) Plücker with moment divided by SAD.
    """
    if isinstance(sad_mm, (int, float)):
        sad = torch.tensor(float(sad_mm), device=plucker.device, dtype=plucker.dtype)
    else:
        sad = sad_mm.to(device=plucker.device, dtype=plucker.dtype)
    direction, moment = plucker.split(3, dim=-1)
    moment = moment / sad.clamp_min(EPS).unsqueeze(-1)
    return torch.cat([direction, moment], dim=-1)


def patch_pool_plucker(plucker_pixel: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Average-pool a per-pixel Plücker map into a per-patch map.

    The 2D context encoder embeds patch_size x patch_size patches of the X-ray
    image. We give each patch a single 6-D ray by averaging the ray field
    inside the patch. Direction averaging is approximate, so we re-normalise
    post-pool; for the angular spread inside one 16x16 patch this is fine.

    Args:
        plucker_pixel: (..., H, W, 6) per-pixel Plücker.
        patch_size: int patch edge.

    Returns:
        (..., H/patch, W/patch, 6) per-patch Plücker.
    """
    *prefix, h, w, c = plucker_pixel.shape
    assert c == 6, f"expected last dim 6, got {c}"
    assert h % patch_size == 0 and w % patch_size == 0, (
        f"H={h} W={w} must be divisible by patch_size={patch_size}"
    )
    ph, pw = h // patch_size, w // patch_size
    x = plucker_pixel.view(*prefix, ph, patch_size, pw, patch_size, 6)
    x = x.mean(dim=(-4, -2))
    direction, moment = x.split(3, dim=-1)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.cat([direction, moment], dim=-1)
