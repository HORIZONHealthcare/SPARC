"""Portable project-path handling for IBD and local runs.

New manifests store paths relative to the project root (for example,
``datasets/ct_rate/...``). Existing absolute paths remain supported.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_SUBDIRS = {
    "datasets",
    "processed",
    "outputs",
    "pretrained_cache",
}


def project_root() -> Path:
    """Root under which datasets and outputs live.

    Set PROJECT_ROOT (or DATA_ROOT) to wherever you keep them. There is no
    site-specific default: this is deliberately deployment-neutral.
    """
    for var in ("PROJECT_ROOT", "DATA_ROOT"):
        value = os.environ.get(var)
        if value:
            return Path(os.path.expanduser(os.path.expandvars(value)))
    raise RuntimeError(
        "Neither PROJECT_ROOT nor DATA_ROOT is set; point one at the directory "
        "holding your splits/ manifests and dataset volumes."
    )

def _expand(value: str | os.PathLike[str]) -> str:
    raw = os.fspath(value)
    if "$PROJECT_ROOT" in raw or "${PROJECT_ROOT}" in raw:
        root = str(project_root())
        raw = raw.replace("${PROJECT_ROOT}", root).replace("$PROJECT_ROOT", root)
    return os.path.expanduser(os.path.expandvars(raw))


def resolve_project_path(value: str | os.PathLike[str]) -> str:
    """Resolve project-root-relative data paths; preserve other relative paths."""
    path = Path(_expand(value))
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] in PROJECT_SUBDIRS:
        return str(project_root() / path)
    return str(path)


def project_relative_path(value: str | os.PathLike[str]) -> str:
    """Store a path relative to PROJECT_ROOT when it lives underneath it."""
    path = Path(_expand(value))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def expand_env(obj):
    """Expand ${VAR} / $VAR in every string of a loaded config.

    Configs here reference data and run directories through ${DATA_ROOT} and
    ${OUTPUT_ROOT}; without this the placeholders would reach pandas / torch
    verbatim. Strings with no variable in them are returned unchanged, so
    plain relative paths keep working.
    """
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return _expand(obj) if isinstance(obj, str) else obj
