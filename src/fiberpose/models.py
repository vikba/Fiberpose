"""Locate, download and cache pretrained Fiberpose models.

Weights are not shipped in the wheel. Models are looked up by file name in
``models_dir()``; registered models missing there are downloaded from the
release assets on first use and verified against the hashes below.

Environment overrides:
    FIBERPOSE_MODELS_DIR  folder with .onnx models (and where downloads go)
    FIBERPOSE_MODEL       path to the .onnx file used when no model is configured
"""

from __future__ import annotations

import os
from pathlib import Path

import pooch

DEFAULT_MODEL = "R108_dt_R111_fg.onnx"

# TODO: upload the registered files as assets of this release before publishing.
_BASE_URL = "https://github.com/vikba/Fiberpose/releases/download/models-v1/"

# Model file -> files to download; external-data files (.onnx.data) must sit
# next to their .onnx graph.
_MODELS: dict[str, tuple[str, ...]] = {
    "R108_dt_R111_fg.onnx": ("R108_dt_R111_fg.onnx",),
    "R108_dt_fg.onnx": ("R108_dt_fg.onnx",),
}

_REGISTRY = {
    "R108_dt_R111_fg.onnx": "sha256:8f89b9161cb454968d36fc7e6ea9c6d770a6dd45a891f0aa308d86e99778c6c0",
    "R108_dt_fg.onnx": "sha256:bb515be44e1daa220f8ba457fb1f9d856a2c3865149f4b6b97e12aefdea37536",
}

_LEGACY_PREFIX = "checkpoints/"


def models_dir() -> Path:
    """Folder that holds .onnx models.

    Order: ``$FIBERPOSE_MODELS_DIR``, then ``<repo>/checkpoints`` when running
    from a source checkout, then the per-user download cache.
    """
    override = os.environ.get("FIBERPOSE_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    checkout_models = repo_root / "checkpoints"
    if (repo_root / "pyproject.toml").is_file() and checkout_models.is_dir():
        return checkout_models
    return Path(pooch.os_cache("fiberpose")) / "models"


def normalize_model_entry(entry: str | None) -> str:
    """Strip the legacy ``checkpoints/`` prefix used by older configs."""
    entry = str(entry or "").strip().replace("\\", "/")
    if entry.startswith(_LEGACY_PREFIX):
        entry = entry[len(_LEGACY_PREFIX):]
    return entry


def discover_models() -> list[str]:
    """File names of the .onnx models available in ``models_dir()``, plus registered ones."""
    folder = models_dir()
    local = sorted(p.name for p in folder.glob("*.onnx")) if folder.is_dir() else []
    return local + [name for name in _MODELS if name not in local]


def resolve_model_path(entry: str | None = None) -> str:
    """Map a config/GUI model entry to an absolute path, downloading it if needed.

    An empty entry means the default model (or ``$FIBERPOSE_MODEL`` if set).
    Absolute paths are returned unchanged; names are looked up in ``models_dir()``.
    """
    name = normalize_model_entry(entry)
    if not name:
        override = os.environ.get("FIBERPOSE_MODEL")
        if override:
            return str(Path(override).expanduser())
        name = DEFAULT_MODEL
    if os.path.isabs(name):
        return name
    local = models_dir() / name
    if local.is_file() or name not in _MODELS:
        return str(local)
    return fetch_model(name)


def fetch_model(name: str = DEFAULT_MODEL) -> str:
    """Download a registered model into ``models_dir()`` and return its path."""
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}; registered: {', '.join(_MODELS)}")
    fetcher = pooch.create(path=models_dir(), base_url=_BASE_URL, registry=_REGISTRY)
    paths = [fetcher.fetch(fname) for fname in _MODELS[name]]
    return paths[0]
