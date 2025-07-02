"""Utilities for managing model caches."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get_cache_dir"]


def get_cache_dir() -> str:
    """Return the directory used to cache downloaded models."""
    return os.environ.get("MODEL_CACHE", str(Path.home() / ".cache" / "creator"))

