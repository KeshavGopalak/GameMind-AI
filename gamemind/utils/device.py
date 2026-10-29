"""Device selection helpers."""

from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return requested
