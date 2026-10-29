"""Project-relative path handling.

GameMind avoids hard-coded absolute paths. Paths in YAML configurations are
resolved relative to the project root unless they are already absolute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

PathLike = Union[str, Path]


def resolve_project_path(path: PathLike | None, *, base: PathLike | None = None) -> Path:
    """Resolve *path* against the project root (or *base*) without requiring existence."""
    if path is None:
        raise ValueError("A path is required.")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(base) if base is not None else PROJECT_ROOT) / candidate


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and its parents) and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_name(value: str, *, fallback: str = "run") -> str:
    """Turn user text into a filename-safe stem; never returns path separators."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned[:80] or fallback
