"""JSON/JSONL serialization helpers used for metadata, status, and metrics."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp suitable for logs and metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    """Convert common scientific-Python objects into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # gym.Space
        return space_to_dict(value)
    return repr(value)


def to_jsonable(value: Any) -> Any:
    """Round-trip a value through JSON so nested non-primitive types are handled."""
    return json.loads(json.dumps(value, default=json_default))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically so a dashboard never reads a half-written status file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2)
            handle.write("\n")
        # Windows can briefly lock the destination while a dashboard/editor reads it.
        # Retry the atomic replace instead of losing a status update.
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON object to a newline-delimited file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True))
        handle.write("\n")


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning *default* if the file is absent or malformed."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank or malformed lines (for dashboards)."""
    records: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        return []
    return records


def space_to_dict(space: Any) -> dict[str, Any]:
    """Return a compact, human-readable description of a Gymnasium space."""
    result: dict[str, Any] = {
        "class": type(space).__name__,
        "dtype": str(getattr(space, "dtype", "")),
        "shape": list(getattr(space, "shape", ()) or ()),
    }
    if hasattr(space, "n"):
        result["n"] = int(space.n)
    for attr in ("low", "high"):
        if hasattr(space, attr):
            value = np.asarray(getattr(space, attr))
            if value.size <= 16:
                result[attr] = value.tolist()
            else:
                result[attr] = {"min": float(np.min(value)), "max": float(np.max(value))}
    return result


def library_versions() -> dict[str, str]:
    """Return versions of the most relevant runtime dependencies when available."""
    import importlib

    versions: dict[str, str] = {"python": platform.python_version()}
    for module_name in ("numpy", "gymnasium", "torch", "stable_baselines3", "cv2", "pygame"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # optional dependency or broken install; informational only
            versions[module_name] = "not installed"
    return versions


def git_metadata() -> dict[str, str | None]:
    """Read the current commit without failing outside a Git checkout."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None}
