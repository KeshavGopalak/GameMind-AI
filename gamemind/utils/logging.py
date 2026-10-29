"""Consistent console logging for scripts and long-running training jobs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def configure_logging(verbose: int = 1, log_file: Path | str | None = None) -> None:
    """Configure GameMind's logger once.

    Parameters
    ----------
    verbose:
        0 = warnings only, 1 = informational progress, 2 = debug details.
    log_file:
        Optional file that receives the same messages (used by the dashboard).
    """
    global _CONFIGURED
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(int(verbose), logging.INFO)
    root = logging.getLogger("gamemind")
    root.setLevel(level)
    root.propagate = False
    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(level)
        if log_file is not None:
            _add_file_handler(root, log_file, level)
        return

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        _add_file_handler(root, log_file, level)
    _CONFIGURED = True


def _add_file_handler(logger: logging.Logger, log_file: Path | str, level: int) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    for existing in logger.handlers:
        if isinstance(existing, logging.FileHandler) and Path(existing.baseFilename) == path.resolve():
            existing.setLevel(level)
            return
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``gamemind`` namespace."""
    return logging.getLogger(f"gamemind.{name}")


def close_file_handlers() -> None:
    """Close run-specific log files (important on Windows, which locks open files)."""
    root = logging.getLogger("gamemind")
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - best-effort close
                pass
