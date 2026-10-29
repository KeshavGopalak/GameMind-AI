"""Atomic run-status files consumed by the CLI and Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gamemind.utils.logging import get_logger
from gamemind.utils.serialization import atomic_write_json, utc_now_iso

logger = get_logger("status")


class TrainingStatus:
    """Small JSON document describing the current state of a training run."""

    def __init__(self, path: Path, base: dict[str, Any]) -> None:
        self.path = Path(path)
        self._payload: dict[str, Any] = {
            **base,
            "state": "starting",
            "started_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self._write()

    def _write(self) -> None:
        self._payload["updated_at"] = utc_now_iso()
        try:
            atomic_write_json(self.path, self._payload)
        except OSError as exc:
            # Status is for observability; a read-only log volume should not kill training.
            logger.warning("Could not write training status to %s: %s", self.path, exc)

    def update(self, **fields: Any) -> None:
        self._payload.update(fields)
        self._write()

    def finish(self, state: str, **fields: Any) -> None:
        self._payload.update(fields)
        self._payload["state"] = state
        self._payload["finished_at"] = utc_now_iso()
        self._write()

    @property
    def payload(self) -> dict[str, Any]:
        return dict(self._payload)
