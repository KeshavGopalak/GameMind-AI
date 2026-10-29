"""A tiny pygame window for displaying frames from screen-capture adapters.

The example game and Gymnasium environments display themselves. This viewer is
for adapters whose only visual surface is a screenshot (typically a desktop
game). It is intentionally optional and only imported on the human-render path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gamemind.utils.errors import AdapterError
from gamemind.utils.logging import get_logger

logger = get_logger("viewer")


class FrameViewer:
    """Display RGB frames in a resizable pygame window."""

    def __init__(self, title: str = "GameMind AI", fps: int = 30, scale: float = 1.0) -> None:
        self.title = title
        self.fps = max(1, int(fps))
        self.scale = max(0.25, float(scale))
        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self.closed = False

    def show(self, frame: np.ndarray, caption: str | None = None) -> bool:
        """Show one RGB frame. Returns ``False`` when the user closes the window."""
        if self.closed:
            return False
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise AdapterError("pygame is required for the live viewer. Install pygame.") from exc

        if self._pygame is None:
            self._pygame = pygame
            pygame.init()
            pygame.display.set_caption(self.title)
            height, width = frame.shape[:2]
            size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
            self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self._clock = pygame.time.Clock()
        assert self._screen is not None and self._clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.closed = True
                return False

        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(self._pygame.transform.smoothscale(surface, self._screen.get_size()), (0, 0))
        if caption:
            self._pygame.display.set_caption(f"{self.title} — {caption}")
        self._pygame.display.flip()
        return True

    def close(self) -> None:
        """Close the viewer if it was opened."""
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
                self._pygame.quit()
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("Ignoring pygame shutdown error", exc_info=True)
        self._pygame = None
        self._screen = None
        self._clock = None
        self.closed = True

    def __enter__(self) -> "FrameViewer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
