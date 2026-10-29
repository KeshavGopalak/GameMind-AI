"""Small visual detectors used by optional reward and termination components.

These are honest, limited tools: template matching can recognise a fixed image
such as a "Game Over" banner, but it is not OCR, score reading, or general game
understanding. If a game exposes a real API, prefer that over pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamemind.utils.errors import ConfigError, RewardError
from gamemind.utils.paths import resolve_project_path


@dataclass(frozen=True)
class Region:
    """A rectangular screen region in pixels."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_config(cls, value: object, *, where: str) -> "Region | None":
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ConfigError(
                f"{where} must be [x, y, width, height] or null, got {value!r}."
            )
        x, y, width, height = value
        for item in (x, y, width, height):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError(f"{where} values must be integers, got {value!r}.")
        if width <= 0 or height <= 0:
            raise ConfigError(f"{where} width and height must be positive, got {value!r}.")
        if x < 0 or y < 0:
            raise ConfigError(f"{where} x and y must be non-negative, got {value!r}.")
        return cls(x=x, y=y, width=width, height=height)

    def crop(self, frame: np.ndarray, *, where: str = "frame") -> np.ndarray:
        height, width = frame.shape[:2]
        if self.x + self.width > width or self.y + self.height > height:
            raise RewardError(
                f"{where}: region {self.as_list()} extends beyond the {width}x{height} frame."
            )
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def load_template(path: str | Path, *, where: str = "template") -> np.ndarray:
    """Load a template image as a grayscale array, rejecting missing/empty files."""
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise RewardError(
            f"{where} image not found: {resolved}. Use a path relative to the project root "
            "or an absolute path, and make sure the screenshot exists."
        )
    image = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise RewardError(f"{where} image could not be decoded (empty or unsupported): {resolved}")
    return image


class TemplateMatcher:
    """Detect a fixed image within an optional frame region using normalized SSD."""

    def __init__(
        self,
        template_path: str | Path,
        *,
        region: Region | None = None,
        threshold: float = 0.85,
        where: str = "template detector",
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ConfigError(f"{where}.threshold must be between 0 and 1, got {threshold}.")
        self.template_path = str(template_path)
        self.template = load_template(template_path, where=where)
        self.region = region
        self.threshold = float(threshold)

    def score(self, frame: np.ndarray) -> float:
        """Return a 0..1 similarity score (1 means a very close visual match)."""
        if frame is None:
            raise RewardError(
                "A visual detector needs raw frames, but this adapter supplied none. "
                "Use a pixel-observation game adapter or remove the visual detector."
            )
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise RewardError(f"Expected a 2D/3D frame, got shape {frame.shape}.")
        search = self.region.crop(gray, where="template detector") if self.region else gray
        template = self.template
        if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
            raise RewardError(
                f"Template {self.template_path} ({template.shape[1]}x{template.shape[0]}) is "
                f"larger than the search region ({search.shape[1]}x{search.shape[0]}). "
                "Choose a larger region or a smaller template screenshot."
            )
        result = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED)
        min_value = float(np.min(result))
        return float(np.clip(1.0 - min_value, 0.0, 1.0))

    def matches(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold
