"""Optional visual episode-termination detectors.

No generic screenshot adapter can know when an arbitrary game is over. The
built-in detector can recognise a fixed "Game Over" image. Otherwise use
``episode.max_steps`` as a time limit, or implement a custom adapter that reads
the game's real state.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gamemind.config import EpisodeConfig
from gamemind.utils.errors import ConfigError, RewardError
from gamemind.utils.vision import Region, TemplateMatcher


class TemplateTerminationDetector:
    """End an episode after a template matches on several consecutive frames."""

    def __init__(
        self,
        template: str,
        *,
        region: Region | None = None,
        threshold: float = 0.85,
        consecutive: int = 1,
    ) -> None:
        if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 1:
            raise ConfigError("episode.terminate_on.consecutive must be a positive integer.")
        self.matcher = TemplateMatcher(
            template, region=region, threshold=threshold, where="episode termination detector"
        )
        self.consecutive = consecutive
        self._matches = 0

    def reset(self) -> None:
        self._matches = 0

    def update(self, frame: np.ndarray) -> bool:
        if frame is None:
            raise RewardError(
                "A template termination detector needs a raw frame, but the adapter supplied none."
            )
        if self.matcher.score(frame) >= self.matcher.threshold:
            self._matches += 1
        else:
            self._matches = 0
        return self._matches >= self.consecutive


def build_termination_detector(config: EpisodeConfig) -> TemplateTerminationDetector | None:
    """Build the detector declared in ``episode.terminate_on``, if any."""
    raw: Mapping[str, Any] | None = config.terminate_on
    if not raw:
        return None
    type_name = str(raw.get("type", "")).strip().lower()
    if type_name != "template":
        raise ConfigError(
            f"Unsupported episode.terminate_on.type {type_name!r}. "
            "Supported: 'template'. For other games, use max_steps or a custom adapter."
        )
    allowed = {"type", "template", "region", "threshold", "consecutive"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in episode.terminate_on: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    template = raw.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ConfigError("episode.terminate_on.template must be an image path.")
    region = Region.from_config(raw.get("region"), where="episode.terminate_on.region")
    threshold = raw.get("threshold", 0.85)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ConfigError("episode.terminate_on.threshold must be a number between 0 and 1.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ConfigError("episode.terminate_on.threshold must be between 0 and 1.")
    consecutive = raw.get("consecutive", 1)
    return TemplateTerminationDetector(
        template,
        region=region,
        threshold=float(threshold),
        consecutive=consecutive,
    )
