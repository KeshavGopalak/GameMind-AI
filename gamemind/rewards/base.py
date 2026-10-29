"""Reward interfaces shared by every game adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gamemind.config import RewardConfig
from gamemind.utils.errors import RewardError


@dataclass
class RewardContext:
    """Everything a reward component may legitimately inspect for one step."""

    signals: Mapping[str, float]
    native_reward: float = 0.0
    frame: np.ndarray | None = None
    info: Mapping[str, Any] | None = None
    episode_step: int = 0
    episode_reward: float = 0.0

    def signal(self, key: str) -> float:
        """Read a named numeric signal, with an actionable error when it is absent."""
        if not key:
            raise RewardError("A reward component requested an empty signal name.")
        if key not in self.signals:
            available = ", ".join(sorted(self.signals)) or "(none)"
            raise RewardError(
                f"Reward signal {key!r} was not supplied by the game adapter. "
                f"Available signals: {available}."
            )
        raw = self.signals[key]
        if isinstance(raw, (bool, np.bool_)):
            return float(raw)
        if isinstance(raw, (int, float, np.integer, np.floating)):
            value = float(raw)
            if not np.isfinite(value):
                raise RewardError(f"Reward signal {key!r} is not finite ({raw!r}).")
            return value
        raise RewardError(
            f"Reward signal {key!r} must be numeric or boolean, got {type(raw).__name__}."
        )


class RewardComponent:
    """Base class for one additive reward term."""

    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self) -> None:
        """Reset per-episode state (progress trackers, template edges, ...)."""

    def evaluate(self, context: RewardContext) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class RewardFunction:
    """Sum components, optionally clip, and keep a per-component breakdown."""

    def __init__(
        self,
        components: list[RewardComponent],
        *,
        clip: tuple[float, float] | None = None,
    ) -> None:
        self.components = components
        self.clip = clip

    def reset(self) -> None:
        for component in self.components:
            component.reset()

    def evaluate(self, context: RewardContext) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        total = 0.0
        for component in self.components:
            try:
                value = float(component.evaluate(context))
            except RewardError:
                raise
            except Exception as exc:
                raise RewardError(
                    f"Reward component {component.name!r} ({type(component).__name__}) failed: {exc}"
                ) from exc
            if not np.isfinite(value):
                raise RewardError(
                    f"Reward component {component.name!r} returned a non-finite value {value!r}."
                )
            breakdown[component.name] = value
            total += value
        if self.clip is not None:
            total = float(np.clip(total, self.clip[0], self.clip[1]))
        if not np.isfinite(total):
            raise RewardError(f"Total reward is not finite ({total!r}).")
        return total, breakdown


def build_reward_function(config: RewardConfig) -> RewardFunction:
    """Build a reward function from validated YAML components."""
    from gamemind.rewards.components import create_component

    components: list[RewardComponent] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(config.components):
        component = create_component(raw, index=index)
        if component.name in seen_names:
            raise RewardError(
                f"Duplicate reward component name {component.name!r}. "
                "Give one of them a different 'name'."
            )
        seen_names.add(component.name)
        components.append(component)
    return RewardFunction(components, clip=config.clip)
