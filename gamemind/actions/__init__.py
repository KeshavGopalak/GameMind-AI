"""Configurable discrete actions and desktop input backends."""

from __future__ import annotations

import gymnasium as gym

from gamemind.actions.backends import InputBackend, PyAutoGUIBackend
from gamemind.config import ActionsConfig
from gamemind.utils.errors import ConfigError


def build_discrete_action_space(actions: ActionsConfig, *, where: str = "actions") -> gym.spaces.Discrete:
    """Create a Discrete space from configured bindings."""
    if not actions.bindings:
        raise ConfigError(
            f"{where}: no action bindings are configured. Add actions.bindings entries "
            "(including a no-op with empty keys) or let the adapter use its native action space."
        )
    return gym.spaces.Discrete(len(actions.bindings))


def action_label(actions: ActionsConfig, index: int) -> str:
    """Return a human-readable action name, or a numbered fallback."""
    try:
        return actions.binding(int(index)).name
    except (IndexError, TypeError, ValueError):
        return f"action_{index}"


__all__ = [
    "InputBackend",
    "PyAutoGUIBackend",
    "action_label",
    "build_discrete_action_space",
]
