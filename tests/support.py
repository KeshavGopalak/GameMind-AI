"""Small deterministic environments and fake desktop backends for tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.config import GameConfig
from gamemind.rewards.base import RewardContext


class CountingEnv(gym.Env):
    """Minimal env whose observation/bump behavior makes wrapper tests obvious."""

    metadata = {"render_modes": []}

    def __init__(self, *, terminate_after: int = 100) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (1, 4, 4), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.terminate_after = terminate_after
        self.count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros((1, 4, 4), dtype=np.uint8), {}

    def step(self, action):
        del action
        self.count += 1
        observation = np.full((1, 4, 4), min(self.count, 255), dtype=np.uint8)
        terminated = self.count >= self.terminate_after
        return observation, 1.0, terminated, False, {"count": self.count}


class FakeCapture:
    width = 96
    height = 64
    closed = False

    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        self.frames = frames or [np.zeros((self.height, self.width, 3), dtype=np.uint8)]
        self.index = 0

    def grab(self) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)].copy()
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeInput:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.held: set[str] = set()
        self.focus_calls = 0
        self.release_calls = 0
        self.fail_focus = False
        self.closed = False

    def validate_keys(self, keys: Sequence[str]) -> None:
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("keys must be strings")

    def ensure_focus(self, window_title: str | None) -> None:
        self.focus_calls += 1
        if self.fail_focus:
            from gamemind.utils.errors import InputBackendError

            raise InputBackendError("focus lost in test")

    def key_down(self, key: str) -> None:
        self.events.append(("down", key))
        self.held.add(key)

    def key_up(self, key: str) -> None:
        self.events.append(("up", key))
        self.held.discard(key)

    def click(self, button: str) -> None:
        self.events.append(("click", button))

    def move_rel(self, dx: int, dy: int) -> None:
        self.events.append(("move", f"{dx},{dy}"))

    def release_all(self) -> None:
        self.release_calls += 1
        for key in sorted(self.held):
            self.events.append(("release", key))
        self.held.clear()

    def close(self) -> None:
        self.release_all()
        self.closed = True


def make_desktop_config(*, terminate_on: dict[str, Any] | None = None) -> GameConfig:
    return GameConfig.from_dict(
        {
            "game": {
                "name": "desktop_test",
                "adapter": "desktop",
                "params": {
                    "region": [0, 0, 96, 64],
                    "window_title": "Test Game",
                    "require_window_focus": True,
                    "start_delay": 0.0,
                    "reset_keys": ["r"],
                    "reset_delay": 0.0,
                    "reset_key_duration": 0.0,
                },
            },
            "observation": {"mode": "grayscale", "width": 32, "height": 32, "stack": 1},
            "actions": {
                "frame_skip": 1,
                "duration": 0.0,
                "max_duration": 0.2,
                "bindings": [
                    {"name": "noop", "keys": []},
                    {"name": "left", "keys": ["a"]},
                    {"name": "jump", "keys": ["space"]},
                ],
            },
            "reward": {"components": [{"type": "constant", "value": 0.0}]},
            "episode": {"max_steps": 3, "terminate_on": terminate_on},
            "training": {"n_envs": 1, "n_steps": 8, "batch_size": 8},
        }
    )


def custom_reward(context: RewardContext, bonus: float = 1.0) -> float:
    """Custom reward used to test module:function loading."""
    return bonus + float(context.native_reward)
