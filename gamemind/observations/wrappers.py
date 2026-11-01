"""Gymnasium wrappers for pixels, action repetition, frame stacking, validation."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.config import ObservationConfig
from gamemind.observations.processing import ObservationProcessor
from gamemind.utils.errors import ObservationError


class PixelObservationWrapper(gym.ObservationWrapper):
    """Apply the shared pixel pipeline to a raw RGB observation."""

    def __init__(self, env: gym.Env, observation_config: ObservationConfig) -> None:
        if not observation_config.is_pixel:
            raise ObservationError(
                "PixelObservationWrapper requires observation.mode 'rgb' or 'grayscale'. "
                "Switch the config or do not wrap a state-space environment."
            )
        self.processor = ObservationProcessor(observation_config)
        super().__init__(env)
        self.observation_space = self.processor.observation_space

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.processor.process(observation, context="PixelObservationWrapper")


class ActionRepeat(gym.Wrapper):
    """Apply the same action for several game ticks and sum their rewards."""

    def __init__(self, env: gym.Env, repeat: int) -> None:
        if repeat < 1:
            raise ValueError("ActionRepeat requires repeat >= 1.")
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation: Any = None
        terminated = truncated = False
        info: dict[str, Any] = {}
        repeats = 0
        breakdown: dict[str, float] = {}
        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            for name, value in info.get("reward_breakdown", {}).items():
                breakdown[name] = breakdown.get(name, 0.0) + float(value)
            repeats += 1
            if terminated or truncated:
                break
        assert observation is not None
        info = dict(info or {})
        info["action_repeats"] = repeats
        if breakdown:
            info["reward_breakdown"] = breakdown
        return observation, total_reward, bool(terminated), bool(truncated), info


class FrameStackChannels(gym.Wrapper):
    """Stack frames along the channel axis, producing ``(C * n, H, W)``.

    Stable-Baselines3 expects channel-first image observations. Stacking on a
    new leading axis would produce invalid 4D observations, so this wrapper
    concatenates frames into the channel dimension. On reset, the first frame is
    repeated to fill the stack (the same convention as SB3's VecFrameStack).
    """

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        if num_stack < 1:
            raise ValueError("FrameStackChannels requires num_stack >= 1.")
        if not isinstance(env.observation_space, spaces.Box):
            raise ObservationError("Frame stacking requires a Box observation space.")
        shape = env.observation_space.shape
        if shape is None or len(shape) != 3:
            raise ObservationError(
                "Frame stacking expects channel-first image observations with shape (C, H, W); "
                f"got {shape}. Wrap the environment with PixelObservationWrapper first."
            )
        super().__init__(env)
        self.num_stack = int(num_stack)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_stack)
        channels, height, width = shape
        low = float(np.min(env.observation_space.low))
        high = float(np.max(env.observation_space.high))
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(channels * self.num_stack, height, width),
            dtype=env.observation_space.dtype,
        )

    def _get_stacked(self) -> np.ndarray:
        if len(self._frames) != self.num_stack:
            raise ObservationError(
                f"Internal frame stack has {len(self._frames)} frames, expected {self.num_stack}. "
                "Call reset() before step()."
            )
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            raise ObservationError(
                "Frame stacking received an unexpected reset observation "
                f"(shape {getattr(observation, 'shape', None)})."
            )
        self._frames.clear()
        for _ in range(self.num_stack):
            self._frames.append(np.array(observation, copy=True))
        return self._get_stacked(), info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            raise ObservationError(
                "Frame stacking received an unexpected step observation "
                f"(shape {getattr(observation, 'shape', None)})."
            )
        self._frames.append(np.array(observation, copy=True))
        return self._get_stacked(), reward, terminated, truncated, info


class ObservationValidator(gym.ObservationWrapper):
    """Fail loudly when an adapter or wrapper violates its declared space."""

    def observation(self, observation: Any) -> Any:
        try:
            contained = self.observation_space.contains(observation)
        except (TypeError, ValueError) as exc:
            raise ObservationError(
                f"Observation does not match space {self.observation_space}: {exc}"
            ) from exc
        if not contained:
            value = np.asarray(observation)
            raise ObservationError(
                "Observation is outside its declared Gymnasium space. "
                f"Expected shape={self.observation_space.shape}, "
                f"dtype={self.observation_space.dtype}; got shape={value.shape}, "
                f"dtype={value.dtype}. Check the adapter's get_observation_space()."
            )
        return observation
