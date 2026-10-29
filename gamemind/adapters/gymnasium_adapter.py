"""Adapter for any registered Gymnasium environment.

This supports both native state observations (CartPole, LunarLander) and
pixel observations rendered by the environment. The observation mode is explicit:
GameMind never silently switches from pixels to the environment's state vector.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.adapters.base import GameAdapter
from gamemind.config import GameConfig, RewardConfig
from gamemind.rewards import build_reward_function
from gamemind.utils.errors import AdapterError, ConfigError, ObservationError
from gamemind.utils.logging import get_logger

logger = get_logger("gym_adapter")


class GymnasiumAdapter(GameAdapter):
    """Wrap a Gymnasium ``env_id`` with GameMind's reward/observation pipeline."""

    def __init__(self, config: GameConfig, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode=render_mode)
        raw = config.params
        allowed = {"env_id", "kwargs", "max_episode_steps", "disable_env_checker"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(
                f"Unknown gymnasium adapter params: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}. Use 'env_id' for the Gymnasium ID."
            )
        env_id = raw.get("env_id")
        if not isinstance(env_id, str) or not env_id.strip():
            raise ConfigError(
                "gymnasium adapter requires game.params.env_id, for example 'CartPole-v1'."
            )
        self.env_id = env_id.strip()
        kwargs = raw.get("kwargs", {}) or {}
        if not isinstance(kwargs, dict):
            raise ConfigError("game.params.kwargs must be a mapping.")
        self._kwargs = dict(kwargs)
        max_steps = raw.get("max_episode_steps")
        if max_steps is not None:
            if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
                raise ConfigError("game.params.max_episode_steps must be a positive integer or null.")
        self._make_max_steps = max_steps
        disable_checker = raw.get("disable_env_checker", False)
        if not isinstance(disable_checker, bool):
            raise ConfigError("game.params.disable_env_checker must be true or false.")
        self._disable_env_checker = disable_checker

        if not config.reward.components:
            logger.info(
                "No reward components configured for %s; using the environment's native reward.",
                self.env_id,
            )
            self.reward_function = build_reward_function(
                RewardConfig(components=({"type": "native", "scale": 1.0},))
            )

        self._env: gym.Env | None = None
        self._current_state: Any = None
        self._native_terminated = False
        self._native_truncated = False
        self._underlying_info: dict[str, Any] = {}
        self._inner_render_mode: str | None = None

    def _choose_inner_render_mode(self) -> str | None:
        if self.observation_config.is_pixel:
            return "human" if self.render_mode == "human" else "rgb_array"
        return self.render_mode

    def connect(self) -> None:
        if self._connected:
            return
        self._inner_render_mode = self._choose_inner_render_mode()
        make_kwargs: dict[str, Any] = {
            "render_mode": self._inner_render_mode,
            "disable_env_checker": self._disable_env_checker,
        }
        if self._make_max_steps is not None:
            make_kwargs["max_episode_steps"] = self._make_max_steps
        make_kwargs.update(self._kwargs)
        try:
            env = gym.make(self.env_id, **make_kwargs)
        except Exception as exc:
            raise AdapterError(
                f"Could not create Gymnasium environment {self.env_id!r}: {exc}. "
                "Check the ID, install any extras (for example LunarLander needs Box2D), "
                "and ensure rendering is supported if you selected pixel mode."
            ) from exc

        try:
            self.action_space = env.action_space
            if self.observation_config.is_pixel:
                sample_obs, _ = env.reset(seed=self.config.seed)
                del sample_obs  # state is intentionally not used in pixel mode
                try:
                    frame = env.render()
                except Exception as exc:
                    raise AdapterError(
                        f"Pixel observations require {self.env_id!r} to support render('rgb_array'): {exc}"
                    ) from exc
                if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ObservationError(
                        f"{self.env_id!r}.render() did not return an RGB uint8 frame "
                        f"(got {type(frame).__name__}, shape={getattr(frame, 'shape', None)}). "
                        "Use observation.mode: state for this environment."
                    )
                if frame.dtype != np.uint8:
                    raise ObservationError(
                        f"{self.env_id!r}.render() returned dtype {frame.dtype}; expected uint8."
                    )
                self._declare_observation_space(
                    spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8)
                )
            else:
                self._declare_observation_space(env.observation_space)
            self._env = env
            self._connected = True
        except Exception:
            env.close()
            raise

    def _require_env(self) -> gym.Env:
        if self._env is None:
            raise AdapterError("Gymnasium adapter is not connected. Call connect() first.")
        return self._env

    def _reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        env = self._require_env()
        try:
            state, info = env.reset(seed=seed, options=options)
        except Exception as exc:
            raise AdapterError(f"Resetting {self.env_id!r} failed: {exc}") from exc
        self._current_state = state
        self._native_terminated = False
        self._native_truncated = False
        self._underlying_info = dict(info or {})
        if self.observation_config.is_pixel:
            observation = self.observe()
            return observation, dict(info or {})
        return state, dict(info or {})

    def _apply_action(self, action: Any) -> None:
        env = self._require_env()
        try:
            result = env.step(action)
        except Exception as exc:
            raise AdapterError(f"Stepping {self.env_id!r} failed: {exc}") from exc
        if not isinstance(result, tuple) or len(result) != 5:
            raise AdapterError(
                f"{self.env_id!r}.step() must return (obs, reward, terminated, truncated, info). "
                "Legacy 4-value Gym/Gymnasium environments are not supported."
            )
        state, reward, terminated, truncated, info = result
        self._current_state = state
        self._native_reward = float(reward)
        self._native_terminated = bool(terminated)
        self._native_truncated = bool(truncated)
        self._underlying_info = dict(info or {})
        self._step_info = {"underlying": dict(info or {})}

    def observe(self) -> Any:
        if self.observation_config.is_pixel:
            env = self._require_env()
            try:
                frame = env.render()
            except Exception as exc:
                raise AdapterError(f"Could not render {self.env_id!r} for a pixel observation: {exc}") from exc
            if not isinstance(frame, np.ndarray):
                raise ObservationError(f"{self.env_id!r}.render() returned no pixel frame.")
            return frame
        if self._current_state is None:
            raise AdapterError("No state observation available; call reset() first.")
        return self._current_state

    def is_episode_over(self) -> bool:
        detected = super().is_episode_over() if self.termination_detector else False
        return bool(self._native_terminated or detected)

    def _is_truncated(self) -> bool:
        return bool(self._native_truncated or super()._is_truncated())

    def _reward_signals(self):
        return {}

    def _extra_info(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "native_reward": float(self._native_reward),
            "underlying": dict(self._underlying_info),
        }

    def render(self) -> np.ndarray | None:
        env = self._require_env()
        frame = env.render()
        return frame if isinstance(frame, np.ndarray) else None

    def _close_resources(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._connected = False
