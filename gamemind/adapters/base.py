"""The standardized GameAdapter interface.

Every game adapter is a Gymnasium environment. It additionally exposes:

* ``connect()``            acquire the game/window/environment
* ``observe()``            return the raw observation (pixels or state)
* ``calculate_reward()``   evaluate the configured reward components
* ``is_episode_over()``    game-specific termination (separate from truncation)
* ``get_action_space()`` / ``get_observation_space()``
* ``render()``             return a frame and/or update a human window
* ``close()``              release input keys and close game resources

The base class implements the Gymnasium reset/step protocol once, so adapters
only supply game-specific behavior. Observation preprocessing, action repetition,
and frame stacking are applied by :func:`gamemind.envs.build_env` wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from gamemind.actions import action_label
from gamemind.config import GameConfig
from gamemind.rewards import RewardContext, RewardFunction, build_reward_function
from gamemind.rewards.termination import build_termination_detector
from gamemind.utils.errors import AdapterError, ConfigError, ObservationError
from gamemind.utils.logging import get_logger

logger = get_logger("adapter")


class GameAdapter(gym.Env, ABC):
    """Base class for all game integrations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: GameConfig, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode if render_mode is not None else config.render_mode
        self.observation_config = config.observation
        self.action_config = config.actions
        self.reward_function: RewardFunction = build_reward_function(config.reward)
        self.termination_detector = build_termination_detector(config.episode)
        if self.termination_detector is not None and not config.observation.is_pixel:
            raise ConfigError(
                "A visual episode.terminate_on detector requires pixel observations "
                "(observation.mode: rgb or grayscale)."
            )
        self._connected = False
        self._closed = False
        self._needs_reset = True
        self._episode_done = False
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_max_steps: int | None = config.episode.max_steps
        self._native_reward = 0.0
        self._last_raw_observation: Any | None = None
        self._last_reward_breakdown: dict[str, float] = {}
        self._last_info: dict[str, Any] = {}
        self._step_info: dict[str, Any] = {}
        self._last_action: Any | None = None
        self._last_action_name: str = ""
        self._rng = np.random.default_rng(config.seed)
        self.window_closed = False
        self._raw_observation_space: gym.Space | None = None
        # Set by each concrete adapter before reset(): gym.spaces.Discrete/Box/...
        self.action_space: gym.Space

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Connect to the game. Must be safe to call before reset()."""

    def _close_resources(self) -> None:
        """Subclass hook for closing windows, emulators, or releasing inputs."""

    def close(self) -> None:
        """Release resources exactly once. Safe to call multiple times."""
        if self._closed:
            return
        try:
            self._close_resources()
        finally:
            self._closed = True
            self._connected = False
            self._needs_reset = True
            self._episode_done = True

    def _require_connected(self) -> None:
        if self._closed:
            raise AdapterError("This adapter is closed. Create a new environment to continue.")
        if not self._connected:
            raise AdapterError(
                f"Adapter for game {self.config.name!r} is not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def get_action_space(self) -> gym.Space:
        """Return the adapter's declared action space."""
        if not hasattr(self, "action_space"):
            raise AdapterError("Adapter has not set action_space.")
        return self.action_space

    def get_observation_space(self) -> gym.Space:
        """Return the adapter's *raw* observation space (before wrappers)."""
        if self._raw_observation_space is None:
            raise AdapterError(
                "Adapter has not declared its raw observation space. Set "
                "self.observation_space and self._raw_observation_space during connect()."
            )
        return self._raw_observation_space

    def _declare_observation_space(self, space: gym.Space) -> None:
        self._raw_observation_space = space
        self.observation_space = space

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the game and return a raw observation in the declared space."""
        self._require_connected()
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)  # initializes Gymnasium's np_random
        self._last_action = None
        self._last_action_name = ""
        self._needs_reset = False
        self._episode_done = False
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_reward_breakdown = {}
        self._step_info = {}
        self.reward_function.reset()
        if self.termination_detector is not None:
            self.termination_detector.reset()
        observation, info = self._reset(seed=seed, options=options)
        observation = self._validate_raw_observation(observation, where="reset")
        if info is None:
            info = {}
        if not isinstance(info, Mapping):
            raise AdapterError(f"reset() info must be a dict, got {type(info).__name__}.")
        self._last_raw_observation = observation
        self._last_info = dict(info)
        return observation, dict(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the game one tick and apply the configured reward function."""
        self._require_connected()
        if self._needs_reset:
            raise AdapterError(
                "step() was called before reset(). Gymnasium environments must be reset first."
            )
        if self._episode_done:
            raise AdapterError(
                "The episode has ended. Call reset() before step() again. "
                "Stable-Baselines3's vector environments do this automatically."
            )
        normalized_action = self._normalize_action(action)
        self._last_action = normalized_action
        self._step_info = {}
        self._apply_action(normalized_action)
        self._episode_step += 1

        observation = self.observe()
        observation = self._validate_raw_observation(observation, where="step")
        self._last_raw_observation = observation

        terminated = bool(self.is_episode_over())
        truncated = bool(self._is_truncated())
        reward = self.calculate_reward()
        self._episode_reward += reward
        self._episode_done = terminated or truncated

        info = self._build_info()
        self._last_info = info
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Hooks implemented by concrete adapters
    # ------------------------------------------------------------------
    @abstractmethod
    def _reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset game-specific state and return the first raw observation."""

    @abstractmethod
    def _apply_action(self, action: Any) -> None:
        """Apply one normalized action to the game. Do not sleep for frame_skip here."""

    @abstractmethod
    def observe(self) -> Any:
        """Return the current raw observation (screen frame or native state)."""

    @abstractmethod
    def render(self) -> np.ndarray | None:
        """Render/return an RGB frame for human viewing when requested."""

    def is_episode_over(self) -> bool:
        """Game-specific termination (not the time limit).

        The default uses the optional visual termination detector. Adapters with
        access to real game state should override this method and, when useful,
        combine their result with ``super().is_episode_over()``.
        """
        if self.termination_detector is None:
            return False
        if self._last_raw_observation is None:
            raise AdapterError("No frame is available for the termination detector.")
        return bool(self.termination_detector.update(self._last_raw_observation))

    def _reward_signals(self) -> Mapping[str, float]:
        """Named game signals available to configured reward components."""
        return {}

    def _extra_info(self) -> dict[str, Any]:
        """Game-specific info for logging/tests (never fed to the policy automatically)."""
        return {}

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------
    def calculate_reward(self) -> float:
        """Evaluate the configured reward components for the latest step."""
        context = RewardContext(
            signals=self._reward_signals(),
            native_reward=float(self._native_reward),
            frame=self._last_raw_observation if self.observation_config.is_pixel else None,
            info=self._last_info,
            episode_step=self._episode_step,
            episode_reward=self._episode_reward,
        )
        reward, breakdown = self.reward_function.evaluate(context)
        self._last_reward_breakdown = breakdown
        return reward

    def _is_truncated(self) -> bool:
        """Default time-limit truncation, independent of game termination."""
        return bool(
            self._episode_max_steps is not None
            and self._episode_step >= self._episode_max_steps
        )

    # ------------------------------------------------------------------
    # Observation/action validation
    # ------------------------------------------------------------------
    def _validate_raw_observation(self, observation: Any, *, where: str) -> Any:
        if self._raw_observation_space is None:
            raise AdapterError(
                "Adapter observation space is undeclared. Call _declare_observation_space() "
                "before returning observations."
            )
        if self.observation_config.is_pixel and not isinstance(observation, np.ndarray):
            raise ObservationError(
                f"{where}: pixel mode requires a NumPy frame, got {type(observation).__name__}. "
                "Do not silently fall back to privileged game state."
            )
        try:
            valid = self._raw_observation_space.contains(observation)
        except (TypeError, ValueError) as exc:
            raise ObservationError(f"{where}: observation failed space validation: {exc}") from exc
        if not valid:
            raise ObservationError(
                f"{where}: observation does not match the adapter's declared space "
                f"{self._raw_observation_space}. Shape: "
                f"{getattr(observation, 'shape', None)}, dtype: {getattr(observation, 'dtype', None)}."
            )
        return observation

    def _normalize_action(self, action: Any) -> Any:
        space = self.get_action_space()
        if isinstance(space, gym.spaces.Discrete):
            if isinstance(action, np.ndarray):
                if action.size != 1:
                    raise ValueError(
                        f"Discrete action must contain exactly one value, got shape {action.shape}."
                    )
                action = action.reshape(-1)[0]
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
                raise ValueError(
                    f"Discrete action must be an integer, got {type(action).__name__} ({action!r})."
                )
            index = int(action)
            if not space.contains(index):
                raise ValueError(
                    f"Action {index} is outside the valid range 0..{space.n - 1} "
                    f"for game {self.config.name!r}."
                )
            self._last_action_name = (
                action_label(self.action_config, index)
                if self.action_config.has_bindings
                else f"action_{index}"
            )
            return index
        if isinstance(space, gym.spaces.Box):
            array = np.asarray(action, dtype=space.dtype)
            if array.shape != space.shape:
                raise ValueError(
                    f"Continuous action shape {array.shape} does not match {space.shape} "
                    f"for game {self.config.name!r}."
                )
            if not space.contains(array):
                raise ValueError(
                    f"Continuous action {array!r} is outside the action space bounds {space}."
                )
            self._last_action_name = "continuous"
            return array
        try:
            if not space.contains(action):
                raise ValueError(f"Action {action!r} is outside {space}.")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid action {action!r} for space {space}: {exc}") from exc
        self._last_action_name = str(action)
        return action

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "game": self.config.name,
            "episode_step": self._episode_step,
            "episode_reward": float(self._episode_reward),
            "action": self._last_action,
            "action_name": self._last_action_name,
            "reward_breakdown": dict(self._last_reward_breakdown),
        }
        info.update(self._step_info)
        info.update(self._extra_info())
        return info

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------
    def set_max_episode_steps(self, steps: int | None) -> None:
        """Set a hard time limit; ``None`` disables truncation."""
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
            raise ConfigError("max_episode_steps must be a positive integer or None.")
        self._episode_max_steps = steps

    def get_action_labels(self) -> list[str]:
        """Return action binding names when configured, otherwise generic labels."""
        space = self.get_action_space()
        if isinstance(space, gym.spaces.Discrete):
            if self.action_config.has_bindings:
                return [binding.name for binding in self.action_config.bindings]
            return [f"action_{i}" for i in range(space.n)]
        return []
