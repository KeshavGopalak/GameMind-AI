"""GameAdapter for the self-contained Target Chase pixel game."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.adapters.base import GameAdapter
from gamemind.actions import build_discrete_action_space
from gamemind.config import ActionBinding, ActionsConfig, GameConfig, RewardConfig
from gamemind.rewards import build_reward_function
from gamemind.utils.errors import ConfigError, ObservationError
from gamemind.utils.logging import get_logger
from games.example_game.game import TargetChaseGame

logger = get_logger("example_adapter")

_DEFAULT_BINDINGS = (
    ActionBinding("noop", params={"move": [0, 0]}),
    ActionBinding("left", params={"move": [-1, 0]}),
    ActionBinding("right", params={"move": [1, 0]}),
    ActionBinding("up", params={"move": [0, -1]}),
    ActionBinding("down", params={"move": [0, 1]}),
)

_DEFAULT_REWARD = RewardConfig(
    components=(
        {
            "type": "progress",
            "key": "distance",
            "mode": "decrease",
            "reset_on": "target_collected",
            "scale": 0.5,
            "name": "closer",
        },
        {"type": "signal", "key": "target_collected", "scale": 10.0, "name": "target"},
        {"type": "signal", "key": "wall_collision", "scale": -0.5, "name": "wall_penalty"},
        {"type": "constant", "value": -0.01, "name": "time_penalty"},
    )
)


class ExampleGameAdapter(GameAdapter):
    """Wrap Target Chase as a pixel-observation Gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, config: GameConfig, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode=render_mode)
        if not config.observation.is_pixel:
            raise ConfigError(
                "example_game is a pixel game. Set observation.mode to 'grayscale' "
                "(recommended) or 'rgb'. State observations are intentionally not provided."
            )
        params = config.params
        allowed = {
            "grid_size",
            "cell_size",
            "n_obstacles",
            "max_steps",
            "fps",
            "show_compass",
            "start",
            "target",
            "obstacles",
            "terminate_on_collect",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ConfigError(
                f"Unknown example_game params: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        self.game = TargetChaseGame(
            grid_size=params.get("grid_size", 10),
            cell_size=params.get("cell_size", 32),
            n_obstacles=params.get("n_obstacles", 7),
            max_steps=params.get("max_steps", 120),
            seed=config.seed,
            fps=params.get("fps", 15),
            show_compass=params.get("show_compass", True),
            fixed_start=params.get("start"),
            fixed_target=params.get("target"),
            fixed_obstacles=params.get("obstacles"),
            terminate_on_collect=params.get("terminate_on_collect", False),
        )
        self.metadata = {**self.metadata, "render_fps": self.game.fps}

        bindings = self._resolve_bindings(config.actions)
        self.action_config = ActionsConfig(
            frame_skip=config.actions.frame_skip,
            duration=config.actions.duration,
            max_duration=config.actions.max_duration,
            bindings=bindings,
        )
        self.action_space = build_discrete_action_space(
            self.action_config, where="example_game actions"
        )
        self._declare_observation_space(
            spaces.Box(
                low=0,
                high=255,
                shape=(self.game.grid_size * self.game.cell_size, self.game.grid_size * self.game.cell_size, 3),
                dtype=np.uint8,
            )
        )
        if not config.reward.components:
            logger.info(
                "No reward components configured for example_game; using its small default shaping."
            )
            self.reward_function = build_reward_function(_DEFAULT_REWARD)
        self._last_signals: dict[str, float] = {}
        self._episode_limit = config.episode.max_steps or self.game.max_steps

    def _resolve_bindings(self, actions: ActionsConfig) -> tuple[ActionBinding, ...]:
        if not actions.bindings:
            return _DEFAULT_BINDINGS
        resolved: list[ActionBinding] = []
        for binding in actions.bindings:
            move = binding.params.get("move")
            if (
                not isinstance(move, (list, tuple))
                or len(move) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or abs(v) > 1 for v in move)
            ):
                raise ConfigError(
                    f"Action {binding.name!r} for example_game needs params.move: [dx, dy] "
                    "with dx/dy in -1..1. Example: {move: [1, 0]}."
                )
            resolved.append(binding)
        return tuple(resolved)

    def _move_for(self, action: int) -> tuple[int, int]:
        binding = self.action_config.binding(action)
        move = binding.params["move"]
        return int(move[0]), int(move[1])

    # ------------------------------------------------------------------
    # GameAdapter hooks
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self.set_max_episode_steps(self._episode_limit)

    def _reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        frame = self.game.reset(seed=seed)
        self._last_signals = {
            "distance": float(self.game.distance_to_target()),
            "distance_delta": 0.0,
            "target_collected": 0.0,
            "wall_collision": 0.0,
        }
        self.window_closed = False
        info = {
            "agent_cell": self.game.agent,
            "target_cell": self.game.target,
            "grid_size": self.game.grid_size,
            "success": self.game.success,
        }
        return frame, info

    def _apply_action(self, action: int) -> None:
        dx, dy = self._move_for(int(action))
        signals = self.game.move(dx, dy)
        self._last_signals = signals.as_dict()
        self._step_info = {
            "agent_cell": self.game.agent,
            "target_cell": self.game.target,
            "targets_collected": self.game.targets_collected,
            "wall_hits": self.game.wall_hits,
            "distance": self.game.distance_to_target(),
            "success": self.game.success,
        }

    def observe(self) -> np.ndarray:
        if self.game.frame is None:
            raise ObservationError("Target Chase has no rendered frame; call reset() first.")
        return self.game.frame.copy()

    def is_episode_over(self) -> bool:
        # A collected target normally respawns. With terminate_on_collect it is
        # a genuine success terminal; the time limit is handled as truncation.
        return bool(self.game.success)

    def _reward_signals(self) -> Mapping[str, float]:
        return {
            **self._last_signals,
            "alive": 1.0,
            "targets_collected": float(self.game.targets_collected),
            "wall_hits": float(self.game.wall_hits),
        }

    def _extra_info(self) -> dict[str, Any]:
        return {
            "targets_collected": self.game.targets_collected,
            "wall_hits": self.game.wall_hits,
            "success": self.game.success,
        }

    def render(self) -> np.ndarray | None:
        frame = self.game.render_frame()
        if self.render_mode == "human":
            self.game.display_frame(frame)
            self.window_closed = self.game.window_closed
        return frame

    def _close_resources(self) -> None:
        self.game.close()
        self.window_closed = True
