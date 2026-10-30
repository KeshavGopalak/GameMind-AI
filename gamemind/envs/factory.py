"""Build single and vectorized environments with the exact same wrapper stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from gamemind.adapters.registry import create_adapter
from gamemind.config import GameConfig
from gamemind.observations import (
    ActionRepeat,
    FrameStackChannels,
    ObservationValidator,
    PixelObservationWrapper,
)
from gamemind.utils.errors import ConfigError
from gamemind.utils.logging import get_logger

logger = get_logger("envs")


def build_env(
    config: GameConfig,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create and connect the configured adapter, then apply the wrapper stack.

    The same function is used for training, evaluation, and playback. That is
    what guarantees the saved model sees exactly the same observation layout at
    inference time as it did during training.
    """
    adapter = create_adapter(config, render_mode=render_mode)
    try:
        adapter.connect()
        env: gym.Env = adapter
        if config.observation.is_pixel:
            env = PixelObservationWrapper(env, config.observation)
        if config.actions.frame_skip > 1:
            env = ActionRepeat(env, config.actions.frame_skip)
        if config.observation.stack > 1:
            env = FrameStackChannels(env, config.observation.stack)
        env = ObservationValidator(env)
        return env
    except Exception:
        adapter.close()
        raise


def make_env_thunk(
    config: GameConfig,
    *,
    render_mode: str | None = None,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Return a picklable-by-cloudpickle thunk for a vector environment."""

    def _thunk() -> gym.Env:
        env = build_env(config, render_mode=render_mode)
        if rank:
            logger.debug("Creating environment replica %d for %s", rank, config.name)
        return env

    return _thunk


def build_vec_env(
    config: GameConfig,
    *,
    n_envs: int = 1,
    seed: int | None = None,
    monitor_path: Path | str | None = None,
    subproc: bool = False,
    render_mode: str | None = None,
) -> Any:
    """Build a Stable-Baselines3 vector environment.

    Desktop adapters and human rendering are restricted to one in-process
    environment. Multiple replicas of a keyboard-controlled game would send
    conflicting inputs to the same window.
    """
    if n_envs < 1:
        raise ConfigError("n_envs must be at least 1.")
    if config.adapter.strip().lower() in {"desktop", "geometry_dash"} and n_envs > 1:
        raise ConfigError(
            "Desktop-controlled adapters must use n_envs: 1. Multiple copies would control "
            "the same window simultaneously. Reduce training.n_envs or the --n-envs override."
        )
    if render_mode == "human" and n_envs > 1:
        raise ConfigError("Human rendering requires n_envs=1.")

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    thunks = [make_env_thunk(config, render_mode=render_mode, rank=index) for index in range(n_envs)]
    try:
        if subproc and n_envs > 1:
            if config.adapter.strip().lower() in {"desktop", "geometry_dash"}:
                raise ConfigError("Subprocess vectorization is not supported for desktop input.")
            vec_env = SubprocVecEnv(thunks, start_method="spawn")
        else:
            vec_env = DummyVecEnv(thunks)
    except Exception:
        # vec env constructors normally close partially created envs, but a failed
        # SubprocVecEnv can leave resources behind. Nothing else is safe to clean here.
        raise
    if seed is not None:
        vec_env.seed(int(seed))
    if monitor_path is not None:
        monitor = Path(monitor_path)
        monitor.parent.mkdir(parents=True, exist_ok=True)
        vec_env = VecMonitor(vec_env, filename=str(monitor))
    return vec_env
