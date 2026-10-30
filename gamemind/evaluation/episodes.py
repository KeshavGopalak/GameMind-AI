"""Run a trained policy without training it, optionally with a live window."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gamemind.config import GameConfig
from gamemind.envs.factory import build_env
from gamemind.training.model_io import assert_model_env_compatible, load_model_bundle
from gamemind.utils.device import resolve_device
from gamemind.utils.logging import get_logger
from gamemind.utils.serialization import atomic_write_json, utc_now_iso

logger = get_logger("evaluation")


@dataclass
class EpisodeResult:
    episode: int
    reward: float
    length: int
    terminated: bool
    truncated: bool
    action_counts: dict[str, int] = field(default_factory=dict)
    # Optional game metrics. Geometry Dash supplies approximate visual progress
    # and survival time; other games leave these as None.
    progress_pct: float | None = None
    max_progress_pct: float | None = None
    survival_seconds: float | None = None


@dataclass
class EvaluationResult:
    game: str
    model_path: str
    algorithm: str
    episodes: list[EpisodeResult]
    deterministic: bool
    user_closed: bool = False
    started_at: str = field(default_factory=utc_now_iso)

    @property
    def mean_reward(self) -> float:
        return float(np.mean([episode.reward for episode in self.episodes])) if self.episodes else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std([episode.reward for episode in self.episodes])) if self.episodes else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean([episode.length for episode in self.episodes])) if self.episodes else 0.0

    @property
    def best_reward(self) -> float:
        return float(max((episode.reward for episode in self.episodes), default=0.0))

    @property
    def mean_progress_pct(self) -> float | None:
        values = [episode.max_progress_pct for episode in self.episodes if episode.max_progress_pct is not None]
        return float(np.mean(values)) if values else None

    @property
    def mean_survival_seconds(self) -> float | None:
        values = [episode.survival_seconds for episode in self.episodes if episode.survival_seconds is not None]
        return float(np.mean(values)) if values else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "model_path": self.model_path,
            "algorithm": self.algorithm,
            "deterministic": self.deterministic,
            "started_at": self.started_at,
            "finished_at": utc_now_iso(),
            "episodes": [episode.__dict__ for episode in self.episodes],
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_length": self.mean_length,
            "best_reward": self.best_reward,
            "mean_progress_pct": self.mean_progress_pct,
            "mean_survival_seconds": self.mean_survival_seconds,
            "user_closed": self.user_closed,
        }


def _render_fps(env: gym.Env, override: int | None) -> int:
    if override is not None:
        return max(0, int(override))
    metadata = getattr(env, "metadata", None) or {}
    if not metadata:
        metadata = getattr(env.unwrapped, "metadata", None) or {}
    return max(1, int(metadata.get("render_fps", 30)))


def _window_closed(env: gym.Env) -> bool:
    try:
        return bool(getattr(env.unwrapped, "window_closed", False))
    except Exception:
        return False


def run_episodes(
    config: GameConfig,
    model_path: str | Path,
    *,
    episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
    seed: int | None = None,
    fps: int | None = None,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> EvaluationResult:
    """Load a model and play episodes without updating its weights.

    This function never calls ``model.learn``. It uses the same environment
    builder and wrapper stack as training, and validates the saved spaces first.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    env = build_env(config, render_mode="human" if render else None)
    model = None
    result: EvaluationResult | None = None
    try:
        model, metadata, algorithm = load_model_bundle(
            model_path, env=None, device=resolve_device(device)
        )
        assert_model_env_compatible(model, env)
        playback_fps = _render_fps(env, fps)
        result = EvaluationResult(
            game=config.name,
            model_path=str(model_path),
            algorithm=algorithm,
            episodes=[],
            deterministic=deterministic,
        )
        logger.info(
            "Playing %s with %s (%s), %d episode(s), render=%s, fps=%s.",
            config.name,
            algorithm.upper(),
            "deterministic" if deterministic else "stochastic",
            episodes,
            render,
            playback_fps if render else "unpaced",
        )
        for episode_index in range(1, episodes + 1):
            episode_seed = None if seed is None else int(seed) + episode_index - 1
            observation, info = env.reset(seed=episode_seed)
            del info
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            action_counts: dict[str, int] = {}
            last_info: dict[str, Any] = {}
            if render:
                try:
                    env.render()
                except Exception as exc:
                    logger.warning("Could not draw the initial frame: %s", exc)
            while not (terminated or truncated):
                if render:
                    step_started = time.perf_counter()
                action, _states = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                last_info = info if isinstance(info, dict) else {}
                total_reward += float(reward)
                length += 1
                action_name = str(info.get("action_name", action))
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
                if render:
                    try:
                        env.render()
                    except Exception as exc:
                        logger.warning("Rendering stopped: %s", exc)
                        result.user_closed = True
                        break
                    if _window_closed(env):
                        logger.info("Game window closed by user.")
                        result.user_closed = True
                        break
                    if playback_fps > 0:
                        remaining = (1.0 / playback_fps) - (time.perf_counter() - step_started)
                        if remaining > 0:
                            time.sleep(remaining)
            def _metric(key: str) -> float | None:
                value = last_info.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    number = float(value)
                    return number if np.isfinite(number) else None
                return None

            episode_result = EpisodeResult(
                episode=episode_index,
                reward=total_reward,
                length=length,
                terminated=bool(terminated),
                truncated=bool(truncated),
                action_counts=action_counts,
                progress_pct=_metric("progress_pct"),
                max_progress_pct=_metric("max_progress_pct"),
                survival_seconds=_metric("survival_seconds"),
            )
            result.episodes.append(episode_result)
            logger.info(
                "Episode %d/%d: reward=%.2f length=%d terminated=%s truncated=%s",
                episode_index,
                episodes,
                total_reward,
                length,
                terminated,
                truncated,
            )
            if episode_result.max_progress_pct is not None:
                logger.info(
                    "  visual progress: %.2f%% (max); survival: %.2fs",
                    episode_result.max_progress_pct,
                    episode_result.survival_seconds or 0.0,
                )
            if result.user_closed:
                break
        logger.info(
            "Evaluation: mean reward %.2f ± %.2f, mean length %.1f over %d episode(s).",
            result.mean_reward,
            result.std_reward,
            result.mean_length,
            len(result.episodes),
        )
        if output_path is not None:
            atomic_write_json(Path(output_path), result.as_dict())
            logger.info("Wrote evaluation results to %s", output_path)
        return result
    finally:
        if model is not None:
            # The model's environment (if any) is not needed; our env owns the resources.
            pass
        env.close()
