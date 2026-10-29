"""Callbacks for metrics, pause/stop control, checkpoints, and evaluation."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from gamemind.config import GameConfig
from gamemind.training.model_io import save_model_bundle, write_model_metadata
from gamemind.training.status import TrainingStatus
from gamemind.utils.errors import TrainingError
from gamemind.utils.logging import get_logger
from gamemind.utils.serialization import append_jsonl, utc_now_iso

logger = get_logger("callbacks")


class TrainingControlCallback(BaseCallback):
    """Track metrics, write a live status file, and honor PAUSE/STOP files."""

    def __init__(
        self,
        *,
        status: TrainingStatus,
        run_dir: Path,
        checkpoint_dir: Path,
        config: GameConfig,
        algorithm: str,
        policy: str,
        run_id: str,
        initial_timesteps: int,
        additional_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.status = status
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.algorithm = algorithm
        self.policy = policy
        self.run_id = run_id
        self.initial_timesteps = int(initial_timesteps)
        self.additional_timesteps = max(1, int(additional_timesteps))
        self.stop_file = self.run_dir / "STOP"
        self.pause_file = self.run_dir / "PAUSE"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self._rewards: deque[float] = deque(maxlen=config.training.stats_window_size)
        self._lengths: deque[int] = deque(maxlen=config.training.stats_window_size)
        self._episode_count = 0
        self._last_episode_reward: float | None = None
        self._best_episode_reward: float | None = None
        self._started = time.monotonic()
        self._last_status_write = 0.0
        self._last_control_check = 0.0
        self.stopped = False
        self.stop_model_path: Path | None = None
        self._last_timesteps = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        self._started = time.monotonic()
        self.status.update(
            state="training",
            timesteps=int(self.model.num_timesteps),
            target_timesteps=self.initial_timesteps + self.additional_timesteps,
            message="Training started.",
        )
        self._write_status(force=True)

    def _on_step(self) -> bool:
        if not self._check_controls():
            return False
        self._collect_episodes()
        now = time.monotonic()
        if now - self._last_status_write >= 0.75:
            self._write_status()
        return True

    def _on_training_end(self) -> None:
        self._write_status(force=True)

    # ------------------------------------------------------------------
    # Status and metrics
    # ------------------------------------------------------------------
    def _collect_episodes(self) -> None:
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        if not isinstance(infos, (list, tuple)):
            return
        for info in infos:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            try:
                reward = float(episode["r"])
                length = int(episode["l"])
            except (KeyError, TypeError, ValueError):
                continue
            self._episode_count += 1
            self._rewards.append(reward)
            self._lengths.append(length)
            self._last_episode_reward = reward
            if self._best_episode_reward is None or reward > self._best_episode_reward:
                self._best_episode_reward = reward
            append_jsonl(
                self.episodes_path,
                {
                    "episode": self._episode_count,
                    "reward": reward,
                    "length": length,
                    "timesteps": int(self.model.num_timesteps),
                    "timestamp": utc_now_iso(),
                },
            )

    def metrics(self) -> dict[str, Any]:
        elapsed = max(1e-9, time.monotonic() - self._started)
        steps_done = max(0, int(self.model.num_timesteps) - self.initial_timesteps)
        progress = min(1.0, steps_done / self.additional_timesteps)
        return {
            "timesteps": int(self.model.num_timesteps),
            "initial_timesteps": self.initial_timesteps,
            "steps_this_run": steps_done,
            "target_timesteps": self.initial_timesteps + self.additional_timesteps,
            "progress": progress,
            "episodes": self._episode_count,
            "ep_rew_mean": float(np.mean(self._rewards)) if self._rewards else None,
            "ep_len_mean": float(np.mean(self._lengths)) if self._lengths else None,
            "last_episode_reward": self._last_episode_reward,
            "best_episode_reward": self._best_episode_reward,
            "fps": steps_done / elapsed if steps_done else 0.0,
        }

    def _write_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_write < 0.25:
            return
        self._last_status_write = now
        self.status.update(**self.metrics())

    # ------------------------------------------------------------------
    # Pause/stop files
    # ------------------------------------------------------------------
    def _check_controls(self) -> bool:
        now = time.monotonic()
        if now - self._last_control_check < 0.15:
            return True
        self._last_control_check = now
        if self.stop_file.exists():
            self._stop_and_checkpoint()
            return False
        if not self.pause_file.exists():
            return True

        logger.info("Training paused via %s. Delete the file or press Resume in the dashboard.", self.pause_file)
        self.status.update(state="paused", **self.metrics(), message="Paused.")
        last_heartbeat = 0.0
        while self.pause_file.exists() and not self.stop_file.exists():
            time.sleep(0.2)
            now = time.monotonic()
            if now - last_heartbeat > 1.0:
                last_heartbeat = now
                self.status.update(state="paused", **self.metrics())
        if self.stop_file.exists():
            self._stop_and_checkpoint()
            return False
        logger.info("Training resumed.")
        self.status.update(state="training", **self.metrics(), message="Resumed.")
        return True

    def _stop_and_checkpoint(self) -> None:
        steps = int(self.model.num_timesteps)
        self.status.update(state="stopping", **self.metrics(), message="Saving a checkpoint before stopping...")
        path = self.checkpoint_dir / f"stopped_{steps}_steps.zip"
        logger.info("Saving stop checkpoint to %s", path)
        try:
            save_model_bundle(
                self.model,
                path,
                self.config,
                algorithm=self.algorithm,
                policy=self.policy,
                run_id=self.run_id,
                extra={"stopped": True, "reason": "stop_requested", "num_timesteps": steps},
            )
        except Exception as exc:
            self.status.update(state="error", error=f"Stop checkpoint failed: {exc}")
            raise TrainingError(f"Could not save the stop checkpoint: {exc}") from exc
        self.stopped = True
        self.stop_model_path = path
        self.status.update(
            state="stopped",
            **self.metrics(),
            message="Training stopped. A checkpoint was saved.",
            stopped_model_path=str(path),
        )


class BundleCheckpointCallback(BaseCallback):
    """Save a model + metadata every N environment timesteps."""

    def __init__(
        self,
        *,
        frequency: int,
        checkpoint_dir: Path,
        config: GameConfig,
        algorithm: str,
        policy: str,
        run_id: str,
        status: TrainingStatus | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.frequency = int(frequency)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.algorithm = algorithm
        self.policy = policy
        self.run_id = run_id
        self.status = status
        self._last_saved = 0
        self.last_checkpoint: Path | None = None

    def _on_training_start(self) -> None:
        self._last_saved = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        if self.frequency <= 0:
            return True
        current = int(self.model.num_timesteps)
        if current - self._last_saved < self.frequency:
            return True
        path = self.checkpoint_dir / f"checkpoint_{current}_steps.zip"
        logger.info("Saving periodic checkpoint at %d timesteps to %s", current, path)
        save_model_bundle(
            self.model,
            path,
            self.config,
            algorithm=self.algorithm,
            policy=self.policy,
            run_id=self.run_id,
            extra={"checkpoint": True, "num_timesteps": current},
        )
        self.last_checkpoint = path
        self._last_saved = current
        if self.status is not None:
            self.status.update(last_checkpoint=str(path))
        return True


class MetadataEvalCallback(EvalCallback):
    """EvalCallback that also writes sidecar metadata and status for the best model."""

    def __init__(
        self,
        *args: Any,
        config: GameConfig,
        algorithm: str,
        policy: str,
        run_id: str,
        status: TrainingStatus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gamemind_config = config
        self.algorithm = algorithm
        self.policy = policy
        self.run_id = run_id
        self.status = status
        self._best_signature: tuple[int, int] | None = None
        self.best_model_path_out: Path | None = None

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.best_model_save_path is not None:
            best_path = Path(self.best_model_save_path) / "best_model.zip"
            if best_path.exists():
                stat = best_path.stat()
                signature = (int(stat.st_mtime_ns), int(stat.st_size))
                if signature != self._best_signature:
                    write_model_metadata(
                        best_path,
                        self.model,
                        self.gamemind_config,
                        algorithm=self.algorithm,
                        policy=self.policy,
                        run_id=self.run_id,
                        extra={
                            "best_model": True,
                            "eval_mean_reward": float(self.best_mean_reward),
                            "num_timesteps": int(self.model.num_timesteps),
                        },
                    )
                    self._best_signature = signature
                    self.best_model_path_out = best_path
        if self.status is not None and np.isfinite(self.best_mean_reward):
            self.status.update(
                best_eval_reward=float(self.best_mean_reward),
                last_eval_reward=float(self.last_mean_reward)
                if np.isfinite(self.last_mean_reward)
                else None,
                best_model_path=str(self.best_model_path_out) if self.best_model_path_out else None,
            )
        return result
