"""Training orchestration with checkpoints, pause/stop control, and evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import CallbackList

from gamemind.algorithms.factory import build_model, select_policy
from gamemind.config import GameConfig
from gamemind.envs.factory import build_vec_env
from gamemind.training.callbacks import (
    BundleCheckpointCallback,
    MetadataEvalCallback,
    TrainingControlCallback,
)
from gamemind.training.model_io import (
    load_model_bundle,
    metadata_path_for,
    save_config_snapshot,
    save_model_bundle,
)
from gamemind.training.status import TrainingStatus
from gamemind.utils.device import resolve_device
from gamemind.utils.errors import ConfigError, GameMindError, ModelCompatibilityError, TrainingError
from gamemind.utils.logging import close_file_handlers, configure_logging, get_logger
from gamemind.utils.paths import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    ensure_dir,
    resolve_project_path,
    safe_name,
)
from gamemind.utils.seeding import seed_everything
from gamemind.utils.serialization import atomic_write_json, utc_now_iso

logger = get_logger("trainer")


@dataclass
class TrainingResult:
    """A summary of one training invocation."""

    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    status_path: Path
    episodes_path: Path
    model_path: Path | None = None
    best_model_path: Path | None = None
    completed: bool = False
    stopped: bool = False
    interrupted: bool = False
    final_timesteps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "status_path": str(self.status_path),
            "episodes_path": str(self.episodes_path),
            "model_path": str(self.model_path) if self.model_path else None,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "completed": self.completed,
            "stopped": self.stopped,
            "interrupted": self.interrupted,
            "final_timesteps": self.final_timesteps,
            "metrics": self.metrics,
            "error": self.error,
        }


class Trainer:
    """Run one PPO/A2C experiment described by a :class:`GameConfig`."""

    def __init__(
        self,
        config: GameConfig,
        *,
        model_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        run_name: str | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        self.config = config
        self.model_path = resolve_project_path(
            model_path or (DEFAULT_MODEL_DIR / f"{safe_name(config.name)}.zip")
        )
        self.checkpoint_root = resolve_project_path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self.log_root = resolve_project_path(log_dir or DEFAULT_LOG_DIR)
        self.run_name = run_name
        self.resume_from = resolve_project_path(resume_from) if resume_from else None
        self.run_id = self._make_run_id()
        self.run_dir = self.log_root / "runs" / self.run_id
        self.checkpoint_dir = self.checkpoint_root / safe_name(config.name) / self.run_id
        self.status_path = self.run_dir / "status.json"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self._status: TrainingStatus | None = None

    def _make_run_id(self) -> str:
        if self.run_name:
            candidate = safe_name(self.run_name, fallback="run")
            target = self.log_root / "runs" / candidate
            if target.exists():
                raise ConfigError(
                    f"Run directory already exists: {target}. Choose another --run-name or delete it."
                )
            return candidate
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{safe_name(self.config.name)}"
        candidate = base
        counter = 2
        while (self.log_root / "runs" / candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    # ------------------------------------------------------------------
    def run(self) -> TrainingResult:
        """Train, save, and return a summary. Raises only for real setup/training failures."""
        config = self.config
        training = config.training
        run_dir = ensure_dir(self.run_dir)
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.model_path.parent)
        configure_logging(training.verbose, log_file=run_dir / "train.log")
        seed_everything(training.seed)
        save_config_snapshot(config, run_dir / "config.resolved.yaml")
        logger.info("Starting run %s for game %s.", self.run_id, config.name)
        logger.info("Run directory: %s", run_dir)

        policy_label = training.policy or "auto"
        base_status = {
            "run_id": self.run_id,
            "game": config.name,
            "adapter": config.adapter,
            "algorithm": training.algorithm,
            "policy": policy_label,
            "run_dir": str(run_dir),
            "model_path": str(self.model_path),
            "checkpoint_dir": str(self.checkpoint_dir),
            "requested_timesteps": training.timesteps,
            "n_envs": training.n_envs,
            "seed": training.seed,
            "resume_from": str(self.resume_from) if self.resume_from else None,
        }
        self._status = TrainingStatus(self.status_path, base_status)
        result = TrainingResult(
            run_id=self.run_id,
            run_dir=run_dir,
            checkpoint_dir=self.checkpoint_dir,
            status_path=self.status_path,
            episodes_path=self.episodes_path,
        )
        train_env = None
        eval_env = None
        model = None
        control: TrainingControlCallback | None = None
        initial_timesteps = 0
        try:
            if (
                config.adapter.strip().lower() in {"desktop", "geometry_dash"}
                and training.eval_freq > 0
            ):
                raise ConfigError(
                    "Periodic evaluation is disabled for desktop-controlled adapters because a "
                    "second environment would send inputs to the same game window. Set "
                    "training.eval_freq: 0."
                )
            train_env = build_vec_env(
                config,
                n_envs=training.n_envs,
                seed=training.seed,
                monitor_path=run_dir / "monitor.csv",
                subproc=training.subproc,
            )
            if self.resume_from is not None:
                model, metadata, loaded_algorithm = load_model_bundle(
                    self.resume_from,
                    algorithm=training.algorithm,
                    env=train_env,
                    device=resolve_device(training.device),
                )
                meta_game = metadata.get("game")
                if meta_game and safe_name(str(meta_game)) != safe_name(config.name):
                    raise ModelCompatibilityError(
                        f"Checkpoint was trained for game {meta_game!r}, but the current config is "
                        f"{config.name!r}. Use the matching config to continue training."
                    )
                initial_timesteps = int(model.num_timesteps)
                policy_label = str(metadata.get("policy") or type(model.policy).__name__)
                logger.info(
                    "Resuming %s from %s at %d timesteps; training %d additional steps.",
                    loaded_algorithm.upper(),
                    self.resume_from,
                    initial_timesteps,
                    training.timesteps,
                )
                model.tensorboard_log = (
                    str(run_dir / "tensorboard") if training.tensorboard else None
                )
                # Ensure a fresh SB3 logger writes into this run's TensorBoard folder.
                model._logger = None  # type: ignore[attr-defined]
                model.set_env(train_env)
            else:
                select_policy(config, train_env)  # validate before constructing the model
                tensorboard = str(run_dir / "tensorboard") if training.tensorboard else None
                model = build_model(config, train_env, tensorboard_log=tensorboard)
                policy_label = select_policy(config, train_env)
            assert model is not None
            self._status.update(
                state="training",
                policy=policy_label,
                initial_timesteps=initial_timesteps,
                target_timesteps=initial_timesteps + training.timesteps,
            )

            eval_callback = None
            if training.eval_freq > 0:
                eval_env = build_vec_env(
                    config,
                    n_envs=1,
                    seed=training.seed + 9973,
                    monitor_path=run_dir / "eval_monitor.csv",
                    subproc=False,
                )
                best_dir = ensure_dir(self.checkpoint_dir / "best")
                # SB3 callbacks receive one call per vector-env step. Convert the
                # user-facing environment-step interval into callback calls.
                callback_eval_freq = max(1, training.eval_freq // max(1, training.n_envs))
                eval_callback = MetadataEvalCallback(
                    eval_env=eval_env,
                    n_eval_episodes=training.n_eval_episodes,
                    eval_freq=callback_eval_freq,
                    log_path=str(run_dir / "eval"),
                    best_model_save_path=str(best_dir) if training.save_best else None,
                    deterministic=True,
                    render=False,
                    verbose=0,
                    warn=False,
                    config=config,
                    algorithm=training.algorithm,
                    policy=policy_label,
                    run_id=self.run_id,
                    status=self._status,
                )

            control = TrainingControlCallback(
                status=self._status,
                run_dir=run_dir,
                checkpoint_dir=self.checkpoint_dir,
                config=config,
                algorithm=training.algorithm,
                policy=policy_label,
                run_id=self.run_id,
                initial_timesteps=initial_timesteps,
                additional_timesteps=training.timesteps,
                verbose=training.verbose,
            )
            checkpoint = BundleCheckpointCallback(
                frequency=training.checkpoint_freq,
                checkpoint_dir=self.checkpoint_dir,
                config=config,
                algorithm=training.algorithm,
                policy=policy_label,
                run_id=self.run_id,
                status=self._status,
                verbose=0,
            )
            callbacks = [control, checkpoint]
            if eval_callback is not None:
                callbacks.append(eval_callback)
            callback_list = CallbackList(callbacks)

            logger.info(
                "Training %s for %d additional timesteps with %d environment(s).",
                training.algorithm.upper(),
                training.timesteps,
                training.n_envs,
            )
            model.learn(
                total_timesteps=training.timesteps,
                callback=callback_list,
                log_interval=100,
                reset_num_timesteps=self.resume_from is None,
                progress_bar=training.progress_bar,
            )
            result.final_timesteps = int(model.num_timesteps)
            result.metrics = control.metrics()
            if control.stopped:
                stopped_path = self.model_path.with_name(self.model_path.stem + "_stopped.zip")
                save_model_bundle(
                    model,
                    stopped_path,
                    config,
                    algorithm=training.algorithm,
                    policy=policy_label,
                    run_id=self.run_id,
                    extra={
                        "stopped": True,
                        "num_timesteps": result.final_timesteps,
                        "training_metrics": result.metrics,
                    },
                )
                result.model_path = stopped_path
                result.stopped = True
                result.best_model_path = (
                    eval_callback.best_model_path_out if eval_callback is not None else None
                )
                self._status.finish(
                    "stopped",
                    **result.metrics,
                    model_path=str(stopped_path),
                    stopped_model_path=str(stopped_path),
                    message="Training stopped early. Load the stopped model to inspect or continue.",
                )
            else:
                save_model_bundle(
                    model,
                    self.model_path,
                    config,
                    algorithm=training.algorithm,
                    policy=policy_label,
                    run_id=self.run_id,
                    extra={
                        "completed": True,
                        "num_timesteps": result.final_timesteps,
                        "training_metrics": result.metrics,
                    },
                )
                best_path = None
                if eval_callback is not None:
                    best_path = eval_callback.best_model_path_out
                if best_path is None:
                    candidate = self.checkpoint_dir / "best" / "best_model.zip"
                    if candidate.exists():
                        best_path = candidate
                result.model_path = self.model_path
                result.best_model_path = best_path
                result.completed = True
                self._status.finish(
                    "completed",
                    **result.metrics,
                    model_path=str(self.model_path),
                    best_model_path=str(best_path) if best_path else None,
                    message="Training completed.",
                )
            logger.info("Final model: %s", result.model_path)
            if result.best_model_path:
                logger.info("Best evaluation model: %s", result.best_model_path)
            atomic_write_json(run_dir / "result.json", result.as_dict())
            return result
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving an interrupted model where possible.")
            result.interrupted = True
            if model is not None:
                interrupted_path = self.model_path.with_name(
                    self.model_path.stem + "_interrupted.zip"
                )
                try:
                    save_model_bundle(
                        model,
                        interrupted_path,
                        config,
                        algorithm=training.algorithm,
                        policy=policy_label,
                        run_id=self.run_id,
                        extra={
                            "interrupted": True,
                            "num_timesteps": int(model.num_timesteps),
                            "training_metrics": control.metrics() if control else {},
                        },
                    )
                    result.model_path = interrupted_path
                    result.final_timesteps = int(model.num_timesteps)
                    result.metrics = control.metrics() if control else {}
                    self._status.finish(
                        "interrupted",
                        model_path=str(interrupted_path),
                        message="Training interrupted by Ctrl+C. Partial progress was saved.",
                        **result.metrics,
                    )
                except Exception as exc:
                    logger.error("Could not save interrupted model: %s", exc)
                    result.error = f"Interrupted and failed to save: {exc}"
                    self._status.finish("interrupted", error=result.error)
            else:
                self._status.finish("interrupted", error="Interrupted before a model was created.")
            atomic_write_json(run_dir / "result.json", result.as_dict())
            return result
        except Exception as exc:
            message = str(exc)
            logger.exception("Training failed: %s", message)
            self._status.finish("error", error=message)
            result.error = message
            atomic_write_json(run_dir / "result.json", result.as_dict())
            if isinstance(exc, GameMindError):
                raise
            raise TrainingError(
                f"Training failed: {message}. See {self.status_path} and "
                f"{run_dir / 'train.log'} for details."
            ) from exc
        finally:
            if eval_env is not None:
                try:
                    eval_env.close()
                except Exception:
                    logger.debug("Error closing eval environment", exc_info=True)
            if train_env is not None:
                try:
                    train_env.close()
                except Exception:
                    logger.debug("Error closing training environment", exc_info=True)
            close_file_handlers()
