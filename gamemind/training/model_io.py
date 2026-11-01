"""Model persistence with sidecar metadata and environment compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from gamemind.algorithms.factory import get_algorithm_class, maskable_ppo_available
from gamemind.config import GameConfig, load_game_config
from gamemind.utils.errors import ConfigError, ModelCompatibilityError, TrainingError
from gamemind.utils.logging import get_logger
from gamemind.utils.paths import PROJECT_ROOT, resolve_project_path, safe_name
from gamemind.utils.serialization import (
    atomic_write_json,
    git_metadata,
    library_versions,
    read_json,
    space_to_dict,
    utc_now_iso,
)

logger = get_logger("models")


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a model path, accepting an omitted ``.zip`` extension."""
    resolved = resolve_project_path(path)
    if resolved.is_dir():
        raise ModelCompatibilityError(f"Model path is a directory, not a .zip file: {resolved}")
    if resolved.suffix == "" and not resolved.exists():
        with_zip = resolved.with_suffix(".zip")
        if with_zip.exists():
            return with_zip
    if not resolved.exists():
        raise ModelCompatibilityError(
            f"Model file not found: {resolved}. Train one first, for example:\n"
            "  python scripts/train.py --game example_game --timesteps 50000"
        )
    return resolved


def metadata_path_for(model_path: str | Path) -> Path:
    """Sidecar path for a model: ``models/foo.zip`` -> ``models/foo.metadata.json``."""
    path = Path(model_path)
    if path.suffix.lower() == ".zip":
        return path.with_suffix(".metadata.json")
    return path.with_name(path.name + ".metadata.json")


def save_config_snapshot(config: GameConfig, path: Path) -> Path:
    """Write the fully resolved config used for a run (not the original with comments)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False)
    return path


def write_model_metadata(
    model_path: str | Path,
    model: Any,
    config: GameConfig,
    *,
    algorithm: str,
    policy: str,
    run_id: str | None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the sidecar metadata that makes a saved model reproducible."""
    model_path = Path(model_path)
    config_path = None
    if config.source_path:
        try:
            config_path = str(Path(config.source_path).resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            config_path = str(config.source_path)
    payload: dict[str, Any] = {
        "format_version": 1,
        "created_at": utc_now_iso(),
        "game": config.name,
        "adapter": config.adapter,
        "algorithm": algorithm,
        "policy": policy,
        "run_id": run_id,
        "num_timesteps": int(getattr(model, "num_timesteps", 0) or 0),
        "config_path": config_path,
        "config": config.as_dict(),
        "observation_space": space_to_dict(getattr(model, "observation_space", None)),
        "action_space": space_to_dict(getattr(model, "action_space", None)),
        "library_versions": library_versions(),
        "git": git_metadata(),
    }
    if extra:
        payload.update(extra)
    path = metadata_path_for(model_path)
    atomic_write_json(path, payload)
    return path


def save_model_bundle(
    model: Any,
    model_path: str | Path,
    config: GameConfig,
    *,
    algorithm: str,
    policy: str,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a Stable-Baselines3 model and its metadata atomically enough for users."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    try:
        model.save(str(path))
    except Exception as exc:
        raise TrainingError(f"Could not save model to {path}: {exc}") from exc
    write_model_metadata(
        path, model, config, algorithm=algorithm, policy=policy, run_id=run_id, extra=extra
    )
    return path


def _spaces_equal(left: gym.Space | None, right: gym.Space | None) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, spaces.Box) and isinstance(right, spaces.Box):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return bool(np.array_equal(left.low, right.low) and np.array_equal(left.high, right.high))
    if isinstance(left, spaces.Discrete) and isinstance(right, spaces.Discrete):
        return bool(left.n == right.n)
    if isinstance(left, spaces.MultiDiscrete) and isinstance(right, spaces.MultiDiscrete):
        return bool(np.array_equal(left.nvec, right.nvec))
    if isinstance(left, spaces.MultiBinary) and isinstance(right, spaces.MultiBinary):
        return left.n == right.n
    if isinstance(left, spaces.Dict) and isinstance(right, spaces.Dict):
        if set(left.spaces) != set(right.spaces):
            return False
        return all(_spaces_equal(left.spaces[key], right.spaces[key]) for key in left.spaces)
    if isinstance(left, spaces.Tuple) and isinstance(right, spaces.Tuple):
        return len(left.spaces) == len(right.spaces) and all(
            _spaces_equal(a, b) for a, b in zip(left.spaces, right.spaces)
        )
    return left == right


def assert_model_env_compatible(model: Any, env: gym.Env) -> None:
    """Fail before evaluation if a model was trained for a different observation/action layout."""
    saved_obs = getattr(model, "observation_space", None)
    saved_act = getattr(model, "action_space", None)
    if not _spaces_equal(saved_obs, env.observation_space):
        raise ModelCompatibilityError(
            "The model's observation space does not match the current environment.\n"
            f"  saved:   {saved_obs}\n"
            f"  current: {env.observation_space}\n"
            "Use the config saved with the model (model.metadata.json) or retrain with "
            "the current observation settings."
        )
    if not _spaces_equal(saved_act, env.action_space):
        raise ModelCompatibilityError(
            "The model's action space does not match the current environment.\n"
            f"  saved:   {saved_act}\n"
            f"  current: {env.action_space}\n"
            "Use the same action bindings/configuration that were used for training."
        )


def load_model_bundle(
    model_path: str | Path,
    *,
    algorithm: str | None = None,
    env: gym.Env | None = None,
    device: str = "auto",
) -> tuple[Any, dict[str, Any], str]:
    """Load a model, its metadata, and the resolved algorithm name.

    The returned model is never trained by this function. If ``env`` is supplied,
    observation and action spaces are validated before use.
    """
    path = resolve_model_path(model_path)
    metadata = read_json(metadata_path_for(path), default={}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    meta_algorithm = str(metadata.get("algorithm", "")).strip().lower() or None
    requested = str(algorithm).strip().lower() if algorithm else None
    if requested and meta_algorithm and requested != meta_algorithm:
        raise ModelCompatibilityError(
            f"Requested algorithm {requested!r} but {path.name} records {meta_algorithm!r}. "
            "Use the algorithm recorded in the model metadata or retrain."
        )
    if requested:
        candidates = [requested]
    elif meta_algorithm:
        candidates = [meta_algorithm]
    else:
        candidates = ["ppo", "a2c"] + (["maskable_ppo"] if maskable_ppo_available() else [])

    errors: list[str] = []
    for candidate in candidates:
        try:
            cls = get_algorithm_class(candidate)
            model = cls.load(str(path), env=env, device=device)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if env is not None:
            assert_model_env_compatible(model, env)
        logger.info("Loaded %s model from %s", candidate, path)
        return model, metadata, candidate
    raise ModelCompatibilityError(
        f"Could not load model {path}. Tried: {', '.join(candidates)}.\n"
        + "\n".join(f"  - {error}" for error in errors)
    )


def assert_observation_mode_compatible(metadata: Mapping[str, Any], config: GameConfig) -> None:
    """Reject switching between edge maps and ordinary pixels on existing models."""
    snapshot = metadata.get("config")
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get("observation"), Mapping):
        saved_mode = snapshot["observation"].get("mode")
        if saved_mode != config.observation.mode and "canny" in {saved_mode, config.observation.mode}:
            raise ModelCompatibilityError(
                "Canny and non-Canny observations are not interchangeable, even when their "
                "shapes match. Use the saved preprocessing or train a separate model."
            )


def config_for_model(
    model_path: str | Path,
    *,
    config_path: str | Path | None = None,
    expected_game: str | None = None,
) -> tuple[GameConfig, dict[str, Any]]:
    """Reconstruct the training config saved with a model, unless explicitly overridden."""
    path = resolve_model_path(model_path)
    metadata = read_json(metadata_path_for(path), default={}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if config_path is not None:
        config = load_game_config(config_path)
    elif isinstance(metadata.get("config"), Mapping):
        try:
            config = GameConfig.from_dict(metadata["config"])
        except Exception as exc:
            raise ConfigError(
                f"The config snapshot in {metadata_path_for(path)} is invalid: {exc}. "
                "Pass --config with the matching YAML file."
            ) from exc
        if metadata.get("config_path"):
            config.source_path = metadata["config_path"]
    else:
        raise ConfigError(
            f"No config snapshot found for {path}. Pass --config path/to/game.yaml "
            "so the environment and preprocessing can be recreated."
        )
    assert_observation_mode_compatible(metadata, config)
    meta_game = metadata.get("game")
    if expected_game and meta_game and safe_name(str(expected_game)) != safe_name(str(meta_game)):
        raise ModelCompatibilityError(
            f"Model was trained for game {meta_game!r}, but --game {expected_game!r} was requested. "
            "Use the matching game or omit --game to use the saved config."
        )
    if expected_game and not meta_game and safe_name(str(config.name)) != safe_name(str(expected_game)):
        raise ModelCompatibilityError(
            f"The provided config describes {config.name!r}, not requested game {expected_game!r}."
        )
    return config, metadata


def model_summary(model_path: str | Path) -> dict[str, Any]:
    """Read metadata for dashboards and CLI listings."""
    path = resolve_model_path(model_path)
    metadata = read_json(metadata_path_for(path), default={}) or {}
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "game": metadata.get("game"),
        "algorithm": metadata.get("algorithm"),
        "policy": metadata.get("policy"),
        "num_timesteps": metadata.get("num_timesteps"),
        "created_at": metadata.get("created_at"),
        "observation_space": metadata.get("observation_space"),
        "action_space": metadata.get("action_space"),
    }
