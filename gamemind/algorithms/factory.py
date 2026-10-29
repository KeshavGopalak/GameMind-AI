"""Stable-Baselines3 algorithm and policy selection."""

from __future__ import annotations

import inspect
from typing import Any, Type

import gymnasium as gym
from gymnasium import spaces

from gamemind.config import GameConfig, TrainingConfig
from gamemind.utils.device import resolve_device
from gamemind.utils.errors import ConfigError, TrainingError
from gamemind.utils.logging import get_logger

logger = get_logger("algorithms")

_POLICY_ALIASES = {"MlpPolicy", "CnnPolicy", "MultiInputPolicy"}


def maskable_ppo_available() -> bool:
    """Whether the optional ``sb3-contrib`` package can be imported."""
    try:
        import sb3_contrib  # noqa: F401

        return True
    except ImportError:
        return False


def algorithm_display_name(name: str) -> str:
    return {"ppo": "PPO", "a2c": "A2C", "maskable_ppo": "MaskablePPO"}.get(
        name.lower(), name
    )


def get_algorithm_class(name: str) -> Type[Any]:
    """Resolve a supported Stable-Baselines3 algorithm class."""
    normalized = str(name).strip().lower()
    if normalized == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if normalized == "a2c":
        from stable_baselines3 import A2C

        return A2C
    if normalized == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise TrainingError(
                "MaskablePPO requires the optional sb3-contrib package. Install it with:\n"
                "  pip install sb3-contrib\n"
                "Use algorithm: ppo unless your environment implements action_masks()."
            ) from exc
        return MaskablePPO
    raise ConfigError(
        f"Unsupported algorithm {name!r}. Supported: ppo, a2c, maskable_ppo (optional)."
    )


def select_policy(config: GameConfig, env: gym.Env) -> str:
    """Choose a Stable-Baselines3 policy alias, validating explicit overrides."""
    requested = config.training.policy
    space = env.observation_space
    is_pixel = config.observation.is_pixel

    if requested is not None:
        if ":" in requested:
            raise ConfigError(
                "Custom policy classes are not supported through a string path yet. "
                "Use MlpPolicy/CnnPolicy/MultiInputPolicy, or set training.extra_params."
                "policy_kwargs.features_extractor_class for a custom extractor."
            )
        if requested not in _POLICY_ALIASES:
            raise ConfigError(
                f"Unknown training.policy {requested!r}. Choose from "
                f"{sorted(_POLICY_ALIASES)} or leave it null for automatic selection."
            )
        if requested == "CnnPolicy" and not (isinstance(space, spaces.Box) and len(space.shape) == 3):
            raise ConfigError(
                "CnnPolicy requires a 3D image observation space (channels, height, width). "
                f"Current space: {space}. Use observation.mode: grayscale or rgb, or MlpPolicy."
            )
        if requested == "MultiInputPolicy" and not isinstance(space, spaces.Dict):
            raise ConfigError(
                f"MultiInputPolicy requires a Dict observation space, got {type(space).__name__}."
            )
        if requested == "MlpPolicy" and not isinstance(space, spaces.Box):
            raise ConfigError(f"MlpPolicy requires a Box observation space, got {type(space).__name__}.")
        return requested

    if isinstance(space, spaces.Dict):
        return "MultiInputPolicy"
    if is_pixel or (isinstance(space, spaces.Box) and len(space.shape) == 3):
        if not (isinstance(space, spaces.Box) and len(space.shape) == 3):
            raise ConfigError(
                "Pixel observation mode produced a non-image space "
                f"({space}). Check the adapter's get_observation_space()."
            )
        return "CnnPolicy"
    if isinstance(space, spaces.Box):
        return "MlpPolicy"
    raise ConfigError(
        f"No automatic policy supports observation space {space}. "
        "Supported: Box (Mlp/Cnn) and Dict (MultiInput). Implement a custom adapter/policy "
        "for other observation types."
    )


def _managed_kwargs(config: GameConfig, env: gym.Env) -> dict[str, Any]:
    training = config.training
    return {
        "policy": select_policy(config, env),
        "env": env,
        "learning_rate": training.learning_rate,
        "gamma": training.gamma,
        "n_steps": training.n_steps,
        "batch_size": training.batch_size,
        "n_epochs": training.n_epochs,
        "gae_lambda": training.gae_lambda,
        "clip_range": training.clip_range,
        "ent_coef": training.ent_coef,
        "vf_coef": training.vf_coef,
        "max_grad_norm": training.max_grad_norm,
        "stats_window_size": training.stats_window_size,
        "tensorboard_log": None,
        "verbose": training.verbose,
        "seed": training.seed,
        "device": resolve_device(training.device),
        "progress_bar": training.progress_bar,
    }


def _validate_action_space(env: gym.Env) -> None:
    space = env.action_space
    if not isinstance(space, (spaces.Discrete, spaces.Box)):
        raise ConfigError(
            "The built-in PPO/A2C configurations support Discrete and Box action spaces. "
            f"Got {space}. Extend the adapter and pass algorithm-specific arguments in "
            "training.extra_params if you need another action type."
        )


def _validate_maskable_environment(env: gym.Env) -> None:
    candidate = env
    if hasattr(env, "envs") and getattr(env, "envs", None):
        candidate = env.envs[0]
    if not callable(getattr(candidate, "action_masks", None)):
        raise TrainingError(
            "MaskablePPO requires the environment to expose an action_masks() method. "
            "This adapter does not provide one; use algorithm: ppo instead, or implement "
            "action_masks() in a custom adapter."
        )


def build_model(
    config: GameConfig,
    env: gym.Env,
    *,
    tensorboard_log: str | None,
) -> Any:
    """Construct a new Stable-Baselines3 model with validated configuration."""
    training: TrainingConfig = config.training
    _validate_action_space(env)
    if training.algorithm == "maskable_ppo":
        _validate_maskable_environment(env)
    cls = get_algorithm_class(training.algorithm)
    policy = select_policy(config, env)
    managed = _managed_kwargs(config, env)
    managed["policy"] = policy
    managed["tensorboard_log"] = tensorboard_log if training.tensorboard else None

    try:
        signature = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual Python implementations
        signature = {}
    accepted = set(signature)
    kwargs = {key: value for key, value in managed.items() if key in accepted}

    extras = dict(training.extra_params)
    conflicts = sorted(set(extras) & set(managed))
    if conflicts:
        raise ConfigError(
            "training.extra_params may not override managed settings: "
            f"{', '.join(conflicts)}. Use the dedicated training fields instead."
        )
    unknown = sorted(set(extras) - set(accepted))
    if unknown:
        raise ConfigError(
            f"Unknown extra_params for {algorithm_display_name(training.algorithm)}: "
            f"{', '.join(unknown)}. Accepted constructor arguments: "
            f"{', '.join(sorted(accepted))}."
        )
    kwargs.update(extras)
    logger.info(
        "Building %s with policy %s, seed=%s, device=%s.",
        algorithm_display_name(training.algorithm),
        policy,
        training.seed,
        kwargs.get("device", "auto"),
    )
    try:
        model = cls(**kwargs)
    except Exception as exc:
        raise TrainingError(
            f"Could not construct {algorithm_display_name(training.algorithm)}: {exc}. "
            "Check hyperparameters, policy compatibility, and installed PyTorch."
        ) from exc
    return model
