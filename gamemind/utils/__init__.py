"""Small, dependency-light helpers shared across the framework."""

from gamemind.utils.errors import (
    AdapterError,
    ConfigError,
    GameMindError,
    InputBackendError,
    ModelCompatibilityError,
    ObservationError,
    RewardError,
    TrainingError,
)
from gamemind.utils.paths import PROJECT_ROOT, ensure_dir, resolve_project_path

__all__ = [
    "PROJECT_ROOT",
    "AdapterError",
    "ConfigError",
    "GameMindError",
    "InputBackendError",
    "ModelCompatibilityError",
    "ObservationError",
    "RewardError",
    "TrainingError",
    "ensure_dir",
    "resolve_project_path",
]
