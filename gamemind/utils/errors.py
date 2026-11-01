"""Framework exception types.

Using specific exceptions makes failures easier to test and gives CLI users a
clearer message than a bare ``AssertionError`` from a deep learning library.
"""

from __future__ import annotations


class GameMindError(Exception):
    """Base class for expected GameMind AI errors."""


class ConfigError(GameMindError, ValueError):
    """A configuration file or CLI value is invalid."""


class AdapterError(GameMindError, RuntimeError):
    """A game adapter could not connect or interact with its game."""


class InputBackendError(AdapterError):
    """A keyboard/mouse input operation failed or focus was lost."""


class ObservationError(GameMindError, ValueError):
    """An observation had the wrong type, shape, dtype, or value range."""


class RewardError(GameMindError, RuntimeError):
    """A reward component was misconfigured or could not be evaluated."""


class TrainingError(GameMindError, RuntimeError):
    """Training could not start or could not continue safely."""


class ModelCompatibilityError(GameMindError, RuntimeError):
    """A saved model does not match the environment it is being loaded into."""


class LibraryError(GameMindError, ValueError):
    """A game/level/session record is invalid or would damage existing data."""


class WorkerError(GameMindError, RuntimeError):
    """A background training worker could not be started or controlled."""
