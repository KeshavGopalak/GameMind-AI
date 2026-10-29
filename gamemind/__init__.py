"""GameMind AI: a modular framework for training reinforcement-learning agents on games.

The public building blocks are deliberately small:

* :class:`gamemind.config.GameConfig` describes an experiment.
* :class:`gamemind.adapters.base.GameAdapter` is the contract every game implements.
* :func:`gamemind.envs.build_env` creates a Gymnasium-compatible environment.
* :class:`gamemind.training.Trainer` trains, checkpoints, and resumes agents.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
