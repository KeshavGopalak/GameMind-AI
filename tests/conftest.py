"""Shared pytest setup. Forces a headless SDL driver for reproducible CI runs."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from gamemind.config import GameConfig, load_game_config  # noqa: E402
from gamemind.utils.logging import close_file_handlers  # noqa: E402


@pytest.fixture(autouse=True)
def _release_log_files():
    """Windows keeps log files locked; close them so tmp_path cleanup works."""
    yield
    close_file_handlers()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


def _small_example_config() -> GameConfig:
    config = load_game_config(ROOT / "configs" / "example_game.yaml")
    return replace(
        config,
        observation=replace(config.observation, width=48, height=48, stack=2),
        episode=replace(config.episode, max_steps=8),
        training=replace(
            config.training,
            timesteps=64,
            n_envs=1,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            checkpoint_freq=0,
            eval_freq=0,
            save_best=False,
            tensorboard=False,
            verbose=0,
            device="cpu",
        ),
    )


@pytest.fixture
def example_config() -> GameConfig:
    return _small_example_config()


@pytest.fixture(scope="session")
def trained_cartpole(tmp_path_factory: pytest.TempPathFactory):
    """A real small PPO training run, reused by persistence/evaluation tests."""
    from gamemind.training import Trainer

    root = tmp_path_factory.mktemp("gamemind_cartpole")
    raw = load_game_config(ROOT / "configs" / "cartpole.yaml")
    config = replace(
        raw,
        training=replace(
            raw.training,
            timesteps=64,
            n_envs=1,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            checkpoint_freq=32,
            eval_freq=0,
            save_best=False,
            tensorboard=False,
            verbose=0,
            device="cpu",
            seed=11,
        ),
    )
    trainer = Trainer(
        config,
        model_path=root / "models" / "cartpole.zip",
        checkpoint_dir=root / "checkpoints",
        log_dir=root / "logs",
        run_name="pytest-smoke",
    )
    result = trainer.run()
    return config, result, root
