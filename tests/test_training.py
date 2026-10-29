from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gamemind.config import load_game_config
from gamemind.envs import build_env
from gamemind.evaluation import run_episodes
from gamemind.training import Trainer
from gamemind.training.model_io import (
    assert_model_env_compatible,
    config_for_model,
    load_model_bundle,
    metadata_path_for,
    resolve_model_path,
)
from gamemind.utils.errors import ModelCompatibilityError
from gamemind.utils.serialization import read_json


def test_small_training_run_saves_model_and_status(trained_cartpole):
    config, result, root = trained_cartpole
    assert result.completed is True
    assert result.final_timesteps >= config.training.timesteps
    assert result.model_path is not None and result.model_path.exists()
    assert metadata_path_for(result.model_path).exists()
    assert result.status_path.exists()
    status = read_json(result.status_path, default={})
    assert status["state"] == "completed"
    assert status["timesteps"] >= config.training.timesteps

    metadata = read_json(metadata_path_for(result.model_path), default={})
    assert metadata["game"] == "cartpole"
    assert metadata["algorithm"] == "ppo"
    assert metadata["num_timesteps"] >= config.training.timesteps
    assert metadata["observation_space"]["shape"] == [4]
    assert metadata["config"]["training"]["timesteps"] == 64
    assert list(result.checkpoint_dir.glob("checkpoint_*.zip"))


def test_model_evaluation_does_not_require_training(trained_cartpole):
    config, result, root = trained_cartpole
    evaluation = run_episodes(
        config,
        result.model_path,
        episodes=1,
        deterministic=True,
        render=False,
        seed=123,
        device="cpu",
    )
    assert len(evaluation.episodes) == 1
    assert evaluation.episodes[0].length > 0
    assert evaluation.mean_reward == evaluation.episodes[0].reward


def test_model_space_mismatch_is_rejected(trained_cartpole, project_root):
    _, result, _ = trained_cartpole
    model, metadata, algorithm = load_model_bundle(result.model_path, device="cpu")
    assert algorithm == "ppo"
    assert metadata["game"] == "cartpole"
    pixel_config = load_game_config(project_root / "configs" / "example_game.yaml")
    pixel_env = build_env(
        replace(
            pixel_config,
            observation=replace(pixel_config.observation, width=32, height=32, stack=2),
        )
    )
    try:
        with pytest.raises(ModelCompatibilityError, match="observation space"):
            assert_model_env_compatible(model, pixel_env)
    finally:
        pixel_env.close()


def test_resume_training_from_checkpoint(trained_cartpole):
    config, result, root = trained_cartpole
    previous_steps = result.final_timesteps
    resume_config = replace(
        config,
        training=replace(
            config.training,
            timesteps=32,
            n_steps=32,
            batch_size=32,
            n_envs=1,
            checkpoint_freq=0,
            eval_freq=0,
            save_best=False,
            tensorboard=False,
            verbose=0,
            device="cpu",
        ),
    )
    trainer = Trainer(
        resume_config,
        model_path=root / "models" / "cartpole_resumed.zip",
        checkpoint_dir=root / "checkpoints",
        log_dir=root / "logs",
        run_name="pytest-resume",
        resume_from=result.model_path,
    )
    resumed = trainer.run()
    assert resumed.completed is True
    assert resumed.model_path is not None and resumed.model_path.exists()
    assert resumed.final_timesteps > previous_steps
    assert resumed.final_timesteps == previous_steps + 32
    assert metadata_path_for(resumed.model_path).exists()


def test_resolve_model_path_without_zip_suffix(trained_cartpole):
    _, result, _ = trained_cartpole
    assert result.model_path is not None
    without_suffix = result.model_path.with_suffix("")
    assert resolve_model_path(without_suffix) == result.model_path


def test_status_file_is_valid_json(trained_cartpole):
    _, result, _ = trained_cartpole
    payload = read_json(result.status_path, default=None)
    assert isinstance(payload, dict)
    assert payload["run_id"] == result.run_id
    assert "progress" in payload


def test_metadata_config_recreates_matching_environment(trained_cartpole):
    _, result, _ = trained_cartpole
    restored_config, metadata = config_for_model(result.model_path)
    assert restored_config.name == "cartpole"
    assert restored_config.training.timesteps == 64
    env = build_env(restored_config)
    try:
        model, metadata2, algorithm = load_model_bundle(result.model_path, env=env)
        assert algorithm == "ppo"
        assert metadata2["game"] == "cartpole"
        observation, _ = env.reset(seed=2)
        action, _ = model.predict(observation, deterministic=True)
        assert env.action_space.contains(int(action))
    finally:
        env.close()


def test_crop_is_applied_before_resize(project_root):
    from gamemind.config import ObservationConfig
    from gamemind.observations.processing import ObservationProcessor

    processor = ObservationProcessor(
        ObservationConfig(
            mode="grayscale",
            width=32,
            height=32,
            stack=1,
            crop=(0.0, 0.25, 0.0, 0.0),
        )
    )
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :10] = 255  # left margin should be cropped away
    result = processor.process(frame)
    assert result.shape == (1, 32, 32)
    assert result.max() == 0
