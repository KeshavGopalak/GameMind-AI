from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from gamemind.config import (
    ActionsConfig,
    GameConfig,
    RewardConfig,
    apply_config_overrides,
    find_config_for_game,
    load_game_config,
)
from gamemind.utils.errors import ConfigError


def test_all_shipped_configs_parse(project_root):
    names = [
        "default",
        "cartpole",
        "lunarlander",
        "example_game",
        "example_game_random",
        "desktop_game.example",
        "geometry_dash",
    ]
    for name in names:
        config = load_game_config(project_root / "configs" / f"{name}.yaml")
        assert config.name
        assert config.adapter
        assert config.training.timesteps > 0


def test_default_config_is_example_game(project_root):
    config = load_game_config(project_root / "configs" / "default.yaml")
    assert config.name == "example_game"
    assert config.observation.mode == "grayscale"


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        GameConfig.from_dict(
            {
                "game": {"name": "x", "adapter": "gymnasium"},
                "observaton": {"mode": "state"},
            }
        )


def test_state_observation_cannot_stack():
    with pytest.raises(ConfigError, match="stack must be 1"):
        GameConfig.from_dict(
            {
                "game": {"name": "x", "adapter": "gymnasium"},
                "observation": {"mode": "state", "stack": 4},
            }
        )


def test_unsupported_resolution_is_rejected():
    with pytest.raises(ConfigError, match="width"):
        GameConfig.from_dict(
            {
                "game": {"name": "x", "adapter": "gymnasium"},
                "observation": {"mode": "grayscale", "width": 8, "height": 84},
            }
        )


def test_invalid_ppo_batch_size_is_rejected():
    with pytest.raises(ConfigError, match="batch_size must divide"):
        GameConfig.from_dict(
            {
                "game": {"name": "x", "adapter": "gymnasium"},
                "training": {"n_steps": 32, "n_envs": 1, "batch_size": 10},
            }
        )


def test_a2c_does_not_apply_ppo_batch_divisibility():
    config = GameConfig.from_dict(
        {
            "game": {"name": "x", "adapter": "gymnasium"},
            "training": {"algorithm": "a2c", "n_steps": 32, "n_envs": 1, "batch_size": 10},
        }
    )
    assert config.training.algorithm == "a2c"


def test_duplicate_action_names_are_rejected():
    with pytest.raises(ConfigError, match="Duplicate action"):
        ActionsConfig.from_dict(
            {
                "bindings": [
                    {"name": "left", "keys": ["a"]},
                    {"name": "left", "keys": ["d"]},
                ]
            }
        )


def test_invalid_action_duration_is_rejected():
    with pytest.raises(ConfigError, match="duration"):
        ActionsConfig.from_dict({"duration": 2.0, "max_duration": 0.5})


def test_unknown_action_binding_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        ActionsConfig.from_dict({"bindings": [{"name": "left", "keys": ["a"], "typo": 1}]})


def test_reward_component_requires_type():
    with pytest.raises(ConfigError, match="requires a 'type'"):
        RewardConfig.from_dict({"components": [{"value": 1.0}]})


def test_config_roundtrip():
    original = GameConfig.from_dict(
        {
            "game": {"name": "x", "adapter": "example_game"},
            "observation": {"mode": "grayscale", "width": 64, "height": 48, "stack": 2},
            "actions": {"bindings": [{"name": "noop", "params": {"move": [0, 0]}}]},
            "episode": {"max_steps": 12},
            "training": {
                "timesteps": 123,
                "n_envs": 1,
                "n_steps": 16,
                "batch_size": 16,
            },
        }
    )
    restored = GameConfig.from_dict(original.as_dict())
    assert restored.name == original.name
    assert restored.observation.width == 64
    assert restored.training.timesteps == 123
    assert restored.actions.bindings[0].name == "noop"


def test_overrides_switch_state_mode_and_reset_stack(project_root):
    config = load_game_config(project_root / "configs" / "example_game.yaml")
    updated = apply_config_overrides(
        config,
        observation_mode="state",
        timesteps=1234,
        seed=7,
        stack=1,
        algorithm="a2c",
    )
    assert updated.observation.mode == "state"
    assert updated.observation.stack == 1
    assert updated.training.timesteps == 1234
    assert updated.training.algorithm == "a2c"
    assert updated.training.seed == 7
    assert updated.seed == 7


def test_find_config_for_game(project_root):
    path = find_config_for_game("cartpole")
    assert path.name == "cartpole.yaml"
    with pytest.raises(ConfigError, match="No configuration found"):
        find_config_for_game("not_a_real_game")


def test_config_path_missing_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_game_config("configs/does_not_exist.yaml")
