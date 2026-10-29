from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gamemind.adapters.desktop_adapter import DesktopGameAdapter
from gamemind.adapters.registry import available_adapters, create_adapter
from gamemind.config import GameConfig, RewardConfig, load_game_config
from gamemind.envs import build_env
from gamemind.observations import ObservationValidator, PixelObservationWrapper
from gamemind.utils.errors import AdapterError, InputBackendError
from tests.support import FakeCapture, FakeInput, make_desktop_config


def test_builtin_adapters_are_registered():
    assert {"gymnasium", "example_game", "desktop"}.issubset(set(available_adapters()))


def test_example_game_wrapped_spaces_and_determinism(example_config):
    env = build_env(example_config)
    try:
        assert env.action_space.n == 5
        assert env.observation_space.shape == (2, 48, 48)
        assert env.observation_space.dtype == np.uint8
        first, info = env.reset(seed=5)
        second, _ = env.reset(seed=5)
        assert np.array_equal(first, second)
        assert first.shape == (2, 48, 48)
        observation, reward, terminated, truncated, step_info = env.step(2)
        assert observation.shape == (2, 48, 48)
        assert np.isfinite(reward)
        assert isinstance(step_info, dict)
        with pytest.raises(ValueError, match="outside the valid range"):
            env.step(99)
    finally:
        env.close()


def test_example_game_raw_adapter_passes_gym_checker(example_config):
    adapter = create_adapter(example_config)
    adapter.connect()
    try:
        check_env(adapter, skip_render_check=True, skip_close_check=True)
        observation, _ = adapter.reset(seed=9)
        assert observation.shape[2] == 3
        assert observation.dtype == np.uint8
    finally:
        adapter.close()


def test_example_game_time_limit_and_post_done_guard(example_config):
    config = replace(
        example_config,
        observation=replace(example_config.observation, stack=1, width=32, height=32),
        episode=replace(example_config.episode, max_steps=3),
        reward=RewardConfig(components=({"type": "constant", "value": 0.0},)),
    )
    env = build_env(config)
    try:
        env.reset(seed=1)
        for index in range(3):
            observation, reward, terminated, truncated, info = env.step(0)
            assert observation.shape == (1, 32, 32)
            if index < 2:
                assert not truncated
        assert truncated is True
        with pytest.raises(AdapterError, match="reset"):
            env.step(0)
    finally:
        env.close()


def test_example_game_target_reward(example_config):
    config = replace(
        example_config,
        observation=replace(example_config.observation, stack=1),
        reward=RewardConfig(
            components=(
                {"type": "signal", "key": "target_collected", "scale": 10.0, "name": "target"},
                {"type": "constant", "value": -0.01, "name": "time"},
            )
        ),
    )
    env = build_env(config)
    try:
        env.reset(seed=4)
        adapter = env.unwrapped
        adapter.game.agent = (0, 0)
        adapter.game.target = (1, 0)
        adapter.game.obstacles = set()
        adapter.game.frame = adapter.game.render_frame()
        observation, reward, terminated, truncated, info = env.step(2)  # right
        assert reward == pytest.approx(9.99)
        assert info["targets_collected"] == 1
        assert observation.shape == (1, 48, 48)
    finally:
        env.close()


def test_example_game_fixed_layout_and_success(example_config):
    config = replace(
        example_config,
        params={
            "grid_size": 5,
            "cell_size": 32,
            "n_obstacles": 0,
            "max_steps": 20,
            "start": [2, 4],
            "target": [2, 2],
            "obstacles": [[0, 0]],
            "show_compass": True,
            "terminate_on_collect": True,
        },
        observation=replace(example_config.observation, stack=1),
        episode=replace(example_config.episode, max_steps=20),
        reward=RewardConfig(
            components=(
                {"type": "signal", "key": "target_collected", "scale": 10.0, "name": "target"},
                {"type": "constant", "value": -0.01, "name": "time"},
            )
        ),
    )
    env = build_env(config)
    try:
        env.reset(seed=0)
        adapter = env.unwrapped
        assert adapter.game.agent == (2, 4)
        assert adapter.game.target == (2, 2)
        assert adapter.game.obstacles == {(0, 0)}
        observation, reward, terminated, truncated, info = env.step(3)  # up
        assert observation.shape == (1, 48, 48)
        assert terminated is False
        assert info["success"] is False
        observation, reward, terminated, truncated, info = env.step(3)  # up onto target
        assert reward == pytest.approx(9.99)
        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        with pytest.raises(AdapterError, match="reset"):
            env.step(0)
    finally:
        env.close()


def test_gymnasium_cartpole_state(project_root):
    config = load_game_config(project_root / "configs" / "cartpole.yaml")
    config = replace(config, episode=replace(config.episode, max_steps=500))
    env = build_env(config)
    try:
        observation, info = env.reset(seed=1)
        assert observation.shape == (4,)
        next_observation, reward, terminated, truncated, info = env.step(0)
        assert next_observation.shape == (4,)
        assert reward == pytest.approx(1.0)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_gymnasium_pixel_adapter(project_root):
    config = GameConfig.from_dict(
        {
            "game": {
                "name": "cartpole_pixels_test",
                "adapter": "gymnasium",
                "params": {"env_id": "CartPole-v1"},
            },
            "observation": {"mode": "grayscale", "width": 48, "height": 48, "stack": 2},
            "reward": {"components": [{"type": "native"}]},
            "episode": {"max_steps": 20},
            "training": {"n_envs": 1, "n_steps": 8, "batch_size": 8},
        }
    )
    env = build_env(config)
    try:
        observation, _ = env.reset(seed=3)
        assert observation.shape == (2, 48, 48)
        assert observation.dtype == np.uint8
        observation, reward, terminated, truncated, info = env.step(0)
        assert observation.shape == (2, 48, 48)
        assert np.isfinite(reward)
    finally:
        env.close()


def test_desktop_adapter_with_fake_backends():
    config = make_desktop_config()
    capture = FakeCapture()
    inputs = FakeInput()
    adapter = DesktopGameAdapter(config, capture_backend=capture, input_backend=inputs)
    adapter.connect()
    try:
        assert adapter.observation_space.shape == (64, 96, 3)
        observation, info = adapter.reset(seed=0)
        assert observation.shape == (64, 96, 3)
        assert ("down", "r") in inputs.events
        assert inputs.held == set()
        observation, reward, terminated, truncated, info = adapter.step(1)
        assert reward == 0.0
        assert ("down", "a") in inputs.events
        assert ("release", "a") in inputs.events
        assert inputs.held == set()
        assert info["action_name"] == "left"
        assert inputs.focus_calls >= 1
        adapter.step(0)  # step 2
        observation, reward, terminated, truncated, info = adapter.step(0)  # step 3
        assert truncated is True
    finally:
        adapter.close()
    assert inputs.closed is True
    assert inputs.held == set()


def test_desktop_adapter_pixel_wrapper_pipeline():
    config = make_desktop_config()
    observation_config = replace(config.observation, width=32, height=32, stack=2)
    adapter = DesktopGameAdapter(
        replace(config, observation=observation_config),
        capture_backend=FakeCapture(),
        input_backend=FakeInput(),
    )
    adapter.connect()
    env = ObservationValidator(PixelObservationWrapper(adapter, observation_config))
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (1, 32, 32)
        assert observation.dtype == np.uint8
    finally:
        env.close()


def test_desktop_focus_loss_releases_keys():
    config = make_desktop_config()
    capture = FakeCapture()
    inputs = FakeInput()
    inputs.fail_focus = True
    adapter = DesktopGameAdapter(config, capture_backend=capture, input_backend=inputs)
    adapter.connect()
    try:
        with pytest.raises(InputBackendError, match="focus"):
            adapter.reset(seed=0)
        assert inputs.held == set()
    finally:
        adapter.close()


def test_desktop_template_termination(tmp_path):
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[4:20, 4:20] = 255
    template = frame[4:20, 4:20]
    template_path = tmp_path / "game_over.png"
    assert cv2.imwrite(str(template_path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
    config = make_desktop_config(
        terminate_on={
            "type": "template",
            "template": str(template_path),
            "region": [0, 0, 32, 32],
            "threshold": 0.9,
            "consecutive": 1,
        }
    )
    adapter = DesktopGameAdapter(
        config,
        capture_backend=FakeCapture([frame]),
        input_backend=FakeInput(),
    )
    adapter.connect()
    try:
        adapter.reset(seed=0)
        observation, reward, terminated, truncated, info = adapter.step(0)
        assert terminated is True
        assert truncated is False
    finally:
        adapter.close()
