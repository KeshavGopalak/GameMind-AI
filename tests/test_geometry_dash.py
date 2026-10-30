from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

from gamemind.adapters.geometry_dash import (
    GeometryDashAdapter,
    GeometryDashVision,
    WindowInfo,
)
from gamemind.config import GameConfig, load_game_config
from gamemind.observations import ObservationValidator, PixelObservationWrapper
from gamemind.utils.errors import AdapterError, ObservationError
from tests.support import FakeInput


SCREEN_WIDTH = 480
SCREEN_HEIGHT = 270


def frame_with_progress(
    progress_pct: float,
    *,
    present: bool = True,
    fill_color: tuple[int, int, int] = (245, 178, 35),
) -> np.ndarray:
    """A synthetic Geometry Dash-like frame with a white bar and coloured fill."""
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (24, 20, 32)
    if not present:
        return frame
    x, y, width, height = 120, 9, 240, 22
    frame[y : y + height, x : x + width] = (245, 245, 245)
    frame[y + 3 : y + height - 3, x + 3 : x + width - 3] = (105, 24, 30)
    inner_width = width - 16
    filled = int(round(progress_pct / 100.0 * inner_width))
    if filled:
        frame[y + 6 : y + height - 6, x + 8 : x + 8 + filled] = fill_color
    return frame


class LiveCapture:
    width = SCREEN_WIDTH
    height = SCREEN_HEIGHT

    def __init__(self, frame: np.ndarray | None = None) -> None:
        self.frame = frame if frame is not None else frame_with_progress(0.0)
        self.closed = False

    def grab(self) -> np.ndarray:
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


class SequenceCapture(LiveCapture):
    def __init__(self, frames: list[np.ndarray]) -> None:
        super().__init__(frames[0])
        self.frames = frames
        self.index = 0

    def grab(self) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)].copy()
        self.index += 1
        return frame


class FakeWindowLocator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def find(self, title: str) -> WindowInfo:
        self.calls += 1
        if self.fail:
            raise AdapterError("Geometry Dash window not found in test")
        return WindowInfo(title=title, left=0, top=0, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)


def make_geometry_config(**overrides) -> GameConfig:
    params = {
        "region": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
        "monitor": 1,
        "window_title": "Geometry Dash",
        "require_window_focus": True,
        "start_delay": 0.0,
        "reset_keys": ["enter"],
        "resume_keys": ["space"],
        "reset_delay": 0.0,
        "reset_key_duration": 0.0,
        "reset_timeout": 0.2,
        "reset_poll_interval": 0.005,
        "progress_region": None,
        "bar_search_top_fraction": 0.09,
        "bar_min_width_fraction": 0.15,
        "bar_max_height_fraction": 0.08,
        "bar_min_aspect": 5.0,
        "bar_min_area_fraction": 0.0001,
        "bar_left_inset": 8,
        "bar_right_inset": 8,
        "bar_min_fill_columns": 4,
        "progress_fill_hsv_low": [8, 90, 90],
        "progress_fill_hsv_high": [45, 255, 255],
        "death_min_progress": 0.8,
        "reset_progress_max": 0.4,
        "completion_pct": 99.0,
        "completion_frames": 2,
        "lost_gameplay_frames": 2,
        "start_progress_threshold": 5.0,
    }
    params.update(overrides.pop("params", {}))
    actions = overrides.pop(
        "actions",
        {
            "frame_skip": 1,
            "duration": 0.0,
            "max_duration": 0.2,
            "bindings": [
                {"name": "noop", "keys": []},
                {"name": "jump", "keys": ["space"], "duration": 0.0},
                {"name": "hold_jump", "keys": ["space"], "duration": 0.1},
                {"name": "release_jump", "keys": [], "duration": 0.0},
            ],
        },
    )
    payload = {
        "game": {
            "name": "geometry_dash_test",
            "adapter": "geometry_dash",
            "params": params,
        },
        "observation": {"mode": "grayscale", "width": 64, "height": 64, "stack": 1},
        "actions": actions,
        "reward": {
            "components": [
                {"type": "signal", "key": "died", "scale": -5.0, "name": "death"},
                {"type": "signal", "key": "completed", "scale": 50.0, "name": "complete"},
                {"type": "progress", "key": "progress_pct", "scale": 1.0, "reset_on": "died"},
            ]
        },
        "episode": {"max_steps": 8},
        "training": {
            "algorithm": "ppo",
            "timesteps": 64,
            "n_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "checkpoint_freq": 0,
            "eval_freq": 0,
            "save_best": False,
            "tensorboard": False,
            "verbose": 0,
            "device": "cpu",
        },
    }
    payload.update(overrides)
    return GameConfig.from_dict(payload)


def make_adapter(
    *,
    capture: LiveCapture | None = None,
    inputs: FakeInput | None = None,
    config: GameConfig | None = None,
    locator: FakeWindowLocator | None = None,
) -> GeometryDashAdapter:
    return GeometryDashAdapter(
        config or make_geometry_config(),
        capture_backend=capture or LiveCapture(),
        input_backend=inputs or FakeInput(),
        window_locator=locator or FakeWindowLocator(),
    )


def test_geometry_dash_is_registered_and_shipped_config_parses(project_root):
    from gamemind.adapters.registry import available_adapters

    assert "geometry_dash" in available_adapters()
    config = load_game_config(project_root / "configs" / "geometry_dash.yaml")
    assert config.name == "geometry_dash"
    assert config.observation.mode == "grayscale"
    assert config.observation.stack == 4
    assert config.training.n_envs == 1


def test_vision_reads_bar_and_does_not_invent_progress():
    vision = GeometryDashVision()
    empty = vision.read(frame_with_progress(0.0))
    assert empty.found is True
    assert empty.progress_pct == pytest.approx(0.0, abs=0.01)
    quarter = vision.read(frame_with_progress(25.0))
    assert quarter.found is True
    assert 20.0 <= quarter.progress_pct <= 30.0
    full = vision.read(frame_with_progress(100.0))
    assert full.found is True
    assert full.progress_pct >= 95.0
    missing = vision.read(frame_with_progress(50.0, present=False))
    assert missing.found is False
    assert missing.progress_pct == 0.0


def test_vision_calibrates_to_player_colour():
    vision = GeometryDashVision()
    empty = frame_with_progress(0.0, fill_color=(3, 117, 236))
    assert vision.calibrate(empty) is True
    assert vision.read(empty).progress_pct == pytest.approx(0.0, abs=0.01)
    blue = vision.read(frame_with_progress(25.0, fill_color=(3, 117, 236)))
    assert blue.found is True
    assert 20.0 <= blue.progress_pct <= 30.0


def test_vision_rejects_bad_pixels_and_manual_region():
    vision = GeometryDashVision()
    with pytest.raises(ObservationError, match="uint8"):
        vision.read(np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32))
    with pytest.raises(ObservationError, match="region"):
        GeometryDashVision(progress_region=[0, 0, 999, 30]).read(frame_with_progress(10.0))
    manual = GeometryDashVision(progress_region=[120, 9, 240, 22])
    reading = manual.read(frame_with_progress(50.0))
    assert reading.found is True
    assert 45.0 <= reading.progress_pct <= 55.0


def test_geometry_dash_raw_adapter_passes_gym_checker():
    adapter = make_adapter()
    adapter.connect()
    try:
        check_env(adapter, skip_render_check=True, skip_close_check=True)
    finally:
        adapter.close()


def test_window_lookup_happens_before_connect_and_reset_uses_configured_key():
    locator = FakeWindowLocator()
    inputs = FakeInput()
    adapter = make_adapter(inputs=inputs, locator=locator)
    adapter.connect()
    try:
        assert locator.calls == 1
        observation, info = adapter.reset(seed=0)
        assert observation.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert info["bar_found"] is True
        assert info["progress_pct"] == pytest.approx(0.0, abs=0.01)
        assert ("down", "enter") in inputs.events
        assert ("up", "enter") in inputs.events
        assert inputs.held == set()
    finally:
        adapter.close()


def test_window_not_found_is_actionable_and_sends_no_input():
    inputs = FakeInput()
    adapter = make_adapter(inputs=inputs, locator=FakeWindowLocator(fail=True))
    with pytest.raises(AdapterError, match="not found"):
        adapter.connect()
    assert inputs.events == []


def test_death_detection_uses_progress_reset_and_terminates():
    capture = LiveCapture(frame_with_progress(0.0))
    adapter = make_adapter(capture=capture)
    adapter.connect()
    try:
        adapter.reset(seed=0)
        capture.frame = frame_with_progress(10.0)
        _, reward, terminated, truncated, info = adapter.step(0)
        assert terminated is False
        assert truncated is False
        assert info["progress_pct"] >= 8.0
        capture.frame = frame_with_progress(0.0)
        _, reward, terminated, truncated, info = adapter.step(0)
        assert terminated is True
        assert truncated is False
        assert info["died"] is True
        assert reward == pytest.approx(-5.0)
    finally:
        adapter.close()


def test_completion_requires_consecutive_frames_and_terminates():
    capture = LiveCapture(frame_with_progress(0.0))
    adapter = make_adapter(capture=capture)
    adapter.connect()
    try:
        adapter.reset(seed=0)
        capture.frame = frame_with_progress(100.0)
        _, reward, terminated, _, info = adapter.step(0)
        assert terminated is False
        assert info["completed"] is False
        _, reward, terminated, _, info = adapter.step(0)
        assert terminated is True
        assert info["completed"] is True
        assert reward == pytest.approx(50.0)
    finally:
        adapter.close()


def test_missing_bar_ends_episode_without_fake_death():
    capture = LiveCapture(frame_with_progress(0.0))
    adapter = make_adapter(capture=capture)
    adapter.connect()
    try:
        adapter.reset(seed=0)
        capture.frame = frame_with_progress(0.0, present=False)
        _, _, terminated, _, info = adapter.step(0)
        assert terminated is False
        assert info["bar_found"] is False
        _, _, terminated, _, info = adapter.step(0)
        assert terminated is True
        assert info["gameplay_lost"] is True
        assert info["died"] is False
    finally:
        adapter.close()


def test_paused_game_reset_tries_resume_key():
    paused = frame_with_progress(0.0, present=False)
    running = frame_with_progress(0.0)
    capture = SequenceCapture([paused, paused, running, running])
    inputs = FakeInput()
    adapter = make_adapter(capture=capture, inputs=inputs)
    adapter.connect()
    try:
        _, info = adapter.reset(seed=0)
        assert info["bar_found"] is True
        assert info["resume_attempts"] == 1
        assert ("down", "space") in inputs.events
        assert inputs.held == set()
    finally:
        adapter.close()


def test_reset_failure_is_clear_and_releases_keys():
    capture = LiveCapture(frame_with_progress(0.0, present=False))
    inputs = FakeInput()
    adapter = make_adapter(capture=capture, inputs=inputs)
    adapter.connect()
    try:
        with pytest.raises(AdapterError, match="playable Geometry Dash level"):
            adapter.reset(seed=0)
        assert inputs.held == set()
    finally:
        adapter.close()


def test_jump_actions_press_and_release_space():
    inputs = FakeInput()
    adapter = make_adapter(inputs=inputs)
    adapter.connect()
    try:
        adapter.reset(seed=0)
        _, _, _, _, info = adapter.step(1)  # jump
        assert info["action_name"] == "jump"
        assert ("down", "space") in inputs.events
        assert ("release", "space") in inputs.events
        assert inputs.held == set()
        adapter.step(2)  # hold_jump
        assert inputs.held == set()
        adapter.step(3)  # release_jump
        assert inputs.held == set()
    finally:
        adapter.close()


def test_pixel_wrapper_and_short_ppo_run(tmp_path):
    capture = LiveCapture(frame_with_progress(0.0))
    adapter = make_adapter(capture=capture)
    adapter.connect()
    env = ObservationValidator(PixelObservationWrapper(adapter, adapter.observation_config))
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (1, 64, 64)
        assert observation.dtype == np.uint8
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        model_path = tmp_path / "geometry_dash_model.zip"
        model.save(model_path)
        loaded = PPO.load(model_path, env=env, device="cpu")
        action, _ = loaded.predict(observation, deterministic=True)
        assert int(action) in range(adapter.action_space.n)
    finally:
        env.close()
