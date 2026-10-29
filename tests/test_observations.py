from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from gamemind.config import ObservationConfig
from gamemind.observations import (
    ActionRepeat,
    FrameStackChannels,
    ObservationValidator,
    PixelObservationWrapper,
)
from gamemind.observations.processing import ObservationProcessor
from gamemind.utils.errors import ObservationError
from tests.support import CountingEnv


def test_rgb_processor_shape_and_range():
    processor = ObservationProcessor(
        ObservationConfig(mode="rgb", width=32, height=32, stack=1)
    )
    frame = np.random.default_rng(0).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    result = processor.process(frame)
    assert result.shape == (3, 32, 32)
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255
    assert processor.observation_space.shape == (3, 32, 32)


def test_grayscale_and_normalized_processor():
    frame = np.full((50, 60, 3), 255, dtype=np.uint8)
    gray = ObservationProcessor(
        ObservationConfig(mode="grayscale", width=32, height=32, stack=1, normalize=True)
    ).process(frame)
    assert gray.shape == (1, 32, 32)
    assert gray.dtype == np.float32
    assert gray.min() == pytest.approx(1.0)
    assert gray.max() == pytest.approx(1.0)


def test_rgba_input_drops_alpha():
    processor = ObservationProcessor(ObservationConfig(mode="rgb", width=32, height=32))
    frame = np.zeros((40, 40, 4), dtype=np.uint8)
    frame[..., 3] = 255
    result = processor.process(frame)
    assert result.shape == (3, 32, 32)


def test_invalid_frames_raise_clear_errors():
    processor = ObservationProcessor(ObservationConfig(mode="grayscale", width=32, height=32))
    with pytest.raises(ObservationError, match="NumPy array"):
        processor.process([[1, 2, 3]])
    with pytest.raises(ObservationError, match="uint8"):
        processor.process(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(ObservationError, match="channels"):
        processor.process(np.zeros((10, 10, 2), dtype=np.uint8))
    with pytest.raises(ObservationError, match="too small"):
        processor.process(np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(ObservationError, match="2D grayscale"):
        processor.process(np.zeros((10,), dtype=np.uint8))


def test_processor_rejects_state_mode():
    with pytest.raises(ObservationError, match="pixel modes"):
        ObservationProcessor(ObservationConfig(mode="state"))


def test_frame_stack_repeat_on_reset_and_shapes():
    env = FrameStackChannels(CountingEnv(), 3)
    observation, info = env.reset(seed=1)
    assert observation.shape == (3, 4, 4)
    assert np.array_equal(observation[0], observation[1])
    assert np.array_equal(observation[1], observation[2])
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == (3, 4, 4)
    assert reward == 1.0
    assert not terminated and not truncated


def test_action_repeat_sums_rewards_and_stops_on_termination():
    env = ActionRepeat(CountingEnv(terminate_after=2), 5)
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(0)
    assert reward == 2.0
    assert terminated is True
    assert info["action_repeats"] == 2
    assert observation.shape == (1, 4, 4)


def test_observation_validator_rejects_bad_space():
    base = CountingEnv()
    env = ObservationValidator(base)
    env.reset(seed=0)
    # Bypass the wrapper's observation conversion to simulate a bad adapter.
    base.observation_space = spaces.Box(0, 255, (1, 4, 4), dtype=np.uint8)
    env.observation_space = spaces.Box(0, 255, (2, 2), dtype=np.uint8)
    with pytest.raises(ObservationError):
        env.step(0)


def test_pixel_wrapper_from_hwc_env():
    class RawEnv(CountingEnv):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(0, 255, (8, 6, 3), dtype=np.uint8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros((8, 6, 3), dtype=np.uint8), {}

        def step(self, action):
            del action
            self.count += 1
            return (
                np.full((8, 6, 3), 100, dtype=np.uint8),
                1.0,
                False,
                False,
                {},
            )

    env = PixelObservationWrapper(
        RawEnv(), ObservationConfig(mode="grayscale", width=32, height=32, stack=1)
    )
    observation, _ = env.reset(seed=0)
    assert observation.shape == (1, 32, 32)
    assert observation.dtype == np.uint8
    observation, *_ = env.step(0)
    assert observation.shape == (1, 32, 32)
