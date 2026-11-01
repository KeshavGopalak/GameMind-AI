from __future__ import annotations

import cv2
import numpy as np
import pytest

from gamemind.config import EpisodeConfig, RewardConfig
from gamemind.rewards import (
    RewardContext,
    build_reward_function,
    build_termination_detector,
)
from gamemind.utils.errors import ConfigError, RewardError
from gamemind.utils.vision import TemplateMatcher


def context(**kwargs) -> RewardContext:
    values = {
        "signals": {},
        "native_reward": 0.0,
        "frame": None,
        "info": {},
        "episode_step": 0,
        "episode_reward": 0.0,
    }
    values.update(kwargs)
    return RewardContext(**values)


def test_additive_components_and_breakdown():
    function = build_reward_function(
        RewardConfig(
            components=(
                {"type": "native", "scale": 2.0, "name": "native"},
                {"type": "constant", "value": -0.1, "name": "time"},
                {"type": "signal", "key": "score", "scale": 0.5, "name": "score"},
                {"type": "event", "key": "hit", "value": -2.0, "name": "hit"},
            )
        )
    )
    reward, breakdown = function.evaluate(context(signals={"score": 4.0, "hit": 1.0}, native_reward=1.0))
    assert reward == pytest.approx(2.0 - 0.1 + 2.0 - 2.0)
    assert breakdown["native"] == 2.0
    assert breakdown["score"] == 2.0
    assert breakdown["hit"] == -2.0


def test_missing_signal_has_actionable_error():
    function = build_reward_function(
        RewardConfig(components=({"type": "signal", "key": "does_not_exist"},))
    )
    with pytest.raises(RewardError, match="does_not_exist"):
        function.evaluate(context())


def test_signal_default_is_supported():
    function = build_reward_function(
        RewardConfig(
            components=({"type": "signal", "key": "optional", "scale": 2.0, "default": 0.5},)
        )
    )
    reward, _ = function.evaluate(context())
    assert reward == pytest.approx(1.0)


def test_progress_reward_direction_and_reset_on_event():
    function = build_reward_function(
        RewardConfig(
            components=(
                {
                    "type": "progress",
                    "key": "distance",
                    "mode": "decrease",
                    "reset_on": "collected",
                    "scale": 2.0,
                    "name": "progress",
                },
            )
        )
    )
    assert function.evaluate(context(signals={"distance": 10.0, "collected": 0.0}))[0] == 0.0
    assert function.evaluate(context(signals={"distance": 7.0, "collected": 0.0}))[0] == 6.0
    # A collection respawns the target; do not reward the artificial distance jump.
    assert function.evaluate(context(signals={"distance": 20.0, "collected": 1.0}))[0] == 0.0
    assert function.evaluate(context(signals={"distance": 18.0, "collected": 0.0}))[0] == 4.0
    function.reset()
    assert function.evaluate(context(signals={"distance": 5.0, "collected": 0.0}))[0] == 0.0


def test_high_water_progress_does_not_reward_jitter_or_restart():
    function = build_reward_function(RewardConfig(components=(
        {"type": "progress", "key": "max_progress_pct", "reset_on": "died"},
    )))
    values = [0, 2, 2, 3, 3]
    assert [function.evaluate(context(signals={"max_progress_pct": x, "died": 0}))[0] for x in values] == [0, 2, 0, 1, 0]
    assert function.evaluate(context(signals={"max_progress_pct": 3, "died": 1}))[0] == 0
    function.reset()
    assert function.evaluate(context(signals={"max_progress_pct": 0, "died": 0}))[0] == 0


def test_clipping_and_survival():
    function = build_reward_function(
        RewardConfig(
            components=(
                {"type": "constant", "value": 10.0},
                {"type": "survival", "value": 1.0},
            ),
            clip=(-3.0, 3.0),
        )
    )
    reward, _ = function.evaluate(context(signals={"alive": 1.0}))
    assert reward == 3.0
    reward, _ = function.evaluate(context(signals={"alive": 0.0}))
    assert reward == 3.0  # 10 + 0 still clips to 3


def test_unknown_component_type_is_rejected():
    with pytest.raises(ConfigError, match="Unknown reward component"):
        build_reward_function(RewardConfig(components=({"type": "magic"},)))


def test_custom_reward_loading():
    function = build_reward_function(
        RewardConfig(
            components=(
                {"type": "custom", "path": "tests.support:custom_reward", "params": {"bonus": 2.5}},
            )
        )
    )
    reward, _ = function.evaluate(context(native_reward=1.0))
    assert reward == pytest.approx(3.5)


def _template_image(path):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[8:18, 6:16] = 255
    template = frame[8:18, 6:16]
    assert cv2.imwrite(str(path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
    return frame


def test_template_matcher_and_once_per_episode_reward(tmp_path):
    template_path = tmp_path / "target.png"
    frame = _template_image(template_path)
    matcher = TemplateMatcher(template_path, region=None, threshold=0.9)
    assert matcher.score(frame) > 0.99
    assert matcher.matches(frame)
    function = build_reward_function(
        RewardConfig(
            components=(
                {
                    "type": "template",
                    "template": str(template_path),
                    "threshold": 0.9,
                    "scale": 4.0,
                    "once_per_episode": True,
                    "name": "found",
                },
            )
        )
    )
    first, _ = function.evaluate(context(frame=frame))
    second, _ = function.evaluate(context(frame=frame))
    assert first == 4.0
    assert second == 0.0
    function.reset()
    assert function.evaluate(context(frame=frame))[0] == 4.0


def test_template_termination_detector_consecutive(tmp_path):
    template_path = tmp_path / "game_over.png"
    frame = _template_image(template_path)
    detector = build_termination_detector(
        EpisodeConfig(
            terminate_on={
                "type": "template",
                "template": str(template_path),
                "threshold": 0.9,
                "consecutive": 2,
            }
        )
    )
    assert detector is not None
    assert detector.update(frame) is False
    assert detector.update(frame) is True
    detector.reset()
    assert detector.update(frame) is False


def test_missing_template_file_is_clear(tmp_path):
    with pytest.raises(RewardError, match="not found"):
        TemplateMatcher(tmp_path / "missing.png")
