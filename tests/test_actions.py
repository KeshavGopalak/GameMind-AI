from __future__ import annotations

import pytest

from gamemind.actions import action_label, build_discrete_action_space
from gamemind.config import ActionsConfig
from gamemind.utils.errors import ConfigError


def test_discrete_space_and_labels():
    actions = ActionsConfig.from_dict(
        {
            "bindings": [
                {"name": "noop"},
                {"name": "left", "keys": ["a"]},
                {"name": "right", "keys": ["d"]},
            ]
        }
    )
    space = build_discrete_action_space(actions)
    assert space.n == 3
    assert action_label(actions, 0) == "noop"
    assert action_label(actions, 2) == "right"
    assert action_label(actions, 99) == "action_99"


def test_no_bindings_is_an_error():
    with pytest.raises(ConfigError, match="no action bindings"):
        build_discrete_action_space(ActionsConfig())


def test_mouse_binding_validation():
    actions = ActionsConfig.from_dict(
        {"bindings": [{"name": "click", "mouse_click": ["left"], "mouse_move": [10, -5]}]}
    )
    binding = actions.bindings[0]
    assert binding.mouse_click == ("left",)
    assert binding.mouse_move == (10, -5)
    with pytest.raises(ConfigError, match="mouse button"):
        ActionsConfig.from_dict({"bindings": [{"name": "bad", "mouse_click": ["thumb"]}]})


def test_duplicate_keys_in_one_action_rejected():
    with pytest.raises(ConfigError, match="twice"):
        ActionsConfig.from_dict({"bindings": [{"name": "x", "keys": ["a", "A"]}]})


def test_pyautogui_key_pause_is_opt_out_and_releases(monkeypatch):
    import sys
    from types import SimpleNamespace
    from gamemind.actions.backends import PyAutoGUIBackend
    events = []
    fake = SimpleNamespace(KEYBOARD_KEYS=["space"], FAILSAFE=True, PAUSE=0,
        keyDown=lambda key, _pause=True: events.append(("down", key, _pause)),
        keyUp=lambda key, _pause=True: events.append(("up", key, _pause)))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    backend = PyAutoGUIBackend(key_pause=False)
    try:
        backend.key_down("space")
        backend.release_all()
        assert events == [("down", "space", False), ("up", "space", False)]
        assert not backend._held
    finally:
        backend.close()


def test_frame_skip_bounds():
    with pytest.raises(ConfigError, match="frame_skip"):
        ActionsConfig(frame_skip=0)
    with pytest.raises(ConfigError, match="frame_skip"):
        ActionsConfig(frame_skip=31)
