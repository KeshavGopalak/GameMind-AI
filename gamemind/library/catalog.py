"""What GameMind can actually connect to in this install.

The Add Game form is driven by this catalog. An adapter appears here only when
GameMind ships a real implementation for it. Adding a game whose adapter is not
in this catalog is refused with an explanation rather than silently creating a
profile that cannot train.
"""

from __future__ import annotations

from dataclasses import dataclass

from gamemind.adapters.registry import adapter_available, available_adapters


@dataclass(frozen=True)
class AdapterOption:
    adapter: str
    label: str
    summary: str
    requirements: tuple[str, ...]
    template_config: str | None = None
    needs_env_id: bool = False
    level_workflow: bool = False
    ready: bool = True


CATALOG: tuple[AdapterOption, ...] = (
    AdapterOption(
        adapter="geometry_dash",
        label="Geometry Dash",
        summary=(
            "Controls the real Geometry Dash window with ordinary keyboard input and learns "
            "from screen pixels. Progress is estimated from the on-screen bar."
        ),
        requirements=(
            "Geometry Dash installed and running on a Windows desktop session.",
            "A level selected manually in game, with Auto-Retry enabled.",
            "Fullscreen or borderless window on the primary monitor; no overlay covering it.",
            "The game window focused while training.",
        ),
        template_config="configs/geometry_dash.yaml",
        level_workflow=True,
    ),
    AdapterOption(
        adapter="example_game",
        label="Target Chase (built-in)",
        summary=(
            "The built-in pixel game. Needs no external window, so it is the safest way to "
            "check that training and the library work end to end."
        ),
        requirements=("Nothing beyond the Python dependencies.",),
        template_config="configs/example_game.yaml",
    ),
    AdapterOption(
        adapter="gymnasium",
        label="Gymnasium environment",
        summary=(
            "Wraps any installed Gymnasium environment id. State observations use the "
            "environment's own numbers; pixel observations need rgb_array rendering."
        ),
        requirements=(
            "A Gymnasium environment id that imports in this Python environment.",
            "The package providing that environment installed in the same venv.",
        ),
        template_config="configs/cartpole.yaml",
        needs_env_id=True,
    ),
    AdapterOption(
        adapter="desktop",
        label="Custom desktop game",
        summary=(
            "The general desktop template: capture a window and send keys you configure. "
            "It is deliberately not preconfigured: rewards, termination and reset keys must "
            "be written for the specific game."
        ),
        requirements=(
            "A window title, capture region, action bindings and reward signals you define.",
            "A YAML configuration file you provide; nothing is guessed for you.",
        ),
        ready=False,
    ),
)

_BY_ADAPTER = {option.adapter: option for option in CATALOG}

ADAPTER_HELP = (
    "GameMind ships adapters for: "
    + ", ".join(option.adapter for option in CATALOG)
    + ". A different game needs a new adapter class before it can train; adding a name to "
    "the library does not create that support."
)


def catalog() -> tuple[AdapterOption, ...]:
    return CATALOG


def option_for(adapter: str) -> AdapterOption | None:
    return _BY_ADAPTER.get(str(adapter or "").strip().lower())


def readiness(adapter: str) -> tuple[bool, str]:
    """Return (usable, explanation) for *adapter* in this installation."""
    key = str(adapter or "").strip().lower()
    if not adapter_available(key):
        return False, (
            f"No adapter named {key!r} is registered in this installation. {ADAPTER_HELP}"
        )
    option = option_for(key)
    if option is not None and not option.ready:
        return False, (
            f"The {option.label} adapter exists, but it needs a configuration written for "
            "the specific game (window title, actions, rewards, reset keys). Provide a YAML "
            "configuration; GameMind will not pretend the game is supported without one."
        )
    return True, ""


def installed_adapters() -> list[str]:
    return available_adapters()
