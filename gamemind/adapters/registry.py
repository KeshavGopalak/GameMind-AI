"""Adapter discovery, built-in factories, and custom adapter loading."""

from __future__ import annotations

import importlib
from typing import Callable

from gamemind.adapters.base import GameAdapter
from gamemind.config import GameConfig
from gamemind.utils.errors import AdapterError, ConfigError

AdapterFactory = Callable[[GameConfig, str | None], GameAdapter]
_FACTORIES: dict[str, AdapterFactory] = {}


def register_adapter(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    """Decorator for registering a reusable adapter factory."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("Adapter name must not be empty.")

    def decorator(factory: AdapterFactory) -> AdapterFactory:
        _FACTORIES[key] = factory
        return factory

    return decorator


def _gymnasium_factory(config: GameConfig, render_mode: str | None) -> GameAdapter:
    from gamemind.adapters.gymnasium_adapter import GymnasiumAdapter

    return GymnasiumAdapter(config, render_mode=render_mode)


def _example_factory(config: GameConfig, render_mode: str | None) -> GameAdapter:
    from gamemind.adapters.example_game_adapter import ExampleGameAdapter

    return ExampleGameAdapter(config, render_mode=render_mode)


def _desktop_factory(config: GameConfig, render_mode: str | None) -> GameAdapter:
    from gamemind.adapters.desktop_adapter import DesktopGameAdapter

    return DesktopGameAdapter(config, render_mode=render_mode)


def _geometry_dash_factory(config: GameConfig, render_mode: str | None) -> GameAdapter:
    from gamemind.adapters.geometry_dash import GeometryDashAdapter

    return GeometryDashAdapter(config, render_mode=render_mode)


def _register_builtins() -> None:
    if _FACTORIES:
        return
    _FACTORIES["gymnasium"] = _gymnasium_factory
    _FACTORIES["example_game"] = _example_factory
    _FACTORIES["desktop"] = _desktop_factory
    _FACTORIES["geometry_dash"] = _geometry_dash_factory


def available_adapters() -> list[str]:
    _register_builtins()
    return sorted(_FACTORIES)


def load_custom_adapter_class(path: str) -> type[GameAdapter]:
    """Import a custom adapter from ``my_module:MyAdapter``."""
    if ":" not in path:
        raise ConfigError(
            f"Unknown adapter {path!r}. Built-ins: {', '.join(available_adapters())}. "
            "For a user-written adapter use 'my_module:MyAdapter' or register a factory."
        )
    module_name, class_name = path.split(":", 1)
    module_name, class_name = module_name.strip(), class_name.strip()
    if not module_name or not class_name:
        raise ConfigError(f"Invalid custom adapter path {path!r}; expected module:ClassName.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"Could not import adapter module {module_name!r}: {exc}. "
            "Run from the project root or add the module's parent to PYTHONPATH."
        ) from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, GameAdapter):
        raise ConfigError(
            f"{module_name}:{class_name} must be a class subclassing GameAdapter."
        )
    return cls


def create_adapter(config: GameConfig, render_mode: str | None = None) -> GameAdapter:
    """Instantiate (but do not connect) the adapter named by the configuration."""
    _register_builtins()
    key = config.adapter.strip().lower()
    if key in _FACTORIES:
        return _FACTORIES[key](config, render_mode)
    cls = load_custom_adapter_class(config.adapter)
    try:
        adapter = cls(config=config, render_mode=render_mode)
    except TypeError as exc:
        raise AdapterError(
            f"Custom adapter {config.adapter!r} must accept (config, render_mode=None): {exc}"
        ) from exc
    if not isinstance(adapter, GameAdapter):
        raise AdapterError(f"Custom adapter {config.adapter!r} did not produce a GameAdapter.")
    return adapter
