"""Built-in reward components.

A component reads either a game-provided signal, the environment's native reward,
the raw frame (through an optional visual detector), or a user-supplied Python
function. Components are additive. This keeps reward shaping explicit and
inspectable instead of hiding it inside a game adapter.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Mapping

import numpy as np

from gamemind.rewards.base import RewardComponent, RewardContext
from gamemind.utils.errors import ConfigError, RewardError
from gamemind.utils.vision import Region, TemplateMatcher

_COMMON_KEYS = {"type", "name"}


def _check_unknown(raw: Mapping[str, Any], allowed: set[str], type_name: str) -> None:
    unknown = sorted(set(raw) - allowed - _COMMON_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) for reward component {type_name!r}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed | _COMMON_KEYS))}."
        )


def _name(raw: Mapping[str, Any], type_name: str, index: int) -> str:
    value = raw.get("name", f"{type_name}_{index}")
    value = str(value).strip()
    if not value:
        raise ConfigError(f"reward.components[{index}].name must not be empty.")
    return value


def _number(
    raw: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
    required: bool = False,
    minimum: float | None = None,
) -> float:
    if key not in raw:
        if required:
            raise ConfigError(f"Reward component {raw.get('type')!r} requires '{key}'.")
        assert default is not None
        value = default
    else:
        value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError(f"Reward component field {key!r} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"Reward component field {key!r} must be finite, got {value!r}.")
    if minimum is not None and result < minimum:
        raise ConfigError(f"Reward component field {key!r} must be >= {minimum}, got {result}.")
    return result


def _key(raw: Mapping[str, Any], required: bool = True) -> str:
    value = raw.get("key")
    if value is None:
        if required:
            raise ConfigError(f"Reward component {raw.get('type')!r} requires a 'key'.")
        return ""
    value = str(value).strip()
    if not value:
        raise ConfigError("Reward component signal key must not be empty.")
    return value


class NativeReward(RewardComponent):
    """Use the reward returned by the underlying Gymnasium environment."""

    def __init__(self, name: str, scale: float = 1.0) -> None:
        super().__init__(name)
        self.scale = scale

    def evaluate(self, context: RewardContext) -> float:
        return self.scale * float(context.native_reward)


class ConstantReward(RewardComponent):
    """Add a fixed amount every step (commonly a time penalty)."""

    def __init__(self, name: str, value: float) -> None:
        super().__init__(name)
        self.value = value

    def evaluate(self, context: RewardContext) -> float:
        return self.value


class SignalReward(RewardComponent):
    """Scale a named numeric signal supplied by the adapter."""

    def __init__(self, name: str, key: str, scale: float = 1.0, default: float | None = None) -> None:
        super().__init__(name)
        self.key = key
        self.scale = scale
        self.default = default

    def evaluate(self, context: RewardContext) -> float:
        if self.key not in context.signals and self.default is not None:
            return self.scale * self.default
        return self.scale * context.signal(self.key)


class EventReward(RewardComponent):
    """Award ``value`` on steps where a boolean/count signal is positive.

    Adapters should emit a signal only on the step an event occurs (rather than
    leaving it at 1 for many frames), otherwise this reward repeats.
    """

    def __init__(self, name: str, key: str, value: float = 1.0) -> None:
        super().__init__(name)
        self.key = key
        self.value = value

    def evaluate(self, context: RewardContext) -> float:
        if self.key not in context.signals:
            return 0.0
        return self.value if context.signal(self.key) > 0.0 else 0.0


class ProgressReward(RewardComponent):
    """Reward movement of a signal in a chosen direction.

    With ``mode: decrease`` and a distance-to-target signal, moving closer yields
    a positive reward. The first step of an episode has no previous value and
    therefore contributes zero.
    """

    def __init__(
        self,
        name: str,
        key: str,
        scale: float = 1.0,
        mode: str = "increase",
        reset_on: str | None = None,
    ) -> None:
        super().__init__(name)
        self.key = key
        self.scale = scale
        self.mode = mode
        self.reset_on = reset_on
        self._previous: float | None = None

    def reset(self) -> None:
        self._previous = None

    def evaluate(self, context: RewardContext) -> float:
        current = context.signal(self.key)
        if self.reset_on and self.reset_on in context.signals and context.signal(self.reset_on) > 0:
            # A collection/respawn changes the meaning of the distance signal.
            # Reset the baseline instead of rewarding a spurious jump.
            self._previous = current
            return 0.0
        if self._previous is None:
            self._previous = current
            return 0.0
        delta = current - self._previous
        self._previous = current
        directional = -delta if self.mode == "decrease" else delta
        return self.scale * directional


class SurvivalReward(RewardComponent):
    """Reward each step while the configured ``alive`` signal remains positive.

    If the signal is absent, the agent is assumed to be alive. This is useful for
    games where only a terminal screen can be detected.
    """

    def __init__(self, name: str, value: float = 0.01, key: str = "alive") -> None:
        super().__init__(name)
        self.value = value
        self.key = key

    def evaluate(self, context: RewardContext) -> float:
        alive = True if self.key not in context.signals else context.signal(self.key) > 0.0
        return self.value if alive else 0.0


class TemplateMatchReward(RewardComponent):
    """Reward a visual template match, optionally once per episode."""

    def __init__(
        self,
        name: str,
        template: str,
        *,
        scale: float = 1.0,
        region: Region | None = None,
        threshold: float = 0.85,
        once_per_episode: bool = True,
        cooldown: int = 0,
    ) -> None:
        super().__init__(name)
        self.scale = scale
        self.region = region
        self.once_per_episode = once_per_episode
        self.cooldown = cooldown
        self.matcher = TemplateMatcher(
            template, region=region, threshold=threshold, where=f"reward component {name!r}"
        )
        self._fired = False
        self._last_fired_step = -10_000

    def reset(self) -> None:
        self._fired = False
        self._last_fired_step = -10_000

    def evaluate(self, context: RewardContext) -> float:
        if self.once_per_episode and self._fired:
            return 0.0
        if context.frame is None:
            raise RewardError(
                f"Reward component {self.name!r} uses a template but the adapter supplied no frame."
            )
        if context.episode_step - self._last_fired_step <= self.cooldown and self._last_fired_step >= 0:
            return 0.0
        if self.matcher.score(context.frame) >= self.matcher.threshold:
            self._fired = True
            self._last_fired_step = context.episode_step
            return self.scale
        return 0.0


class CustomReward(RewardComponent):
    """Call ``module:function`` with ``(context, **params)``."""

    def __init__(self, name: str, path: str, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(name)
        self.path = path
        self.params = dict(params or {})
        self._function = _import_callable(path)

    def evaluate(self, context: RewardContext) -> float:
        try:
            return float(self._function(context, **self.params))
        except RewardError:
            raise
        except Exception as exc:
            raise RewardError(
                f"Custom reward function {self.path!r} failed: {exc}. "
                "Its signature must be fn(context, **params) and return a number."
            ) from exc


def _import_callable(path: str):
    if not isinstance(path, str) or ":" not in path:
        raise ConfigError(
            "Custom reward component 'path' must look like 'my_module:my_function'."
        )
    module_name, function_name = path.split(":", 1)
    module_name = module_name.strip()
    function_name = function_name.strip()
    if not module_name or not function_name:
        raise ConfigError(f"Invalid custom reward path {path!r}; expected module:function.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"Could not import reward module {module_name!r}: {exc}. "
            "Run from the project root or put the module on PYTHONPATH."
        ) from exc
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ConfigError(f"Module {module_name!r} has no callable {function_name!r}.")
    return function


def create_component(raw: Mapping[str, Any], *, index: int) -> RewardComponent:
    """Validate one component mapping and instantiate it."""
    type_name = str(raw.get("type", "")).strip().lower()
    name = _name(raw, type_name, index)

    if type_name == "native":
        _check_unknown(raw, {"scale"}, type_name)
        return NativeReward(name, scale=_number(raw, "scale", default=1.0))
    if type_name == "constant":
        _check_unknown(raw, {"value"}, type_name)
        return ConstantReward(name, value=_number(raw, "value", required=True))
    if type_name == "signal":
        _check_unknown(raw, {"key", "scale", "default"}, type_name)
        default = _number(raw, "default") if "default" in raw else None
        return SignalReward(
            name, key=_key(raw), scale=_number(raw, "scale", default=1.0), default=default
        )
    if type_name == "event":
        _check_unknown(raw, {"key", "value"}, type_name)
        return EventReward(name, key=_key(raw), value=_number(raw, "value", default=1.0))
    if type_name == "progress":
        _check_unknown(raw, {"key", "scale", "mode", "reset_on"}, type_name)
        mode = str(raw.get("mode", "increase")).strip().lower()
        if mode not in {"increase", "decrease"}:
            raise ConfigError(
                f"Reward component {name!r}: mode must be 'increase' or 'decrease', got {mode!r}."
            )
        reset_on = raw.get("reset_on")
        if reset_on is not None:
            reset_on = str(reset_on).strip()
            if not reset_on:
                raise ConfigError(f"Reward component {name!r}: reset_on must be a signal name.")
        return ProgressReward(
            name,
            key=_key(raw),
            scale=_number(raw, "scale", default=1.0),
            mode=mode,
            reset_on=reset_on,
        )
    if type_name == "survival":
        _check_unknown(raw, {"key", "value"}, type_name)
        return SurvivalReward(
            name, value=_number(raw, "value", default=0.01), key=_key(raw, required=False) or "alive"
        )
    if type_name == "template":
        _check_unknown(
            raw,
            {"template", "scale", "region", "threshold", "once_per_episode", "cooldown"},
            type_name,
        )
        template = raw.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ConfigError(f"Reward component {name!r} requires a template image path.")
        region = Region.from_config(raw.get("region"), where=f"reward component {name!r}.region")
        once = raw.get("once_per_episode", True)
        if not isinstance(once, bool):
            raise ConfigError(f"Reward component {name!r}: once_per_episode must be true or false.")
        cooldown = raw.get("cooldown", 0)
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
            raise ConfigError(f"Reward component {name!r}: cooldown must be a non-negative integer.")
        return TemplateMatchReward(
            name,
            template,
            scale=_number(raw, "scale", default=1.0),
            region=region,
            threshold=_number(raw, "threshold", default=0.85, minimum=0.0),
            once_per_episode=once,
            cooldown=cooldown,
        )
    if type_name == "custom":
        _check_unknown(raw, {"path", "params"}, type_name)
        path = raw.get("path")
        if not isinstance(path, str) or ":" not in path:
            raise ConfigError(f"Reward component {name!r} requires path='module:function'.")
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise ConfigError(f"Reward component {name!r}: params must be a mapping.")
        return CustomReward(name, path, params)

    raise ConfigError(
        f"Unknown reward component type {type_name!r}. Supported types: "
        "native, constant, signal, event, progress, survival, template, custom."
    )
