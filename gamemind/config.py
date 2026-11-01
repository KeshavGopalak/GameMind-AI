"""Typed configuration objects and strict YAML parsing.

All configuration is validated here, before any environment, neural network, or
desktop input backend is created. That keeps beginner mistakes cheap: an invalid
resolution or batch size produces a clear message instead of a failure deep in
Stable-Baselines3 after a long wait.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from gamemind.utils.errors import ConfigError
from gamemind.utils.paths import PROJECT_ROOT, resolve_project_path

SUPPORTED_ALGORITHMS = ("ppo", "a2c", "maskable_ppo")
OBSERVATION_MODES = ("state", "rgb", "grayscale", "canny")
MOUSE_BUTTONS = ("left", "right", "middle")


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping (YAML key/value block), got {type(value).__name__}.")
    return dict(value)


def _check_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {where}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}."
        )


def _as_int(value: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}.")
    return value


def _as_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite, got {value!r}.")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {result}.")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {result}.")
    return result


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# Observation configuration
# ---------------------------------------------------------------------------
@dataclass
class ObservationConfig:
    """How raw game observations become what the neural network sees.

    Modes:
        ``state``     use the environment's native vector/array observation.
        ``rgb``       use screen frames, keeping three colour channels.
        ``grayscale`` use screen frames, converted to one channel.
    """

    mode: str = "state"
    width: int = 84
    height: int = 84
    stack: int = 1
    normalize: bool = False
    # Fractional crop [top, left, bottom, right], each 0..0.45. Applied before resize.
    crop: tuple[float, float, float, float] | None = None
    canny_low: int = 50
    canny_high: int = 150

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in OBSERVATION_MODES:
            raise ConfigError(
                f"observation.mode must be one of {OBSERVATION_MODES}, got {self.mode!r}."
            )
        self.width = _as_int(self.width, "observation.width", minimum=32, maximum=1024)
        self.height = _as_int(self.height, "observation.height", minimum=32, maximum=1024)
        self.stack = _as_int(self.stack, "observation.stack", minimum=1, maximum=16)
        self.normalize = _as_bool(self.normalize, "observation.normalize")
        self.canny_low = _as_int(self.canny_low, "observation.canny_low", minimum=0, maximum=255)
        self.canny_high = _as_int(self.canny_high, "observation.canny_high", minimum=0, maximum=255)
        if self.canny_low >= self.canny_high:
            raise ConfigError("observation.canny_low must be less than canny_high.")
        if self.crop is not None:
            if not isinstance(self.crop, (list, tuple)) or len(self.crop) != 4:
                raise ConfigError(
                    "observation.crop must be null or [top, left, bottom, right] fractions."
                )
            values = tuple(
                _as_float(value, f"observation.crop[{index}]", minimum=0.0, maximum=0.45)
                for index, value in enumerate(self.crop)
            )
            assert all(value is not None for value in values)
            top, left, bottom, right = (float(value) for value in values)
            if top + bottom >= 0.8 or left + right >= 0.8:
                raise ConfigError(
                    "observation.crop removes too much of the frame; keep top+bottom < 0.8 and "
                    "left+right < 0.8."
                )
            self.crop = (top, left, bottom, right)
        if self.mode == "state":
            if self.stack != 1:
                raise ConfigError(
                    "observation.stack must be 1 in state mode. Frame stacking is only for pixels; "
                    "use observation.mode: grayscale to capture motion."
                )
            if self.normalize:
                raise ConfigError(
                    "observation.normalize is only supported for pixel observations. "
                    "Set normalize: false in state mode."
                )
            if self.crop is not None:
                raise ConfigError("observation.crop is only supported for pixel observations.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ObservationConfig":
        raw = _mapping(data, "observation")
        _check_unknown(
            raw, {"mode", "width", "height", "stack", "normalize", "crop", "canny_low", "canny_high"}, "observation"
        )
        defaults = cls()
        return cls(
            mode=raw.get("mode", defaults.mode),
            width=raw.get("width", defaults.width),
            height=raw.get("height", defaults.height),
            stack=raw.get("stack", defaults.stack),
            normalize=raw.get("normalize", defaults.normalize),
            crop=raw.get("crop", defaults.crop),
            canny_low=raw.get("canny_low", defaults.canny_low),
            canny_high=raw.get("canny_high", defaults.canny_high),
        )

    @property
    def is_pixel(self) -> bool:
        return self.mode in {"rgb", "grayscale", "canny"}


# ---------------------------------------------------------------------------
# Action configuration
# ---------------------------------------------------------------------------
@dataclass
class ActionBinding:
    """One discrete action and the keys/mouse events it sends (when applicable)."""

    name: str
    keys: tuple[str, ...] = ()
    mouse_click: tuple[str, ...] = ()
    mouse_move: tuple[int, int] | None = None
    duration: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ConfigError("Every action binding needs a non-empty name.")
        if len(self.name) > 64:
            raise ConfigError(f"Action name {self.name!r} is too long (max 64 characters).")
        keys: list[str] = []
        for key in self.keys:
            cleaned = str(key).strip().lower()
            if not cleaned:
                raise ConfigError(f"Action {self.name!r} contains an empty key name.")
            if cleaned in keys:
                raise ConfigError(f"Action {self.name!r} lists key {cleaned!r} twice.")
            keys.append(cleaned)
        self.keys = tuple(keys)
        clicks: list[str] = []
        for button in self.mouse_click:
            cleaned = str(button).strip().lower()
            if cleaned not in MOUSE_BUTTONS:
                raise ConfigError(
                    f"Action {self.name!r} has mouse button {button!r}; "
                    f"choose from {MOUSE_BUTTONS}."
                )
            clicks.append(cleaned)
        self.mouse_click = tuple(clicks)
        if self.mouse_move is not None:
            if not isinstance(self.mouse_move, (list, tuple)) or len(self.mouse_move) != 2:
                raise ConfigError(
                    f"Action {self.name!r}: mouse_move must be two integers [dx, dy]."
                )
            dx, dy = self.mouse_move
            self.mouse_move = (
                _as_int(dx, f"actions.mouse_move[x] for {self.name}", minimum=-10000, maximum=10000),
                _as_int(dy, f"actions.mouse_move[y] for {self.name}", minimum=-10000, maximum=10000),
            )
        if self.duration is not None:
            self.duration = _as_float(
                self.duration, f"actions.duration for {self.name}", minimum=0.0, maximum=10.0
            )
        if not isinstance(self.params, Mapping):
            raise ConfigError(f"Action {self.name!r}: params must be a mapping.")
        self.params = dict(self.params)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionBinding":
        raw = _mapping(data, "action binding")
        _check_unknown(
            raw,
            {"name", "keys", "mouse_click", "mouse_move", "duration", "params"},
            "action binding",
        )
        if "name" not in raw:
            raise ConfigError("Every action binding requires a name.")
        keys = raw.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        clicks = raw.get("mouse_click", [])
        if isinstance(clicks, str):
            clicks = [clicks]
        return cls(
            name=raw["name"],
            keys=tuple(keys or ()),
            mouse_click=tuple(clicks or ()),
            mouse_move=raw.get("mouse_move"),
            duration=raw.get("duration"),
            params=raw.get("params", {}) or {},
        )


@dataclass
class ActionsConfig:
    """Discrete action definitions plus shared timing/frame-skip settings."""

    frame_skip: int = 1
    duration: float = 0.05
    max_duration: float = 1.0
    bindings: tuple[ActionBinding, ...] = ()

    def __post_init__(self) -> None:
        self.frame_skip = _as_int(self.frame_skip, "actions.frame_skip", minimum=1, maximum=30)
        self.max_duration = _as_float(
            self.max_duration, "actions.max_duration", minimum=0.01, maximum=5.0
        )
        self.duration = _as_float(
            self.duration, "actions.duration", minimum=0.0, maximum=self.max_duration
        )
        self.bindings = tuple(self.bindings)
        if len(self.bindings) > 100:
            raise ConfigError("At most 100 action bindings are supported.")
        names = [binding.name for binding in self.bindings]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigError(f"Duplicate action binding name(s): {', '.join(duplicates)}.")
        for binding in self.bindings:
            if binding.duration is not None and binding.duration > self.max_duration:
                raise ConfigError(
                    f"actions.max_duration ({self.max_duration}) is smaller than the duration "
                    f"for action {binding.name!r} ({binding.duration})."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ActionsConfig":
        raw = _mapping(data, "actions")
        _check_unknown(raw, {"frame_skip", "duration", "max_duration", "bindings"}, "actions")
        defaults = cls()
        raw_bindings = raw.get("bindings", []) or []
        if not isinstance(raw_bindings, list):
            raise ConfigError("actions.bindings must be a YAML list.")
        bindings = tuple(ActionBinding.from_dict(item) for item in raw_bindings)
        return cls(
            frame_skip=raw.get("frame_skip", defaults.frame_skip),
            duration=raw.get("duration", defaults.duration),
            max_duration=raw.get("max_duration", defaults.max_duration),
            bindings=bindings,
        )

    @property
    def has_bindings(self) -> bool:
        return bool(self.bindings)

    def binding(self, index: int) -> ActionBinding:
        if not 0 <= index < len(self.bindings):
            raise IndexError(f"Action index {index} is outside 0..{len(self.bindings) - 1}.")
        return self.bindings[index]


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """A list of reward components plus an optional final clipping range."""

    components: tuple[dict[str, Any], ...] = ()
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        parsed: list[dict[str, Any]] = []
        for index, component in enumerate(self.components or ()):
            if not isinstance(component, Mapping):
                raise ConfigError(f"reward.components[{index}] must be a mapping.")
            item = dict(component)
            if not item.get("type"):
                raise ConfigError(f"reward.components[{index}] requires a 'type' field.")
            item["type"] = str(item["type"]).strip().lower()
            parsed.append(item)
        self.components = tuple(parsed)
        if self.clip is not None:
            if isinstance(self.clip, (int, float)) and not isinstance(self.clip, bool):
                bound = abs(_as_float(self.clip, "reward.clip", minimum=1e-6))
                self.clip = (-bound, bound)
            elif isinstance(self.clip, (list, tuple)) and len(self.clip) == 2:
                low = _as_float(self.clip[0], "reward.clip[0]")
                high = _as_float(self.clip[1], "reward.clip[1]")
                assert low is not None and high is not None
                if low > high:
                    raise ConfigError(f"reward.clip lower bound {low} exceeds upper bound {high}.")
                self.clip = (low, high)
            else:
                raise ConfigError("reward.clip must be null, a number, or [low, high].")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RewardConfig":
        raw = _mapping(data, "reward")
        _check_unknown(raw, {"components", "clip"}, "reward")
        components = raw.get("components", []) or []
        if not isinstance(components, list):
            raise ConfigError("reward.components must be a YAML list.")
        return cls(components=tuple(components), clip=raw.get("clip"))


# ---------------------------------------------------------------------------
# Episode configuration
# ---------------------------------------------------------------------------
@dataclass
class EpisodeConfig:
    """Episode length and an optional visual termination detector."""

    max_steps: int | None = None
    terminate_on: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None:
            self.max_steps = _as_int(
                self.max_steps, "episode.max_steps", minimum=1, maximum=1_000_000
            )
        if self.terminate_on is not None:
            raw = _mapping(self.terminate_on, "episode.terminate_on")
            if not raw.get("type"):
                raise ConfigError("episode.terminate_on requires a 'type' field.")
            raw["type"] = str(raw["type"]).strip().lower()
            self.terminate_on = raw

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "EpisodeConfig":
        raw = _mapping(data, "episode")
        _check_unknown(raw, {"max_steps", "terminate_on"}, "episode")
        return cls(max_steps=raw.get("max_steps"), terminate_on=raw.get("terminate_on"))


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Hyperparameters and training-loop behavior shared by all supported games."""

    algorithm: str = "ppo"
    timesteps: int = 50_000
    seed: int = 42
    policy: str | None = None
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 4
    checkpoint_freq: int = 10_000
    eval_freq: int = 0
    n_eval_episodes: int = 5
    save_best: bool = True
    tensorboard: bool = True
    device: str = "auto"
    verbose: int = 1
    progress_bar: bool = False
    stats_window_size: int = 100
    subproc: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.algorithm = str(self.algorithm).strip().lower()
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ConfigError(
                f"training.algorithm must be one of {SUPPORTED_ALGORITHMS}, "
                f"got {self.algorithm!r}."
            )
        self.timesteps = _as_int(
            self.timesteps, "training.timesteps", minimum=1, maximum=100_000_000
        )
        self.seed = _as_int(self.seed, "training.seed", minimum=0, maximum=2**32 - 1)
        self.learning_rate = _as_float(
            self.learning_rate, "training.learning_rate", minimum=1e-8, maximum=1.0
        )
        self.gamma = _as_float(self.gamma, "training.gamma", minimum=0.0, maximum=1.0)
        self.n_steps = _as_int(self.n_steps, "training.n_steps", minimum=8, maximum=1_000_000)
        self.batch_size = _as_int(self.batch_size, "training.batch_size", minimum=2, maximum=1_000_000)
        self.n_epochs = _as_int(self.n_epochs, "training.n_epochs", minimum=1, maximum=1000)
        self.gae_lambda = _as_float(self.gae_lambda, "training.gae_lambda", minimum=0.0, maximum=1.0)
        self.clip_range = _as_float(self.clip_range, "training.clip_range", minimum=0.0, maximum=1.0)
        self.ent_coef = _as_float(self.ent_coef, "training.ent_coef", minimum=0.0, maximum=10.0)
        self.vf_coef = _as_float(self.vf_coef, "training.vf_coef", minimum=0.0, maximum=10.0)
        self.max_grad_norm = _as_float(self.max_grad_norm, "training.max_grad_norm", minimum=0.0)
        self.n_envs = _as_int(self.n_envs, "training.n_envs", minimum=1, maximum=32)
        self.checkpoint_freq = _as_int(
            self.checkpoint_freq, "training.checkpoint_freq", minimum=0, maximum=100_000_000
        )
        self.eval_freq = _as_int(
            self.eval_freq, "training.eval_freq", minimum=0, maximum=100_000_000
        )
        self.n_eval_episodes = _as_int(
            self.n_eval_episodes, "training.n_eval_episodes", minimum=1, maximum=100
        )
        self.save_best = _as_bool(self.save_best, "training.save_best")
        self.tensorboard = _as_bool(self.tensorboard, "training.tensorboard")
        self.progress_bar = _as_bool(self.progress_bar, "training.progress_bar")
        self.subproc = _as_bool(self.subproc, "training.subproc")
        self.device = str(self.device).strip().lower()
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("training.device must be one of: auto, cpu, cuda, mps.")
        self.verbose = _as_int(self.verbose, "training.verbose", minimum=0, maximum=2)
        self.stats_window_size = _as_int(
            self.stats_window_size, "training.stats_window_size", minimum=1, maximum=10_000
        )
        if self.policy is not None:
            self.policy = str(self.policy).strip()
            if not self.policy:
                raise ConfigError("training.policy must be null or a non-empty policy name.")
        if not isinstance(self.extra_params, Mapping):
            raise ConfigError("training.extra_params must be a mapping.")
        self.extra_params = dict(self.extra_params)
        if self.algorithm == "ppo" and (self.n_steps * self.n_envs) % self.batch_size != 0:
            raise ConfigError(
                "For PPO, training.batch_size must divide training.n_steps * training.n_envs. "
                f"Currently {self.n_steps} * {self.n_envs} = {self.n_steps * self.n_envs}, "
                f"which is not divisible by batch_size={self.batch_size}."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrainingConfig":
        raw = _mapping(data, "training")
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        _check_unknown(raw, allowed, "training")
        defaults = cls()
        return cls(**{key: raw.get(key, getattr(defaults, key)) for key in allowed})


# ---------------------------------------------------------------------------
# Full experiment configuration
# ---------------------------------------------------------------------------
@dataclass
class GameConfig:
    """A complete experiment: game adapter, observations, actions, rewards, training."""

    name: str
    adapter: str
    params: dict[str, Any] = field(default_factory=dict)
    render_mode: str | None = None
    seed: int = 42
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    source_path: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ConfigError("game.name must not be empty.")
        self.adapter = str(self.adapter).strip()
        if not self.adapter:
            raise ConfigError("game.adapter must not be empty (e.g. 'gymnasium', 'example_game', 'desktop').")
        if not isinstance(self.params, Mapping):
            raise ConfigError("game.params must be a mapping.")
        self.params = dict(self.params)
        if self.render_mode is not None:
            self.render_mode = str(self.render_mode).strip().lower()
            if self.render_mode not in {"human", "rgb_array"}:
                raise ConfigError("game.render_mode must be null, 'human', or 'rgb_array'.")
        self.seed = _as_int(self.seed, "game.seed", minimum=0, maximum=2**32 - 1)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, source_path: str | Path | None = None
    ) -> "GameConfig":
        raw = _mapping(data, "configuration root")
        _check_unknown(
            raw,
            {"version", "description", "game", "observation", "actions", "reward", "episode", "training"},
            "configuration root",
        )
        version = raw.get("version", 1)
        if version != 1:
            raise ConfigError(f"Unsupported config version {version!r}; this release supports version 1.")
        game_raw = _mapping(raw.get("game"), "game")
        _check_unknown(game_raw, {"name", "adapter", "params", "render_mode", "seed"}, "game")
        for required in ("name", "adapter"):
            if required not in game_raw:
                raise ConfigError(f"game.{required} is required.")
        return cls(
            name=game_raw["name"],
            adapter=game_raw["adapter"],
            params=game_raw.get("params", {}) or {},
            render_mode=game_raw.get("render_mode"),
            seed=game_raw.get("seed", 42),
            observation=ObservationConfig.from_dict(raw.get("observation")),
            actions=ActionsConfig.from_dict(raw.get("actions")),
            reward=RewardConfig.from_dict(raw.get("reward")),
            episode=EpisodeConfig.from_dict(raw.get("episode")),
            training=TrainingConfig.from_dict(raw.get("training")),
            source_path=str(source_path) if source_path is not None else None,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the nested YAML/JSON form accepted by :meth:`from_dict`."""
        game: dict[str, Any] = {
            "name": self.name,
            "adapter": self.adapter,
            "params": dict(self.params),
            "seed": self.seed,
        }
        if self.render_mode is not None:
            game["render_mode"] = self.render_mode
        actions_payload = asdict(self.actions)
        actions_payload["bindings"] = [
            {
                "name": binding.name,
                "keys": list(binding.keys),
                "mouse_click": list(binding.mouse_click),
                "mouse_move": list(binding.mouse_move) if binding.mouse_move is not None else None,
                "duration": binding.duration,
                "params": dict(binding.params),
            }
            for binding in self.actions.bindings
        ]
        reward_payload = asdict(self.reward)
        reward_payload["components"] = [dict(component) for component in self.reward.components]
        reward_payload["clip"] = list(self.reward.clip) if self.reward.clip is not None else None
        return {
            "version": 1,
            "game": game,
            "observation": asdict(self.observation),
            "actions": actions_payload,
            "reward": reward_payload,
            "episode": asdict(self.episode),
            "training": asdict(self.training),
        }

    def with_overrides(self, **overrides: Any) -> "GameConfig":
        """Return a validated copy with selected fields replaced."""
        unknown = set(overrides) - set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        if unknown:
            raise ConfigError(f"Unknown config override(s): {', '.join(sorted(unknown))}")
        return replace(self, **overrides)


def load_game_config(path: str | Path) -> GameConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {config_path}:\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"Configuration must be a YAML mapping, got {type(data).__name__}.")
    return GameConfig.from_dict(data, source_path=config_path.resolve())


def find_config_for_game(game: str) -> Path:
    """Resolve ``--game name`` to ``configs/name.yaml`` without guessing silently."""
    from gamemind.utils.paths import DEFAULT_CONFIG_DIR

    if not game or not str(game).strip():
        raise ConfigError("--game must not be empty.")
    candidate = Path(game)
    if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.exists():
        resolved = resolve_project_path(candidate)
        if resolved.exists():
            return resolved
    cleaned = Path(str(game)).name
    stem = cleaned[:-5] if cleaned.lower().endswith(".yaml") else cleaned
    for suffix in (".yaml", ".yml"):
        path = DEFAULT_CONFIG_DIR / f"{stem}{suffix}"
        if path.exists():
            return path
    available = sorted(p.stem for p in DEFAULT_CONFIG_DIR.glob("*.yaml"))
    raise ConfigError(
        f"No configuration found for game {game!r}. Looked in {DEFAULT_CONFIG_DIR}. "
        f"Available configs: {', '.join(available) or '(none)'}. "
        "Pass --config path/to/file.yaml for a custom game."
    )


def apply_config_overrides(
    config: GameConfig,
    *,
    timesteps: int | None = None,
    algorithm: str | None = None,
    seed: int | None = None,
    observation_mode: str | None = None,
    width: int | None = None,
    height: int | None = None,
    stack: int | None = None,
    normalize: bool | None = None,
    frame_skip: int | None = None,
    action_duration: float | None = None,
    policy: str | None = None,
    n_envs: int | None = None,
    eval_freq: int | None = None,
    device: str | None = None,
    run_timesteps_override: bool = False,
) -> GameConfig:
    """Apply beginner-facing CLI/dashboard overrides while keeping validation central.

    ``run_timesteps_override`` exists so callers can distinguish "leave timesteps
    unchanged" from an explicit value. It is accepted for readability and has no
    behavioural effect.
    """
    del run_timesteps_override
    observation = config.observation
    action_config = config.actions
    training = config.training
    game_seed = config.seed

    if observation_mode is not None:
        mode = str(observation_mode).strip().lower()
        if mode == "state":
            observation = replace(observation, mode="state", stack=1, normalize=False)
        else:
            observation = replace(observation, mode=mode)
    if width is not None or height is not None:
        observation = replace(
            observation,
            width=width if width is not None else observation.width,
            height=height if height is not None else observation.height,
        )
    if stack is not None:
        observation = replace(observation, stack=stack)
    if normalize is not None:
        observation = replace(observation, normalize=normalize)
    if frame_skip is not None:
        action_config = replace(action_config, frame_skip=frame_skip)
    if action_duration is not None:
        if action_duration > action_config.max_duration:
            raise ConfigError(
                f"Action duration {action_duration}s exceeds actions.max_duration "
                f"({action_config.max_duration}s). Raise max_duration or choose a shorter duration."
            )
        action_config = replace(action_config, duration=action_duration)

    training_fields: dict[str, Any] = {}
    if timesteps is not None:
        training_fields["timesteps"] = timesteps
    if algorithm is not None:
        training_fields["algorithm"] = algorithm
    if seed is not None:
        training_fields["seed"] = seed
        game_seed = seed
    if policy is not None:
        training_fields["policy"] = policy
    if n_envs is not None:
        training_fields["n_envs"] = n_envs
    if eval_freq is not None:
        training_fields["eval_freq"] = eval_freq
    if device is not None:
        training_fields["device"] = device
    if training_fields:
        training = replace(training, **training_fields)

    return replace(
        config,
        seed=game_seed,
        observation=observation,
        actions=action_config,
        training=training,
    )
