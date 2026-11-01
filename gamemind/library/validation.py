"""Static and live validation for a level profile.

Validation is deliberately split in two:

* **Static checks** parse the level configuration and confirm that the adapter,
  observation mode, and completion settings are internally consistent. They need
  no game window and are what the test suite exercises.
* **Live checks** connect to the real game, reset it, capture one frame, and send
  one no-op action. They prove that the window is found and that the adapter can
  read the progress bar. They do *not* prove death detection, completion
  detection, or that a long training run will work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gamemind.adapters.registry import adapter_available
from gamemind.config import GameConfig
from gamemind.library.models import GameProfile, LevelProfile
from gamemind.library.store import GameLibrary
from gamemind.utils.errors import GameMindError
from gamemind.utils.serialization import utc_now_iso


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class ValidationReport:
    game_id: str
    level_id: str
    checks: list[Check] = field(default_factory=list)
    live: bool = False
    observed: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def static_ok(self) -> bool:
        return all(check.ok for check in self.checks if not check.name.startswith("live:"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "live": self.live,
            "created_at": self.created_at,
            "ok": self.ok,
            "static_ok": self.static_ok,
            "checks": [check.to_dict() for check in self.checks],
            "observed": self.observed,
        }


def static_checks(game: GameProfile, profile: LevelProfile, config: GameConfig) -> list[Check]:
    checks: list[Check] = []
    checks.append(
        Check(
            "Adapter available",
            adapter_available(game.adapter),
            f"adapter={game.adapter}",
        )
    )
    checks.append(
        Check(
            "Configuration matches the game",
            config.adapter.strip().lower() == game.adapter.strip().lower(),
            f"config adapter={config.adapter}",
        )
    )
    desktop = config.adapter.strip().lower() in {"geometry_dash", "desktop"}
    if desktop:
        checks.append(
            Check(
                "Learns from pixels only",
                config.observation.is_pixel,
                f"observation.mode={config.observation.mode}",
            )
        )
        checks.append(
            Check(
                "Single environment",
                config.training.n_envs == 1,
                f"n_envs={config.training.n_envs}",
            )
        )
    if config.adapter.strip().lower() == "geometry_dash":
        title = str(config.params.get("window_title") or "").strip()
        checks.append(
            Check(
                "Game window title configured",
                bool(title),
                f"window_title={title or '(empty)'}",
            )
        )
        checks.append(
            Check(
                "Window focus is required",
                config.params.get("require_window_focus") is True,
                "GameMind will not send keys to an unfocused window.",
            )
        )
        completion_on = bool(config.params.get("allow_bar_completion"))
        checks.append(
            Check(
                "Completion setting matches the level profile",
                completion_on == profile.completion_enabled,
                (
                    "Bar completion enabled; treat it as an estimate until observed."
                    if completion_on
                    else "Bar completion disabled; sessions cannot be marked as a verified win."
                ),
            )
        )
    if profile.completion_mode == "bar" and "completion_pct" not in config.params:
        checks.append(
            Check("Completion threshold present", False, "game.params.completion_pct is missing.")
        )
    return checks


def live_checks(config: GameConfig) -> tuple[list[Check], dict[str, Any]]:
    """Connect to the game, reset, take one frame, and send one no-op step."""
    from gamemind.envs import build_env

    checks: list[Check] = []
    observed: dict[str, Any] = {}
    env = None
    try:
        env = build_env(config)
        observation, info = env.reset()
        observed["reset_info"] = {
            key: info.get(key)
            for key in ("bar_found", "progress_pct", "max_progress_pct", "started", "attempt")
            if key in info
        }
        checks.append(Check("live: game window reachable and reset", True, "reset() succeeded"))
        checks.append(
            Check(
                "live: observation shape",
                getattr(observation, "shape", None) is not None,
                str(getattr(observation, "shape", None)),
            )
        )
        _, _, terminated, truncated, step_info = env.step(0)
        observed["step_info"] = {
            key: step_info.get(key)
            for key in ("bar_found", "progress_pct", "action_name", "died", "completed")
            if key in step_info
        }
        bar_found = step_info.get("bar_found")
        checks.append(
            Check(
                "live: progress bar detected",
                bool(bar_found) if bar_found is not None else False,
                "The adapter found the outlined progress bar in the captured frame."
                if bar_found
                else "No progress bar found. The game may be on a menu, paused, or the HUD needs calibration.",
            )
        )
        observed["terminated"] = bool(terminated)
        observed["truncated"] = bool(truncated)
        observed["not_verified"] = [
            "death detection",
            "completion detection",
            "reset synchronization over many attempts",
            "long-run stability",
        ]
    except GameMindError as exc:
        checks.append(Check("live: connect to the game", False, str(exc)))
    except Exception as exc:  # unexpected driver/OS failure
        checks.append(Check("live: connect to the game", False, f"{type(exc).__name__}: {exc}"))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return checks, observed


def validate_level(
    library: GameLibrary,
    game: GameProfile,
    level: LevelProfile,
    *,
    live: bool = False,
) -> ValidationReport:
    """Run validation and persist the report next to the level profile."""
    report = ValidationReport(game_id=game.id, level_id=level.id, live=live)
    try:
        config = library.level_config(game.id, level.id)
    except Exception as exc:
        report.checks.append(Check("Load level configuration", False, str(exc)))
        library.save_validation(game.id, level.id, report.to_dict())
        return report
    report.checks.append(Check("Load level configuration", True, str(config.source_path or "")))
    report.checks.extend(static_checks(game, level, config))
    if live:
        checks, observed = live_checks(config)
        report.checks.extend(checks)
        report.observed = observed
    library.save_validation(game.id, level.id, report.to_dict())
    return report
