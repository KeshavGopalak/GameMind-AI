"""Compatibility-aware initialization and transfer of policies.

Weights are only reused when the pieces that define "what the network sees and
can do" match. Game/level identity is *not* enough on its own: a checkpoint from
another level is transferred only when the level asks for it and every
observation/action check passes.

Nothing here mutates a saved model, and nothing copies a progress percentage from
one level to another. Transferring a policy is an experiment; its result is
measured by the target level's own evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gamemind.config import GameConfig
from gamemind.library.models import LevelProfile, SessionRecord
from gamemind.library.sessions import (
    best_validated_session,
    evaluation_progress,
    latest_resumable_session,
)
from gamemind.library.store import GameLibrary
from gamemind.training.model_io import metadata_path_for
from gamemind.utils.paths import resolve_project_path
from gamemind.utils.serialization import read_json

# Fields of the configuration that change what the policy can learn from or do.
# Comparing these is what makes "compatible" a check instead of an assumption.
_OBSERVATION_KEYS = ("mode", "width", "height", "stack", "normalize", "crop")
_ACTION_KEYS = ("frame_skip",)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class Initialization:
    """Where a new session should start from."""

    path: str | None = None
    source: str = "scratch"
    session_id: str | None = None
    source_level_id: str | None = None
    reason: str = ""
    checks: list[Check] = field(default_factory=list)

    @property
    def transferring(self) -> bool:
        return self.source == "transfer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "session_id": self.session_id,
            "source_level_id": self.source_level_id,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
        }


def _saved_config(metadata: dict[str, Any]) -> dict[str, Any] | None:
    config = metadata.get("config")
    return config if isinstance(config, dict) else None


def _saved_observation(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("observation") or {}


def _saved_actions(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("actions") or {}


def compatibility_checks(
    target: GameConfig,
    source_config: dict[str, Any] | None,
) -> list[Check]:
    checks: list[Check] = []
    if not source_config:
        checks.append(
            Check(
                "Saved configuration available",
                False,
                "The checkpoint has no metadata sidecar, so its observation and action layout "
                "cannot be verified. Retrain instead of guessing.",
            )
        )
        return checks

    source_adapter = str(source_config.get("game", {}).get("adapter") or "").strip().lower()
    checks.append(
        Check(
            "Adapter matches",
            source_adapter == target.adapter.strip().lower(),
            f"saved={source_adapter or '(unknown)'} current={target.adapter}",
        )
    )

    saved_obs = _saved_observation(source_config)
    current_obs = {
        "mode": target.observation.mode,
        "width": target.observation.width,
        "height": target.observation.height,
        "stack": target.observation.stack,
        "normalize": target.observation.normalize,
        "crop": list(target.observation.crop) if target.observation.crop else None,
    }
    saved_obs_view = {key: saved_obs.get(key) for key in _OBSERVATION_KEYS}
    saved_obs_view["crop"] = list(saved_obs_view["crop"]) if saved_obs_view.get("crop") else None
    differences = [
        f"{key}: saved={saved_obs_view.get(key)!r} current={current_obs[key]!r}"
        for key in _OBSERVATION_KEYS
        if saved_obs_view.get(key) != current_obs[key]
    ]
    checks.append(
        Check(
            "Observation preprocessing matches",
            not differences,
            "; ".join(differences) if differences else "mode, size, stack, normalization and crop match",
        )
    )

    saved_actions = _saved_actions(source_config)
    saved_bindings = saved_actions.get("bindings") or []
    current_bindings = [
        {
            "name": binding.name,
            "keys": list(binding.keys),
            "mouse_click": list(binding.mouse_click),
            "duration": binding.duration,
        }
        for binding in target.actions.bindings
    ]
    saved_view = [
        {
            "name": item.get("name"),
            "keys": list(item.get("keys") or []),
            "mouse_click": list(item.get("mouse_click") or []),
            "duration": item.get("duration"),
        }
        for item in saved_bindings
        if isinstance(item, dict)
    ]
    checks.append(
        Check(
            "Action bindings match",
            saved_view == current_bindings,
            (
                f"{len(saved_view)} saved vs {len(current_bindings)} current action(s); "
                "names, keys and durations must match exactly"
            ),
        )
    )
    saved_skip = saved_actions.get("frame_skip")
    checks.append(
        Check(
            "Frame skip matches",
            saved_skip == target.actions.frame_skip,
            f"saved={saved_skip} current={target.actions.frame_skip}",
        )
    )
    saved_episode = (source_config.get("episode") or {}).get("max_steps")
    checks.append(
        Check(
            "Episode length matches",
            saved_episode == target.episode.max_steps,
            f"saved={saved_episode} current={target.episode.max_steps}",
        )
    )
    return checks


def _read_source_config(path: str) -> dict[str, Any] | None:
    try:
        metadata = read_json(metadata_path_for(resolve_project_path(path)), default={})
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict):
        return None
    return _saved_config(metadata)


def _usable(path: str | None) -> bool:
    if not path:
        return False
    try:
        return resolve_project_path(path).exists()
    except (OSError, ValueError):
        return False


def choose_initialization(library: GameLibrary, level: LevelProfile) -> Initialization:
    """Pick a starting point for the next session on *level*.

    * ``scratch``           always start a fresh policy.
    * ``best_same_level``   reuse this level's best evaluated checkpoint.
    * ``transfer_game``     fall back to another level's compatible policy when
                            this level has no usable checkpoint.
    """
    target = library.level_config(level.game_id, level.id)

    if level.init_policy == "scratch":
        return Initialization(
            source="scratch",
            reason="This level is configured to start from a fresh policy.",
        )

    best = best_validated_session(library, level.game_id, level.id)
    if best is not None and _usable(best.model_path or best.best_model_path):
        best_path = best.model_path or best.best_model_path
        checks = compatibility_checks(target, _read_source_config(str(best_path)))
        return Initialization(
            path=best.model_path or best.best_model_path,
            source="same_level_best",
            session_id=best.session_id,
            source_level_id=level.id,
            reason=(
                f"Best evaluated checkpoint for this level "
                f"({evaluation_progress(best.evaluation) or 0.0:.2f}% mean max progress)."
            ),
            checks=checks,
        )

    latest = latest_resumable_session(library, level.game_id, level.id)
    if latest is not None and _usable(latest.model_path):
        return Initialization(
            path=latest.model_path,
            source="same_level_latest",
            session_id=latest.session_id,
            source_level_id=level.id,
            reason="Most recent saved checkpoint for this level (no evaluation to rank by yet).",
        )

    if level.init_policy != "transfer_game":
        return Initialization(
            source="scratch",
            reason="No checkpoint for this level yet; starting from a fresh policy.",
        )

    return find_transfer_source(library, level, target)


def find_transfer_source(
    library: GameLibrary, level: LevelProfile, target: GameConfig | None = None
) -> Initialization:
    """Find another level in the same game whose policy is safe to initialize from."""
    target = target or library.level_config(level.game_id, level.id)
    candidates: list[tuple[float, str, SessionRecord, list[Check]]] = []
    rejections: list[str] = []

    for other in library.list_levels(level.game_id, include_archived=False):
        if other.id == level.id:
            continue
        session = best_validated_session(library, level.game_id, other.id)
        if session is None:
            session = latest_resumable_session(library, level.game_id, other.id)
        if session is None or not _usable(session.model_path):
            continue
        source_config = _read_source_config(session.model_path)
        checks = compatibility_checks(target, source_config)
        if not all(check.ok for check in checks):
            failed = ", ".join(check.name for check in checks if not check.ok)
            rejections.append(f"{other.id}: {failed}")
            continue
        score = evaluation_progress(session.evaluation)
        candidates.append(
            (
                float(score) if score is not None else -1.0,
                other.id,
                session,
                checks,
            )
        )

    if not candidates:
        reason = "No compatible checkpoint exists in this game yet."
        if rejections:
            reason += " Incompatible candidates: " + "; ".join(rejections)
        return Initialization(source="scratch", reason=reason)

    candidates.sort(key=lambda item: (item[0], item[2].final_timesteps), reverse=True)
    score, source_level_id, session, checks = candidates[0]
    return Initialization(
        path=session.model_path,
        source="transfer",
        session_id=session.session_id,
        source_level_id=source_level_id,
        reason=(
            f"Initializing from level {source_level_id!r} (session {session.session_id}, "
            f"evaluated {score:.2f}% mean max progress). This is a starting point, not credit "
            "for that level's progress."
        ),
        checks=checks,
    )


def level_model_inventory(library: GameLibrary, level: LevelProfile) -> list[dict[str, Any]]:
    """Models stored for one level, with the metadata needed to judge them."""
    items: list[dict[str, Any]] = []
    directory = library.level_models_dir(level.game_id, level.id)
    if not directory.exists():
        return items
    for path in sorted(directory.glob("*.zip")):
        metadata = read_json(metadata_path_for(path), default={}) or {}
        items.append(
            {
                "path": str(path),
                "name": path.name,
                "num_timesteps": metadata.get("num_timesteps"),
                "created_at": metadata.get("created_at"),
                "run_id": metadata.get("run_id"),
                "algorithm": metadata.get("algorithm"),
                "compatible": not any(
                    not check.ok for check in compatibility_checks(
                        library.level_config(level.game_id, level.id),
                        _saved_config(metadata) if isinstance(metadata, dict) else None,
                    )
                ),
            }
        )
    return items


__all__ = [
    "Check",
    "Initialization",
    "choose_initialization",
    "compatibility_checks",
    "find_transfer_source",
    "level_model_inventory",
]
