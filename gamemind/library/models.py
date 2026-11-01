"""Records stored by the GameMind game/level library.

Every record is a small JSON document. Models and checkpoints stay in the
existing ``models/`` and ``checkpoints/`` roots; these records only reference
them. That keeps the library readable, diffable, and safe to back up without
copying gigabytes of weights.

The records are deliberately explicit about what they do *not* know:

* a session ending never implies the level was beaten,
* a progress percentage observed during training is an estimate,
* skills carry evidence and a confidence value, not assumed understanding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from gamemind.utils.errors import LibraryError
from gamemind.utils.serialization import utc_now_iso

GAME_ADAPTERS_HELP = (
    "Supported adapters are listed by scripts/test_setup.py and in the Add Game form."
)

_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")
_VALID_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

LEVEL_KINDS = ("official", "custom")
COMPLETION_MODES = ("disabled", "bar")
INIT_POLICIES = ("scratch", "best_same_level", "transfer_game")
SESSION_STATES = ("running", "finished", "stopped", "interrupted", "error", "imported")
QUEUE_STATES = ("pending", "running", "paused", "stopped", "completed", "error")
SKILL_KINDS = ("control", "observation", "reset", "temporal", "strategy")


def slugify(value: str, *, fallback: str = "item", max_length: int = 64) -> str:
    """Turn free text into a stable lowercase id."""
    cleaned = _SLUG_TOKEN.sub("_", str(value).strip().lower()).strip("_")
    return (cleaned[:max_length] or fallback)


def validate_id(value: str, *, what: str = "id") -> str:
    text = str(value).strip()
    if not _VALID_ID.match(text):
        raise LibraryError(
            f"Invalid {what} {value!r}. Use lowercase letters, digits, '-' or '_' "
            "and start with a letter or digit."
        )
    return text


def _optional_text(value: Any, *, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        raise LibraryError(f"Text field is too long ({len(text)} > {limit} characters).")
    return text


def _choice(value: Any, allowed: tuple[str, ...], *, what: str) -> str:
    text = str(value).strip().lower()
    if text not in allowed:
        raise LibraryError(f"{what} must be one of {', '.join(allowed)}; got {value!r}.")
    return text


def _finite(value: Any, *, what: str, minimum: float = 0.0, maximum: float = 100.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LibraryError(f"{what} must be a number.")
    number = float(value)
    if not (minimum <= number <= maximum):
        raise LibraryError(f"{what} must be between {minimum} and {maximum}.")
    return number


# ---------------------------------------------------------------------------
# Game profiles
# ---------------------------------------------------------------------------
@dataclass
class GameProfile:
    """One installed game. Its adapter decides how GameMind connects to it."""

    id: str
    name: str
    adapter: str
    description: str = ""
    setup_notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    archived: bool = False

    def __post_init__(self) -> None:
        self.id = validate_id(self.id, what="game id")
        self.name = _optional_text(self.name, limit=200).strip()
        if not self.name:
            raise LibraryError("A game needs a display name.")
        self.adapter = _optional_text(self.adapter, limit=200).strip()
        if not self.adapter:
            raise LibraryError("A game needs an adapter name.")
        self.description = _optional_text(self.description)
        self.setup_notes = _optional_text(self.setup_notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "adapter": self.adapter,
            "description": self.description,
            "setup_notes": self.setup_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": bool(self.archived),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameProfile":
        if not isinstance(data, Mapping):
            raise LibraryError("Game profile must be a mapping.")
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            adapter=data.get("adapter", ""),
            description=data.get("description", ""),
            setup_notes=data.get("setup_notes", ""),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            archived=bool(data.get("archived", False)),
        )


# ---------------------------------------------------------------------------
# Level profiles
# ---------------------------------------------------------------------------
@dataclass
class LevelProfile:
    """One level inside a game, plus how GameMind should treat finishing it."""

    id: str
    game_id: str
    name: str
    kind: str = "official"
    entry: str = "manual"
    completion_mode: str = "disabled"
    completion_pct: float = 99.0
    completion_frames: int = 2
    init_policy: str = "best_same_level"
    notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    archived: bool = False

    def __post_init__(self) -> None:
        self.id = validate_id(self.id, what="level id")
        self.game_id = validate_id(self.game_id, what="game id")
        self.name = _optional_text(self.name, limit=200).strip()
        if not self.name:
            raise LibraryError("A level needs a display name.")
        self.kind = _choice(self.kind, LEVEL_KINDS, what="level kind")
        self.entry = _optional_text(self.entry, limit=200).strip() or "manual"
        self.completion_mode = _choice(
            self.completion_mode, COMPLETION_MODES, what="completion mode"
        )
        self.completion_pct = _finite(self.completion_pct, what="completion percentage", minimum=50.0)
        if isinstance(self.completion_frames, bool) or not isinstance(self.completion_frames, int):
            raise LibraryError("completion_frames must be an integer.")
        if not 1 <= self.completion_frames <= 20:
            raise LibraryError("completion_frames must be between 1 and 20.")
        self.init_policy = _choice(self.init_policy, INIT_POLICIES, what="initialization policy")
        self.notes = _optional_text(self.notes)

    @property
    def completion_enabled(self) -> bool:
        return self.completion_mode != "disabled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "game_id": self.game_id,
            "name": self.name,
            "kind": self.kind,
            "entry": self.entry,
            "completion_mode": self.completion_mode,
            "completion_pct": self.completion_pct,
            "completion_frames": self.completion_frames,
            "init_policy": self.init_policy,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": bool(self.archived),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LevelProfile":
        if not isinstance(data, Mapping):
            raise LibraryError("Level profile must be a mapping.")
        return cls(
            id=data.get("id", ""),
            game_id=data.get("game_id", ""),
            name=data.get("name", ""),
            kind=data.get("kind", "official"),
            entry=data.get("entry", "manual"),
            completion_mode=data.get("completion_mode", "disabled"),
            completion_pct=data.get("completion_pct", 99.0),
            completion_frames=data.get("completion_frames", 2),
            init_policy=data.get("init_policy", "best_same_level"),
            notes=data.get("notes", ""),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            archived=bool(data.get("archived", False)),
        )


# ---------------------------------------------------------------------------
# Session records
# ---------------------------------------------------------------------------
@dataclass
class SessionRecord:
    """One training session for one level, written at start and updated at the end."""

    session_id: str
    game_id: str
    level_id: str
    run_id: str
    state: str = "running"
    reason: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    requested_timesteps: int = 0
    initial_timesteps: int = 0
    final_timesteps: int = 0
    model_path: str | None = None
    best_model_path: str | None = None
    checkpoint_dir: str | None = None
    run_dir: str | None = None
    status_path: str | None = None
    config_path: str | None = None
    resume_from: str | None = None
    initialized_from: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] | None = None
    evaluation_error: str | None = None
    completion: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = validate_id(self.session_id, what="session id")
        self.game_id = validate_id(self.game_id, what="game id")
        self.level_id = validate_id(self.level_id, what="level id")
        self.run_id = validate_id(self.run_id, what="run id")
        self.state = _choice(self.state, SESSION_STATES, what="session state")
        if isinstance(self.metrics, Mapping):
            self.metrics = dict(self.metrics)
        if self.completion is None:
            self.completion = {}

    @property
    def verified(self) -> bool:
        return bool(self.completion.get("verified"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "game_id": self.game_id,
            "level_id": self.level_id,
            "run_id": self.run_id,
            "state": self.state,
            "reason": self.reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "requested_timesteps": self.requested_timesteps,
            "initial_timesteps": self.initial_timesteps,
            "final_timesteps": self.final_timesteps,
            "model_path": self.model_path,
            "best_model_path": self.best_model_path,
            "checkpoint_dir": self.checkpoint_dir,
            "run_dir": self.run_dir,
            "status_path": self.status_path,
            "config_path": self.config_path,
            "resume_from": self.resume_from,
            "initialized_from": self.initialized_from,
            "metrics": self.metrics,
            "evaluation": self.evaluation,
            "evaluation_error": self.evaluation_error,
            "completion": self.completion,
            "error": self.error,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionRecord":
        if not isinstance(data, Mapping):
            raise LibraryError("Session record must be a mapping.")
        return cls(
            session_id=data.get("session_id", ""),
            game_id=data.get("game_id", ""),
            level_id=data.get("level_id", ""),
            run_id=data.get("run_id", ""),
            state=data.get("state", "running"),
            reason=str(data.get("reason") or ""),
            started_at=str(data.get("started_at") or utc_now_iso()),
            finished_at=data.get("finished_at"),
            requested_timesteps=int(data.get("requested_timesteps") or 0),
            initial_timesteps=int(data.get("initial_timesteps") or 0),
            final_timesteps=int(data.get("final_timesteps") or 0),
            model_path=data.get("model_path"),
            best_model_path=data.get("best_model_path"),
            checkpoint_dir=data.get("checkpoint_dir"),
            run_dir=data.get("run_dir"),
            status_path=data.get("status_path"),
            config_path=data.get("config_path"),
            resume_from=data.get("resume_from"),
            initialized_from=data.get("initialized_from"),
            metrics=dict(data.get("metrics") or {}),
            evaluation=data.get("evaluation"),
            evaluation_error=data.get("evaluation_error"),
            completion=dict(data.get("completion") or {}),
            error=data.get("error"),
            notes=str(data.get("notes") or ""),
            extra=dict(data.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
@dataclass
class SkillRecord:
    """Validated, reusable knowledge for one game.

    A skill is a note with evidence. It is never a substitute for a trained
    policy and never shared across games implicitly.
    """

    skill_id: str
    game_id: str
    name: str
    kind: str
    summary: str
    confidence: float = 0.0
    validated: bool = False
    conditions: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.skill_id = validate_id(self.skill_id, what="skill id")
        self.game_id = validate_id(self.game_id, what="game id")
        self.name = _optional_text(self.name, limit=200).strip()
        if not self.name:
            raise LibraryError("A skill needs a name.")
        self.kind = _choice(self.kind, SKILL_KINDS, what="skill kind")
        self.summary = _optional_text(self.summary)
        self.confidence = _finite(self.confidence, what="confidence", minimum=0.0, maximum=1.0)
        self.validated = bool(self.validated)
        self.conditions = dict(self.conditions or {})
        self.evidence = [dict(item) for item in (self.evidence or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "game_id": self.game_id,
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "confidence": self.confidence,
            "validated": self.validated,
            "conditions": self.conditions,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillRecord":
        return cls(
            skill_id=data.get("skill_id", ""),
            game_id=data.get("game_id", ""),
            name=data.get("name", ""),
            kind=data.get("kind", "strategy"),
            summary=data.get("summary", ""),
            confidence=data.get("confidence", 0.0),
            validated=data.get("validated", False),
            conditions=dict(data.get("conditions") or {}),
            evidence=list(data.get("evidence") or []),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
        )


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
@dataclass
class QueueTask:
    level_id: str
    state: str = "pending"
    session_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    verified_completion: bool = False
    best_progress_pct: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "state": self.state,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "verified_completion": bool(self.verified_completion),
            "best_progress_pct": self.best_progress_pct,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueueTask":
        return cls(
            level_id=validate_id(data.get("level_id", ""), what="level id"),
            state=str(data.get("state") or "pending"),
            session_id=data.get("session_id"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            verified_completion=bool(data.get("verified_completion", False)),
            best_progress_pct=data.get("best_progress_pct"),
            error=data.get("error"),
        )


@dataclass
class QueueRecord:
    """An ordered multi-level training plan that survives restarts."""

    queue_id: str
    game_id: str
    tasks: list[QueueTask] = field(default_factory=list)
    timesteps_per_level: int = 50_000
    evaluate: bool = True
    eval_episodes: int = 3
    continue_after_completion: bool = False
    status: str = "pending"
    current_index: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    error: str | None = None

    def __post_init__(self) -> None:
        self.queue_id = validate_id(self.queue_id, what="queue id")
        self.game_id = validate_id(self.game_id, what="game id")
        self.tasks = [task if isinstance(task, QueueTask) else QueueTask.from_dict(task) for task in self.tasks]
        if not self.tasks:
            raise LibraryError("A queue needs at least one level.")
        if len(self.tasks) > 200:
            raise LibraryError("A queue supports at most 200 level entries.")
        if isinstance(self.timesteps_per_level, bool) or not isinstance(self.timesteps_per_level, int):
            raise LibraryError("timesteps_per_level must be an integer.")
        if self.timesteps_per_level < 1:
            raise LibraryError("timesteps_per_level must be at least 1.")
        self.status = _choice(self.status, QUEUE_STATES, what="queue status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "game_id": self.game_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "timesteps_per_level": self.timesteps_per_level,
            "evaluate": self.evaluate,
            "eval_episodes": self.eval_episodes,
            "continue_after_completion": self.continue_after_completion,
            "status": self.status,
            "current_index": self.current_index,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueueRecord":
        return cls(
            queue_id=data.get("queue_id", ""),
            game_id=data.get("game_id", ""),
            tasks=[QueueTask.from_dict(item) for item in data.get("tasks") or []],
            timesteps_per_level=int(data.get("timesteps_per_level") or 0),
            evaluate=bool(data.get("evaluate", True)),
            eval_episodes=int(data.get("eval_episodes") or 3),
            continue_after_completion=bool(data.get("continue_after_completion", False)),
            status=data.get("status", "pending"),
            current_index=int(data.get("current_index") or 0),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            error=data.get("error"),
        )
