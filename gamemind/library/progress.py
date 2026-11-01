"""Human-readable level status, stagnation detection, and diagnostics.

These functions only interpret persisted records. They never infer that a level
was beaten from a reward, a timestep budget, or an estimated percentage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gamemind.library.models import LevelProfile, SessionRecord
from gamemind.library.sessions import evaluation_score, level_summary, session_best_progress
from gamemind.library.store import GameLibrary

STATUS_LABELS: dict[str, str] = {
    "not_started": "Not started",
    "configured": "Configured",
    "training": "Training",
    "evaluated": "Evaluated",
    "improving": "Improving",
    "stagnating": "Stagnating",
    "completed": "Completed",
    "needs_attention": "Needs attention",
}

# A level must gain this many (estimated) percentage points before a session
# counts as an improvement. The bar reading itself is approximate, so tiny gains
# are treated as noise.
DEFAULT_MIN_DELTA = 0.5
DEFAULT_STAGNATION_WINDOW = 3


@dataclass
class LevelProgress:
    game_id: str
    level_id: str
    name: str
    status: str
    detail: str = ""
    sessions: int = 0
    attempts: int = 0
    total_timesteps: int = 0
    best_eval_progress_pct: float | None = None
    best_eval_score: float | None = None
    best_eval_metric: str | None = None
    previous_eval_progress_pct: float | None = None
    delta: float | None = None
    training_best_progress_pct: float | None = None
    verified_completion: bool = False
    completion_mode: str = "disabled"
    last_training_at: str | None = None
    last_state: str | None = None
    best_model_path: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "name": self.name,
            "status": self.status,
            "label": self.label,
            "detail": self.detail,
            "sessions": self.sessions,
            "attempts": self.attempts,
            "total_timesteps": self.total_timesteps,
            "best_eval_progress_pct": self.best_eval_progress_pct,
            "best_eval_score": self.best_eval_score,
            "best_eval_metric": self.best_eval_metric,
            "previous_eval_progress_pct": self.previous_eval_progress_pct,
            "delta": self.delta,
            "training_best_progress_pct": self.training_best_progress_pct,
            "verified_completion": self.verified_completion,
            "completion_mode": self.completion_mode,
            "last_training_at": self.last_training_at,
            "last_state": self.last_state,
            "best_model_path": self.best_model_path,
            "notes": list(self.notes),
        }


def _evaluated_sessions(sessions: list[SessionRecord]) -> list[SessionRecord]:
    return [session for session in sessions if evaluation_score(session.evaluation) is not None]


def improvement_delta(sessions: list[SessionRecord]) -> float | None:
    """Difference between the latest and previous comparable evaluation."""
    evaluated = _evaluated_sessions(sessions)
    if len(evaluated) < 2:
        return None
    latest = evaluation_score(evaluated[-1].evaluation)
    previous = evaluation_score(evaluated[-2].evaluation)
    if latest is None or previous is None or latest[0] != previous[0]:
        return None
    return round(latest[1] - previous[1], 3)


def is_stagnating(
    sessions: list[SessionRecord],
    *,
    window: int = DEFAULT_STAGNATION_WINDOW,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> bool:
    """True when recent evaluated sessions did not beat the earlier best.

    At least ``window + 1`` evaluated sessions are required, so a single unlucky
    run is never labelled stagnation.
    """
    from gamemind.library.sessions import evaluation_progress

    evaluated = _evaluated_sessions(sessions)
    if len(evaluated) < window + 1:
        return False
    scores = [(evaluation_score(session.evaluation) or ("", 0.0))[1] for session in evaluated]
    baseline = max(scores[: -window])
    recent = scores[-window:]
    return max(recent) < baseline + min_delta


def describe_level(
    library: GameLibrary,
    level: LevelProfile,
    *,
    running_session_id: str | None = None,
) -> LevelProgress:
    summary = level_summary(library, level)
    sessions = library.list_sessions(level.game_id, level.id)
    validation = library.load_validation(level.game_id, level.id)
    notes: list[str] = []

    evaluated = _evaluated_sessions(sessions)
    latest_eval = None
    previous_eval = None
    metric: str | None = None
    if evaluated:
        scored = evaluation_score(evaluated[-1].evaluation)
        if scored is not None:
            metric, latest_eval = scored
        if len(evaluated) >= 2:
            previous_scored = evaluation_score(evaluated[-2].evaluation)
            if previous_scored is not None and previous_scored[0] == metric:
                previous_eval = previous_scored[1]

    delta = None
    if latest_eval is not None and previous_eval is not None:
        delta = round(latest_eval - previous_eval, 3)

    last = sessions[-1] if sessions else None
    status: str
    detail: str

    if level.archived:
        status, detail = "configured", "This level profile is archived."
    elif summary["verified_completion"]:
        status = "completed"
        count = (summary.get("completion_evidence") or {}).get("count", 0)
        detail = (
            f"Verified by the level's completion detector ({count} signal(s)). "
            "Completion from the progress bar is an estimate; watch the recording before relying on it."
        )
    elif running_session_id and last is not None and last.session_id == running_session_id:
        status = "training"
        detail = "A training session is running for this level."
    elif last is not None and last.state == "error":
        status = "needs_attention"
        detail = f"The last session failed: {last.error or 'see the run log'}."
        notes.append("Open Training → last log to see the full error.")
    elif not sessions:
        if validation and validation.get("static_ok"):
            status, detail = "configured", "Configuration validated; no training session yet."
        else:
            status, detail = "not_started", "No training session yet."
            notes.append("Run Validate to check the level configuration before training.")
    elif is_stagnating(sessions):
        status = "stagnating"
        detail = (
            f"The last {DEFAULT_STAGNATION_WINDOW} evaluated sessions did not beat the earlier "
            "best. Diagnose before starting another long run."
        )
    elif delta is not None and delta >= DEFAULT_MIN_DELTA:
        status = "improving"
        unit = "percentage points" if metric == "progress_pct" else "reward"
        detail = f"Latest evaluation improved by {delta:+.2f} {unit}."
    elif len(evaluated) >= 2:
        status, detail = "evaluated", "Evaluations exist, but the latest one did not improve."
    elif evaluated:
        status, detail = "evaluated", "One evaluation is recorded; improvement needs a comparison."
    else:
        status = "configured"
        detail = "Training has run, but no evaluation was recorded, so there is no comparable result."
        notes.append(
            "Train again with evaluation enabled to measure progress; training reward is not a score."
        )

    if level.completion_mode == "disabled" and sessions:
        notes.append(
            "Completion detection is disabled for this level, so it can never be marked beaten."
        )

    last_training_at = None
    if last is not None:
        last_training_at = last.finished_at or last.started_at

    return LevelProgress(
        game_id=level.game_id,
        level_id=level.id,
        name=level.name,
        status=status,
        detail=detail,
        sessions=len(sessions),
        attempts=summary.get("attempts") or 0,
        total_timesteps=summary.get("total_timesteps") or 0,
        best_eval_progress_pct=summary.get("best_eval_progress_pct"),
        best_eval_score=summary.get("best_eval_score"),
        best_eval_metric=summary.get("best_eval_metric"),
        previous_eval_progress_pct=previous_eval,
        delta=delta,
        training_best_progress_pct=summary.get("training_best_progress_pct"),
        verified_completion=bool(summary["verified_completion"]),
        completion_mode=level.completion_mode,
        last_training_at=last_training_at,
        last_state=last.state if last else None,
        best_model_path=summary.get("best_model_path"),
        notes=notes,
    )


def session_rows(library: GameLibrary, level: LevelProfile) -> list[dict[str, Any]]:
    """Compact session history rows for the UI and for tests."""
    rows: list[dict[str, Any]] = []
    for session in reversed(library.list_sessions(level.game_id, level.id)):
        evaluation = session.evaluation or {}
        rows.append(
            {
                "session_id": session.session_id,
                "started_at": session.started_at,
                "finished_at": session.finished_at,
                "state": session.state,
                "reason": session.reason,
                "final_timesteps": session.final_timesteps,
                "requested_timesteps": session.requested_timesteps,
                "best_progress_pct": session_best_progress(session),
                "eval_mean_progress_pct": evaluation.get("mean_progress_pct"),
                "eval_mean_reward": evaluation.get("mean_reward"),
                "eval_score": (evaluation_score(evaluation) or (None, None))[1],
                "verified_completion": session.verified,
                "model_path": session.model_path,
                "error": session.error,
            }
        )
    return rows


def diagnose(
    library: GameLibrary, level: LevelProfile, progress: LevelProgress | None = None
) -> list[str]:
    """Concrete next checks when a level is not improving.

    These are suggestions based on stored measurements, not conclusions about
    the game. None of them starts a long run automatically.
    """
    state = progress or describe_level(library, level)
    suggestions: list[str] = []
    if state.status == "needs_attention":
        suggestions.append("Read the failed session's log before retrying; the cause is recorded.")
    if not library.load_validation(level.game_id, level.id):
        suggestions.append("Validate the level configuration; a missing adapter or window title stops runs immediately.")
    if state.status in {"stagnating", "evaluated"}:
        suggestions.append(
            "Run a deterministic 5-attempt evaluation and compare with the stored best, rather than judging by training reward."
        )
        suggestions.append(
            "Check the run's episodes.jsonl action counts. A policy that emits one action almost every step is not exploring."
        )
        suggestions.append(
            "Confirm the progress bar reading matches the percentage shown in game before trusting any improvement."
        )
    if state.total_timesteps and state.sessions:
        suggestions.append(
            f"{state.total_timesteps:,} timesteps across {state.sessions} session(s) have not produced a "
            "verified win. A longer run is only justified after the checks above."
        )
    if level.completion_mode == "disabled":
        suggestions.append(
            "Completion detection is off. Enable it for this level only after watching a manual win confirm the bar signal."
        )
    return suggestions


__all__ = [
    "DEFAULT_MIN_DELTA",
    "DEFAULT_STAGNATION_WINDOW",
    "LevelProgress",
    "STATUS_LABELS",
    "describe_level",
    "diagnose",
    "improvement_delta",
    "is_stagnating",
    "session_rows",
]
