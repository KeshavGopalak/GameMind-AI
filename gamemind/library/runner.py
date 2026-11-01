"""Run one training session for a level and persist everything it produced.

This module is the bridge between the library records and the existing training
engine. It does not reimplement training: it loads the level configuration,
chooses a safe starting point, calls :class:`gamemind.training.Trainer`, then
writes a session record, an evaluation record, and verified-completion evidence.

The record is written *before* training starts and rewritten afterwards, so an
interrupted session is still visible in the UI instead of vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gamemind.library.models import LevelProfile, SessionRecord
from gamemind.library.sessions import (
    detect_completion,
    model_path_for,
    new_session_id,
    record_from_run,
    run_dir_for,
    run_name_for,
    session_best_progress,
)
from gamemind.library.store import GameLibrary
from gamemind.library.transfer import choose_initialization
from gamemind.utils.errors import GameMindError, LibraryError, ModelCompatibilityError
from gamemind.utils.logging import close_file_handlers, get_logger
from gamemind.utils.serialization import utc_now_iso

logger = get_logger("library")

_STATE_FROM_RESULT = {
    "completed": "finished",
    "stopped": "stopped",
    "interrupted": "interrupted",
}


@dataclass
class SessionRequest:
    game_id: str
    level_id: str
    timesteps: int
    session_id: str | None = None
    evaluate: bool = True
    eval_episodes: int = 3
    device: str = "auto"
    notes: str = ""
    stop_on_completion: bool = False


def _state_for(result: Any, error: str | None) -> tuple[str, str]:
    if error:
        return "error", error
    if getattr(result, "interrupted", False):
        return "interrupted", "Interrupted (Ctrl+C or process signal). Partial progress was saved."
    if getattr(result, "stopped", False):
        return "stopped", "Stopped on request. A checkpoint was saved before stopping."
    if getattr(result, "completed", False):
        return "finished", "Training budget finished normally."
    return "error", "Training ended without a result state."


def run_session(library: GameLibrary, request: SessionRequest) -> SessionRecord:
    """Train one level once, saving the session record and its evaluation."""
    game = library.get_game(request.game_id)
    level = library.get_level(request.game_id, request.level_id)
    if level.archived:
        raise LibraryError(f"Level {level.id!r} is archived; unarchive it before training.")
    if request.timesteps < 1:
        raise LibraryError("Timesteps must be at least 1.")

    session_id = request.session_id or new_session_id()
    run_name = run_name_for(game.id, level.id, session_id)
    model_path = model_path_for(library, game.id, level.id, session_id)
    run_dir = run_dir_for(library, game.id, level.id, run_name)
    checkpoint_dir = library.level_checkpoints_dir(game.id, level.id)
    if run_dir.exists():
        raise LibraryError(
            f"Run directory already exists for session {session_id!r}: {run_dir}. "
            "Each session needs a unique id."
        )

    config = library.level_config(game.id, level.id)
    config = replace(config, training=replace(config.training, timesteps=int(request.timesteps)))
    initialization = choose_initialization(library, level)

    previous_best = _previous_best_progress(library, level)

    record = SessionRecord(
        session_id=session_id,
        game_id=game.id,
        level_id=level.id,
        run_id=run_name,
        state="running",
        reason="Training started.",
        requested_timesteps=int(request.timesteps),
        model_path=str(model_path),
        checkpoint_dir=str(checkpoint_dir),
        run_dir=str(run_dir),
        status_path=str(run_dir / "status.json"),
        config_path=str(library.level_config_path(game.id, level.id)),
        resume_from=initialization.path,
        initialized_from=initialization.to_dict(),
        notes=request.notes,
        started_at=utc_now_iso(),
    )
    library.save_session(record)
    logger.info(
        "Session %s for %s/%s: %d timesteps, initialization=%s",
        session_id,
        game.id,
        level.id,
        request.timesteps,
        initialization.source,
    )

    error: str | None = None
    result = None
    try:
        from gamemind.training import Trainer

        trainer = Trainer(
            config,
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=library.level_dir(game.id, level.id),
            run_name=run_name,
            resume_from=initialization.path,
            allow_transfer=initialization.transferring,
            stop_on_completion=request.stop_on_completion and level.completion_enabled,
        )
        result = trainer.run()
    except ModelCompatibilityError as exc:
        error = f"Initialization model is not compatible with this level: {exc}"
        logger.error(error)
    except GameMindError as exc:
        error = str(exc)
        logger.error("Session %s failed: %s", session_id, error)
    except Exception as exc:  # unexpected failure must still be recorded
        logger.exception("Session %s crashed", session_id)
        error = f"{type(exc).__name__}: {exc}"

    state, reason = _state_for(result, error)
    record.state = state
    record.reason = reason
    record.error = error
    record.finished_at = utc_now_iso()

    evaluation_payload: dict[str, Any] | None = None
    if result is not None:
        record.final_timesteps = int(getattr(result, "final_timesteps", 0) or 0)
        record.model_path = str(result.model_path) if result.model_path else record.model_path
        record.best_model_path = str(result.best_model_path) if result.best_model_path else None
        record.metrics = dict(getattr(result, "metrics", {}) or {})
        if result.error and not error:
            record.error = result.error

    record_from_run(level=level, session=record)

    if (
        request.evaluate
        and record.state != "error"
        and record.model_path
        and Path(record.model_path).exists()
    ):
        evaluation_payload, evaluation_error = _evaluate(
            config, record.model_path, session_id, game.id, level.id, request
        )
        if evaluation_payload is not None:
            record.evaluation = evaluation_payload
            library.save_evaluation(
                game.id, level.id, evaluation_payload, evaluation_id=f"{session_id}-eval"
            )
        record.evaluation_error = evaluation_error

    # Completion evidence now includes this session's evaluation, if any.
    record.completion = detect_completion(
        level,
        _episodes_of(record),
        [evaluation_payload] if evaluation_payload else [],
    )

    new_best = session_best_progress(record)
    record.extra = {
        "applicable_skills": _applicable_skill_ids(library, game.id, config),
        "previous_best_progress_pct": previous_best,
        "session_best_progress_pct": new_best,
        "regressed": bool(
            previous_best is not None and new_best is not None and new_best + 0.5 < previous_best
        ),
        "retained_best_model": _retained_best_model(library, level),
        "attempts": int(record.metrics.get("episodes") or 0),
    }
    library.save_session(record)
    close_file_handlers()
    logger.info("Session %s finished with state %s.", session_id, record.state)
    return record


def _applicable_skill_ids(library: GameLibrary, game_id: str, config: Any) -> list[str]:
    """Skills that are compatible with this environment, recorded for reference.

    These are notes with matching conditions, not weights: the policy still
    starts from the checkpoint chosen by the level's initialization policy.
    """
    from gamemind.library.skills import applicable_skills

    try:
        return [skill.skill_id for skill in applicable_skills(library, game_id, config)]
    except Exception:
        return []


def _episodes_of(record: SessionRecord) -> list[dict[str, Any]]:
    from gamemind.library.sessions import read_training_episodes

    return read_training_episodes(record.run_dir)


def _previous_best_progress(library: GameLibrary, level: LevelProfile) -> float | None:
    values = [
        session_best_progress(session)
        for session in library.list_sessions(level.game_id, level.id)
        if session.state != "running"
    ]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _retained_best_model(library: GameLibrary, level: LevelProfile) -> str | None:
    from gamemind.library.sessions import best_validated_session

    best = best_validated_session(library, level.game_id, level.id)
    return best.model_path if best is not None else None


def _evaluate(
    config: Any,
    model_path: str,
    session_id: str,
    game_id: str,
    level_id: str,
    request: SessionRequest,
) -> tuple[dict[str, Any] | None, str | None]:
    """Play deterministic episodes without training. Never fatal for the session."""
    from gamemind.evaluation import run_episodes

    try:
        result = run_episodes(
            config,
            model_path,
            episodes=max(1, int(request.eval_episodes)),
            deterministic=True,
            render=False,
            seed=config.training.seed,
            device=request.device,
        )
        payload = result.as_dict()
        payload["session_id"] = session_id
        payload["level_id"] = level_id
        payload["game_id"] = game_id
        payload["evaluation_id"] = f"{session_id}-eval"
        payload["notes"] = (
            "Progress values are estimated from the game's on-screen indicator. "
            "They are comparable only between runs of this same level and configuration."
        )
        return payload, None
    except GameMindError as exc:
        logger.warning("Post-training evaluation failed: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.exception("Post-training evaluation crashed")
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_session_model(
    library: GameLibrary,
    level: LevelProfile,
    *,
    model_path: str,
    episodes: int = 5,
    device: str = "auto",
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved model and store the result in the level's history."""
    from gamemind.evaluation import run_episodes
    from gamemind.library.sessions import new_session_id

    config = library.level_config(level.game_id, level.id)
    result = run_episodes(
        config,
        model_path,
        episodes=max(1, int(episodes)),
        deterministic=True,
        render=False,
        seed=config.training.seed,
        device=device,
    )
    payload = result.as_dict()
    payload["level_id"] = level.id
    payload["game_id"] = level.game_id
    identifier = evaluation_id or new_session_id("eval-")
    payload["evaluation_id"] = identifier
    payload["notes"] = (
        "Deterministic evaluation of a saved model. Progress is an estimate from the game's "
        "on-screen indicator; compare only within this level."
    )
    library.save_evaluation(level.game_id, level.id, payload, evaluation_id=identifier)
    return payload


__all__ = ["SessionRequest", "evaluate_session_model", "run_session"]
