"""Session records: creation, completion verification, and best-model selection.

A training session is the unit of persistence. Each one stores its configuration,
model paths, metrics, evaluation results, and the reason it ended. A level's
history is the ordered list of these records; nothing important lives only in a
terminal log.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from gamemind.library.models import LevelProfile, SessionRecord
from gamemind.library.store import GameLibrary
from gamemind.utils.errors import LibraryError
from gamemind.utils.paths import resolve_project_path
from gamemind.utils.serialization import read_json, read_jsonl

# Session states that mean "training reached a conclusion". A resumed session
# can start from any of these.
CONCLUDED_STATES = ("finished", "stopped", "interrupted", "imported")


def new_session_id(prefix: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(3)
    base = f"{stamp}-{token}"
    return f"{prefix}{base}" if prefix else base


def run_name_for(game_id: str, level_id: str, session_id: str) -> str:
    return f"{game_id}__{level_id}__{session_id}"


def model_path_for(
    library: GameLibrary, game_id: str, level_id: str, session_id: str
) -> Path:
    return library.level_model_path(game_id, level_id, session_id)


def run_dir_for(library: GameLibrary, game_id: str, level_id: str, run_name: str) -> Path:
    return library.level_run_dir(game_id, level_id, run_name)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    return False


def read_training_episodes(run_dir: str | Path | None) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    path = Path(run_dir) / "episodes.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def detect_completion(
    level: LevelProfile,
    episodes: Iterable[Mapping[str, Any]],
    evaluations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Decide whether a level was *verified* complete.

    Only a game-specific completion signal counts. Training ending, a timestep
    budget running out, or a high progress reward never marks a level complete.
    """
    if not level.completion_enabled:
        return {
            "verified": False,
            "mode": level.completion_mode,
            "count": 0,
            "evidence": [],
            "reason": (
                "Completion detection is disabled for this level. Training time, reward, "
                "and estimated progress are not accepted as proof that the level was beaten."
            ),
        }
    evidence: list[dict[str, Any]] = []
    for record in episodes:
        if _truthy(record.get("completed")):
            evidence.append(
                {
                    "source": "training_episode",
                    "episode": record.get("episode"),
                    "timesteps": record.get("timesteps"),
                    "max_progress_pct": record.get("max_progress_pct"),
                    "timestamp": record.get("timestamp"),
                }
            )
    for evaluation in evaluations:
        for episode in evaluation.get("episodes") or []:
            if not isinstance(episode, Mapping):
                continue
            if _truthy(episode.get("completed")):
                evidence.append(
                    {
                        "source": "evaluation",
                        "evaluation_id": evaluation.get("evaluation_id"),
                        "episode": episode.get("episode"),
                        "max_progress_pct": episode.get("max_progress_pct"),
                        "timestamp": evaluation.get("finished_at"),
                    }
                )
    if evidence:
        reason = (
            f"The level's completion detector fired {len(evidence)} time(s). "
            "Bar-based completion is an estimate and should be watched before it is trusted."
            if level.completion_mode == "bar"
            else f"Completion detector ({level.completion_mode}) fired {len(evidence)} time(s)."
        )
    else:
        reason = (
            "No verified completion signal was recorded in this session. "
            "Progress and reward are not treated as a win."
        )
    return {
        "verified": bool(evidence),
        "mode": level.completion_mode,
        "count": len(evidence),
        "evidence": evidence[:20],
        "reason": reason,
    }


def evaluation_progress(evaluation: Mapping[str, Any] | None) -> float | None:
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("mean_progress_pct")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def evaluation_score(evaluation: Mapping[str, Any] | None) -> tuple[str, float] | None:
    """The metric used to compare two evaluations of the same level.

    Estimated level progress is used when the game reports it. Other games are
    compared by mean episode reward, which is meaningful only within one game and
    reward configuration.
    """
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("mean_progress_pct")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "progress_pct", float(value)
    value = evaluation.get("mean_reward")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "reward", float(value)
    return None


def evaluation_best_progress(evaluation: Mapping[str, Any] | None) -> float | None:
    if not isinstance(evaluation, Mapping):
        return None
    values = [
        float(episode["max_progress_pct"])
        for episode in evaluation.get("episodes") or []
        if isinstance(episode, Mapping)
        and isinstance(episode.get("max_progress_pct"), (int, float))
        and not isinstance(episode.get("max_progress_pct"), bool)
    ]
    return max(values) if values else None


def session_best_progress(session: SessionRecord) -> float | None:
    """Best recorded progress for one session. Evaluation wins over training logs."""
    from_eval = evaluation_progress(session.evaluation)
    if from_eval is not None:
        return from_eval
    value = session.metrics.get("best_progress_pct") if session.metrics else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def session_best_eval_progress(session: SessionRecord) -> float | None:
    return evaluation_best_progress(session.evaluation)


def _model_exists(path: str | None) -> bool:
    if not path:
        return False
    try:
        return resolve_project_path(path).exists()
    except (OSError, ValueError):
        return False


def best_validated_session(
    library: GameLibrary, game_id: str, level_id: str
) -> SessionRecord | None:
    """The session with the best evaluated progress for this level.

    Training runs without an evaluation are not eligible: without a comparable
    measurement there is no evidence that one policy is better than another.
    """
    candidates = [
        session
        for session in library.list_sessions(game_id, level_id)
        if evaluation_score(session.evaluation) is not None
        and _model_exists(session.model_path or session.best_model_path)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda session: (
            (evaluation_score(session.evaluation) or ("", 0.0))[1],
            session.final_timesteps,
        ),
        reverse=True,
    )
    return candidates[0]


def latest_resumable_session(
    library: GameLibrary, game_id: str, level_id: str
) -> SessionRecord | None:
    """Most recent session that left a usable model behind."""
    for session in reversed(library.list_sessions(game_id, level_id)):
        if session.state in CONCLUDED_STATES and _model_exists(session.model_path):
            return session
    return None


def level_summary(library: GameLibrary, level: LevelProfile) -> dict[str, Any]:
    """Aggregate the persisted facts for one level."""
    sessions = library.list_sessions(level.game_id, level.id)
    evaluations = library.list_evaluations(level.game_id, level.id)
    best_session = best_validated_session(library, level.game_id, level.id)

    best_eval_progress = None
    best_eval_source = None
    best_eval_score = None
    best_eval_metric = None
    for session in sessions:
        value = evaluation_progress(session.evaluation)
        if value is not None and (best_eval_progress is None or value > best_eval_progress):
            best_eval_progress = value
            best_eval_source = session.session_id
        scored = evaluation_score(session.evaluation)
        if scored is not None and (best_eval_score is None or scored[1] > best_eval_score):
            best_eval_metric, best_eval_score = scored
    for evaluation in evaluations:
        value = evaluation.get("mean_progress_pct")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if best_eval_progress is None or float(value) > best_eval_progress:
                best_eval_progress = float(value)
                best_eval_source = str(evaluation.get("evaluation_id") or "evaluation")
        scored = evaluation_score(evaluation)
        if scored is not None and (best_eval_score is None or scored[1] > best_eval_score):
            best_eval_metric, best_eval_score = scored

    newest = sessions[-1] if sessions else None
    verified = any(session.verified for session in sessions)
    completion_evidence = next(
        (session.completion for session in reversed(sessions) if session.verified), {}
    )
    attempts = sum(
        int(session.metrics.get("episodes") or 0) for session in sessions if session.metrics
    )
    return {
        "level_id": level.id,
        "game_id": level.game_id,
        "name": level.name,
        "kind": level.kind,
        "completion_mode": level.completion_mode,
        "archived": level.archived,
        "sessions": len(sessions),
        "attempts": attempts,
        "last_session": newest.to_dict() if newest else None,
        "best_eval_progress_pct": best_eval_progress,
        "best_eval_source": best_eval_source,
        "best_eval_score": best_eval_score,
        "best_eval_metric": best_eval_metric,
        "best_model_path": (best_session.model_path if best_session else None),
        "best_session_id": (best_session.session_id if best_session else None),
        "training_best_progress_pct": max(
            (
                float(session.metrics["best_progress_pct"])
                for session in sessions
                if isinstance(session.metrics.get("best_progress_pct"), (int, float))
            ),
            default=None,
        ),
        "total_timesteps": sum(int(session.final_timesteps or 0) for session in sessions),
        "verified_completion": verified,
        "completion_evidence": completion_evidence,
    }


def record_from_run(
    *,
    level: LevelProfile,
    session: SessionRecord,
    evaluation: Mapping[str, Any] | None = None,
) -> SessionRecord:
    """Fill a session record from the files a finished run left on disk."""
    run_dir = Path(session.run_dir) if session.run_dir else None
    result_payload: dict[str, Any] = {}
    if run_dir is not None and (run_dir / "result.json").exists():
        value = read_json(run_dir / "result.json", default={})
        if isinstance(value, dict):
            result_payload = value
    if result_payload:
        session.final_timesteps = int(
            result_payload.get("final_timesteps") or session.final_timesteps or 0
        )
        session.model_path = result_payload.get("model_path") or session.model_path
        session.best_model_path = (
            result_payload.get("best_model_path") or session.best_model_path
        )
        metrics = result_payload.get("metrics")
        if isinstance(metrics, dict) and metrics:
            session.metrics = metrics
        if result_payload.get("error"):
            session.error = str(result_payload["error"])

    episodes = read_training_episodes(session.run_dir)
    evaluations = [evaluation] if isinstance(evaluation, Mapping) else []
    session.completion = detect_completion(level, episodes, evaluations)
    return session


__all__ = [
    "CONCLUDED_STATES",
    "best_validated_session",
    "detect_completion",
    "evaluation_best_progress",
    "evaluation_progress",
    "evaluation_score",
    "latest_resumable_session",
    "level_summary",
    "model_path_for",
    "new_session_id",
    "read_training_episodes",
    "record_from_run",
    "run_dir_for",
    "run_name_for",
    "session_best_eval_progress",
    "session_best_progress",
]
