"""Multi-level training queues that survive interruptions.

A queue is an ordered list of levels to train. Each entry produces its own
session record, so a queue never merges two levels' results. Progress is saved
after every level, and a queue that was interrupted resumes from its first
unfinished entry instead of starting over.

A level only moves to "completed" when that level's own completion detector
fired. Finishing the timestep budget moves the entry to "done".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gamemind.library.control import TrainingController
from gamemind.library.models import LevelProfile, QueueRecord, QueueTask
from gamemind.library.progress import describe_level
from gamemind.library.runner import SessionRequest, run_session
from gamemind.library.sessions import (
    new_session_id,
    run_dir_for,
    run_name_for,
    session_best_progress,
)
from gamemind.library.store import GameLibrary
from gamemind.utils.errors import LibraryError
from gamemind.utils.serialization import atomic_write_json, read_json, utc_now_iso


def queue_dir(library: GameLibrary) -> Path:
    directory = library.root / "queues"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def queue_path(library: GameLibrary, queue_id: str) -> Path:
    return queue_dir(library) / f"{queue_id}.json"


def load_queue(library: GameLibrary, queue_id: str) -> QueueRecord:
    path = queue_path(library, queue_id)
    if not path.exists():
        raise LibraryError(f"Queue {queue_id!r} does not exist.")
    return QueueRecord.from_dict(read_json(path, default={}) or {})


def save_queue(library: GameLibrary, queue: QueueRecord) -> Path:
    queue.updated_at = utc_now_iso()
    path = queue_path(library, queue.queue_id)
    atomic_write_json(path, queue.to_dict())
    return path


def list_queues(library: GameLibrary, game_id: str | None = None) -> list[QueueRecord]:
    queues: list[QueueRecord] = []
    for path in sorted(queue_dir(library).glob("*.json")):
        payload = read_json(path, default=None)
        if not isinstance(payload, dict):
            continue
        try:
            record = QueueRecord.from_dict(payload)
        except LibraryError:
            continue
        if game_id and record.game_id != game_id:
            continue
        queues.append(record)
    queues.sort(key=lambda record: record.created_at, reverse=True)
    return queues


def create_queue(
    library: GameLibrary,
    game_id: str,
    level_ids: list[str],
    *,
    timesteps_per_level: int,
    evaluate: bool = True,
    eval_episodes: int = 3,
    continue_after_completion: bool = False,
    queue_id: str | None = None,
) -> QueueRecord:
    if not level_ids:
        raise LibraryError("Select at least one level for the queue.")
    library.get_game(game_id)
    seen: set[str] = set()
    tasks: list[QueueTask] = []
    skipped: list[str] = []
    for level_id in level_ids:
        if level_id in seen:
            continue
        seen.add(level_id)
        level = library.get_level(game_id, level_id)
        if level.archived:
            skipped.append(f"{level_id} (archived)")
            continue
        progress = describe_level(library, level)
        if progress.verified_completion and not continue_after_completion:
            skipped.append(f"{level_id} (already verified complete)")
            continue
        tasks.append(QueueTask(level_id=level_id))
    if not tasks:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        raise LibraryError(
            "Every selected level is already verified complete or archived. "
            "Enable 'Continue training completed levels' to train them again." + detail
        )
    identifier = queue_id or new_session_id("queue-")
    queue = QueueRecord(
        queue_id=identifier,
        game_id=game_id,
        tasks=tasks,
        timesteps_per_level=int(timesteps_per_level),
        evaluate=bool(evaluate),
        eval_episodes=int(eval_episodes),
        continue_after_completion=bool(continue_after_completion),
        status="pending",
    )
    save_queue(library, queue)
    return queue


def recover_queue(library: GameLibrary, queue_id: str) -> QueueRecord:
    """Prepare an interrupted queue to continue.

    An entry that was mid-training when the process died goes back to pending;
    its partial session record is kept so nothing is silently discarded.
    """
    queue = load_queue(library, queue_id)
    changed = False
    for task in queue.tasks:
        if task.state == "running":
            task.state = "pending"
            task.finished_at = utc_now_iso()
            task.error = "Interrupted while training; will run again."
            changed = True
    if queue.status in {"running", "error"}:
        queue.status = "pending"
        queue.error = None
        changed = True
    if changed:
        save_queue(library, queue)
    return queue


def pending_tasks(queue: QueueRecord) -> list[QueueTask]:
    return [task for task in queue.tasks if task.state in {"pending", "interrupted"}]


def _next_task(queue: QueueRecord) -> QueueTask | None:
    for index, task in enumerate(queue.tasks):
        if task.state in {"pending", "interrupted"}:
            queue.current_index = index
            return task
    queue.current_index = len(queue.tasks)
    return None


def run_queue(
    library: GameLibrary,
    queue_id: str,
    *,
    controller: TrainingController | None = None,
    device: str = "auto",
) -> QueueRecord:
    """Execute pending entries in order, saving after every step."""
    queue = load_queue(library, queue_id)
    if queue.status == "completed":
        return queue

    for task in queue.tasks:
        if task.state == "running":
            task.state = "pending"
    queue.status = "running"
    queue.error = None
    save_queue(library, queue)

    controller = controller or TrainingController(library)
    try:
        while True:
            if controller.control_flags().get("stop"):
                queue.status = "stopped"
                break
            task = _next_task(queue)
            if task is None:
                queue.status = "completed"
                break
            level = library.get_level(queue.game_id, task.level_id)
            progress = describe_level(library, level)
            if progress.verified_completion and not queue.continue_after_completion:
                task.state = "completed"
                task.verified_completion = True
                task.finished_at = utc_now_iso()
                save_queue(library, queue)
                continue

            session_id = new_session_id()
            run_dir = run_dir_for(
                library, queue.game_id, level.id, run_name_for(queue.game_id, level.id, session_id)
            )
            controller.set_current(run_dir=str(run_dir), level_id=level.id)
            task.state = "running"
            task.session_id = session_id
            task.started_at = utc_now_iso()
            task.error = None
            save_queue(library, queue)

            record = run_session(
                library,
                SessionRequest(
                    game_id=queue.game_id,
                    level_id=level.id,
                    timesteps=queue.timesteps_per_level,
                    session_id=session_id,
                    evaluate=queue.evaluate,
                    eval_episodes=queue.eval_episodes,
                    device=device,
                    stop_on_completion=True,
                    notes=f"Queue {queue.queue_id} entry {queue.current_index + 1}.",
                ),
            )
            task.finished_at = utc_now_iso()
            task.session_id = record.session_id
            task.verified_completion = record.verified
            task.best_progress_pct = session_best_progress(record)
            if record.state == "error":
                task.state = "error"
                task.error = record.error or "The session failed."
                queue.status = "error"
                queue.error = task.error
                save_queue(library, queue)
                break
            task.state = "completed" if record.verified else "done"
            save_queue(library, queue)
            if controller.control_flags().get("stop"):
                queue.status = "stopped"
                break
    finally:
        if queue.status == "running":
            queue.status = "stopped"
        save_queue(library, queue)
    return queue


def queue_summary(library: GameLibrary, queue: QueueRecord) -> dict[str, Any]:
    """A compact, display-ready view of a queue."""
    rows: list[dict[str, Any]] = []
    for task in queue.tasks:
        try:
            level = library.get_level(queue.game_id, task.level_id)
            progress = describe_level(library, level)
            name = level.name
            best = progress.best_eval_progress_pct
            done = progress.verified_completion
            status = progress.label
        except LibraryError:
            name = task.level_id
            best = None
            done = task.verified_completion
            status = task.state
        rows.append(
            {
                "level_id": task.level_id,
                "level": name,
                "task_state": task.state,
                "level_status": status,
                "best_progress_pct": best,
                "verified_completion": bool(done or task.verified_completion),
                "session_id": task.session_id,
                "error": task.error,
            }
        )
    return {
        "queue_id": queue.queue_id,
        "game_id": queue.game_id,
        "status": queue.status,
        "current_index": queue.current_index,
        "timesteps_per_level": queue.timesteps_per_level,
        "continue_after_completion": queue.continue_after_completion,
        "tasks": rows,
        "remaining": len(pending_tasks(queue)),
        "error": queue.error,
        "updated_at": queue.updated_at,
    }


def ensure_level_task(library: GameLibrary, queue: QueueRecord, level: LevelProfile) -> None:
    """Add a level to an existing queue if it is not already there."""
    if any(task.level_id == level.id for task in queue.tasks):
        return
    queue.tasks.append(QueueTask(level_id=level.id))
    save_queue(library, queue)


__all__ = [
    "create_queue",
    "ensure_level_task",
    "list_queues",
    "load_queue",
    "pending_tasks",
    "queue_summary",
    "recover_queue",
    "run_queue",
    "save_queue",
]
