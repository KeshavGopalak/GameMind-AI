"""Start, pause, resume, and stop background training without duplicates.

Training never runs inside the Streamlit process. The UI (or a script) writes a
job file and launches ``scripts/library_worker.py``; that worker owns the
trainer. This module is the single place that decides whether a worker is
already running, so a double-click cannot start two agents on one game window.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gamemind.library.models import LevelProfile, QueueRecord
from gamemind.library.store import GameLibrary
from gamemind.utils.errors import LibraryError, WorkerError
from gamemind.utils.paths import PROJECT_ROOT, ensure_dir
from gamemind.utils.process import launch_detached, pid_alive, stop_process
from gamemind.utils.serialization import atomic_write_json, read_json, utc_now_iso

WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "library_worker.py"


@dataclass
class WorkerState:
    pid: int
    kind: str
    job_path: str
    status: str = "running"
    game_id: str | None = None
    level_id: str | None = None
    session_id: str | None = None
    queue_id: str | None = None
    run_dir: str | None = None
    log_path: str | None = None
    started_at: str = utc_now_iso()
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "kind": self.kind,
            "job_path": self.job_path,
            "status": self.status,
            "game_id": self.game_id,
            "level_id": self.level_id,
            "session_id": self.session_id,
            "queue_id": self.queue_id,
            "run_dir": self.run_dir,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerState":
        return cls(
            pid=int(data.get("pid") or 0),
            kind=str(data.get("kind") or "session"),
            job_path=str(data.get("job_path") or ""),
            status=str(data.get("status") or "running"),
            game_id=data.get("game_id"),
            level_id=data.get("level_id"),
            session_id=data.get("session_id"),
            queue_id=data.get("queue_id"),
            run_dir=data.get("run_dir"),
            log_path=data.get("log_path"),
            started_at=str(data.get("started_at") or utc_now_iso()),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
        )


class TrainingController:
    """Owns the one background worker a library is allowed to have."""

    def __init__(self, library: GameLibrary) -> None:
        self.library = library
        self.state_dir = ensure_dir(library.state_dir)
        self.state_path = self.state_dir / "worker.json"
        self.lock_path = self.state_dir / "worker.lock"
        self.control_path = self.state_dir / "control.json"
        self.current_path = self.state_dir / "current_session.json"
        self.jobs_dir = ensure_dir(self.state_dir / "jobs")
        self.log_dir = ensure_dir(self.state_dir / "logs")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def state(self) -> WorkerState | None:
        payload = read_json(self.state_path, default=None)
        if not isinstance(payload, dict):
            return None
        try:
            return WorkerState.from_dict(payload)
        except (TypeError, ValueError):
            return None

    def is_running(self) -> bool:
        state = self.state()
        if state is None or state.status != "running":
            return False
        return pid_alive(state.pid)

    def reconcile(self) -> WorkerState | None:
        """Mark a worker whose process vanished without updating its own record."""
        state = self.state()
        if state is None:
            return None
        if state.status == "running" and not pid_alive(state.pid):
            state.status = "lost"
            state.finished_at = state.finished_at or utc_now_iso()
            state.error = state.error or (
                "The worker process is no longer running. Check the worker log; the session "
                "record keeps whatever was saved before it stopped."
            )
            self._write_state(state)
        return state

    def _write_state(self, state: WorkerState) -> None:
        atomic_write_json(self.state_path, state.to_dict())

    def mark_finished(self, status: str, *, error: str | None = None) -> None:
        """Called from inside the worker when it stops."""
        state = self.state()
        if state is None:
            return
        state.status = status
        state.finished_at = utc_now_iso()
        if error:
            state.error = error
        if state.run_dir:
            state.run_dir = state.run_dir
        self._write_state(state)

    def set_current(self, *, run_dir: str | None, level_id: str | None = None) -> None:
        atomic_write_json(
            self.current_path,
            {"run_dir": run_dir, "level_id": level_id, "updated_at": utc_now_iso()},
        )

    # ------------------------------------------------------------------
    # Starting work
    # ------------------------------------------------------------------
    def _acquire_lock(self) -> None:
        for _attempt in range(2):
            try:
                handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = self.state()
                if existing is not None and pid_alive(existing.pid):
                    raise WorkerError(
                        f"A training worker is already running (PID {existing.pid}). "
                        "Stop it or wait for it to finish."
                    )
                try:
                    self.lock_path.unlink()
                except OSError:
                    raise WorkerError(
                        "Another start is already in progress. Wait a moment and refresh."
                    ) from None
                continue
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
            return
        raise WorkerError("Could not acquire the worker lock; another start is in progress.")

    def _release_lock(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def _launch(self, job: dict[str, Any], *, kind: str) -> WorkerState:
        if self.is_running():
            current = self.state()
            raise WorkerError(
                f"Training is already running (PID {current.pid if current else '?'}). "
                "Use Stop first; GameMind will not start a second agent."
            )
        if not WORKER_SCRIPT.exists():
            raise WorkerError(f"Worker script is missing: {WORKER_SCRIPT}")
        self._acquire_lock()
        try:
            job_id = str(job["job_id"])
            job_path = self.jobs_dir / f"{job_id}.json"
            atomic_write_json(job_path, job)
            log_path = self.log_dir / f"{job_id}.log"
            pid = launch_detached(
                [str(WORKER_SCRIPT), "--job", str(job_path)],
                log_path=log_path,
                env={"GAMEMIND_LIBRARY_ROOT": str(self.library.root)},
            )
            state = WorkerState(
                pid=pid,
                kind=kind,
                job_path=str(job_path),
                status="running",
                game_id=job.get("game_id"),
                level_id=job.get("level_id"),
                session_id=job.get("session_id"),
                queue_id=job.get("queue_id"),
                run_dir=job.get("run_dir"),
                log_path=str(log_path),
            )
            self._write_state(state)
            atomic_write_json(self.control_path, {"pause": False, "stop": False})
            return state
        finally:
            self._release_lock()

    def start_session(
        self,
        level: LevelProfile,
        *,
        timesteps: int,
        evaluate: bool = True,
        eval_episodes: int = 3,
        device: str = "auto",
        stop_on_completion: bool = True,
        notes: str = "",
    ) -> WorkerState:
        from gamemind.library.sessions import new_session_id, run_dir_for, run_name_for

        if int(timesteps) < 1:
            raise LibraryError("Timesteps must be at least 1.")
        session_id = new_session_id()
        run_name = run_name_for(level.game_id, level.id, session_id)
        run_dir = run_dir_for(self.library, level.game_id, level.id, run_name)
        job = {
            "job_id": session_id,
            "kind": "session",
            "library_root": str(self.library.root),
            "game_id": level.game_id,
            "level_id": level.id,
            "session_id": session_id,
            "timesteps": int(timesteps),
            "evaluate": bool(evaluate),
            "eval_episodes": int(eval_episodes),
            "device": str(device),
            "stop_on_completion": bool(stop_on_completion),
            "notes": notes,
            "run_dir": str(run_dir),
        }
        return self._launch(job, kind="session")

    def start_evaluation(
        self,
        level: LevelProfile,
        *,
        model_path: str,
        episodes: int = 5,
        device: str = "auto",
    ) -> WorkerState:
        """Evaluate a saved model in the background (it plays the real game)."""
        from gamemind.library.sessions import new_session_id

        if not Path(model_path).exists():
            raise LibraryError(f"Model file not found: {model_path}")
        identifier = new_session_id("evaljob-")
        job = {
            "job_id": identifier,
            "kind": "evaluate",
            "library_root": str(self.library.root),
            "game_id": level.game_id,
            "level_id": level.id,
            "model_path": str(model_path),
            "episodes": max(1, int(episodes)),
            "device": str(device),
            "run_dir": None,
        }
        return self._launch(job, kind="evaluate")

    def start_queue(self, queue: QueueRecord) -> WorkerState:
        job = {
            "job_id": f"queue-{queue.queue_id}",
            "kind": "queue",
            "library_root": str(self.library.root),
            "game_id": queue.game_id,
            "queue_id": queue.queue_id,
            "run_dir": None,
        }
        return self._launch(job, kind="queue")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _control_target_dirs(self) -> list[Path]:
        targets: list[Path] = []
        state = self.state()
        if state is not None and state.run_dir:
            targets.append(Path(state.run_dir))
        current = read_json(self.current_path, default=None)
        if isinstance(current, dict) and current.get("run_dir"):
            path = Path(str(current["run_dir"]))
            if path not in targets:
                targets.append(path)
        return targets

    def pause(self) -> None:
        if not self.is_running():
            raise WorkerError("No training worker is running.")
        for directory in self._control_target_dirs():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "PAUSE").write_text("requested by GameMind\n", encoding="utf-8")
        atomic_write_json(self.control_path, {"pause": True, "stop": False})

    def resume(self) -> None:
        if not self.is_running():
            raise WorkerError("No training worker is running.")
        for directory in self._control_target_dirs():
            try:
                (directory / "PAUSE").unlink()
            except FileNotFoundError:
                pass
        atomic_write_json(self.control_path, {"pause": False, "stop": False})

    def stop(self) -> None:
        """Ask the worker to stop and save a checkpoint, then verify it exits."""
        state = self.state()
        if state is None:
            raise WorkerError("No training worker has been started.")
        for directory in self._control_target_dirs():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "STOP").write_text("requested by GameMind\n", encoding="utf-8")
        atomic_write_json(self.control_path, {"pause": False, "stop": True})
        if pid_alive(state.pid):
            # The trainer polls the STOP file every ~0.15 s; give it a chance to
            # save a checkpoint before forcing the process to exit.
            import time

            for _ in range(40):
                if not pid_alive(state.pid):
                    return
                time.sleep(0.25)
            stop_process(state.pid)
            self.mark_finished(
                "stopped",
                error="The worker did not stop gracefully and was terminated. "
                "Check the last session record for the checkpoint that was saved.",
            )

    def control_flags(self) -> dict[str, bool]:
        payload = read_json(self.control_path, default={})
        if not isinstance(payload, dict):
            return {"pause": False, "stop": False}
        return {
            "pause": bool(payload.get("pause")),
            "stop": bool(payload.get("stop")),
        }

    def clear_control(self) -> None:
        atomic_write_json(self.control_path, {"pause": False, "stop": False})


def load_job(path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(path), default=None)
    if not isinstance(payload, dict):
        raise WorkerError(f"Job file is missing or invalid: {path}")
    return payload


def job_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


__all__ = [
    "TrainingController",
    "WorkerState",
    "WORKER_SCRIPT",
    "load_job",
]
