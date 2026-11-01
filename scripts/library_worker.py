#!/usr/bin/env python
"""Background worker: runs exactly one training job for the GameMind library.

The dashboard launches this in a separate process so Streamlit reruns, page
navigation, or closing the browser tab never interrupt training.

    python scripts/library_worker.py --job library/state/jobs/<job_id>.json

Normally you never run this by hand; the library's TrainingController writes the
job file and starts it.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamemind.library.control import TrainingController, load_job  # noqa: E402
from gamemind.library.queue import run_queue  # noqa: E402
from gamemind.library.runner import SessionRequest, run_session  # noqa: E402
from gamemind.library.store import GameLibrary  # noqa: E402
from gamemind.utils.errors import GameMindError  # noqa: E402
from gamemind.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one GameMind library job.")
    parser.add_argument("--job", required=True, help="Path to a job JSON written by the library.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(1)
    job = load_job(args.job)
    library = GameLibrary(job.get("library_root"))
    controller = TrainingController(library)
    status = "finished"
    error: str | None = None
    try:
        kind = str(job.get("kind"))
        if kind == "session":
            record = run_session(
                library,
                SessionRequest(
                    game_id=str(job["game_id"]),
                    level_id=str(job["level_id"]),
                    timesteps=int(job["timesteps"]),
                    session_id=job.get("session_id"),
                    evaluate=bool(job.get("evaluate", True)),
                    eval_episodes=int(job.get("eval_episodes") or 3),
                    device=str(job.get("device") or "auto"),
                    notes=str(job.get("notes") or ""),
                    stop_on_completion=bool(job.get("stop_on_completion", False)),
                ),
            )
            if record.state == "error":
                status, error = "error", record.error
            elif record.state in {"stopped", "interrupted"}:
                status = "stopped"
        elif kind == "evaluate":
            from gamemind.library.runner import evaluate_session_model

            level = library.get_level(str(job["game_id"]), str(job["level_id"]))
            payload = evaluate_session_model(
                library,
                level,
                model_path=str(job["model_path"]),
                episodes=int(job.get("episodes") or 5),
                device=str(job.get("device") or "auto"),
            )
            print(
                "Evaluation finished: "
                f"mean_reward={payload.get('mean_reward')} "
                f"mean_progress={payload.get('mean_progress_pct')}"
            )
        elif kind == "queue":
            queue = run_queue(
                library,
                str(job["queue_id"]),
                controller=controller,
                device=str(job.get("device") or "auto"),
            )
            if queue.status == "error":
                status, error = "error", queue.error
            elif queue.status == "stopped":
                status = "stopped"
        else:
            status, error = "error", f"Unknown job kind {kind!r}."
    except GameMindError as exc:
        status, error = "error", str(exc)
    except Exception:
        status, error = "error", traceback.format_exc()
        traceback.print_exc()
    finally:
        try:
            controller.mark_finished(status, error=error)
            controller.clear_control()
        except Exception:
            traceback.print_exc()
    if error:
        print(f"Worker finished with error: {error}", file=sys.stderr)
    return 0 if status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
