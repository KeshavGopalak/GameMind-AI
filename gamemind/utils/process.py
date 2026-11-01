"""Cross-platform process helpers for background training workers.

The dashboard launches training in a separate process so Streamlit reruns never
interrupt a run. These helpers answer the only question the UI must get right:
"is the worker I recorded still the process that is running?"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gamemind.utils.paths import PROJECT_ROOT


def pid_alive(pid: int | None) -> bool:
    """Return True when *pid* refers to a live process."""
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if handle:
                kernel32.CloseHandle(handle)
                return True
            # Access denied usually means the process exists but runs at a higher
            # integrity level, which still counts as alive for our purposes.
            return ctypes.get_last_error() == 5
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def launch_detached(
    args: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Start a background Python process writing stdout/stderr to *log_path*.

    Returns its PID. The process is not tied to the caller's console, so closing
    the dashboard does not stop training.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update({key: str(value) for key, value in env.items()})
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        # DETACHED_PROCESS keeps the worker alive after the dashboard exits;
        # CREATE_NO_WINDOW prevents a stray console window.
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    else:
        start_new_session = True
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [sys.executable, "-u", *args],
            cwd=str(cwd or PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=full_env,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    return int(process.pid)


def stop_process(pid: int | None) -> bool:
    """Terminate a worker we started. Returns True when a signal was delivered."""
    if not pid or not pid_alive(pid):
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=15,
            )
        else:
            os.kill(int(pid), 15)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
