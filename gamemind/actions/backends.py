"""Keyboard/mouse input backends and clean key-release handling.

PyAutoGUI is imported lazily. Gymnasium and the example game do not need any
desktop input libraries, so those dependencies remain optional in practice.
"""

from __future__ import annotations

import atexit
from typing import Protocol, Sequence, runtime_checkable

from gamemind.utils.errors import InputBackendError
from gamemind.utils.logging import get_logger

logger = get_logger("inputs")


@runtime_checkable
class InputBackend(Protocol):
    """Minimal input contract used by the desktop adapter.

    Custom backends (a background input socket, an emulator API, a virtual
    keyboard) can implement these methods and be injected into the adapter.
    """

    def validate_keys(self, keys: Sequence[str]) -> None: ...

    def ensure_focus(self, window_title: str | None) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def click(self, button: str) -> None: ...

    def move_rel(self, dx: int, dy: int) -> None: ...

    def release_all(self) -> None: ...


class PyAutoGUIBackend:
    """PyAutoGUI keyboard and mouse backend with stuck-key protection."""

    def __init__(self, *, failsafe: bool = True) -> None:
        try:
            import pyautogui
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise InputBackendError(
                "pyautogui is required for desktop control. Install requirements.txt "
                "or use the example_game/Gymnasium adapters."
            ) from exc
        self._pyautogui = pyautogui
        self._held: set[str] = set()
        self._valid_keys = {str(key).lower() for key in pyautogui.KEYBOARD_KEYS}
        # Corner escape hatch: moving the real mouse to a screen corner aborts.
        pyautogui.FAILSAFE = bool(failsafe)
        pyautogui.PAUSE = 0.02
        self._closed = False
        atexit.register(self.release_all)

    def validate_keys(self, keys: Sequence[str]) -> None:
        invalid = [key for key in keys if key.lower() not in self._valid_keys]
        if invalid:
            examples = sorted(self._valid_keys)[:12]
            raise InputBackendError(
                f"Unsupported key name(s): {', '.join(invalid)}. Use PyAutoGUI key names such as "
                f"'left', 'space', 'a', 'enter'. Valid examples: {', '.join(examples)}."
            )

    def ensure_focus(self, window_title: str | None) -> None:
        if not window_title:
            return
        try:
            windows = self._pyautogui.getWindowsWithTitle(window_title)
        except Exception as exc:
            raise InputBackendError(
                f"Could not inspect windows for title {window_title!r}: {exc}. "
                "Set require_window_focus: false if your platform does not support this."
            ) from exc
        if not windows:
            raise InputBackendError(
                f"No window containing {window_title!r} was found. Start the game first or "
                "clear game.params.window_title."
            )
        wanted = window_title.strip().lower()
        exact = [
            window
            for window in windows
            if str(getattr(window, "title", "")).strip().lower() == wanted
        ]
        # A game may also publish a small helper window whose title contains the
        # same text. Prefer an exact title, then the largest window.
        window = max(
            exact or windows,
            key=lambda item: int(getattr(item, "width", 0)) * int(getattr(item, "height", 0)),
        )
        try:
            active_title = str(getattr(self._pyautogui.getActiveWindow(), "title", "") or "")
        except Exception:
            active_title = ""
        if wanted in active_title.lower() and not getattr(window, "isMinimized", False):
            # Focus is already correct. Avoid the expensive activate/sleep path
            # on every action; this matters for fast desktop games.
            return
        last_title = active_title
        for attempt in range(3):
            try:
                if getattr(window, "isMinimized", False):
                    window.restore()
                window.activate()
                self._pyautogui.sleep(0.15 if attempt == 0 else 0.25)
                active = self._pyautogui.getActiveWindow()
            except Exception as exc:
                raise InputBackendError(
                    f"Could not focus window {window_title!r}: {exc}. "
                    "Focus the game manually or disable require_window_focus."
                ) from exc
            last_title = str(getattr(active, "title", "") or "")
            if wanted in last_title.lower():
                return
            # Windows can briefly show the Alt+Tab "Task Switching" surface or
            # another transient foreground window. Retry before refusing input;
            # this also gives the user a moment to return focus to the game.
        raise InputBackendError(
            f"Game window focus was lost (active window: {last_title!r}). "
            "GameMind refuses to send inputs to an unknown window."
        )

    def key_down(self, key: str) -> None:
        key = key.lower()
        if key in self._held:
            return
        try:
            self._pyautogui.keyDown(key)
            self._held.add(key)
        except Exception as exc:
            self.release_all()
            raise InputBackendError(f"Could not press key {key!r}: {exc}") from exc

    def key_up(self, key: str) -> None:
        key = key.lower()
        try:
            self._pyautogui.keyUp(key)
        except Exception as exc:
            self.release_all()
            raise InputBackendError(f"Could not release key {key!r}: {exc}") from exc
        finally:
            self._held.discard(key)

    def click(self, button: str) -> None:
        try:
            self._pyautogui.click(button=button)
        except Exception as exc:
            self.release_all()
            raise InputBackendError(f"Mouse click failed: {exc}") from exc

    def move_rel(self, dx: int, dy: int) -> None:
        try:
            self._pyautogui.moveRel(dx, dy, duration=0)
        except Exception as exc:
            self.release_all()
            raise InputBackendError(f"Mouse movement failed: {exc}") from exc

    def release_all(self) -> None:
        """Release every held key. Safe to call repeatedly, including at exit."""
        if self._closed:
            return
        for key in list(self._held):
            try:
                self._pyautogui.keyUp(key)
            except Exception:
                logger.warning("Could not release key %s during cleanup.", key, exc_info=False)
            finally:
                self._held.discard(key)

    def close(self) -> None:
        self.release_all()
        self._closed = True
        try:
            atexit.unregister(self.release_all)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
