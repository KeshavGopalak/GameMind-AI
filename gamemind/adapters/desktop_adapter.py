"""Generic desktop-game adapter: screen capture + keyboard/mouse control.

This adapter is intentionally honest about its limits. It can capture a region,
send configured inputs, and apply time limits or a template-based game-over
detector. It cannot read an arbitrary game's score, lives, internal state, or
even know that an episode ended unless you configure a detector or write a
custom adapter with a real game/emulator API.

Safe defaults:
* all inputs are explicit in YAML (no random clicking/keyboard exploration),
* held keys are released in a ``finally`` block and on close,
* PyAutoGUI's corner failsafe remains enabled,
* an optional window title must be focused before inputs are sent,
* episodes require a finite ``episode.max_steps``.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, Sequence, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.actions import build_discrete_action_space
from gamemind.actions.backends import InputBackend, PyAutoGUIBackend
from gamemind.adapters.base import GameAdapter
from gamemind.config import GameConfig
from gamemind.utils.errors import AdapterError, ConfigError, InputBackendError, ObservationError
from gamemind.utils.logging import get_logger
from gamemind.utils.viewer import FrameViewer

logger = get_logger("desktop")


@runtime_checkable
class ScreenCapture(Protocol):
    """A screen-capture backend. Tests can inject a fake implementation."""

    width: int
    height: int

    def grab(self) -> np.ndarray: ...

    def close(self) -> None: ...


class MSSCapture:
    """Fast screen-region capture using the optional ``mss`` package."""

    def __init__(self, region: Sequence[int] | None, monitor: int = 1) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise AdapterError(
                "mss is required for the desktop adapter. Install requirements.txt or "
                "use a Gymnasium/example-game adapter."
            ) from exc
        self._mss = mss.mss()
        monitors = self._mss.monitors
        if not isinstance(monitor, int) or isinstance(monitor, bool):
            raise ConfigError("game.params.monitor must be an integer.")
        if monitor < 0 or monitor >= len(monitors):
            raise ConfigError(
                f"game.params.monitor={monitor} is invalid. Available monitors: 0..{len(monitors) - 1}."
            )
        if region is None:
            area = dict(monitors[monitor])
        else:
            if len(region) != 4:
                raise ConfigError("game.params.region must be [x, y, width, height] or null.")
            left, top, width, height = (int(value) for value in region)
            if width <= 0 or height <= 0:
                raise ConfigError("game.params.region width and height must be positive.")
            area = {"left": left, "top": top, "width": width, "height": height}
            virtual = monitors[0]
            if (
                left < virtual["left"]
                or top < virtual["top"]
                or left + width > virtual["left"] + virtual["width"]
                or top + height > virtual["top"] + virtual["height"]
            ):
                raise ConfigError(
                    f"Capture region {area} is outside the virtual desktop "
                    f"{virtual}. Check coordinates and display scaling."
                )
        self._area = area
        self.width = int(area["width"])
        self.height = int(area["height"])

    def grab(self) -> np.ndarray:
        try:
            shot = self._mss.grab(self._area)
        except Exception as exc:
            raise AdapterError(f"Screen capture failed: {exc}") from exc
        # mss returns BGRA; convert to RGB and copy so later captures do not mutate it.
        frame = np.asarray(shot)[:, :, :3][:, :, ::-1].copy()
        return frame

    def close(self) -> None:
        try:
            self._mss.close()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("mss close failed", exc_info=True)


class DesktopGameAdapter(GameAdapter):
    """Control an arbitrary desktop game through keyboard/mouse and screenshots."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    # Specialised desktop adapters can add validated configuration keys without
    # weakening the strict template used by the generic adapter.
    extra_param_names: frozenset[str] = frozenset()
    # A real game window is usually visible already. Adapters that control a
    # real desktop game can disable the duplicate OpenCV preview window.
    show_capture_viewer: bool = True

    def __init__(
        self,
        config: GameConfig,
        render_mode: str | None = None,
        *,
        capture_backend: ScreenCapture | None = None,
        input_backend: InputBackend | None = None,
    ) -> None:
        super().__init__(config, render_mode=render_mode)
        if not config.observation.is_pixel:
            raise ConfigError(
                "The desktop adapter observes screenshots. Set observation.mode to "
                "'grayscale' (recommended for controllers) or 'rgb'."
            )
        params = config.params
        allowed = {
            "region",
            "monitor",
            "window_title",
            "require_window_focus",
            "start_delay",
            "reset_keys",
            "reset_delay",
            "reset_key_duration",
            "fps",
        } | set(self.extra_param_names)
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ConfigError(
                f"Unknown desktop adapter params: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        if config.episode.max_steps is None:
            raise ConfigError(
                "Desktop games require episode.max_steps so an experiment cannot control the "
                "keyboard indefinitely. Set a finite value such as 300."
            )
        if not config.actions.bindings:
            raise ConfigError(
                "Desktop games require actions.bindings. Define a no-op and the keys/mouse "
                "actions the agent may use; see configs/desktop_game.example.yaml."
            )
        self.region = params.get("region")
        if self.region is not None and (
            not isinstance(self.region, list) or len(self.region) != 4
        ):
            raise ConfigError("game.params.region must be a list [x, y, width, height] or null.")
        self.monitor = params.get("monitor", 1)
        self.window_title = params.get("window_title")
        if self.window_title is not None and not isinstance(self.window_title, str):
            raise ConfigError("game.params.window_title must be a string or null.")
        self.require_window_focus = params.get("require_window_focus", False)
        if not isinstance(self.require_window_focus, bool):
            raise ConfigError("game.params.require_window_focus must be true or false.")
        if self.require_window_focus and not self.window_title:
            raise ConfigError(
                "require_window_focus: true needs game.params.window_title so inputs are only "
                "sent to the intended game."
            )
        self.start_delay = params.get("start_delay", 2.0)
        if isinstance(self.start_delay, bool) or not isinstance(self.start_delay, (int, float)):
            raise ConfigError("game.params.start_delay must be a number of seconds.")
        if not 0.0 <= float(self.start_delay) <= 30.0:
            raise ConfigError("game.params.start_delay must be between 0 and 30 seconds.")
        reset_keys = params.get("reset_keys", [])
        if isinstance(reset_keys, str):
            reset_keys = [reset_keys]
        if not isinstance(reset_keys, (list, tuple)):
            raise ConfigError("game.params.reset_keys must be a list of key names.")
        self.reset_keys = [str(key).lower() for key in reset_keys]
        self.reset_delay = params.get("reset_delay", 0.5)
        self.reset_key_duration = params.get("reset_key_duration", 0.05)
        for name, value in (("reset_delay", self.reset_delay), ("reset_key_duration", self.reset_key_duration)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"game.params.{name} must be a number of seconds.")
            if not 0.0 <= float(value) <= 30.0:
                raise ConfigError(f"game.params.{name} must be between 0 and 30 seconds.")
        self.fps = params.get("fps", 30)
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or not 1 <= self.fps <= 144:
            raise ConfigError("game.params.fps must be an integer between 1 and 144.")
        self.metadata = {**self.metadata, "render_fps": self.fps}

        self._capture: ScreenCapture | None = capture_backend
        self._input: InputBackend | None = input_backend
        self._owns_capture = capture_backend is None
        self._owns_input = input_backend is None
        if not self._owns_input and self._input is not None:
            self._owns_input = False
        self._viewer: FrameViewer | None = None
        self._last_frame: np.ndarray | None = None
        self.action_space = build_discrete_action_space(
            config.actions, where="desktop actions"
        )
        # The actual raw pixel space is declared in connect(), once the capture size is known.

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        try:
            if self._capture is None:
                self._capture = MSSCapture(self.region, self.monitor)
            if self._input is None:
                self._input = PyAutoGUIBackend(failsafe=True)
            assert self._capture is not None and self._input is not None
            for binding in self.action_config.bindings:
                self._input.validate_keys(binding.keys)
            self._input.validate_keys(self.reset_keys)
            self._declare_observation_space(
                spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._capture.height, self._capture.width, 3),
                    dtype=np.uint8,
                )
            )
            if self.render_mode == "human" and self.show_capture_viewer:
                self._viewer = FrameViewer(
                    title=f"GameMind AI — {self.config.name}", fps=self.fps
                )
            frame = self._capture.grab()
            self._validate_screenshot(frame)
            self._last_frame = frame
            self.set_max_episode_steps(self.config.episode.max_steps)
            self._connected = True
            logger.warning(
                "Desktop adapter connected. Inputs start after %.1fs. Press Ctrl+C to stop; "
                "PyAutoGUI failsafe: move the mouse to a screen corner to abort.",
                float(self.start_delay),
            )
            if self.start_delay:
                time.sleep(float(self.start_delay))
        except Exception:
            self._close_resources()
            raise

    def _require_backends(self) -> tuple[ScreenCapture, InputBackend]:
        if self._capture is None or self._input is None:
            raise AdapterError("Desktop adapter is not connected. Call connect() first.")
        return self._capture, self._input

    def _close_resources(self) -> None:
        if self._input is not None:
            try:
                self._input.release_all()
                close = getattr(self._input, "close", None)
                if callable(close):
                    close()
            except Exception:
                logger.warning("Desktop input cleanup failed", exc_info=True)
        if self._capture is not None and self._owns_capture:
            try:
                self._capture.close()
            except Exception:
                logger.warning("Screen capture cleanup failed", exc_info=True)
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._capture = None if self._owns_capture else self._capture
        self._input = None if self._owns_input else self._input
        self._connected = False
        self._last_frame = None

    # ------------------------------------------------------------------
    # Inputs and observations
    # ------------------------------------------------------------------
    def _validate_screenshot(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ObservationError(f"Capture backend returned {type(frame).__name__}, expected ndarray.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ObservationError(f"Capture frame must be RGB (H, W, 3), got {frame.shape}.")
        if frame.dtype != np.uint8:
            raise ObservationError(f"Capture frame must be uint8, got {frame.dtype}.")
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ObservationError(f"Capture frame is too small: {frame.shape[:2]}.")
        return frame

    def _press_reset_keys(self) -> None:
        _, inputs = self._require_backends()
        if not self.reset_keys:
            return
        if self.require_window_focus:
            inputs.ensure_focus(self.window_title)
        try:
            for key in self.reset_keys:
                inputs.key_down(key)
            time.sleep(float(self.reset_key_duration))
        finally:
            inputs.release_all()

    def _grab(self) -> np.ndarray:
        capture, _ = self._require_backends()
        frame = self._validate_screenshot(capture.grab())
        self._last_frame = frame
        return frame

    def _apply_action(self, action: int) -> None:
        _, inputs = self._require_backends()
        binding = self.action_config.binding(int(action))
        if self.require_window_focus:
            inputs.ensure_focus(self.window_title)
        duration = (
            float(binding.duration)
            if binding.duration is not None
            else float(self.action_config.duration)
        )
        pressed: list[str] = []
        try:
            for key in binding.keys:
                inputs.key_down(key)
                pressed.append(key)
            for button in binding.mouse_click:
                inputs.click(button)
            if binding.mouse_move is not None:
                inputs.move_rel(int(binding.mouse_move[0]), int(binding.mouse_move[1]))
            if duration > 0:
                time.sleep(duration)
        except Exception as exc:
            if isinstance(exc, InputBackendError):
                raise
            raise InputBackendError(
                f"Desktop action {binding.name!r} failed: {exc}. Held keys were released."
            ) from exc
        finally:
            # Releasing all tracked keys is safer than releasing only the ones
            # recorded here if a backend partially applied a binding.
            inputs.release_all()
        self._step_info = {
            "action_keys": list(binding.keys),
            "action_name": binding.name,
            "focus_checked": self.require_window_focus,
        }

    # ------------------------------------------------------------------
    # GameAdapter hooks
    # ------------------------------------------------------------------
    def _reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        _, inputs = self._require_backends()
        inputs.release_all()
        try:
            self._press_reset_keys()
        finally:
            inputs.release_all()
        if self.reset_delay:
            time.sleep(float(self.reset_delay))
        frame = self._grab()
        return frame, {
            "capture_width": frame.shape[1],
            "capture_height": frame.shape[0],
            "reset_keys": list(self.reset_keys),
        }

    def observe(self) -> np.ndarray:
        return self._grab()

    def is_episode_over(self) -> bool:
        # The base implementation uses the optional template detector.
        return super().is_episode_over()

    def _reward_signals(self) -> dict[str, float]:
        return {"alive": 1.0}

    def render(self) -> np.ndarray | None:
        frame = self._last_frame
        if frame is None:
            frame = self._grab()
        if self.render_mode == "human" and self._viewer is not None:
            if not self._viewer.show(frame, caption=self._last_action_name):
                self.window_closed = True
        return frame

    def _extra_info(self) -> dict[str, Any]:
        return {
            "window_title": self.window_title,
            "require_window_focus": self.require_window_focus,
        }
