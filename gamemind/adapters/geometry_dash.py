"""Geometry Dash integration for GameMind AI.

This is a desktop adapter specialised for Geometry Dash. It captures the real
game window's pixels, sends ordinary keyboard input, and derives a small set of
signals from the visible progress bar:

* whether a gameplay progress bar is present,
* approximate level progress (from the filled part of the bar),
* death/restart (progress drops back to the start),
* level completion (the bar reaches ~100%).

It does **not** read Geometry Dash's memory, level coordinates, collision data,
attempt counter, or internal state. Progress is an estimate from pixels. The
adapter is intentionally honest: when the bar cannot be found, it reports
``bar_found: false`` instead of pretending progress is zero, and it never adds a
progress reward for an uncertain reading.

Typical setup (see ``docs/geometry_dash.md``):

1. Launch Geometry Dash and select a level.
2. Use fullscreen/borderless so the game is visible on the primary monitor.
3. Enable Auto-Retry in the game (the adapter uses progress reset as its death
   signal).
4. Run ``python scripts/train.py --config configs/geometry_dash.yaml ...``.

Only a single adapter can control one game window. Multiple parallel
environments would send conflicting keystrokes, so ``n_envs`` must be 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np
from gymnasium.envs.registration import EnvSpec

from gamemind.adapters.desktop_adapter import DesktopGameAdapter
from gamemind.config import GameConfig
from gamemind.utils.errors import AdapterError, ConfigError, InputBackendError, ObservationError
from gamemind.utils.logging import get_logger

logger = get_logger("geometry_dash")


@dataclass(frozen=True)
class WindowInfo:
    """A located desktop window. Coordinates are the OS/window-manager rect."""

    title: str
    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "title": self.title,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@runtime_checkable
class WindowLocator(Protocol):
    """Find the game window without sending any input."""

    def find(self, title: str) -> WindowInfo: ...


class PyAutoGUIWindowLocator:
    """Locate a visible window using PyAutoGUI's cross-platform window helpers."""

    def find(self, title: str) -> WindowInfo:
        if not isinstance(title, str) or not title.strip():
            raise ConfigError("Geometry Dash requires a non-empty game.params.window_title.")
        try:
            import pyautogui
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise AdapterError(
                "pyautogui is required to locate Geometry Dash. Install requirements.txt."
            ) from exc
        try:
            candidates = list(pyautogui.getWindowsWithTitle(title))
        except Exception as exc:  # pragma: no cover - desktop dependent
            raise AdapterError(
                f"Could not inspect desktop windows for {title!r}: {exc}. "
                "Use a normal Windows desktop session, not a headless environment."
            ) from exc
        candidates = [
            window
            for window in candidates
            if getattr(window, "visible", True)
            and int(getattr(window, "width", 0)) >= 100
            and int(getattr(window, "height", 0)) >= 100
        ]
        if not candidates:
            raise AdapterError(
                f"No visible window containing {title!r} was found. Launch Geometry Dash "
                "and keep it out of the system tray, then try again."
            )
        wanted = title.strip().lower()
        exact = [
            window
            for window in candidates
            if str(getattr(window, "title", "")).strip().lower() == wanted
        ]
        # PyAutoGUI can return both the game and a Steam/Geode helper window
        # whose title contains the same text. Prefer an exact title, then the
        # largest visible window.
        window = max(exact or candidates, key=lambda item: int(item.width) * int(item.height))
        return WindowInfo(
            title=str(getattr(window, "title", title)),
            left=int(getattr(window, "left", 0)),
            top=int(getattr(window, "top", 0)),
            width=int(getattr(window, "width", 0)),
            height=int(getattr(window, "height", 0)),
        )


@dataclass(frozen=True)
class ProgressReading:
    """One progress-bar reading from a frame."""

    found: bool
    progress_pct: float = 0.0
    fill_columns: int = 0
    bar: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def progress_fraction(self) -> float:
        return float(np.clip(self.progress_pct / 100.0, 0.0, 1.0))


class GeometryDashVision:
    """Detect and read Geometry Dash's top progress bar from RGB screenshots.

    The detector never assumes a fixed resolution. By default it searches the
    top strip for a long, short, white outlined horizontal component (the bar),
    then measures how far its coloured fill extends from the left. A manual
    ``progress_region`` can be supplied for unusual HUD scaling.

    Colour thresholds are configurable because mods, level colours, or a
    different UI scale can change the fill colour. If the fill is not found, the
    detector reports zero progress for that frame only; the adapter keeps the
    previous reliable reading instead of inventing a positive reward.
    """

    def __init__(
        self,
        *,
        progress_region: Sequence[int] | None = None,
        search_top_fraction: float = 0.09,
        white_threshold: int = 200,
        min_width_fraction: float = 0.15,
        max_height_fraction: float = 0.08,
        min_aspect: float = 5.0,
        min_area_fraction: float = 0.0005,
        left_inset: int = 8,
        right_inset: int = 8,
        min_fill_columns: int = 4,
        fill_color_distance: int = 30,
        fill_hsv_low: Sequence[int] = (8, 90, 90),
        fill_hsv_high: Sequence[int] = (45, 255, 255),
    ) -> None:
        self.progress_region = self._parse_region(progress_region)
        self.search_top_fraction = self._positive_float(
            search_top_fraction, "bar_search_top_fraction", 0.02, 0.45
        )
        self.white_threshold = self._integer(
            white_threshold, "bar_white_threshold", 0, 255
        )
        self.min_width_fraction = self._positive_float(
            min_width_fraction, "bar_min_width_fraction", 0.05, 0.95
        )
        self.max_height_fraction = self._positive_float(
            max_height_fraction, "bar_max_height_fraction", 0.01, 0.2
        )
        self.min_aspect = self._positive_float(min_aspect, "bar_min_aspect", 2.0, 50.0)
        self.min_area_fraction = self._positive_float(
            min_area_fraction, "bar_min_area_fraction", 0.00001, 0.05
        )
        self.left_inset = self._integer(left_inset, "bar_left_inset", 0, 100)
        self.right_inset = self._integer(right_inset, "bar_right_inset", 0, 100)
        self.min_fill_columns = self._integer(
            min_fill_columns, "bar_min_fill_columns", 1, 200
        )
        self.fill_color_distance = self._integer(
            fill_color_distance, "fill_color_distance", 1, 255
        )
        self.fill_hsv_low = self._hsv(fill_hsv_low, "progress_fill_hsv_low")
        self.fill_hsv_high = self._hsv(fill_hsv_high, "progress_fill_hsv_high")
        self._reference_color: np.ndarray | None = None
        if np.any(self.fill_hsv_low > self.fill_hsv_high):
            raise ConfigError(
                "progress_fill_hsv_low must be <= progress_fill_hsv_high for each HSV channel."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read(self, frame: np.ndarray) -> ProgressReading:
        """Read progress from an RGB uint8 frame."""
        height, width = self._validate_frame(frame)
        if self.progress_region is not None:
            x, y, bar_width, bar_height = self.progress_region
            if x < 0 or y < 0 or x + bar_width > width or y + bar_height > height:
                raise ObservationError(
                    f"game.params.progress_region {list(self.progress_region)} is outside "
                    f"the {width}x{height} captured frame. Set it to the bar's pixel bounds "
                    "in the captured image, or set it to null for automatic detection."
                )
            if self.left_inset + self.right_inset >= bar_width:
                raise ConfigError(
                    "progress bar insets are too large for progress_region; reduce "
                    "bar_left_inset/bar_right_inset."
                )
            bar = (x, y, bar_width, bar_height)
            # A manually specified region still has to look like the white
            # outlined bar. This avoids treating the pause/menu screen as
            # gameplay merely because a fixed rectangle exists there.
            gray = cv2.cvtColor(frame[y : y + bar_height, x : x + bar_width], cv2.COLOR_RGB2GRAY)
            white_fraction = float(np.mean(gray >= self.white_threshold))
            if white_fraction < 0.006:
                return ProgressReading(found=False, reason="no white progress-bar outline")
            confidence = 1.0
        else:
            bar = self._find_bar(frame)
            if bar is None:
                return ProgressReading(found=False, reason="no white progress bar found")
            confidence = 1.0

        x, y, bar_width, bar_height = bar
        inner = frame[
            y + 4 : y + bar_height - 4,
            x + self.left_inset : x + bar_width - self.right_inset,
        ]
        if inner.size == 0:
            return ProgressReading(found=False, bar=bar, reason="progress bar is too small")
        columns = self._fill_columns(inner)
        fill_columns = int(np.count_nonzero(columns))
        if fill_columns < self.min_fill_columns:
            progress_pct = 0.0
        else:
            rightmost = int(np.flatnonzero(columns)[-1])
            progress_pct = 100.0 * (rightmost + 1) / float(columns.size)
        return ProgressReading(
            found=True,
            progress_pct=float(np.clip(progress_pct, 0.0, 100.0)),
            fill_columns=fill_columns,
            bar=bar,
            confidence=confidence,
        )

    def calibrate(self, frame: np.ndarray) -> bool:
        """Learn the empty bar's colour profile from a reset frame.

        Geometry Dash draws the fill in the player's colour, which can be blue,
        orange, green, or almost anything else. Comparing each column with the
        empty bar captured at reset is therefore much more reliable than a fixed
        orange HSV range. The right-hand part of a fresh attempt is empty even if
        the attempt has already advanced a few percent.
        """
        self._validate_frame(frame)
        if self.progress_region is not None:
            bar = self.progress_region
        else:
            bar = self._find_bar(frame)
        if bar is None:
            return False
        x, y, bar_width, bar_height = bar
        inner = frame[
            y + 4 : y + bar_height - 4,
            x + self.left_inset : x + bar_width - self.right_inset,
        ]
        if inner.size == 0 or inner.shape[1] < 8:
            return False
        # A fresh attempt normally has progress below the adapter's 5% reset
        # threshold, so the rightmost half of the bar is background. Sampling a
        # broad right-hand area is more robust than one pixel.
        start = max(0, int(inner.shape[1] * 0.55))
        reference = np.median(
            inner[:, start:, :].astype(np.int16), axis=(0, 1)
        ).astype(np.int16)
        if reference.shape != (3,):
            return False
        self._reference_color = reference
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fill_columns(self, inner: np.ndarray) -> np.ndarray:
        """Return a boolean mask of columns that differ from the empty bar."""
        if self._reference_color is not None:
            current = np.median(inner.astype(np.int16), axis=0)
            distance = np.max(np.abs(current - self._reference_color), axis=1)
            columns = distance > self.fill_color_distance
            # The fill grows from the left. Require a contiguous run starting at
            # the left edge (allow a one-pixel anti-aliasing gap). A random
            # background change farther right must not look like progress.
            nonzero = np.flatnonzero(columns)
            if nonzero.size == 0 or nonzero[0] > 1:
                return np.zeros_like(columns, dtype=bool)
            start = int(nonzero[0])
            end = start
            while end + 1 < columns.size and columns[end + 1]:
                end += 1
            result = np.zeros_like(columns, dtype=bool)
            result[start : end + 1] = True
            return result
        hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv,
            np.asarray(self.fill_hsv_low, dtype=np.uint8),
            np.asarray(self.fill_hsv_high, dtype=np.uint8),
        )
        return np.any(mask > 0, axis=0)

    def _find_bar(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        top = max(16, int(round(height * self.search_top_fraction)))
        gray = cv2.cvtColor(frame[:top], cv2.COLOR_RGB2GRAY)
        white = (gray >= self.white_threshold).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(white, 8)
        minimum_area = max(100.0, self.min_area_fraction * width * height)
        best: tuple[int, int, int, int, int] | None = None
        for index in range(1, count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[index]
            )
            if component_width < self.min_width_fraction * width:
                continue
            if component_width > 0.98 * width:
                continue
            if component_height < 5 or component_height > self.max_height_fraction * height:
                continue
            if component_width / max(component_height, 1) < self.min_aspect:
                continue
            if area < minimum_area:
                continue
            candidate = (x, y, component_width, component_height, area)
            if best is None or candidate[2] > best[2]:
                best = candidate
        if best is None:
            return None
        return (best[0], best[1], best[2], best[3])

    def _validate_frame(self, frame: np.ndarray) -> tuple[int, int]:
        if not isinstance(frame, np.ndarray):
            raise ObservationError(f"Expected an RGB NumPy frame, got {type(frame).__name__}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ObservationError(f"Expected an (H, W, 3) RGB frame, got {frame.shape}.")
        if frame.dtype != np.uint8:
            raise ObservationError(f"Expected uint8 pixels, got {frame.dtype}.")
        if frame.shape[0] < 32 or frame.shape[1] < 32:
            raise ObservationError(f"Frame is too small to inspect: {frame.shape[:2]}.")
        return int(frame.shape[0]), int(frame.shape[1])

    @staticmethod
    def _parse_region(value: Sequence[int] | None) -> tuple[int, int, int, int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ConfigError(
                "game.params.progress_region must be [x, y, width, height] or null."
            )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ConfigError("game.params.progress_region values must be integers.")
        x, y, width, height = (int(item) for item in value)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ConfigError(
                "game.params.progress_region requires x >= 0, y >= 0, width > 0, height > 0."
            )
        return (x, y, width, height)

    @staticmethod
    def _positive_float(value: Any, name: str, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number, got {value!r}.")
        number = float(value)
        if not np.isfinite(number) or number < minimum or number > maximum:
            raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value!r}.")
        return number

    @staticmethod
    def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer, got {value!r}.")
        if value < minimum or value > maximum:
            raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value!r}.")
        return value

    @staticmethod
    def _hsv(value: Sequence[int], name: str) -> tuple[int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ConfigError(f"{name} must be [hue, saturation, value].")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ConfigError(f"{name} values must be integers.")
        hue, saturation, brightness = (int(item) for item in value)
        if not (0 <= hue <= 179 and 0 <= saturation <= 255 and 0 <= brightness <= 255):
            raise ConfigError(
                f"{name} must use OpenCV HSV ranges: hue 0..179, saturation/value 0..255."
            )
        return (hue, saturation, brightness)


class GeometryDashAdapter(DesktopGameAdapter):
    """Control Geometry Dash through its real window and screen pixels."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    # The actual game window is already visible; a second screenshot viewer
    # would steal focus and slow the game down.
    show_capture_viewer = False
    extra_param_names = frozenset(
        {
            "progress_region",
            "bar_search_top_fraction",
            "bar_white_threshold",
            "bar_min_width_fraction",
            "bar_max_height_fraction",
            "bar_min_aspect",
            "bar_min_area_fraction",
            "bar_left_inset",
            "bar_right_inset",
            "bar_min_fill_columns",
            "fill_color_distance",
            "progress_fill_hsv_low",
            "progress_fill_hsv_high",
            "death_min_progress",
            "reset_progress_max",
            "completion_pct",
            "completion_frames",
            "lost_gameplay_frames",
            "start_progress_threshold",
            "reset_timeout",
            "reset_poll_interval",
            "resume_keys",
        }
    )

    def __init__(
        self,
        config: GameConfig,
        render_mode: str | None = None,
        *,
        capture_backend: Any | None = None,
        input_backend: Any | None = None,
        window_locator: WindowLocator | None = None,
    ) -> None:
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
                f"Unknown geometry_dash adapter params: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        if not config.observation.is_pixel:
            raise ConfigError(
                "Geometry Dash learns from pixels. Set observation.mode to 'grayscale' or 'rgb'."
            )
        if not config.episode.max_steps or config.episode.max_steps < 1:
            raise ConfigError(
                "Geometry Dash requires a finite episode.max_steps so keyboard control cannot "
                "continue indefinitely."
            )
        if config.training.n_envs != 1:
            raise ConfigError(
                "Geometry Dash requires training.n_envs: 1. Multiple environments would send "
                "conflicting keys to the same game window."
            )
        window_title = params.get("window_title")
        if not isinstance(window_title, str) or not window_title.strip():
            raise ConfigError(
                "Geometry Dash requires game.params.window_title (for example 'Geometry Dash')."
            )
        if params.get("require_window_focus") is not True:
            raise ConfigError(
                "Geometry Dash requires game.params.require_window_focus: true. This prevents "
                "Enter/Space from being sent to another application."
            )
        reset_keys = params.get("reset_keys")
        if not reset_keys:
            raise ConfigError(
                "Geometry Dash requires game.params.reset_keys (for example ['enter']) so each "
                "episode can restart the current attempt."
            )

        self._vision = GeometryDashVision(
            progress_region=params.get("progress_region"),
            search_top_fraction=params.get("bar_search_top_fraction", 0.09),
            white_threshold=params.get("bar_white_threshold", 200),
            min_width_fraction=params.get("bar_min_width_fraction", 0.15),
            max_height_fraction=params.get("bar_max_height_fraction", 0.08),
            min_aspect=params.get("bar_min_aspect", 5.0),
            min_area_fraction=params.get("bar_min_area_fraction", 0.0005),
            left_inset=params.get("bar_left_inset", 8),
            right_inset=params.get("bar_right_inset", 8),
            min_fill_columns=params.get("bar_min_fill_columns", 4),
            fill_color_distance=params.get("fill_color_distance", 30),
            fill_hsv_low=params.get("progress_fill_hsv_low", (8, 90, 90)),
            fill_hsv_high=params.get("progress_fill_hsv_high", (45, 255, 255)),
        )
        self.death_min_progress = self._number(
            params.get("death_min_progress", 0.8), "death_min_progress", 0.0, 100.0
        )
        self.reset_progress_max = self._number(
            params.get("reset_progress_max", 0.4), "reset_progress_max", 0.0, 100.0
        )
        if self.reset_progress_max >= self.death_min_progress:
            raise ConfigError(
                "reset_progress_max must be smaller than death_min_progress, otherwise small "
                "normal progress jitter looks like a death."
            )
        self.completion_pct = self._number(
            params.get("completion_pct", 99.0), "completion_pct", 50.0, 100.0
        )
        self.completion_frames = self._int(
            params.get("completion_frames", 2), "completion_frames", 1, 20
        )
        self.lost_gameplay_frames = self._int(
            params.get("lost_gameplay_frames", 20), "lost_gameplay_frames", 1, 300
        )
        self.start_progress_threshold = self._number(
            params.get("start_progress_threshold", 5.0),
            "start_progress_threshold",
            0.0,
            100.0,
        )
        if self.start_progress_threshold <= self.reset_progress_max:
            raise ConfigError(
                "start_progress_threshold must be larger than reset_progress_max so a restarted "
                "attempt at ~1% is accepted."
            )
        self.reset_timeout = self._number(
            params.get("reset_timeout", 5.0), "reset_timeout", 0.1, 30.0
        )
        self.reset_poll_interval = self._number(
            params.get("reset_poll_interval", 0.05), "reset_poll_interval", 0.005, 1.0
        )
        resume_keys = params.get("resume_keys", ["space"])
        if isinstance(resume_keys, str):
            resume_keys = [resume_keys]
        if not isinstance(resume_keys, (list, tuple)):
            raise ConfigError("game.params.resume_keys must be a list of key names.")
        self.resume_keys = [str(key).strip().lower() for key in resume_keys if str(key).strip()]

        self._window_locator = window_locator or PyAutoGUIWindowLocator()
        self.window_info: WindowInfo | None = None
        self._reading: ProgressReading | None = None
        self._last_progress_pct: float | None = None
        self._max_progress_pct = 0.0
        self._bar_missing_frames = 0
        self._died = False
        self._completed = False
        self._gameplay_lost = False
        self._completed_this_episode = False
        self._completion_hits = 0
        self._episode_started: float | None = None
        self._episode_deaths = 0
        self._reset_attempts = 0
        self._resume_attempts = 0

        try:
            super().__init__(
                config,
                render_mode=render_mode,
                capture_backend=capture_backend,
                input_backend=input_backend,
            )
        except Exception:
            # Base validation raises before it owns any resources, so there is
            # nothing to release here. Keeping this explicit makes failures easy
            # to follow in tests.
            raise
        # Real screen capture and input are asynchronous. Mark the environment
        # nondeterministic so Gymnasium's checker does not claim two identical
        # desktop interactions must produce byte-identical screenshots/info.
        self.spec = EnvSpec(id="GameMindGeometryDash-v0", nondeterministic=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        if self.window_title is None:
            raise AdapterError("Geometry Dash needs game.params.window_title.")
        self.window_info = self._window_locator.find(self.window_title)
        try:
            super().connect()
            _, inputs = self._require_backends()
            inputs.validate_keys(self.resume_keys)
            inputs.ensure_focus(self.window_title)
        except Exception:
            self._close_resources()
            raise
        logger.info(
            "Connected to Geometry Dash window %r (%dx%d). Capture: %s.",
            self.window_info.title,
            self.window_info.width,
            self.window_info.height,
            f"{self._capture.width}x{self._capture.height}"
            if self._capture is not None
            else "unknown",
        )

    def _reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        _, inputs = self._require_backends()
        inputs.release_all()
        self._ensure_focus(inputs)
        self._died = False
        self._completed = False
        self._gameplay_lost = False
        self._completed_this_episode = False
        self._completion_hits = 0
        self._reading = None
        self._last_progress_pct = None
        self._max_progress_pct = 0.0
        self._bar_missing_frames = 0
        self._episode_deaths = 0
        self._reset_attempts = 0
        self._resume_attempts = 0
        self._episode_started = time.monotonic()

        deadline = time.monotonic() + float(self.reset_timeout)
        self._tap_keys(inputs, self.reset_keys)
        self._reset_attempts += 1
        if self.reset_delay:
            time.sleep(float(self.reset_delay))

        resumed = False
        restarted_after_resume = False
        while time.monotonic() < deadline:
            frame = self._grab()
            reading = self._vision.read(frame)
            if reading.found:
                # Fill colour is normally the player's colour. Learn the empty
                # bar's colour now, then re-read so the reset observation and
                # baseline are in the same (contrast) representation as steps.
                if self._vision.calibrate(frame):
                    calibrated = self._vision.read(frame)
                    if calibrated.found:
                        reading = calibrated
                if reading.progress_pct <= self.start_progress_threshold:
                    self._track_reset(reading)
                    info = {
                        "capture_width": int(frame.shape[1]),
                        "capture_height": int(frame.shape[0]),
                        "window": self.window_info.as_dict() if self.window_info else None,
                        "progress_pct": round(reading.progress_pct, 2),
                        "bar_found": True,
                        "reset_attempts": self._reset_attempts,
                        "resume_attempts": self._resume_attempts,
                        "reset_keys": list(self.reset_keys),
                    }
                    return frame, info
                # The reset key did not restart, or a paused game resumed in the
                # middle of an attempt. Try one more restart after resuming.
                if self._reset_attempts < 2:
                    self._tap_keys(inputs, self.reset_keys)
                    self._reset_attempts += 1
                    if self.reset_delay:
                        time.sleep(float(self.reset_delay))
            else:
                if not resumed and self.resume_keys:
                    self._tap_keys(inputs, self.resume_keys)
                    self._resume_attempts += 1
                    resumed = True
                    time.sleep(min(0.25, float(self.reset_poll_interval) * 4))
                elif resumed and not restarted_after_resume and self._reset_attempts < 3:
                    self._tap_keys(inputs, self.reset_keys)
                    self._reset_attempts += 1
                    restarted_after_resume = True
                    if self.reset_delay:
                        time.sleep(float(self.reset_delay))
            time.sleep(float(self.reset_poll_interval))

        raise AdapterError(
            "Could not confirm a playable Geometry Dash level after reset. Open Geometry Dash, "
            "select a level, enable Auto-Retry, make sure the game is not paused or in a menu, "
            "and check game.params.window_title/progress_region. The adapter will not send "
            "keys blindly into a menu."
        )

    def _track_reset(self, reading: ProgressReading) -> None:
        self._reading = reading
        self._last_progress_pct = float(reading.progress_pct)
        self._max_progress_pct = float(reading.progress_pct)
        self._bar_missing_frames = 0
        self._episode_started = time.monotonic()

    def _ensure_focus(self, inputs: Any) -> None:
        if self.require_window_focus:
            try:
                inputs.ensure_focus(self.window_title)
            except InputBackendError:
                raise
            except Exception as exc:
                raise InputBackendError(
                    f"Could not focus Geometry Dash window {self.window_title!r}: {exc}"
                ) from exc

    def _tap_keys(self, inputs: Any, keys: Sequence[str]) -> None:
        if not keys:
            return
        self._ensure_focus(inputs)
        try:
            for key in keys:
                inputs.key_down(key)
                time.sleep(float(self.reset_key_duration))
                inputs.key_up(key)
        finally:
            inputs.release_all()

    # ------------------------------------------------------------------
    # Observations and input
    # ------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        frame = self._grab()
        self._update_reading(frame)
        return frame

    def _update_reading(self, frame: np.ndarray) -> None:
        reading = self._vision.read(frame)
        self._reading = reading
        self._died = False
        self._completed = False
        self._gameplay_lost = False
        if reading.found:
            progress = float(reading.progress_pct)
            if (
                self._last_progress_pct is not None
                and self._last_progress_pct >= self.death_min_progress
                and progress <= self.reset_progress_max
            ):
                self._died = True
                self._episode_deaths += 1
            self._last_progress_pct = progress
            self._max_progress_pct = max(self._max_progress_pct, progress)
            self._bar_missing_frames = 0
            if progress >= self.completion_pct and not self._completed_this_episode:
                self._completion_hits += 1
                if self._completion_hits >= self.completion_frames:
                    self._completed = True
                    self._completed_this_episode = True
            else:
                self._completion_hits = 0
        else:
            self._completion_hits = 0
            self._bar_missing_frames += 1
            if self._bar_missing_frames >= self.lost_gameplay_frames:
                self._gameplay_lost = True

    def is_episode_over(self) -> bool:
        if self._died or self._completed or self._gameplay_lost:
            return True
        return super().is_episode_over()

    def _reward_signals(self) -> Mapping[str, float]:
        found = bool(self._reading is not None and self._reading.found)
        alive = 1.0 if found and not self._died and not self._completed else 0.0
        last_progress = float(self._last_progress_pct or 0.0)
        return {
            "alive": alive,
            "progress_pct": last_progress,
            "progress": last_progress / 100.0,
            "max_progress_pct": float(self._max_progress_pct),
            "survival_seconds": self._survival_seconds(),
            "died": 1.0 if self._died else 0.0,
            "completed": 1.0 if self._completed else 0.0,
            "gameplay_lost": 1.0 if self._gameplay_lost else 0.0,
            "bar_found": 1.0 if found else 0.0,
        }

    def _extra_info(self) -> dict[str, Any]:
        return {
            "window_title": self.window_title,
            "window": self.window_info.as_dict() if self.window_info else None,
            "progress_pct": round(float(self._last_progress_pct or 0.0), 2),
            "max_progress_pct": round(float(self._max_progress_pct), 2),
            "survival_seconds": round(self._survival_seconds(), 3),
            "bar_found": bool(self._reading is not None and self._reading.found),
            "bar_confidence": float(self._reading.confidence) if self._reading else 0.0,
            "died": bool(self._died),
            "completed": bool(self._completed),
            "gameplay_lost": bool(self._gameplay_lost),
            "episode_deaths": int(self._episode_deaths),
            "reset_attempts": int(self._reset_attempts),
        }

    def _survival_seconds(self) -> float:
        if self._episode_started is None:
            return 0.0
        return max(0.0, time.monotonic() - self._episode_started)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"game.params.{name} must be a number, got {value!r}.")
        number = float(value)
        if not np.isfinite(number) or number < minimum or number > maximum:
            raise ConfigError(
                f"game.params.{name} must be between {minimum} and {maximum}, got {value!r}."
            )
        return number

    @staticmethod
    def _int(value: Any, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"game.params.{name} must be an integer, got {value!r}.")
        if value < minimum or value > maximum:
            raise ConfigError(
                f"game.params.{name} must be between {minimum} and {maximum}, got {value!r}."
            )
        return value
