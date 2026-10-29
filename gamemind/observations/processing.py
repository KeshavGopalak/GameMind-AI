"""Screen-frame preprocessing: validation, resize, colour conversion, normalization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from gymnasium import spaces

from gamemind.config import ObservationConfig
from gamemind.utils.errors import ObservationError


def _validate_frame(frame: np.ndarray, *, context: str) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ObservationError(
            f"{context}: expected a NumPy array, got {type(frame).__name__}. "
            "Adapters must return raw RGB uint8 frames from observe()."
        )
    if frame.ndim not in (2, 3):
        raise ObservationError(
            f"{context}: expected a 2D grayscale or 3D colour frame, got shape {frame.shape}."
        )
    if frame.dtype != np.uint8:
        if frame.dtype.kind == "f":
            raise ObservationError(
                f"{context}: expected uint8 pixel values 0..255, got {frame.dtype}. "
                "Convert float frames to uint8 (for example np.clip(frame, 0, 255).astype(np.uint8))."
            )
        raise ObservationError(
            f"{context}: expected uint8 pixel values 0..255, got {frame.dtype}."
        )
    if frame.ndim == 3:
        if frame.shape[2] not in (1, 3, 4):
            raise ObservationError(
                f"{context}: expected 1, 3, or 4 channels, got {frame.shape[2]} "
                f"(shape {frame.shape})."
            )
        if frame.shape[2] == 1:
            frame = frame[:, :, 0]
        elif frame.shape[2] == 4:
            # Most capture backends return BGRA/RGBA; alpha is not useful for learning.
            frame = frame[:, :, :3]
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        raise ObservationError(
            f"{context}: frame is too small ({width}x{height}); expected at least 2x2 pixels."
        )
    if not np.isfinite(frame).all():  # always true for uint8, retained for defensive clarity
        raise ObservationError(f"{context}: frame contains non-finite pixel values.")
    return frame


@dataclass
class ObservationProcessor:
    """Convert one raw RGB frame into the exact tensor layout CnnPolicy expects.

    The processor always returns channel-first arrays: ``(C, height, width)``.
    With ``normalize=False`` (the default) values remain uint8 0..255, which is
    what Stable-Baselines3's ``CnnPolicy`` requires. It performs its own scaling
    internally. Normalized float32 observations are available for custom feature
    extractors, but are deliberately not silently paired with CnnPolicy.
    """

    config: ObservationConfig

    def __post_init__(self) -> None:
        if not self.config.is_pixel:
            raise ObservationError(
                "ObservationProcessor is only for pixel modes ('rgb' or 'grayscale'). "
                f"Got mode={self.config.mode!r}."
            )

    @property
    def channels(self) -> int:
        return 1 if self.config.mode == "grayscale" else 3

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.config.height, self.config.width)

    @property
    def observation_space(self) -> spaces.Box:
        if self.config.normalize:
            return spaces.Box(low=0.0, high=1.0, shape=self.output_shape, dtype=np.float32)
        return spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

    def process(self, frame: np.ndarray, *, context: str = "observation") -> np.ndarray:
        """Validate and preprocess one frame, raising ObservationError on mistakes."""
        rgb_or_gray = _validate_frame(frame, context=context)
        if self.config.crop is not None:
            top_f, left_f, bottom_f, right_f = self.config.crop
            source_height, source_width = rgb_or_gray.shape[:2]
            top = int(round(source_height * top_f))
            bottom = source_height - int(round(source_height * bottom_f))
            left = int(round(source_width * left_f))
            right = source_width - int(round(source_width * right_f))
            if bottom - top < 2 or right - left < 2:
                raise ObservationError(
                    f"{context}: observation.crop leaves a {right - left}x{bottom - top} region; "
                    "keep at least 2x2 pixels after cropping."
                )
            rgb_or_gray = rgb_or_gray[top:bottom, left:right]
        target_size = (self.config.width, self.config.height)
        source_size = (rgb_or_gray.shape[1], rgb_or_gray.shape[0])
        if source_size != target_size:
            interpolation = cv2.INTER_AREA if (
                source_size[0] > target_size[0] or source_size[1] > target_size[1]
            ) else cv2.INTER_LINEAR
            resized = cv2.resize(rgb_or_gray, target_size, interpolation=interpolation)
        else:
            resized = rgb_or_gray.copy()

        if self.config.mode == "grayscale":
            if resized.ndim == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            image = resized[np.newaxis, :, :]
        else:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            image = np.transpose(resized, (2, 0, 1))

        image = np.ascontiguousarray(image)
        if self.config.normalize:
            processed = (image.astype(np.float32) / 255.0).copy()
        else:
            processed = image.astype(np.uint8, copy=False)
        self.validate(processed, context=context)
        return processed

    def validate(self, observation: np.ndarray, *, context: str = "observation") -> None:
        """Check a processed observation against the configured space."""
        if not isinstance(observation, np.ndarray):
            raise ObservationError(f"{context}: processed observation must be a NumPy array.")
        if observation.shape != self.output_shape:
            raise ObservationError(
                f"{context}: expected shape {self.output_shape}, got {observation.shape}."
            )
        if observation.dtype != self.observation_space.dtype:
            raise ObservationError(
                f"{context}: expected dtype {self.observation_space.dtype}, got {observation.dtype}."
            )
        if self.config.normalize:
            if not np.isfinite(observation).all():
                raise ObservationError(f"{context}: normalized observation contains NaN or infinity.")
            if observation.min(initial=0.0) < 0.0 or observation.max(initial=0.0) > 1.0:
                raise ObservationError(
                    f"{context}: normalized observation must be in [0, 1]; "
                    f"found [{observation.min():.3f}, {observation.max():.3f}]."
                )
