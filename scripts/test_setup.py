#!/usr/bin/env python
"""Verify that this machine can run GameMind AI.

Run this before training:

    python scripts/test_setup.py

It performs real imports, creates real environments, and runs one step. It does
not install anything and does not send keyboard/mouse input.
"""

from __future__ import annotations

import importlib
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamemind.config import load_game_config  # noqa: E402
from gamemind.envs import build_env  # noqa: E402
from gamemind.utils.paths import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    ensure_dir,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    icon = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}[status]
    print(f"{icon} {name}" + (f" — {detail}" if detail else ""))


def check(name: str, function) -> None:
    try:
        detail = function()
        record(PASS, name, str(detail or ""))
    except Exception as exc:
        record(FAIL, name, f"{type(exc).__name__}: {exc}")


def check_optional(name: str, function) -> None:
    try:
        detail = function()
        record(PASS, name, str(detail or ""))
    except Exception as exc:
        record(WARN, name, f"optional component unavailable: {type(exc).__name__}: {exc}")


def check_import(name: str, package: str | None = None) -> str:
    module = importlib.import_module(package or name)
    version = getattr(module, "__version__", "")
    return f"{version}".strip() or "imported"


def main() -> int:
    print("GameMind AI setup check")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Project: {ROOT}")
    print("-" * 72)

    check("Python >= 3.11", lambda: _check_python())

    required = [
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("torch", "torch"),
        ("opencv", "cv2"),
        ("pygame", "pygame"),
        ("PyYAML", "yaml"),
        ("tensorboard", "tensorboard"),
        ("pandas", "pandas"),
        ("pytest", "pytest"),
    ]
    for display_name, package in required:
        check(f"import {display_name}", lambda package=package: check_import(package, package))

    check_optional("import streamlit (dashboard)", lambda: check_import("streamlit", "streamlit"))
    check_optional("import mss (desktop capture)", lambda: check_import("mss", "mss"))
    check_optional("import pyautogui (desktop input)", lambda: check_import("pyautogui", "pyautogui"))

    check("parse configs/default.yaml", lambda: _check_config("configs/default.yaml"))
    check("parse configs/cartpole.yaml", lambda: _check_config("configs/cartpole.yaml"))
    check("parse configs/lunarlander.yaml", lambda: _check_config("configs/lunarlander.yaml"))
    check("parse configs/example_game.yaml", lambda: _check_config("configs/example_game.yaml"))
    check(
        "parse configs/example_game_random.yaml",
        lambda: _check_config("configs/example_game_random.yaml"),
    )
    check("parse configs/desktop_game.example.yaml", lambda: _check_config("configs/desktop_game.example.yaml"))

    check("CartPole-v1 reset/step", _check_cartpole)
    check("LunarLander-v3 reset/step (Box2D)", _check_lunarlander)
    check("pygame off-screen rendering", _check_pygame)
    check("built-in Target Chase pixel environment", _check_example_game)
    check("write access to models/checkpoints/logs", _check_directories)
    check("PyTorch device", _check_device)

    print("-" * 72)
    failures = [row for row in _results if row[0] == FAIL]
    warnings = [row for row in _results if row[0] == WARN]
    print(f"{len(_results) - len(failures) - len(warnings)} passed, {len(warnings)} warnings, {len(failures)} failed")
    if warnings:
        print("Warnings:")
        for _, name, detail in warnings:
            print(f"  - {name}: {detail}")
    if failures:
        print("\nSetup is not ready. Fix the failures above before training.")
        print("Desktop capture/input libraries are only required for the desktop adapter.")
        return 1
    print("\nSetup looks good. Try:")
    print("  python scripts/train.py --game example_game --timesteps 50000")
    print("  python scripts/play.py --model models/example_game.zip")
    print("For a live window, make sure a desktop session/display is available.")
    return 0


def _check_python() -> str:
    if sys.version_info < (3, 11):
        raise RuntimeError("Use Python 3.11 or newer (3.11–3.13 are tested).")
    return "compatible"


def _check_config(relative_path: str) -> str:
    config = load_game_config(ROOT / relative_path)
    return f"game={config.name}, adapter={config.adapter}"


def _check_cartpole() -> str:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    try:
        observation, info = env.reset(seed=1)
        result = env.step(env.action_space.sample())
        if not isinstance(info, dict) or len(result) != 5:
            raise RuntimeError("CartPole did not follow the Gymnasium 5-value API")
        return f"observation shape={observation.shape}, dtype={observation.dtype}"
    finally:
        env.close()


def _check_lunarlander() -> str:
    import gymnasium as gym

    try:
        env = gym.make("LunarLander-v3")
    except Exception as exc:
        raise RuntimeError(
            'LunarLander needs Box2D. Install with: pip install "gymnasium[box2d]". '
            f"Original error: {exc}"
        ) from exc
    try:
        observation, _ = env.reset(seed=1)
        env.step(0)
        return f"observation shape={observation.shape}"
    finally:
        env.close()


def _check_pygame() -> str:
    import numpy as np
    import pygame

    surface = pygame.Surface((64, 48))
    surface.fill((20, 40, 60))
    raw = pygame.image.tostring(surface, "RGB")
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(48, 64, 3)
    if frame.shape != (48, 64, 3):
        raise RuntimeError("unexpected pygame surface conversion")
    return f"pygame {pygame.version.ver}"


def _check_example_game() -> str:
    import numpy as np

    config = load_game_config("configs/example_game.yaml")
    # Reduce work; this check should never train.
    from dataclasses import replace

    config = replace(
        config,
        observation=replace(config.observation, width=64, height=64),
        training=replace(config.training, n_envs=1),
    )
    env = build_env(config)
    try:
        observation, _ = env.reset(seed=3)
        observation, reward, terminated, truncated, info = env.step(0)
        if not isinstance(observation, np.ndarray):
            raise RuntimeError("observation is not a NumPy array")
        return f"observation={observation.shape}, reward={reward:.3f}"
    finally:
        env.close()


def _check_directories() -> str:
    paths = [DEFAULT_MODEL_DIR, DEFAULT_CHECKPOINT_DIR, DEFAULT_LOG_DIR]
    for path in paths:
        ensure_dir(path)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write_check_", delete=True):
            pass
    return ", ".join(str(path.name) + "/" for path in paths)


def _check_device() -> str:
    import torch

    if torch.cuda.is_available():
        return f"CUDA available ({torch.cuda.get_device_name(0)})"
    return f"CPU only (torch {torch.__version__})"


if __name__ == "__main__":
    raise SystemExit(main())
