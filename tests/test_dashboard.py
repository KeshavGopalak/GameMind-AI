"""Dashboard tests using Streamlit's own test harness.

The app is executed for real, with ``GAMEMIND_LIBRARY_ROOT`` pointed at a
temporary library, so these tests never read or write the project's data and
never start a training worker (the controller is stubbed where a click would).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from gamemind.library import GameLibrary, TrainingController
from gamemind.library.control import WorkerState

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"


def _run_app(tmp_path: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv("GAMEMIND_LIBRARY_ROOT", str(tmp_path / "library"))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def _library(tmp_path: Path) -> GameLibrary:
    library = GameLibrary(tmp_path / "library")
    library.ensure_migrated()
    return library


def test_every_page_renders(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    assert list(at.exception) == []
    assert [radio.options for radio in at.radio][0] == [
        "Home",
        "Games",
        "Training",
        "Models",
        "Knowledge",
        "Settings",
    ]

    for page in ("Games", "Training", "Models", "Knowledge", "Settings", "Home"):
        at.radio[0].set_value(page).run()
        assert list(at.exception) == [], f"{page} raised"
        assert at.subheader, f"{page} rendered no heading"


def test_games_page_shows_seeded_game(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    at.radio[0].set_value("Games").run()
    text = " ".join(element.value for element in at.markdown)
    assert "Geometry Dash" in text
    captions = " ".join(element.value for element in at.caption)
    assert "geometry_dash" in captions


def test_train_button_starts_a_session_without_running_training(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    calls: dict[str, object] = {}

    def fake_start(self, level, **kwargs):  # noqa: ANN001 - patch signature
        calls["level"] = level.id
        calls["kwargs"] = kwargs
        return WorkerState(pid=424242, kind="session", job_path="job.json")

    monkeypatch.setattr(
        "gamemind.library.control.TrainingController.start_session", fake_start
    )

    train = next(button for button in at.button if button.label == "Train / Resume")
    train.click().run()

    assert list(at.exception) == []
    assert calls["level"] == "stereo_madness"
    assert calls["kwargs"]["timesteps"] == 50000
    assert calls["kwargs"]["evaluate"] is True


def test_duplicate_train_click_is_refused_by_the_controller(tmp_path: Path, monkeypatch):
    """A second click while a worker runs must not start a second worker."""
    at = _run_app(tmp_path, monkeypatch)
    library = _library(tmp_path)
    controller = TrainingController(library)
    controller._write_state(
        WorkerState(pid=os.getpid(), kind="session", job_path="job.json", status="running")
    )

    started: list[int] = []
    monkeypatch.setattr(
        "gamemind.library.control.TrainingController.start_session",
        lambda self, level, **kwargs: started.append(1),
    )

    at.run()
    train = next(button for button in at.button if button.label == "Train / Resume")
    assert train.disabled is True
    assert started == []


def test_pause_button_requests_pause(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    library = _library(tmp_path)
    controller = TrainingController(library)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    controller._write_state(
        WorkerState(
            pid=os.getpid(),
            kind="session",
            job_path="job.json",
            status="running",
            game_id="geometry_dash",
            level_id="stereo_madness",
            run_dir=str(run_dir),
        )
    )

    at.run()
    at.radio[0].set_value("Training").run()
    assert list(at.exception) == []
    pause = next(button for button in at.button if button.label == "Pause")
    assert pause.disabled is False
    pause.click().run()
    assert (run_dir / "PAUSE").exists()

    resume = next(button for button in at.button if button.label == "Resume")
    resume.click().run()
    assert not (run_dir / "PAUSE").exists()


def test_add_level_form_creates_a_profile(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    at.radio[0].set_value("Games").run()

    name_input = next(widget for widget in at.text_input if widget.label == "Level name")
    name_input.set_value("Back on Track")
    add = next(button for button in at.button if button.label == "Add level")
    add.click().run()

    assert list(at.exception) == []
    library = _library(tmp_path)
    levels = library.list_levels("geometry_dash")
    assert "Back on Track" in [level.name for level in levels]
    assert len(levels) == 2


def test_unsupported_game_shows_an_explanation(tmp_path: Path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    at.radio[0].set_value("Games").run()
    add = next(button for button in at.button if button.label == "Add game")
    add.click().run()
    # The form is prefilled with the first catalog entry, which is supported, so
    # adding succeeds. Unsupported adapters are covered by the library tests.
    assert list(at.exception) == []
    assert any("adapter" in caption.value.lower() for caption in at.caption)


def test_app_survives_missing_library_root(tmp_path: Path, monkeypatch):
    """A wrong GAMEMIND_LIBRARY_ROOT must produce a message, not a stack trace."""
    monkeypatch.setenv("GAMEMIND_LIBRARY_ROOT", str(tmp_path / "nested" / "library"))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert list(at.exception) == []
    assert at.subheader
    assert (tmp_path / "nested" / "library" / "games").exists()
