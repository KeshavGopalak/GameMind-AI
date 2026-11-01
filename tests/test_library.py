"""Tests for the persistent game/level library.

These cover profile storage, per-level isolation, session persistence, resume,
compatibility-aware transfer, queue recovery, and worker duplicate prevention.
Real training is exercised twice (once in-process, once through the worker
script) with the built-in Target Chase game; nothing here touches a desktop game.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gamemind.library import (
    GameLibrary,
    SessionRequest,
    TrainingController,
    add_skill,
    choose_initialization,
    create_queue,
    describe_level,
    detect_completion,
    find_transfer_source,
    level_summary,
    list_queues,
    load_queue,
    queue_summary,
    recover_queue,
    run_queue,
    run_session,
    session_rows,
    skills_for_level,
    validate_level,
)
from gamemind.library.control import WorkerState
from gamemind.library.models import LevelProfile, SessionRecord
from gamemind.library.store import level_config_name
from gamemind.library.transfer import compatibility_checks
from gamemind.training.model_io import metadata_path_for
from gamemind.utils.errors import LibraryError, WorkerError
from gamemind.utils.serialization import atomic_write_json, read_json

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def library(tmp_path: Path) -> GameLibrary:
    lib = GameLibrary(tmp_path / "library")
    lib.create_game("Target Chase", "example_game")
    return lib


def _game_id(library: GameLibrary) -> str:
    return library.list_games()[0].id


def _shrink(library: GameLibrary, game_id: str, level_id: str, *, steps: int = 32) -> None:
    """Make a level cheap to train in tests without touching other settings."""
    config = library.level_config(game_id, level_id)
    small = replace(
        config,
        observation=replace(config.observation, width=48, height=48, stack=2),
        episode=replace(config.episode, max_steps=8),
        training=replace(
            config.training,
            timesteps=steps,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            checkpoint_freq=0,
            eval_freq=0,
            save_best=False,
            tensorboard=False,
            verbose=0,
            device="cpu",
        ),
    )
    library.save_level_config_text(game_id, level_id, yaml.safe_dump(small.as_dict(), sort_keys=False))


def _level(library: GameLibrary, name: str = "Puzzle", *, shrink: bool = True) -> LevelProfile:
    game_id = _game_id(library)
    level = library.create_level(game_id, name)
    if shrink:
        _shrink(library, game_id, level.id)
    return level


def _fake_model(library: GameLibrary, level: LevelProfile, session_id: str, config) -> Path:
    path = library.level_model_path(level.game_id, level.id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-model")
    atomic_write_json(metadata_path_for(path), {"config": config.as_dict(), "num_timesteps": 64})
    return path


def _session(
    library: GameLibrary,
    level: LevelProfile,
    session_id: str,
    *,
    evaluation: dict | None = None,
    model: Path | None = None,
    state: str = "finished",
    completion: dict | None = None,
) -> SessionRecord:
    record = SessionRecord(
        session_id=session_id,
        game_id=level.game_id,
        level_id=level.id,
        run_id=f"run-{session_id}",
        state=state,
        final_timesteps=64,
        model_path=str(model) if model else None,
        evaluation=evaluation,
        completion=completion or {"verified": False, "mode": level.completion_mode, "count": 0},
    )
    library.save_session(record)
    return record


# ---------------------------------------------------------------------------
# Profiles and storage
# ---------------------------------------------------------------------------
def test_migration_is_idempotent_and_preserves_edits(tmp_path: Path):
    lib = GameLibrary(tmp_path / "library")
    first = lib.ensure_migrated()
    assert any("geometry_dash" in action for action in first)
    game = lib.get_game("geometry_dash")
    level = lib.list_levels(game.id)[0]
    lib.update_level(game.id, level.id, name="Renamed by the user")

    assert lib.ensure_migrated() == []
    assert lib.list_levels(game.id)[0].name == "Renamed by the user"
    assert len(lib.list_levels(game.id)) == 1


def test_level_profiles_are_isolated(library: GameLibrary):
    game_id = _game_id(library)
    first = _level(library, "First")
    second = _level(library, "Second")

    first_config = library.level_config(game_id, first.id)
    second_config = library.level_config(game_id, second.id)
    assert first_config.name == level_config_name(game_id, first.id)
    assert second_config.name == level_config_name(game_id, second.id)
    assert first_config.name != second_config.name
    assert library.level_models_dir(game_id, first.id) != library.level_models_dir(game_id, second.id)

    library.remove_level(game_id, first.id)
    assert [level.id for level in library.list_levels(game_id)] == [second.id]
    assert library.level_config_path(game_id, second.id).exists()


def test_updating_a_level_keeps_its_own_configuration(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Customized")
    before = library.level_config(game_id, level.id)

    library.update_level(game_id, level.id, name="Renamed", notes="still custom")
    after = library.level_config(game_id, level.id)

    assert after.name == level_config_name(game_id, level.id)
    assert after.observation.width == before.observation.width == 48
    assert after.observation.stack == before.observation.stack == 2
    assert after.episode.max_steps == before.episode.max_steps == 8
    assert after.training.n_steps == before.training.n_steps == 32
    assert library.get_level(game_id, level.id).name == "Renamed"


def test_duplicate_level_copies_settings_but_not_history(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Original")
    _session(library, level, "s1")

    copy = library.duplicate_level(game_id, level.id, new_name="Variant")
    assert copy.id != level.id
    assert copy.name == "Variant"
    assert library.list_sessions(game_id, copy.id) == []
    assert library.list_sessions(game_id, level.id)[0].session_id == "s1"
    assert library.level_config_path(game_id, copy.id).exists()


def test_remove_level_archives_trained_files(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Doomed")
    model = _fake_model(library, level, "s1", library.level_config(game_id, level.id))

    summary = library.remove_level(game_id, level.id)
    assert summary["sessions_removed"] == 0
    assert not model.exists()
    archived = Path(summary["archived_to"])
    assert (archived / "models" / "s1.zip").read_bytes() == b"not-a-real-model"
    with pytest.raises(LibraryError):
        library.get_level(game_id, level.id)


def test_remove_game_refuses_while_levels_exist(library: GameLibrary):
    game_id = _game_id(library)
    _level(library, "Kept")
    with pytest.raises(LibraryError, match="still has"):
        library.remove_game(game_id)
    library.remove_game(game_id, remove_levels=True)
    assert library.list_games() == []


def test_unknown_adapter_is_refused_with_guidance(library: GameLibrary):
    with pytest.raises(LibraryError, match="No adapter"):
        library.create_game("Minecraft", "minecraft_adapter")
    with pytest.raises(LibraryError, match="adapter"):
        library.create_game("Fake", "made_up")
    assert [game.id for game in library.list_games()] == ["target_chase"]


def test_desktop_adapter_needs_a_real_configuration(library: GameLibrary):
    with pytest.raises(LibraryError, match="configuration"):
        library.create_game("Some Desktop Game", "desktop")


def test_gymnasium_game_needs_a_valid_env_id(library: GameLibrary):
    with pytest.raises(LibraryError):
        library.create_game("No env", "gymnasium")
    game = library.create_game("Cart Pole", "gymnasium", params={"env_id": "CartPole-v1"})
    assert game.adapter == "gymnasium"
    config = library.game_config(game.id)
    assert config.params["env_id"] == "CartPole-v1"


def test_level_profile_validation(library: GameLibrary):
    game_id = _game_id(library)
    with pytest.raises(LibraryError):
        library.create_level(game_id, "Bad completion", completion_mode="guess")
    with pytest.raises(LibraryError):
        library.create_level(game_id, "Bad threshold", completion_mode="bar", completion_pct=10.0)
    with pytest.raises(LibraryError):
        library.create_level(game_id, "")


def test_completion_mode_updates_geometry_dash_config(tmp_path: Path):
    lib = GameLibrary(tmp_path / "library")
    lib.ensure_migrated()
    game = lib.get_game("geometry_dash")
    level = lib.list_levels(game.id)[0]
    assert lib.level_config(game.id, level.id).params["allow_bar_completion"] is False

    lib.update_level(game.id, level.id, completion_mode="bar", completion_pct=98.5, completion_frames=3)
    config = lib.level_config(game.id, level.id)
    assert config.params["allow_bar_completion"] is True
    assert config.params["completion_pct"] == 98.5
    assert config.params["completion_frames"] == 3

    lib.update_level(game.id, level.id, completion_mode="disabled")
    assert lib.level_config(game.id, level.id).params["allow_bar_completion"] is False


def test_unsupported_completion_detector_is_refused(library: GameLibrary):
    game_id = _game_id(library)
    with pytest.raises(LibraryError, match="not implemented"):
        library.create_level(game_id, "No detector", completion_mode="bar")
    level = _level(library, "Still fine")
    with pytest.raises(LibraryError, match="not implemented"):
        library.update_level(game_id, level.id, completion_mode="bar")


# ---------------------------------------------------------------------------
# Completion is never inferred
# ---------------------------------------------------------------------------
def test_completion_disabled_never_verifies(library: GameLibrary):
    level = _level(library, "No detector")
    result = detect_completion(
        level,
        [{"episode": 1, "completed": True, "max_progress_pct": 100.0}],
        [{"episodes": [{"completed": True, "max_progress_pct": 100.0}]}],
    )
    assert result["verified"] is False
    assert "not accepted" in result["reason"]


def test_completion_requires_the_detector_signal(library: GameLibrary):
    level = LevelProfile(
        id="detector",
        game_id=_game_id(library),
        name="Detector",
        completion_mode="bar",
        completion_pct=99.0,
    )
    assert detect_completion(level, [{"episode": 1, "max_progress_pct": 100.0}])["verified"] is False
    verified = detect_completion(level, [{"episode": 1, "completed": True}])
    assert verified["verified"] is True
    assert verified["count"] == 1


def test_progress_alone_is_not_a_win(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "High progress")
    _session(
        library,
        level,
        "s1",
        evaluation={"mean_progress_pct": 99.9, "episodes": [{"max_progress_pct": 99.9}]},
    )
    progress = describe_level(library, level)
    assert progress.verified_completion is False
    assert progress.status != "completed"
    assert level_summary(library, level)["verified_completion"] is False


# ---------------------------------------------------------------------------
# Real session, evaluation, and persistence
# ---------------------------------------------------------------------------
def test_session_trains_evaluates_and_persists(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Learns")

    record = run_session(
        library,
        SessionRequest(
            game_id=game_id,
            level_id=level.id,
            timesteps=32,
            evaluate=True,
            eval_episodes=1,
            device="cpu",
        ),
    )

    assert record.state == "finished"
    assert record.model_path and Path(record.model_path).exists()
    assert record.evaluation is not None
    assert record.evaluation["episodes"]
    assert record.final_timesteps >= 32
    assert record.reason

    reopened = GameLibrary(library.root)
    sessions = reopened.list_sessions(game_id, level.id)
    assert [session.session_id for session in sessions] == [record.session_id]
    assert Path(reopened.session_path(game_id, level.id, record.session_id)).exists()
    evaluations = reopened.list_evaluations(game_id, level.id)
    assert len(evaluations) == 1

    progress = describe_level(reopened, level)
    assert progress.status in {"evaluated", "improving", "configured"}
    assert progress.best_eval_score is not None
    assert session_rows(reopened, level)[0]["state"] == "finished"


def test_interrupted_session_record_still_saved(library: GameLibrary):
    """A session that fails leaves a record with the reason, not an empty folder."""
    game_id = _game_id(library)
    level = _level(library, "Fails")
    payload = library.level_config(game_id, level.id).as_dict()
    # A parameter the adapter rejects when it connects: valid YAML, failing env.
    payload["game"]["params"]["no_such_option"] = 1
    library.save_level_config_text(game_id, level.id, yaml.safe_dump(payload, sort_keys=False))

    record = run_session(
        library,
        SessionRequest(game_id=game_id, level_id=level.id, timesteps=32, evaluate=False, device="cpu"),
    )
    assert record.state == "error"
    assert record.error
    assert library.list_sessions(game_id, level.id)[0].state == "error"
    assert describe_level(library, level).status == "needs_attention"


# ---------------------------------------------------------------------------
# Resume and transfer
# ---------------------------------------------------------------------------
def test_initialization_prefers_best_evaluated_model(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Resumable")
    config = library.level_config(game_id, level.id)
    weak = _fake_model(library, level, "s-weak", config)
    strong = _fake_model(library, level, "s-strong", config)
    _session(library, level, "s-weak", evaluation={"mean_progress_pct": 3.0}, model=weak)
    _session(library, level, "s-strong", evaluation={"mean_progress_pct": 7.5}, model=strong)

    choice = choose_initialization(library, level)
    assert choice.source == "same_level_best"
    assert Path(choice.path) == strong


def test_scratch_policy_ignores_saved_models(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Fresh")
    config = library.level_config(game_id, level.id)
    model = _fake_model(library, level, "s1", config)
    _session(library, level, "s1", evaluation={"mean_progress_pct": 9.0}, model=model)
    level = library.update_level(game_id, level.id, init_policy="scratch")

    choice = choose_initialization(library, level)
    assert choice.source == "scratch"
    assert choice.path is None


def test_transfer_between_levels_requires_matching_layout(library: GameLibrary):
    game_id = _game_id(library)
    source = _level(library, "Source")
    target = _level(library, "Target")
    config = library.level_config(game_id, source.id)
    model = _fake_model(library, source, "s1", config)
    _session(library, source, "s1", evaluation={"mean_progress_pct": 12.0}, model=model)
    target = library.update_level(game_id, target.id, init_policy="transfer_game")

    plan = find_transfer_source(library, target)
    assert plan.source == "transfer"
    assert plan.source_level_id == source.id
    assert plan.reason and "not credit" in plan.reason

    # Change the target's action layout: the saved policy can no longer be reused.
    target_config = library.level_config(game_id, target.id)
    payload = target_config.as_dict()
    bindings = payload["actions"]["bindings"]
    bindings.append({"name": "extra", "keys": ["shift"], "duration": 0.02})
    library.save_level_config_text(game_id, target.id, yaml.safe_dump(payload, sort_keys=False))

    blocked = find_transfer_source(library, library.get_level(game_id, target.id))
    assert blocked.source == "scratch"
    assert "Action bindings match" in blocked.reason


def test_compatibility_checks_flag_observation_changes(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Checked")
    config = library.level_config(game_id, level.id)
    saved = config.as_dict()
    saved["observation"]["width"] = 64
    checks = compatibility_checks(config, saved)
    failed = {check.name for check in checks if not check.ok}
    assert "Observation preprocessing matches" in failed


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
def _queue_with_levels(library: GameLibrary, names: list[str], **kwargs) -> tuple[object, list[LevelProfile]]:
    game_id = _game_id(library)
    levels = [_level(library, name) for name in names]
    queue = create_queue(
        library,
        game_id,
        [level.id for level in levels],
        timesteps_per_level=kwargs.pop("timesteps_per_level", 32),
        **kwargs,
    )
    return queue, levels


def test_queue_runs_levels_in_order_and_records_results(library: GameLibrary, monkeypatch):
    queue, levels = _queue_with_levels(library, ["One", "Two"])
    order: list[str] = []

    def fake_run_session(lib: GameLibrary, request: SessionRequest) -> SessionRecord:
        order.append(request.level_id)
        record = SessionRecord(
            session_id=request.session_id or "fake",
            game_id=request.game_id,
            level_id=request.level_id,
            run_id=f"run-{request.level_id}",
            state="finished",
            final_timesteps=request.timesteps,
            completion={"verified": request.level_id == levels[0].id, "mode": "bar", "count": 1},
        )
        lib.save_session(record)
        return record

    monkeypatch.setattr("gamemind.library.queue.run_session", fake_run_session)
    finished = run_queue(library, queue.queue_id)

    assert order == [levels[0].id, levels[1].id]
    assert finished.status == "completed"
    assert [task.state for task in finished.tasks] == ["completed", "done"]
    assert finished.tasks[0].verified_completion is True
    assert finished.tasks[1].verified_completion is False
    summary = queue_summary(library, finished)
    assert summary["remaining"] == 0


def test_queue_stops_on_error(library: GameLibrary, monkeypatch):
    queue, levels = _queue_with_levels(library, ["Boom"])

    def failing(lib: GameLibrary, request: SessionRequest) -> SessionRecord:
        record = SessionRecord(
            session_id=request.session_id or "fake",
            game_id=request.game_id,
            level_id=request.level_id,
            run_id="run-fail",
            state="error",
            error="adapter could not connect",
        )
        lib.save_session(record)
        return record

    monkeypatch.setattr("gamemind.library.queue.run_session", failing)
    finished = run_queue(library, queue.queue_id)
    assert finished.status == "error"
    assert finished.error == "adapter could not connect"
    assert finished.tasks[0].state == "error"


def test_interrupted_queue_recovers_without_losing_records(library: GameLibrary):
    queue, levels = _queue_with_levels(library, ["First", "Second"])
    queue.status = "running"
    queue.tasks[0].state = "done"
    queue.tasks[0].session_id = "kept-session"
    queue.tasks[1].state = "running"
    from gamemind.library import save_queue

    save_queue(library, queue)

    recovered = recover_queue(library, queue.queue_id)
    assert recovered.status == "pending"
    assert recovered.tasks[0].state == "done"
    assert recovered.tasks[0].session_id == "kept-session"
    assert recovered.tasks[1].state == "pending"
    assert "Interrupted" in (recovered.tasks[1].error or "")


def test_queue_skips_already_completed_levels(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Beaten")
    _session(
        library,
        level,
        "win",
        completion={"verified": True, "mode": "bar", "count": 2},
    )
    with pytest.raises(LibraryError, match="verified complete"):
        create_queue(library, game_id, [level.id], timesteps_per_level=32)

    queue = create_queue(
        library,
        game_id,
        [level.id],
        timesteps_per_level=32,
        continue_after_completion=True,
    )
    assert queue.tasks and queue.continue_after_completion


def test_queue_listing_survives_reload(library: GameLibrary):
    queue, _levels = _queue_with_levels(library, ["Listed"])
    reopened = GameLibrary(library.root)
    ids = [item.queue_id for item in list_queues(reopened)]
    assert queue.queue_id in ids
    assert load_queue(reopened, queue.queue_id).tasks[0].level_id == queue.tasks[0].level_id


# ---------------------------------------------------------------------------
# Worker control
# ---------------------------------------------------------------------------
def test_duplicate_worker_is_refused(library: GameLibrary):
    controller = TrainingController(library)
    level = _level(library, "Guarded")
    controller._write_state(
        WorkerState(pid=os.getpid(), kind="session", job_path="job.json", status="running")
    )
    assert controller.is_running() is True
    with pytest.raises(WorkerError, match="already running"):
        controller.start_session(level, timesteps=32)


def test_dead_worker_is_reconciled(library: GameLibrary):
    controller = TrainingController(library)
    controller._write_state(
        WorkerState(pid=2**31 - 1, kind="session", job_path="job.json", status="running")
    )
    assert controller.is_running() is False
    state = controller.reconcile()
    assert state is not None and state.status == "lost"


def test_pause_resume_and_stop_write_control_files(library: GameLibrary, tmp_path: Path):
    controller = TrainingController(library)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    controller._write_state(
        WorkerState(
            pid=os.getpid(),
            kind="session",
            job_path="job.json",
            status="running",
            run_dir=str(run_dir),
        )
    )
    controller.pause()
    assert (run_dir / "PAUSE").exists()
    assert controller.control_flags()["pause"] is True

    controller.resume()
    assert not (run_dir / "PAUSE").exists()
    assert controller.control_flags()["pause"] is False

    # Stop with a dead PID so the test never signals its own process.
    controller._write_state(
        WorkerState(
            pid=2**31 - 1,
            kind="session",
            job_path="job.json",
            status="running",
            run_dir=str(run_dir),
        )
    )
    controller.stop()
    assert (run_dir / "STOP").exists()
    assert controller.control_flags()["stop"] is True


def test_worker_script_runs_a_job_and_updates_state(library: GameLibrary, tmp_path: Path):
    game_id = _game_id(library)
    level = _level(library, "Background")
    controller = TrainingController(library)
    job_path = tmp_path / "job.json"
    atomic_write_json(
        job_path,
        {
            "job_id": "test-job",
            "kind": "session",
            "library_root": str(library.root),
            "game_id": game_id,
            "level_id": level.id,
            "session_id": "20260101-000000-abcdef",
            "timesteps": 32,
            "evaluate": False,
            "eval_episodes": 1,
            "device": "cpu",
            "stop_on_completion": False,
        },
    )
    env = dict(os.environ)
    env["GAMEMIND_LIBRARY_ROOT"] = str(library.root)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "library_worker.py"), "--job", str(job_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    sessions = library.list_sessions(game_id, level.id)
    assert [session.session_id for session in sessions] == ["20260101-000000-abcdef"]
    assert sessions[0].state == "finished"
    assert sessions[0].model_path and Path(sessions[0].model_path).exists()

    state = read_json(controller.state_path, default={})
    assert state == {} or state.get("status") in {"finished", "running"}


# ---------------------------------------------------------------------------
# Skills and validation
# ---------------------------------------------------------------------------
def test_skills_are_per_game_and_condition_checked(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Skilled")
    config = library.level_config(game_id, level.id)
    from gamemind.library.skills import conditions_for

    add_skill(
        library,
        game_id,
        name="Short jump press",
        kind="control",
        summary="A 30 ms press is enough for a short hop in this game's configuration.",
        conditions=conditions_for(config),
        validated=True,
        confidence=0.8,
    )
    applicable, rejected = skills_for_level(library, level)
    assert [skill.name for skill in applicable] == ["Short jump press"]
    assert rejected == []

    payload = config.as_dict()
    payload["observation"]["mode"] = "rgb"
    payload["observation"]["stack"] = 2
    library.save_level_config_text(game_id, level.id, yaml.safe_dump(payload, sort_keys=False))
    applicable, rejected = skills_for_level(library, library.get_level(game_id, level.id))
    assert applicable == []
    assert rejected and "observation mode" in "; ".join(rejected[0][1])


def test_static_validation_passes_and_is_stored(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Validated")
    report = validate_level(library, library.get_game(game_id), level)
    assert report.static_ok
    assert report.live is False
    stored = library.load_validation(game_id, level.id)
    assert stored is not None and stored["static_ok"] is True


def test_geometry_dash_seed_is_conservative(tmp_path: Path):
    lib = GameLibrary(tmp_path / "library")
    lib.ensure_migrated()
    game = lib.get_game("geometry_dash")
    level = lib.list_levels(game.id)[0]
    assert level.completion_mode == "disabled"
    assert level.init_policy == "scratch"
    config = lib.level_config(game.id, level.id)
    assert config.params["allow_bar_completion"] is False
    assert config.observation.is_pixel
    report = validate_level(lib, game, level)
    assert report.static_ok


def test_failed_yaml_edit_keeps_previous_config(library: GameLibrary):
    game_id = _game_id(library)
    level = _level(library, "Fragile")
    before = library.level_config_path(game_id, level.id).read_text(encoding="utf-8")
    with pytest.raises(LibraryError):
        library.save_level_config_text(game_id, level.id, "game: [unclosed")
    assert library.level_config_path(game_id, level.id).read_text(encoding="utf-8") == before
    assert library.level_config(game_id, level.id).name == level_config_name(game_id, level.id)
