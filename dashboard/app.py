"""GameMind AI — one dashboard for games, levels, training, and saved models.

Start it from the project root:

    streamlit run dashboard/app.py

Training never runs inside this process. Every session is launched as a
background worker (``scripts/library_worker.py``) so navigating the page, or
closing it, does not interrupt a run. All state comes from the persistent
library in ``library/`` and the existing ``checkpoints``/``models`` records.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from gamemind.library import (
    ADAPTER_HELP,
    GameLibrary,
    TrainingController,
    add_skill,
    catalog,
    choose_initialization,
    create_queue,
    default_library,
    describe_level,
    diagnose,
    list_queues,
    load_queue,
    option_for,
    queue_summary,
    readiness,
    recover_queue,
    session_rows,
    skills_for_level,
    update_skill,
    validate_level,
)
from gamemind.library.models import COMPLETION_MODES, INIT_POLICIES, LEVEL_KINDS
from gamemind.library.store import level_config_name
from gamemind.library.transfer import level_model_inventory
from gamemind.utils.errors import GameMindError, WorkerError
from gamemind.utils.serialization import read_json
from gamemind.utils.paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_MODEL_DIR

PAGES = ("Home", "Games", "Training", "Models", "Knowledge", "Settings")

COMPLETION_LABELS = {
    "disabled": "Disabled (recommended until the signal is verified in game)",
    "bar": "Progress bar reaches the completion threshold",
}
INIT_LABELS = {
    "best_same_level": "Continue this level's best evaluated model",
    "scratch": "Start from a fresh policy",
    "transfer_game": "Reuse another level's compatible policy when possible",
}
KIND_LABELS = {"official": "Official level", "custom": "Selected / custom level"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _library() -> GameLibrary:
    try:
        return default_library()
    except GameMindError:
        raise
    except Exception as exc:  # a broken library must not crash the whole page
        st.error(f"Could not open the GameMind library: {exc}")
        st.stop()
        raise


def _result_text(progress: Any) -> str:
    if progress is None:
        return "—"
    if progress.best_eval_metric == "progress_pct" and progress.best_eval_score is not None:
        return f"{progress.best_eval_score:.2f}% est. progress"
    if progress.best_eval_metric == "reward" and progress.best_eval_score is not None:
        return f"reward {progress.best_eval_score:.2f}"
    return "no evaluation yet"


def _status_line(progress: Any) -> str:
    return f"**{progress.label}**"


def _active(library: GameLibrary) -> tuple[TrainingController, Any, dict[str, Any], bool]:
    controller = TrainingController(library)
    state = controller.reconcile()
    policy = read_json(controller.state_path, default={})
    status: dict[str, Any] = {}
    if state is not None and state.run_dir:
        status = read_json(Path(state.run_dir) / "status.json", default={}) or {}
    if not status:
        current = read_json(controller.current_path, default={})
        if isinstance(current, dict) and current.get("run_dir"):
            status = read_json(Path(str(current["run_dir"])) / "status.json", default={}) or {}
    return controller, state, status if isinstance(status, dict) else {}, controller.is_running()


def _selected_game(library: GameLibrary):
    games = library.list_games()
    if not games:
        return None
    labels = {f"{game.name}": game.id for game in games}
    stored = st.session_state.get("game_id")
    options = list(labels.values())
    index = options.index(stored) if stored in options else 0
    label = st.selectbox("Game", list(labels), index=index, key="game_selector")
    game_id = labels[label]
    st.session_state["game_id"] = game_id
    return library.get_game(game_id)


def _selected_level(library: GameLibrary, game: Any):
    levels = library.list_levels(game.id)
    if not levels:
        return None
    labels: dict[str, str] = {}
    for level in levels:
        progress = describe_level(library, level)
        labels[f"{level.name} — {progress.label}"] = level.id
    options = list(labels.values())
    stored = st.session_state.get("level_id")
    index = options.index(stored) if stored in options else 0
    label = st.selectbox("Level", list(labels), index=index, key=f"level_selector_{game.id}")
    level_id = labels[label]
    st.session_state["level_id"] = level_id
    return library.get_level(game.id, level_id)


def _progress_bar(status: dict[str, Any]) -> None:
    value = status.get("progress")
    if isinstance(value, (int, float)):
        st.progress(min(1.0, max(0.0, float(value))))


def _status_metrics(status: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("State", str(status.get("state", "unknown")).upper())
    columns[1].metric("Timesteps", f"{int(status.get('timesteps', 0)):,}")
    columns[2].metric("Attempts", int(status.get("episodes", 0) or 0))
    best = status.get("best_progress_pct")
    columns[3].metric("Best progress", f"{float(best):.2f}%" if isinstance(best, (int, float)) else "—")


def _worker_banner(controller: TrainingController, state: Any, running: bool) -> None:
    if running and state is not None:
        st.success(
            f"Training is running in the background (PID {state.pid}, {state.kind}). "
            "You can leave this page; the session keeps going."
        )
    elif state is not None and state.status in {"error", "lost"}:
        st.warning(
            f"The last worker ended with status {state.status!r}. "
            "Open Training to read the log and the session record."
        )


def _train(library: GameLibrary, controller: TrainingController, level: Any, *, timesteps: int,
           evaluate: bool, eval_episodes: int, device: str) -> None:
    if controller.is_running():
        st.error("A worker is already running. Stop it before starting another.")
        return
    try:
        state = controller.start_session(
            level,
            timesteps=int(timesteps),
            evaluate=bool(evaluate),
            eval_episodes=int(eval_episodes),
            device=device,
        )
    except WorkerError as exc:
        st.error(str(exc))
        return
    except GameMindError as exc:
        st.error(f"Could not start training: {exc}")
        return
    st.success(f"Started session {state.session_id} (PID {state.pid}).")
    st.rerun()


def _level_controls(library: GameLibrary, controller: TrainingController, level: Any) -> None:
    settings = library.settings()
    running = controller.is_running()
    columns = st.columns([2, 1, 1, 1])
    timesteps = columns[0].number_input(
        "Timesteps this session",
        min_value=32,
        max_value=5_000_000,
        value=int(settings["default_timesteps"]),
        step=1000,
        key=f"ts_{level.id}",
    )
    evaluate = columns[1].checkbox(
        "Evaluate after", value=bool(settings["evaluate_after_training"]), key=f"ev_{level.id}"
    )
    eval_episodes = columns[2].number_input(
        "Eval attempts", min_value=1, max_value=20, value=int(settings["default_eval_episodes"]),
        key=f"ee_{level.id}",
    )
    start = columns[3].button(
        "Train / Resume",
        type="primary",
        width="stretch",
        disabled=running,
        key=f"train_{level.id}",
        help=(
            "Starts a background session from this level's configured starting point: its best "
            "evaluated model, the latest checkpoint, or a fresh policy."
        ),
    )
    choice = choose_initialization(library, level)
    st.caption(f"Starting point — {choice.reason}")
    if start:
        _train(
            library,
            controller,
            level,
            timesteps=int(timesteps),
            evaluate=bool(evaluate),
            eval_episodes=int(eval_episodes),
            device=str(settings["device"]),
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(library: GameLibrary) -> None:
    st.subheader("Home")
    controller, state, status, running = _active(library)
    games = library.list_games()
    total_levels = sum(len(library.list_levels(game.id)) for game in games)
    completed = sum(
        1
        for game in games
        for level in library.list_levels(game.id)
        if describe_level(library, level).verified_completion
    )
    columns = st.columns(3)
    columns[0].metric("Games", len(games))
    columns[1].metric("Levels", total_levels)
    columns[2].metric("Verified completed", completed)

    _worker_banner(controller, state, running)
    if running and status:
        _progress_bar(status)
        _status_metrics(status)

    st.divider()
    st.markdown("**Quick start**")
    if not games:
        st.info("No games yet. Add Geometry Dash from the Games page.")
        return
    game = _selected_game(library)
    if game is None:
        return
    level = _selected_level(library, game)
    if level is None:
        st.info("This game has no levels yet. Add one on the Games page.")
        return
    progress = describe_level(library, level)
    st.caption(f"{game.name} / {level.name} · {_status_line(progress)} · {_result_text(progress)}")
    _level_controls(library, controller, level)

    st.divider()
    st.markdown("**Recent sessions**")
    rows: list[dict[str, Any]] = []
    for candidate_game in games:
        for candidate_level in library.list_levels(candidate_game.id):
            for row in session_rows(library, candidate_level)[:5]:
                rows.append(
                    {
                        "game": candidate_game.name,
                        "level": candidate_level.name,
                        "started": str(row["started_at"])[:19].replace("T", " "),
                        "state": row["state"],
                        "timesteps": row["final_timesteps"],
                        "result": (
                            f"{row['eval_mean_progress_pct']:.2f}%"
                            if isinstance(row.get("eval_mean_progress_pct"), (int, float))
                            else (
                                f"reward {row['eval_mean_reward']:.2f}"
                                if isinstance(row.get("eval_mean_reward"), (int, float))
                                else "—"
                            )
                        ),
                        "verified": row["verified_completion"],
                    }
                )
    rows.sort(key=lambda item: item["started"], reverse=True)
    if rows:
        st.dataframe(rows[:12], hide_index=True, width="stretch")
    else:
        st.caption("No sessions yet. Starting your first run creates a saved session record.")


def page_games(library: GameLibrary) -> None:
    st.subheader("Games")
    controller, state, status, running = _active(library)
    games = library.list_games()
    if not games:
        st.info("No games yet. Use Add game below.")

    for game in games:
        usable, why = readiness(game.adapter)
        levels = library.list_levels(game.id)
        sessions = sum(len(library.list_sessions(game.id, level.id)) for level in levels)
        best_text = "—"
        best_value = -1.0
        for level in levels:
            progress = describe_level(library, level)
            if progress.best_eval_score is not None and progress.best_eval_score > best_value:
                best_value = progress.best_eval_score
                best_text = f"{level.name}: {_result_text(progress)}"
        last_at = max(
            (session.started_at for level in levels for session in library.list_sessions(game.id, level.id)),
            default=None,
        )
        with st.container(border=True):
            columns = st.columns([3, 2, 2, 2])
            columns[0].markdown(f"**{game.name}**")
            columns[0].caption(f"adapter: {game.adapter}" + ("" if usable else " · not available"))
            columns[1].caption("Levels")
            columns[1].markdown(f"{len(levels)}")
            columns[2].caption("Best recorded result")
            columns[2].markdown(best_text)
            columns[3].caption("Last session")
            columns[3].markdown(str(last_at)[:19].replace("T", " ") if last_at else "never")
            if not usable:
                st.warning(why)
            st.caption(game.description or "")
            buttons = st.columns([1, 1, 1, 3])
            if buttons[0].button("Open", key=f"open_{game.id}", width="stretch"):
                st.session_state["game_id"] = game.id
                st.session_state["nav"] = "Games"
                st.rerun()
            if buttons[1].button("Configure", key=f"cfg_{game.id}", width="stretch"):
                st.session_state["game_id"] = game.id
                st.session_state["nav"] = "Settings"
                st.rerun()
            with buttons[2].popover("Remove", width="stretch"):
                st.caption(
                    "Removing the profile deletes the library record, level settings and session "
                    "history. Model and checkpoint files stay on disk."
                )
                confirm = st.checkbox("I understand", key=f"rm_game_{game.id}")
                if st.button("Remove game", disabled=not confirm, key=f"rm_game_btn_{game.id}"):
                    try:
                        library.remove_game(game.id, remove_levels=True)
                    except GameMindError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pop("game_id", None)
                        st.session_state.pop("level_id", None)
                        st.rerun()
            if len(levels) and st.session_state.get("game_id") == game.id:
                st.divider()
                _game_detail(library, controller, game)

    st.divider()
    _add_game_form(library)


def _game_detail(library: GameLibrary, controller: TrainingController, game: Any) -> None:
    levels = library.list_levels(game.id)
    st.markdown(f"### {game.name}")
    rows = []
    for level in levels:
        progress = describe_level(library, level)
        rows.append(
            {
                "level": level.name,
                "status": progress.label,
                "best result": _result_text(progress),
                "attempts": progress.attempts,
                "sessions": progress.sessions,
                "verified win": progress.verified_completion,
                "last training": str(progress.last_training_at or "—")[:19].replace("T", " "),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No levels yet. Add one below.")

    level = _selected_level(library, game)
    if level is not None:
        progress = describe_level(library, level)
        st.markdown(f"#### {level.name}")
        st.caption(
            f"{_status_line(progress)} · best result: {_result_text(progress)} · "
            f"{progress.sessions} session(s), {progress.attempts} attempt(s)"
        )
        if progress.detail:
            st.caption(progress.detail)
        for note in progress.notes:
            st.caption(f"• {note}")
        _level_controls(library, controller, level)

        with st.expander("Level settings"):
            _edit_level_form(library, level)
        with st.expander("Validation"):
            _validation_panel(library, game, level)
        with st.expander("Diagnostics for this level"):
            suggestions = diagnose(library, level, progress)
            if suggestions:
                for suggestion in suggestions:
                    st.caption(f"• {suggestion}")
            else:
                st.caption("No problems detected from the stored records.")
        with st.expander("Session history"):
            _session_history(library, level)
        with st.expander("Danger zone"):
            _level_danger(library, level)

    st.divider()
    _add_level_form(library, game)


def _add_game_form(library: GameLibrary) -> None:
    st.markdown("### Add game")
    st.caption(
        "GameMind can only train games it has an adapter for. Adding an unsupported name does "
        "not create support."
    )
    options = {f"{item.label} — {item.adapter}": item.adapter for item in catalog()}
    choice = st.selectbox("Supported game type", list(options), key="add_game_adapter")
    adapter = options[choice]
    option = option_for(adapter)
    if option is not None:
        st.caption(option.summary)
        with st.expander("Requirements", expanded=not option.ready):
            for requirement in option.requirements:
                st.caption(f"• {requirement}")
        if not option.ready:
            st.warning(
                "This adapter needs a configuration written for the specific game. Paste one "
                "below; GameMind will not invent rewards or reset keys."
            )
    name = st.text_input("Display name", value=option.label if option else adapter, key="add_game_name")
    env_id = ""
    if option is not None and option.needs_env_id:
        env_id = st.text_input("Gymnasium environment id", placeholder="CartPole-v1", key="add_game_env")
    config_text = ""
    if option is None or not option.ready:
        config_text = st.text_area(
            "Game configuration (YAML)", height=220, key="add_game_yaml",
            placeholder="Paste a complete GameMind game YAML configuration.",
        )
    if st.button("Add game", type="primary", key="add_game_btn"):
        try:
            profile = library.create_game(
                name,
                adapter,
                description=option.summary if option else "",
                setup_notes="\n".join(option.requirements) if option else "",
                config_text=config_text or None,
                params={"env_id": env_id} if env_id else None,
            )
        except GameMindError as exc:
            st.error(f"Could not add the game: {exc}")
        else:
            st.success(f"Added {profile.name}. Open it to add levels.")
            st.session_state["game_id"] = profile.id
            st.rerun()
    st.caption(ADAPTER_HELP)


def _add_level_form(library: GameLibrary, game: Any) -> None:
    st.markdown("### Add level")
    st.caption(
        "Each level gets its own configuration, checkpoints, models, and evaluation history. "
        "The game itself still has to be brought to that level before training."
    )
    name = st.text_input("Level name", key=f"add_level_name_{game.id}")
    kind = st.selectbox(
        "Level type", ["official", "custom"], key=f"add_level_kind_{game.id}",
        format_func=lambda value: KIND_LABELS[value],
    )
    entry = st.text_input(
        "How GameMind reaches this level",
        value="manual",
        key=f"add_level_entry_{game.id}",
        help="GameMind does not navigate menus. 'manual' means you open the level in game.",
    )
    completion_mode = st.selectbox(
        "Completion detection",
        list(COMPLETION_MODES),
        key=f"add_level_completion_{game.id}",
        format_func=lambda value: COMPLETION_LABELS[value],
    )
    completion_pct = 99.0
    completion_frames = 2
    if completion_mode != "disabled":
        completion_pct = st.number_input(
            "Completion threshold (%)", min_value=50.0, max_value=100.0, value=99.0, step=0.5,
            key=f"add_level_pct_{game.id}",
        )
        completion_frames = st.number_input(
            "Consecutive frames at the threshold", min_value=1, max_value=20, value=2,
            key=f"add_level_frames_{game.id}",
        )
        st.caption(
            "A progress-bar win is an estimate. Confirm it once by watching a real completion "
            "before trusting the Completed status."
        )
    init_policy = st.selectbox(
        "Initialization",
        list(INIT_POLICIES),
        index=0,
        key=f"add_level_init_{game.id}",
        format_func=lambda value: INIT_LABELS[value],
    )
    notes = st.text_area("Notes", key=f"add_level_notes_{game.id}", height=70)
    if st.button("Add level", type="primary", key=f"add_level_btn_{game.id}"):
        try:
            profile = library.create_level(
                game.id,
                name,
                kind=kind,
                entry=entry,
                completion_mode=completion_mode,
                completion_pct=float(completion_pct),
                completion_frames=int(completion_frames),
                init_policy=init_policy,
                notes=notes,
            )
        except GameMindError as exc:
            st.error(f"Could not add the level: {exc}")
        else:
            st.success(f"Added level {profile.name}.")
            st.session_state["level_id"] = profile.id
            st.rerun()


def _edit_level_form(library: GameLibrary, level: Any) -> None:
    name = st.text_input("Name", value=level.name, key=f"edit_name_{level.id}")
    kind = st.selectbox(
        "Level type", list(LEVEL_KINDS),
        index=list(LEVEL_KINDS).index(level.kind),
        key=f"edit_kind_{level.id}",
        format_func=lambda value: KIND_LABELS[value],
    )
    entry = st.text_input("Entry method", value=level.entry, key=f"edit_entry_{level.id}")
    completion_mode = st.selectbox(
        "Completion detection",
        list(COMPLETION_MODES),
        index=list(COMPLETION_MODES).index(level.completion_mode),
        key=f"edit_completion_{level.id}",
        format_func=lambda value: COMPLETION_LABELS[value],
    )
    completion_pct = st.number_input(
        "Completion threshold (%)", min_value=50.0, max_value=100.0,
        value=float(level.completion_pct), step=0.5, key=f"edit_pct_{level.id}",
        disabled=completion_mode == "disabled",
    )
    completion_frames = st.number_input(
        "Consecutive frames", min_value=1, max_value=20, value=int(level.completion_frames),
        key=f"edit_frames_{level.id}", disabled=completion_mode == "disabled",
    )
    init_policy = st.selectbox(
        "Initialization",
        list(INIT_POLICIES),
        index=list(INIT_POLICIES).index(level.init_policy),
        key=f"edit_init_{level.id}",
        format_func=lambda value: INIT_LABELS[value],
    )
    notes = st.text_area("Notes", value=level.notes, key=f"edit_notes_{level.id}", height=80)
    columns = st.columns(3)
    if columns[0].button("Save", type="primary", key=f"edit_save_{level.id}"):
        try:
            library.update_level(
                level.game_id,
                level.id,
                name=name,
                kind=kind,
                entry=entry,
                completion_mode=completion_mode,
                completion_pct=float(completion_pct),
                completion_frames=int(completion_frames),
                init_policy=init_policy,
                notes=notes,
            )
        except GameMindError as exc:
            st.error(str(exc))
        else:
            st.success("Level settings saved.")
            st.rerun()
    duplicate_name = st.text_input("Duplicate as", value=f"{level.name} copy", key=f"dup_name_{level.id}")
    if columns[1].button("Duplicate", key=f"dup_{level.id}"):
        try:
            copy = library.duplicate_level(level.game_id, level.id, new_name=duplicate_name)
        except GameMindError as exc:
            st.error(str(exc))
        else:
            st.success(f"Created {copy.name}. History is not copied.")
            st.rerun()
    if columns[2].button(
        "Archive" if not level.archived else "Unarchive", key=f"arch_{level.id}"
    ):
        try:
            library.archive_level(level.game_id, level.id, archived=not level.archived)
        except GameMindError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _validation_panel(library: GameLibrary, game: Any, level: Any) -> None:
    existing = library.load_validation(level.game_id, level.id)
    st.caption(
        "A static check confirms the configuration. A live check connects to the game, resets "
        "the attempt, and reads one frame; it cannot prove death or completion detection."
    )
    columns = st.columns(2)
    if columns[0].button("Validate configuration", key=f"val_static_{level.id}"):
        report = validate_level(library, game, level, live=False)
        st.session_state[f"validation_{level.id}"] = report.to_dict()
    if columns[1].button("Validate with the game open", key=f"val_live_{level.id}"):
        with st.spinner("Connecting to the game and capturing one frame..."):
            report = validate_level(library, game, level, live=True)
        st.session_state[f"validation_{level.id}"] = report.to_dict()
    report = st.session_state.get(f"validation_{level.id}") or existing
    if not report:
        st.caption("No validation recorded yet.")
        return
    st.caption(f"Last validation: {str(report.get('created_at', ''))[:19].replace('T', ' ')}")
    for check in report.get("checks", []):
        marker = "PASS" if check.get("ok") else "FAIL"
        st.caption(f"{marker} — {check.get('name')}: {check.get('detail', '')}")
    observed = report.get("observed") or {}
    if observed.get("not_verified"):
        st.caption("Not verified by this check: " + ", ".join(observed["not_verified"]))


def _session_history(library: GameLibrary, level: Any) -> None:
    rows = session_rows(library, level)
    if not rows:
        st.caption("No sessions recorded yet.")
        return
    display = [
        {
            "started": str(row["started_at"])[:19].replace("T", " "),
            "state": row["state"],
            "reason": row["reason"],
            "timesteps": row["final_timesteps"],
            "progress": row["eval_mean_progress_pct"],
            "reward": row["eval_mean_reward"],
            "win": row["verified_completion"],
            "session": row["session_id"],
        }
        for row in rows
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(f"Session records: {library.sessions_dir(level.game_id, level.id)}")


def _level_danger(library: GameLibrary, level: Any) -> None:
    st.caption(
        "Removing the level deletes its settings and session history. Model and checkpoint files "
        "are intentionally left on disk."
    )
    confirm = st.checkbox("I understand what will be removed", key=f"rm_level_confirm_{level.id}")
    if st.button("Remove level profile", disabled=not confirm, key=f"rm_level_{level.id}"):
        try:
            summary = library.remove_level(level.game_id, level.id)
        except GameMindError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"Removed {level.name}. Sessions removed: {summary['sessions_removed']}. "
                f"Files kept in {summary['models_kept']}"
            )
            st.session_state.pop("level_id", None)
            st.rerun()


def page_training(library: GameLibrary) -> None:
    st.subheader("Training")
    controller, state, status, running = _active(library)
    _worker_banner(controller, state, running)

    if state is not None:
        st.caption(
            f"Worker: PID {state.pid} · {state.kind} · status {state.status} · "
            f"started {str(state.started_at)[:19].replace('T', ' ')}"
        )
        if state.level_id:
            try:
                game = library.get_game(state.game_id)
                level = library.get_level(state.game_id, state.level_id)
                st.caption(f"Game: {game.name} · Level: {level.name}")
            except GameMindError:
                st.caption(f"Game: {state.game_id} · Level: {state.level_id}")
    if status:
        _progress_bar(status)
        _status_metrics(status)
        extra = st.columns(3)
        extra[0].metric(
            "Recent reward",
            f"{float(status['ep_rew_mean']):.2f}"
            if isinstance(status.get("ep_rew_mean"), (int, float))
            else "—",
        )
        extra[1].metric(
            "Mean survival",
            f"{float(status['mean_survival_seconds']):.2f}s"
            if isinstance(status.get("mean_survival_seconds"), (int, float))
            else "—",
        )
        extra[2].metric("Steps/second", f"{float(status.get('fps', 0) or 0):.1f}")

    controls = st.columns(3)
    if controls[0].button("Pause", disabled=not running, width="stretch"):
        try:
            controller.pause()
        except WorkerError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if controls[1].button("Resume", disabled=not running, width="stretch"):
        try:
            controller.resume()
        except WorkerError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if controls[2].button("Stop", disabled=not running, width="stretch"):
        try:
            with st.spinner("Asking the worker to save a checkpoint and stop..."):
                controller.stop()
        except WorkerError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    st.caption(
        "Pause keeps the process alive; Stop saves a checkpoint first. Closing this page does "
        "not stop training."
    )

    st.divider()
    st.markdown("**Level queue**")
    game = _selected_game(library)
    if game is not None:
        levels = library.list_levels(game.id)
        labels = {
            f"{level.name} — {describe_level(library, level).label}": level.id for level in levels
        }
        chosen = st.multiselect("Levels in order", list(labels), key=f"queue_levels_{game.id}")
        columns = st.columns(3)
        timesteps = columns[0].number_input(
            "Timesteps per level", min_value=32, max_value=5_000_000,
            value=int(library.settings()["default_timesteps"]), step=1000, key="queue_ts",
        )
        eval_episodes = columns[1].number_input(
            "Eval attempts per level", min_value=1, max_value=20,
            value=int(library.settings()["default_eval_episodes"]), key="queue_eval",
        )
        continue_completed = columns[2].checkbox(
            "Re-train completed levels", value=False, key="queue_continue",
            help="Off: levels with a verified completion are skipped.",
        )
        if st.button("Start queue", type="primary", disabled=running, key="queue_start"):
            try:
                queue = create_queue(
                    library,
                    game.id,
                    [labels[label] for label in chosen],
                    timesteps_per_level=int(timesteps),
                    eval_episodes=int(eval_episodes),
                    continue_after_completion=bool(continue_completed),
                )
                controller.start_queue(queue)
            except GameMindError as exc:
                st.error(str(exc))
            else:
                st.rerun()

        queues = list_queues(library, game.id)
        if queues:
            st.markdown("**Queues**")
            for queue in queues[:6]:
                summary = queue_summary(library, queue)
                with st.container(border=True):
                    st.markdown(
                        f"`{queue.queue_id}` · **{queue.status}** · "
                        f"{summary['remaining']} of {len(queue.tasks)} remaining"
                    )
                    st.dataframe(summary["tasks"], hide_index=True, width="stretch")
                    columns = st.columns(3)
                    if columns[0].button(
                        "Resume queue",
                        key=f"queue_resume_{queue.queue_id}",
                        disabled=running or queue.status == "completed",
                    ):
                        try:
                            recover_queue(library, queue.queue_id)
                            controller.start_queue(load_queue(library, queue.queue_id))
                        except GameMindError as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()
                    if columns[1].button(
                        "Stop queue",
                        key=f"queue_stop_{queue.queue_id}",
                        disabled=not running,
                    ):
                        try:
                            controller.stop()
                        except WorkerError as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()

    st.divider()
    with st.expander("Worker log", expanded=False):
        log_path = Path(state.log_path) if state is not None and state.log_path else None
        if log_path and log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            except OSError as exc:
                text = f"Could not read the log: {exc}"
            st.code(text or "No output yet.", language="text")
        else:
            st.caption("No worker log yet.")
    if state is not None:
        st.caption(f"Worker state: {controller.state_path}")
        st.caption(f"Job file: {state.job_path}")
    if st.button("Refresh status"):
        st.rerun()


def page_models(library: GameLibrary) -> None:
    st.subheader("Models and checkpoints")
    game = _selected_game(library)
    if game is None:
        return
    level = _selected_level(library, game)
    if level is None:
        return
    progress = describe_level(library, level)
    st.caption(
        f"{_status_line(progress)} · best result: {_result_text(progress)} · "
        f"best model: {progress.best_model_path or '—'}"
    )
    inventory = level_model_inventory(library, level)
    if inventory:
        st.dataframe(
            [
                {
                    "model": item["name"],
                    "timesteps": item["num_timesteps"],
                    "created": str(item["created_at"] or "")[:19].replace("T", " "),
                    "compatible": item["compatible"],
                    "path": item["path"],
                }
                for item in inventory
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No models for this level yet. Train once to create one.")

    controller, state, status, running = _active(library)
    columns = st.columns([2, 1, 1])
    model_choice = columns[0].selectbox(
        "Model to evaluate", [item["name"] for item in inventory] or ["—"],
        key=f"model_select_{level.id}",
    )
    episodes = columns[1].number_input(
        "Attempts", min_value=1, max_value=50, value=5, key=f"model_episodes_{level.id}"
    )
    if columns[2].button("Evaluate", disabled=running or not inventory, key=f"model_eval_{level.id}"):
        selected = next((item for item in inventory if item["name"] == model_choice), None)
        if selected is not None:
            try:
                controller.start_evaluation(
                    level,
                    model_path=selected["path"],
                    episodes=int(episodes),
                    device=str(library.settings()["device"]),
                )
            except GameMindError as exc:
                st.error(str(exc))
            else:
                st.success("Evaluation started in the background. Refresh to see the result.")
    st.caption(
        "Evaluation plays the real game deterministically without training. Progress percentages "
        "are comparable only within this level."
    )

    evaluations = library.list_evaluations(level.game_id, level.id)
    if evaluations:
        st.markdown("**Evaluation history**")
        st.dataframe(
            [
                {
                    "evaluation": item.get("evaluation_id"),
                    "when": str(item.get("finished_at") or item.get("started_at") or "")[:19].replace("T", " "),
                    "attempts": len(item.get("episodes") or []),
                    "mean progress %": item.get("mean_progress_pct"),
                    "mean reward": item.get("mean_reward"),
                    "completed": any(
                        bool(episode.get("completed")) for episode in item.get("episodes") or []
                    ),
                }
                for item in reversed(evaluations)
            ],
            hide_index=True,
            width="stretch",
        )

    checkpoints_root = library.level_checkpoints_dir(level.game_id, level.id)
    with st.expander("Checkpoints on disk"):
        checkpoints = sorted(checkpoints_root.rglob("checkpoint_*.zip"))[-20:]
        if checkpoints:
            for path in checkpoints:
                st.caption(str(path))
        else:
            st.caption(f"No periodic checkpoints yet in {checkpoints_root}")

    with st.expander("Other checkpoints in this project (preserved, unassigned)"):
        legacy = library.legacy_artifacts(game)
        if legacy:
            st.caption(
                "These files predate the library and were not attached to a level, because their "
                "level was not recorded. They are listed for reference only."
            )
            st.dataframe(legacy, hide_index=True, width="stretch")
        else:
            st.caption("None found.")


def page_knowledge(library: GameLibrary) -> None:
    st.subheader("Knowledge and skills")
    st.caption(
        "Skills are notes with evidence and conditions for one game. They are not weights and "
        "never transfer a progress percentage from one level to another."
    )
    game = _selected_game(library)
    if game is None:
        return
    skills = library.list_skills(game.id)
    if skills:
        st.dataframe(
            [
                {
                    "name": skill.name,
                    "kind": skill.kind,
                    "validated": skill.validated,
                    "confidence": round(skill.confidence, 2),
                    "evidence": len(skill.evidence),
                    "summary": skill.summary,
                }
                for skill in skills
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No skills recorded for this game yet.")

    level = _selected_level(library, game)
    if level is not None:
        applicable, rejected = skills_for_level(library, level)
        st.markdown(f"**Applicable to {level.name}**")
        if applicable:
            for skill in applicable:
                st.caption(
                    f"• {skill.name} ({'validated' if skill.validated else 'unvalidated'}, "
                    f"confidence {skill.confidence:.2f})"
                )
        else:
            st.caption("No validated skill matches this level's observation/action settings.")
        if rejected:
            with st.expander("Skills that do not apply here"):
                for skill, reasons in rejected:
                    st.caption(f"• {skill.name}: {'; '.join(reasons)}")

    st.divider()
    with st.expander("Add a skill", expanded=False):
        name = st.text_input("Skill name", key="skill_name")
        kind = st.selectbox(
            "Kind", ["control", "observation", "reset", "temporal", "strategy"], key="skill_kind"
        )
        summary = st.text_area("What was observed, and under what conditions", key="skill_summary")
        validated = st.checkbox(
            "Validated (evidence from more than one situation)",
            value=False,
            key="skill_validated",
        )
        if st.button("Save skill", key="skill_save"):
            try:
                from gamemind.library.skills import conditions_for

                config = library.level_config(game.id, level.id) if level else None
                skill = add_skill(
                    library,
                    game.id,
                    name=name,
                    kind=kind,
                    summary=summary,
                    conditions=conditions_for(config) if config else {},
                    validated=validated,
                    confidence=0.8 if validated else 0.2,
                )
            except GameMindError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved skill {skill.name}.")
                st.rerun()

    if skills:
        st.divider()
        with st.expander("Update a skill"):
            labels = {f"{skill.name} ({skill.kind})": skill.skill_id for skill in skills}
            choice = st.selectbox("Skill", list(labels), key="skill_update_select")
            skill = library.get_skill(labels[choice])
            validated = st.checkbox("Validated", value=skill.validated, key="skill_update_validated")
            confidence = st.slider(
                "Confidence", 0.0, 1.0, value=float(skill.confidence), step=0.05,
                key="skill_update_confidence",
            )
            sessions = library.list_sessions(level.game_id, level.id) if level else []
            if sessions and st.button("Attach the latest session as evidence", key="skill_evidence"):
                from gamemind.library.skills import add_evidence, session_evidence

                latest = sessions[-1]
                add_evidence(library, skill.skill_id, session_evidence(level, latest))
                st.success("Evidence attached.")
                st.rerun()
            if st.button("Save skill changes", key="skill_update_save"):
                update_skill(
                    library,
                    skill.skill_id,
                    validated=validated,
                    confidence=confidence,
                )
                st.success("Skill updated.")
                st.rerun()
            if skill.evidence:
                st.dataframe(skill.evidence, hide_index=True, width="stretch")


def page_settings(library: GameLibrary) -> None:
    st.subheader("Settings")
    settings = library.settings()
    columns = st.columns(3)
    timesteps = columns[0].number_input(
        "Default timesteps per session", min_value=32, max_value=5_000_000,
        value=int(settings["default_timesteps"]), step=1000,
    )
    eval_episodes = columns[1].number_input(
        "Default evaluation attempts", min_value=1, max_value=50,
        value=int(settings["default_eval_episodes"]),
    )
    device = columns[2].selectbox(
        "Device", ["auto", "cpu", "cuda", "mps"],
        index=["auto", "cpu", "cuda", "mps"].index(str(settings["device"])),
    )
    evaluate_after = st.checkbox(
        "Evaluate after each session", value=bool(settings["evaluate_after_training"])
    )
    if st.button("Save settings", type="primary"):
        library.update_settings(
            default_timesteps=int(timesteps),
            default_eval_episodes=int(eval_episodes),
            device=device,
            evaluate_after_training=bool(evaluate_after),
        )
        st.success("Settings saved.")
        st.rerun()
    st.caption(
        "These are the routine controls. PPO hyperparameters live in each game's YAML and stay "
        "out of the way unless you open Advanced settings below."
    )

    game = _selected_game(library)
    if game is not None:
        with st.expander("Advanced: game configuration (YAML)"):
            path = library.game_config_path(game.id)
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            edited = st.text_area("game.yaml", value=text, height=320, key=f"game_yaml_{game.id}")
            if st.button("Save game configuration", key=f"game_yaml_save_{game.id}"):
                try:
                    library.save_game_config_text(game.id, edited)
                except GameMindError as exc:
                    st.error(str(exc))
                else:
                    st.success("Saved. New levels will use these defaults.")
                    st.rerun()
            st.caption(
                "Editing this changes the defaults used when a level is created. Existing levels "
                "keep their own configuration unchanged."
            )
        level = _selected_level(library, game)
        if level is not None:
            with st.expander("Advanced: level configuration (YAML)"):
                level_path = library.level_config_path(game.id, level.id)
                text = level_path.read_text(encoding="utf-8") if level_path.exists() else ""
                edited = st.text_area(
                    "config.yaml", value=text, height=360, key=f"level_yaml_{level.id}"
                )
                if st.button("Save level configuration", key=f"level_yaml_save_{level.id}"):
                    try:
                        library.save_level_config_text(game.id, level.id, edited)
                    except GameMindError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Saved. Run Validate before training.")
                        st.rerun()
                st.caption(f"Level environment name: {level_config_name(game.id, level.id)}")

    with st.expander("Diagnostics"):
        st.caption(f"Library root: {library.root}")
        st.caption(f"Games: {library.games_dir}")
        st.caption(f"Knowledge: {library.knowledge_dir}")
        st.caption(f"Worker state: {library.state_dir}")
        st.caption(f"Project models: {DEFAULT_MODEL_DIR}")
        st.caption(f"Project checkpoints: {DEFAULT_CHECKPOINT_DIR}")
        for candidate_game in library.list_games():
            usable, why = readiness(candidate_game.adapter)
            st.caption(
                f"{candidate_game.name}: adapter {candidate_game.adapter} "
                + ("available" if usable else f"unavailable — {why}")
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="GameMind AI", page_icon="🎮", layout="wide")

    library = _library()
    if st.session_state.get("nav") not in PAGES:
        st.session_state["nav"] = "Home"

    with st.sidebar:
        st.markdown("## GameMind AI")
        st.caption("Train one game at a time. Each game keeps its own agent.")
        page = st.radio(
            "Section",
            PAGES,
            key="nav",
            label_visibility="collapsed",
        )
        st.divider()
        controller, state, status, running = _active(library)
        if running and state is not None:
            st.success(f"Training ({state.kind})")
            st.caption(
                f"{state.game_id or '—'} / {state.level_id or '—'}\n\n"
                f"{int(status.get('timesteps', 0)):,} timesteps"
            )
        elif state is not None:
            st.caption(f"Last worker: {state.status}")
        else:
            st.caption("No training running.")

    try:
        if page == "Home":
            page_home(library)
        elif page == "Games":
            page_games(library)
        elif page == "Training":
            page_training(library)
        elif page == "Models":
            page_models(library)
        elif page == "Knowledge":
            page_knowledge(library)
        else:
            page_settings(library)
    except GameMindError as exc:
        st.error(str(exc))
        with st.expander("Details"):
            st.exception(exc)

    st.divider()
    st.caption(
        "Every session saves a model, a session record, and (when enabled) an evaluation to the "
        "library. Estimated progress is never treated as proof that a level was beaten."
    )


if __name__ == "__main__":
    main()
