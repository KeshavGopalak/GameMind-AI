"""Minimal local Streamlit dashboard for GameMind AI.

Run from the project root:

    streamlit run dashboard/app.py

Training runs as a separate Python process. That keeps the Streamlit rerun model
from interrupting a long PPO run, and lets the STOP/PAUSE files work exactly as
they do from the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import yaml

from gamemind.algorithms.factory import maskable_ppo_available
from gamemind.config import GameConfig, load_game_config
from gamemind.training.model_io import metadata_path_for
from gamemind.utils.serialization import atomic_write_json, read_json, read_jsonl
from gamemind.utils.paths import DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, safe_name

STATE_PATH = DEFAULT_LOG_DIR / "dashboard_state.json"
DASHBOARD_LOG_DIR = DEFAULT_LOG_DIR / "dashboard"


def _load_active_state() -> dict[str, Any]:
    value = read_json(STATE_PATH, default={})
    return value if isinstance(value, dict) else {}


def _save_active_state(payload: dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, payload)


def _pid_alive(pid: int | None) -> bool:
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
            # Access denied usually means the process exists but belongs to a higher integrity level.
            return ctypes.get_last_error() == 5
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _tail(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _read_status(status_path: Path | None) -> dict[str, Any]:
    if status_path is None:
        return {}
    value = read_json(status_path, default={})
    return value if isinstance(value, dict) else {}


def _config_files() -> list[Path]:
    return sorted((ROOT / "configs").glob("*.yaml"))


def _model_files() -> list[Path]:
    paths = list(DEFAULT_MODEL_DIR.glob("*.zip"))
    paths.extend(DEFAULT_MODEL_DIR.rglob("*.zip"))
    # Preserve a stable, de-duplicated order; cap the selector for responsiveness.
    unique = sorted({path.resolve() for path in paths})
    return unique[:300]


def _model_label(path: Path) -> str:
    meta = read_json(metadata_path_for(path), default={})
    if isinstance(meta, dict) and meta:
        return (
            f"{path.relative_to(ROOT)} — {meta.get('game', '?')} / "
            f"{meta.get('algorithm', '?')} / {meta.get('num_timesteps', '?')} steps"
        )
    return f"{path.relative_to(ROOT)} — no metadata"


def _write_geometry_config(
    config: GameConfig,
    run_name: str,
    *,
    jump_key: str | None,
    jump_tap: float | None,
    jump_hold: float | None,
) -> Path:
    """Persist the dashboard's Geometry Dash control edits for the subprocess."""
    if config.adapter != "geometry_dash":
        raise ValueError("Geometry Dash overrides were requested for another adapter.")
    if jump_key is None or not jump_key.strip():
        raise ValueError("Jump key must not be empty.")
    if jump_tap is None or jump_hold is None or jump_tap < 0.0 or jump_hold < 0.0:
        raise ValueError("Jump durations must be zero or positive.")
    bindings = []
    for binding in config.actions.bindings:
        if binding.name == "jump":
            binding = replace(binding, keys=(jump_key.strip().lower(),), duration=float(jump_tap))
        elif binding.name == "hold_jump":
            binding = replace(binding, keys=(jump_key.strip().lower(),), duration=float(jump_hold))
        bindings.append(binding)
    if not any(binding.name in {"jump", "hold_jump"} for binding in bindings):
        raise ValueError("Geometry Dash config has no jump or hold_jump action binding.")
    edited = replace(
        config,
        actions=replace(config.actions, bindings=tuple(bindings)),
    )
    directory = DASHBOARD_LOG_DIR / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(edited.as_dict(), sort_keys=False), encoding="utf-8")
    return path


def _launch_training(
    config_path: Path,
    *,
    timesteps: int,
    algorithm: str,
    observation_mode: str,
    width: int,
    height: int,
    stack: int,
    frame_skip: int,
    action_duration: float | None,
    n_envs: int,
    seed: int,
    eval_freq: int,
    normalize: bool,
    resume_from: str | None,
    no_tensorboard: bool,
    jump_key: str | None = None,
    jump_tap: float | None = None,
    jump_hold: float | None = None,
) -> tuple[dict[str, Any], str]:
    config = load_game_config(config_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"dashboard_{stamp}_{safe_name(config.name)}"
    run_dir = DEFAULT_LOG_DIR / "runs" / run_name
    dashboard_log = DASHBOARD_LOG_DIR / f"{run_name}.log"
    DASHBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_MODEL_DIR / "dashboard" / f"{run_name}.zip"
    effective_config = config_path
    if config.adapter == "geometry_dash":
        effective_config = _write_geometry_config(
            config,
            run_name,
            jump_key=jump_key,
            jump_tap=jump_tap,
            jump_hold=jump_hold,
        )
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(effective_config),
        "--timesteps",
        str(timesteps),
        "--algorithm",
        algorithm,
        "--observation-mode",
        observation_mode,
        "--width",
        str(width),
        "--height",
        str(height),
        "--frame-stack",
        str(stack),
        "--frame-skip",
        str(frame_skip),
        "--n-envs",
        str(n_envs),
        "--seed",
        str(seed),
        "--eval-freq",
        str(eval_freq),
        "--run-name",
        run_name,
        "--output",
        str(output),
    ]
    if action_duration is not None:
        cmd.extend(["--action-duration", str(action_duration)])
    if normalize:
        cmd.append("--normalize")
    if resume_from:
        cmd.extend(["--model", resume_from])
    if no_tensorboard:
        cmd.append("--no-tensorboard")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with dashboard_log.open("ab") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    payload = {
        "run_name": run_name,
        "pid": process.pid,
        "run_dir": str(run_dir),
        "status_path": str(run_dir / "status.json"),
        "episodes_path": str(run_dir / "episodes.jsonl"),
        "dashboard_log": str(dashboard_log),
        "output": str(output),
        "config_path": str(config_path),
        "effective_config_path": str(effective_config),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(cmd),
    }
    _save_active_state(payload)
    return payload, " ".join(cmd)


def _launch_play(model_path: str, episodes: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "play.py"),
        "--model",
        model_path,
        "--episodes",
        str(episodes),
    ]
    subprocess.Popen(cmd, cwd=str(ROOT))
    return cmd


def _run_evaluation(model_path: str, episodes: int) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "evaluate.py"),
        "--model",
        model_path,
        "--episodes",
        str(episodes),
    ]
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _write_control(run_dir: Path, filename: str, present: bool) -> None:
    path = run_dir / filename
    if present:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("requested by dashboard\n", encoding="utf-8")
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    st.set_page_config(page_title="GameMind AI", page_icon="🎮", layout="wide")
    st.title("GameMind AI")
    st.caption(
        "Train, pause, resume, evaluate, and watch reinforcement-learning agents. "
        "Start with the built-in Target Chase game if you are new."
    )

    configs = _config_files()
    if not configs:
        st.error("No configuration files found in configs/.")
        return
    config_names = [path.name for path in configs]
    default_index = config_names.index("example_game.yaml") if "example_game.yaml" in config_names else 0

    active = _load_active_state()
    active_status_path = Path(active["status_path"]) if active.get("status_path") else None
    status = _read_status(active_status_path)
    active_alive = _pid_alive(int(active["pid"])) if active.get("pid") else False
    status_state = str(status.get("state", ""))
    running = active_alive and status_state not in {"completed", "error", "stopped", "interrupted"}

    with st.sidebar:
        st.header("Experiment")
        selected_name = st.selectbox("Game configuration", config_names, index=default_index)
        config_path = ROOT / "configs" / selected_name
        try:
            config = load_game_config(config_path)
        except Exception as exc:
            st.error(f"Could not load config: {exc}")
            return
        st.caption(f"Adapter: `{config.adapter}` · Game: `{config.name}`")
        is_desktop_control = config.adapter in {"desktop", "geometry_dash"}
        if config.adapter == "desktop":
            st.warning(
                "This is the desktop-game template. It will not send input until you edit "
                "the window title and reward/termination settings in the YAML file."
            )
        elif config.adapter == "geometry_dash":
            st.warning(
                "Geometry Dash is controlled through the real desktop window. Open the game, "
                "select a level, enable Auto-Retry, then start training. Keep the game focused "
                "and do not let other windows cover it. Ctrl+C or the mouse-corner failsafe stops it."
            )
        jump_binding = next((b for b in config.actions.bindings if b.name == "jump"), None)
        hold_binding = next((b for b in config.actions.bindings if b.name == "hold_jump"), None)
        jump_key_default = (
            jump_binding.keys[0]
            if jump_binding is not None and jump_binding.keys
            else (hold_binding.keys[0] if hold_binding is not None and hold_binding.keys else "space")
        )
        jump_tap_default = (
            float(jump_binding.duration)
            if jump_binding is not None and jump_binding.duration is not None
            else float(config.actions.duration)
        )
        jump_hold_default = (
            float(hold_binding.duration)
            if hold_binding is not None and hold_binding.duration is not None
            else max(jump_tap_default, 0.12)
        )
        algorithms = ["ppo", "a2c"] + (["maskable_ppo"] if maskable_ppo_available() else [])
        default_algorithm = config.training.algorithm if config.training.algorithm in algorithms else "ppo"
        algorithm = st.selectbox(
            "Algorithm",
            algorithms,
            index=algorithms.index(default_algorithm),
            help="PPO is the recommended default. A2C is a lighter alternative. "
            "MaskablePPO appears only when sb3-contrib is installed.",
        )
        timesteps = st.number_input(
            "Additional timesteps",
            min_value=16,
            max_value=5_000_000,
            value=int(config.training.timesteps),
            step=1000,
        )
        observation_mode = st.selectbox(
            "Observation mode",
            ["state", "grayscale", "rgb"],
            index=["state", "grayscale", "rgb"].index(config.observation.mode),
            help="Pixels use CnnPolicy; state observations use MlpPolicy.",
        )
        if is_desktop_control and observation_mode == "state":
            st.error(
                "Desktop-controlled games must learn from pixels. Choose grayscale or rgb; "
                "the adapter will not substitute hidden game state."
            )
            return
        resolution_default = f"{config.observation.width}x{config.observation.height}"
        resolution = st.text_input("Pixel resolution (WIDTHxHEIGHT)", value=resolution_default)
        try:
            width_text, height_text = resolution.lower().split("x", 1)
            width, height = int(width_text.strip()), int(height_text.strip())
        except ValueError:
            st.error("Resolution must look like 84x84.")
            return
        max_stack = 1 if observation_mode == "state" else 12
        stack = st.number_input(
            "Frame stack",
            min_value=1,
            max_value=int(max_stack),
            value=1 if observation_mode == "state" else int(config.observation.stack),
        )
        frame_skip = st.number_input(
            "Frame skip / action repeat",
            min_value=1,
            max_value=30,
            value=int(config.actions.frame_skip),
        )
        action_duration = st.number_input(
            "Action duration (seconds, desktop)",
            min_value=0.0,
            max_value=float(config.actions.max_duration),
            value=float(config.actions.duration),
            step=0.01,
            format="%.2f",
            disabled=config.adapter == "geometry_dash",
            help=(
                "Geometry Dash uses the per-action jump/hold durations below."
                if config.adapter == "geometry_dash"
                else "Fallback hold duration for desktop action bindings without their own duration."
            ),
        )
        jump_key: str | None = None
        jump_tap: float | None = None
        jump_hold: float | None = None
        if config.adapter == "geometry_dash":
            with st.expander("Geometry Dash jump controls", expanded=True):
                jump_key = st.text_input("Jump key", value=jump_key_default)
                jump_tap = st.number_input(
                    "Short jump press (seconds)",
                    min_value=0.0,
                    max_value=float(config.actions.max_duration),
                    value=min(jump_tap_default, float(config.actions.max_duration)),
                    step=0.01,
                    format="%.2f",
                )
                jump_hold = st.number_input(
                    "Hold jump (seconds)",
                    min_value=0.0,
                    max_value=float(config.actions.max_duration),
                    value=min(jump_hold_default, float(config.actions.max_duration)),
                    step=0.01,
                    format="%.2f",
                )
                st.caption(
                    "These edit the jump/hold_jump bindings for this dashboard run only. "
                    "The release_jump action is always available and every action releases keys."
                )
        n_envs = st.number_input(
            "Parallel environments",
            min_value=1,
            max_value=1 if is_desktop_control else 32,
            value=min(int(config.training.n_envs), 1 if is_desktop_control else 32),
            disabled=is_desktop_control,
        )
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=int(config.training.seed))
        eval_freq = st.number_input(
            "Evaluation frequency (env steps, 0 = off)",
            min_value=0,
            max_value=100_000,
            value=0 if is_desktop_control else int(config.training.eval_freq),
            step=1000,
            disabled=is_desktop_control,
            help=(
                "Periodic evaluation is disabled for desktop games because a second "
                "environment would control the same window."
                if is_desktop_control
                else None
            ),
        )
        normalize = st.checkbox("Normalize pixels to float32 [0,1]", value=bool(config.observation.normalize))
        no_tensorboard = st.checkbox("Disable TensorBoard logging", value=False)

        checkpoints = _model_files()
        resume_options = ["(start from scratch)"] + [str(path) for path in checkpoints]
        resume_choice = st.selectbox("Continue from checkpoint", resume_options, index=0)
        resume_from = None if resume_choice.startswith("(") else resume_choice

        if st.button("Start training", type="primary", use_container_width=True, disabled=running):
            if running:
                st.error("A training run is already active. Stop or wait for it first.")
            else:
                try:
                    payload, command = _launch_training(
                        config_path,
                        timesteps=int(timesteps),
                        algorithm=algorithm,
                        observation_mode=observation_mode,
                        width=width,
                        height=height,
                        stack=int(stack),
                        frame_skip=int(frame_skip),
                        action_duration=float(action_duration) if config.adapter == "desktop" else None,
                        n_envs=int(n_envs),
                        seed=int(seed),
                        eval_freq=int(eval_freq),
                        normalize=bool(normalize),
                        resume_from=resume_from,
                        no_tensorboard=bool(no_tensorboard),
                        jump_key=jump_key,
                        jump_tap=jump_tap,
                        jump_hold=jump_hold,
                    )
                    st.success(f"Launched training run {payload['run_name']} (PID {payload['pid']}).")
                    st.code(command)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not start training: {exc}")

        st.divider()
        st.subheader("Active run")
        if not active:
            st.caption("No run has been launched from this dashboard yet.")
        else:
            st.caption(f"`{active.get('run_name', '?')}` · PID {active.get('pid', '?')}")
            if not active_alive and not status_state:
                st.info("The process is no longer running and did not create a status file. See logs below.")
            if status:
                state = status.get("state", "unknown")
                st.metric("State", str(state).upper())
                progress = status.get("progress")
                if isinstance(progress, (int, float)):
                    st.progress(min(1.0, max(0.0, float(progress))))
                col1, col2 = st.columns(2)
                col1.metric("Timesteps", f"{int(status.get('timesteps', 0)):,}")
                col2.metric("Episodes", int(status.get("episodes", 0) or 0))
                col1.metric(
                    "Reward (recent)",
                    f"{float(status['ep_rew_mean']):.2f}"
                    if status.get("ep_rew_mean") is not None
                    else "—",
                )
                col2.metric(
                    "Length (recent)",
                    f"{float(status['ep_len_mean']):.1f}"
                    if status.get("ep_len_mean") is not None
                    else "—",
                )
                if status.get("progress_pct") is not None:
                    col1.metric("Level progress (latest)", f"{float(status['progress_pct']):.2f}%")
                if status.get("survival_seconds") is not None:
                    col2.metric(
                        "Survival (latest)", f"{float(status['survival_seconds']):.2f}s"
                    )
            if active.get("run_dir"):
                run_dir = Path(active["run_dir"])
                control_cols = st.columns(3)
                if control_cols[0].button("Pause", use_container_width=True, disabled=not running):
                    _write_control(run_dir, "PAUSE", True)
                    st.rerun()
                if control_cols[1].button("Resume", use_container_width=True, disabled=not running):
                    _write_control(run_dir, "PAUSE", False)
                    st.rerun()
                if control_cols[2].button("Stop", use_container_width=True, disabled=not running):
                    _write_control(run_dir, "STOP", True)
                    st.rerun()
            st.caption("Refresh the status and charts with the button below.")
            if st.button("Refresh status", use_container_width=True):
                st.rerun()

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------
    if active:
        run_dir = Path(active.get("run_dir", ""))
        episodes_path = Path(active.get("episodes_path", ""))
        dashboard_log = Path(active.get("dashboard_log", ""))
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Training reward and episode length")
            episodes = read_jsonl(episodes_path)
            if episodes:
                frame = pd.DataFrame(episodes)
                if {"timesteps", "reward"}.issubset(frame.columns):
                    st.caption("Episode reward (higher is better; compare within a game, not across games).")
                    st.line_chart(frame, x="timesteps", y="reward")
                if {"timesteps", "length"}.issubset(frame.columns):
                    st.caption("Episode length (early termination can make this shorter).")
                    st.line_chart(frame, x="timesteps", y="length")
                if {"timesteps", "max_progress_pct"}.issubset(frame.columns):
                    st.caption(
                        "Maximum visual level progress per episode (approximate bar reading)."
                    )
                    st.line_chart(frame, x="timesteps", y="max_progress_pct")
                if {"timesteps", "survival_seconds"}.issubset(frame.columns):
                    st.caption("Survival time per episode (seconds).")
                    st.line_chart(frame, x="timesteps", y="survival_seconds")
            else:
                st.info("No completed episodes have been recorded yet. Refresh after training warms up.")
        with right:
            st.subheader("Run files")
            st.code(
                "\n".join(
                    [
                        f"run: {run_dir}",
                        f"model: {active.get('output', '—')}",
                        f"status: {active.get('status_path', '—')}",
                        f"log: {dashboard_log}",
                    ]
                )
            )
            if status:
                with st.expander("Latest status JSON", expanded=False):
                    st.json(status)
            st.caption("TensorBoard:")
            st.code(f"tensorboard --logdir {run_dir / 'tensorboard'}")
        logs = _tail(dashboard_log)
        with st.expander("Training log", expanded=False):
            st.code(logs or "No log output yet.", language="text")
    else:
        st.info(
            "Choose a configuration in the sidebar and press Start training. "
            "The first recommended run is `example_game` with 50,000 timesteps."
        )

    st.divider()
    st.subheader("Trained models")
    st.caption(
        "Training saves a .zip model and a .metadata.json sidecar. The metadata is what lets "
        "play.py recreate the exact observation and action settings."
    )
    models = _model_files()
    if not models:
        st.info("No models yet. Train one from the sidebar or with scripts/train.py.")
    else:
        labels = {str(path): _model_label(path) for path in models}
        selected_model = st.selectbox("Model", list(labels), format_func=lambda value: labels[value])
        play_episodes = st.number_input("Playback episodes", min_value=1, max_value=50, value=3)
        col_play, col_eval = st.columns(2)
        if col_play.button("Open live playback window", use_container_width=True):
            try:
                command = _launch_play(selected_model, int(play_episodes))
                st.success("Playback launched in a separate window. Close the game window to stop.")
                st.code(" ".join(command))
            except Exception as exc:
                st.error(f"Could not launch playback: {exc}")
        if col_eval.button("Evaluate headlessly", use_container_width=True):
            try:
                with st.spinner("Running evaluation episodes..."):
                    completed = _run_evaluation(selected_model, int(play_episodes))
                if completed.returncode == 0:
                    st.success("Evaluation finished.")
                    st.code(completed.stdout or "(no stdout)")
                else:
                    st.error(f"Evaluation exited with code {completed.returncode}.")
                    st.code((completed.stdout or "") + "\n" + (completed.stderr or ""))
            except Exception as exc:
                st.error(f"Could not evaluate: {exc}")

    with st.expander("Selected config preview", expanded=False):
        st.code(config_path.read_text(encoding="utf-8"), language="yaml")
    st.caption(
        "This dashboard is intentionally small. The CLI is the primary, fully documented interface; "
        "the dashboard reads the same status/metrics files the training engine writes."
    )


if __name__ == "__main__":
    main()
