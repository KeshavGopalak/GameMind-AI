# Troubleshooting and verification notes

## Actual checks performed in this checkout

Environment: **Windows 11 (build 26200)**, **Python 3.13.14**, CPU PyTorch **2.15.0+cpu**,
Gymnasium **1.4.0**, Stable-Baselines3 **2.9.2**, OpenCV **5.0.0**, pygame **2.6.1**.

- `python scripts/test_setup.py`: all checks passed (see the final project report for the count at
  release time).
- `python -m pytest -q`: the complete local test suite passed. The suite includes a real PPO training
  run, checkpoint metadata, resume, headless evaluation, adapter API checks, reward/observation tests,
  desktop adapter tests with fake capture/input backends, and template detection.
- A 50,176-step fixed-layout Target Chase run (`demo_fixed_50k`) completed. Deterministic
  evaluation over 20 episodes reached the target in 20/20 episodes, with mean reward 13.94 and mean
  length 6.0. This is a fixed-layout training puzzle, not a generalization claim. Training is
  stochastic: your numbers will differ.
- A 50,176-step randomized Target Chase experiment and a 51,200-step CartPole run were also executed;
  CartPole reached the 500-step time limit, while the randomized pixel run remained weak under
  deterministic evaluation. The CLI and dashboard paths were exercised in both cases.
- The automated test environment uses `SDL_VIDEODRIVER=dummy`. A real interactive window was not
  visually verified by an automated headless test; the Pygame and captured-frame paths are exercised
  without a display. You must verify a live window on your desktop.

Not verified in this checkout:

- Linux and macOS live displays, desktop permissions, and PyAutoGUI behavior.
- A real third-party desktop game (the desktop adapter is tested with fake capture/input backends and
  template frames, not by controlling an actual game).
- CUDA/MPS performance (CPU was used).
- MaskablePPO (`sb3-contrib` is optional and was not installed for the test run).
- Box2D/LunarLander training quality beyond environment creation; LunarLander is supported but
  50k steps is not a mastery claim.

## Installation and import problems

### `ModuleNotFoundError: No module named 'gamemind'`

Run scripts from the project root. The bundled scripts add the project root to `sys.path`; running
them from another directory can still work, but Python must find the package.

```bat
cd "path\to\GameMind-AI"
.venv\Scripts\python.exe scripts\train.py --game example_game --timesteps 1000
```

Make sure the venv is activated or call `.venv\Scripts\python.exe` explicitly.

### `Box2D` fails for LunarLander

```bash
pip install swig
pip install "gymnasium[box2d]"
```

CartPole and the built-in game do not require Box2D. On platforms without a wheel you may need C++
build tools.

### Pygame opens no window / `No available video device`

- Training the example game does not need a window; only `play.py` with rendering does.
- On Linux, use an X11 session. Wayland screen capture/input may not work.
- On a headless server, run `evaluate.py` (headless by default) or `play.py --no-render`, and do not attempt desktop control.
- The tests set `SDL_VIDEODRIVER=dummy` deliberately.

### PyAutoGUI errors or the desktop adapter refuses to start

- The desktop adapter is optional. Gymnasium and the example game work without it.
- macOS: grant Accessibility and Screen Recording permissions.
- Linux: X11 and the right input permissions are required; Wayland is not reliable.
- `require_window_focus: true` needs an exact substring of the game window title. If no window is
  found, the adapter refuses to send input. This is a safety feature, not a bug.
- The game may be running elevated while Python is not, or vice versa; Windows will block input.
- Check the capture region against display scaling. A region outside the virtual desktop is rejected.

### `pygame` prints a greeting

Harmless. `pygame-ce`/`pygame` can both be present as dependencies; the import resolves to one of
them. The pinned requirements are the combination that was tested.

## Training problems

### "Run directory already exists"

`--run-name` must be unique under `logs/runs/`. Omit `--run-name` to get an automatic timestamped
name, or choose a new name. The dashboard generates a unique name itself.

### PPO rejects my timesteps immediately

- `timesteps` must be at least 1, and PPO needs at least one rollout (`n_steps × n_envs`).
- For PPO, `batch_size` must divide `n_steps × n_envs`. Use the validation error's values.
- `n_envs` must match the batch arithmetic. Example: 4 envs × 256 steps = 1024; batch 128 works.
- Desktop configs must use `n_envs: 1`.

### Training is slow or memory-heavy

- Pixel CNNs are expensive. Reduce `width`/`height` (e.g. 64), use grayscale, and reduce `n_envs`.
- Use `--device cpu` if CUDA is misconfigured or out of memory. A CUDA build may not be installed
  even if `nvidia-smi` works.
- TensorBoard logging is lightweight; disabling it helps only a little.
- External desktop games add real input hold time per step; they are inherently slow.
- CPU PyTorch is fine for CartPole and small pixel games.

### Training stops with `STOP` immediately

A stale run directory can contain a `STOP` or `PAUSE` file. Use a new run name/directory. Inspect:

```text
logs/runs/<run-id>/STOP
logs/runs/<run-id>/PAUSE
```

### Pause seems unresponsive

Pause is checked at a callback boundary (about every training vector step, throttled to 0.15s). It
cannot interrupt a long evaluation or a checkpoint save immediately. Wait, then refresh; the status
file should show `paused`.

### Training was interrupted

- Ctrl+C: look for `models/<game>_interrupted.zip`.
- Dashboard/CLI graceful stop: look for `models/<game>_stopped.zip` and the run's
  `checkpoints/.../stopped_<steps>_steps.zip`.
- If no model exists, use the periodic checkpoint closest to the interruption.
- Resume with `--model PATH --timesteps N` or `--config` plus `--model`.

### Reward is flat, negative, or not improving

- A constant time penalty alone gives no information about success.
- 50k timesteps may be too few for the game, especially with pixels.
- Use a small learning run and watch `episodes.jsonl`, not just one final number.
- Reward components must refer to signals the adapter actually emits. Missing signals raise errors.
- Reward hacking: inspect the reward breakdown and watch playback. The agent may be optimizing a
  detector rather than the game.
- Frame stacking, resolution, and grayscale conversion can remove information the game needs. Try RGB
  or a larger resolution if colour is essential.
- Compare against a random policy. A flat reward does not always mean a broken pipeline.

### Example game looks random after only 128 steps

128 steps is a smoke test, not training. The shipped demonstration uses 50,000 steps; even then,
policy quality depends on seed and reward. Watch the full run and use a longer run if needed.

### LunarLander reward stays negative

LunarLander is significantly harder than CartPole. 50k steps often produces weak or negative
policies. Try larger runs (300k+), tune `gamma`, `ent_coef`, and rewards, and use periodic evaluation.
Do not interpret a poor LunarLander score as a framework failure.

## Model loading, evaluation, and playback

### "Model file not found"

Train first, or use the exact model path from the training summary. The default is
`models/<game>.zip`. `--model` also accepts a path without the `.zip` extension if the file exists.

### "Model's observation space does not match the current environment"

The model was trained with different pixels/state/crop/frame stacking/action bindings. Use the
metadata config:

```bat
python scripts/play.py --model models/example_game.zip
```

Omit `--game`/`--config` to let the sidecar metadata recreate the matching settings. If you pass
`--config`, it must describe the same observation and action spaces. Do not try to bypass the check;
the model would misread the input.

### "Requested algorithm ... but model records ..."

The checkpoint was trained with another algorithm. Load it with the recorded algorithm or use the
matching config. Resume also validates the algorithm.

### Playback opens no window

- `play.py` uses a human rendering mode; a display is required.
- On a server, use `evaluate.py` (headless by default) or `play.py --no-render` instead.
- For the example game, close the Pygame window or press Escape; for desktop games, the real game
  window and GameMind's capture viewer may both be present.
- The model may be playing, but the game itself may not be focused; the desktop adapter's focus rule
  still applies.

### Playback seems to train the model

It does not. `play.py`/`evaluate.py` load the model, call `predict()`, and manually step episodes.
There is no `learn()` call. Model weights and the `num_timesteps` counter are unchanged.

### `fps 0` or low FPS

`--fps` controls the playback loop's pacing. `0` disables added sleeping (the game/input backend may
still be faster or slower). Desktop action duration adds real time per step. Use a value suitable for
your display and game.

## TensorBoard and dashboard

### No TensorBoard metrics

- Check `training.tensorboard: true` and do not pass `--no-tensorboard`.
- Point TensorBoard at the run folder or its parent:
  `tensorboard --logdir logs/runs`.
- PPO updates once per rollout; the first graph points can take time to appear.

### Dashboard cannot launch training

- Start it with the venv's Streamlit: `.venv\Scripts\streamlit run dashboard/app.py`.
- Inspect `logs/dashboard/<run-name>.log` and the run's `train.log`.
- A desktop config with the placeholder window title will fail by design until edited.
- The dashboard uses `sys.executable`, which is the Python running Streamlit; make sure that is the
  project venv.
- Port conflict: `streamlit run dashboard/app.py --server.port 8502`.

### Dashboard is stale

Press **Refresh status**. The dashboard does not install an auto-refresh dependency. Charts appear
after episodes complete, not after every training step.

## Windows-specific notes

- File handles for `train.log` are closed at the end of a run; tests use the same cleanup so
  temporary directories can be deleted. If you kill the process hard, Windows may briefly keep the
  file open.
- The console code page may display non-ASCII symbols as `?`/mojibake. This is cosmetic; logs and JSON
  are UTF-8.
- Use `evaluate.py` (headless by default) or `play.py --no-render` for automated/headless execution. Do not run a trained desktop policy in the
  background while you are using the keyboard; focus rules and a short `start_delay` are there to
  protect you, not to make unattended desktop control risk-free.

## Where to look next

- [training.md](training.md) for every config field and reward component.
- [connecting_games.md](connecting_games.md) for adapter and capture design.
- [installation.md](installation.md) for dependency and display setup.
- `logs/runs/<run-id>/status.json`, `train.log`, `episodes.jsonl`, and `monitor.csv` for the actual
  behavior of a failing run.
