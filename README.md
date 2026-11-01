# GameMind AI

GameMind AI is a modular reinforcement-learning framework for teaching agents to play games from
**screen pixels or native game state**. It is built around one small adapter contract, so the
training engine, reward shaping, observation pipeline, checkpointing, evaluation, and playback stay
the same when you add a new game.

> **Honest scope:** this is a reusable training platform, not a universal pretrained game-playing
> model. A model is trained for a specific game, observation configuration, action set, and reward
> function. 50,000 timesteps can be enough to see learning in a tiny game; it will not master most
> games. Reward design, observation quality, and the game itself matter at least as much as the
> algorithm.

## What works now

- **One adapter interface**, implemented by a Gymnasium adapter, a built-in pixel game, a
  configurable desktop keyboard/mouse adapter, and a specialised **Geometry Dash** adapter with
  visual progress/death detection.
- **Pixel observations**: RGB or grayscale, configurable resolution, optional crop, frame stacking,
  optional normalization, frame skipping, and strict shape/dtype/range validation.
- **Actions**: discrete bindings with keyboard combinations, mouse click/relative movement, a no-op,
  configurable hold duration, and safety cleanup for held keys.
- **Rewards**: additive components for native game reward, constants/time penalties, signals,
  events, progress toward a target, survival, visual template matches, and custom Python functions.
- **Training**: PPO by default, A2C available, optional MaskablePPO; vectorized environments,
  checkpoints, TensorBoard, periodic evaluation, best-model saving, pause/stop, resume, and
  interrupted-run recovery.
- **Evaluation and live playback** from saved models, with an explicit compatibility check so a
  model cannot silently see a different observation layout than it was trained on.
- **A persistent game library** (`library/`) with one independent agent per game, per-level
  checkpoints and history, safe resume, compatibility-checked transfer, shareable skills, and
  multi-level queues that survive interruption.
- **Streamlit dashboard** as the main interface: game cards, add game/level, train, pause, resume,
  stop, queues, models, evaluations, and skills — with training always running in a background
  worker.
- **Tests** for adapters, observations, actions, rewards, training, persistence, evaluation, and the
  example environment's Gymnasium API compliance.

## Quick start

Tested on Windows 11 with Python 3.13.14. Python 3.11–3.13 are supported. Install into a fresh
virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/test_setup.py
```

On macOS/Linux use `source .venv/bin/activate` instead. `LunarLander-v3` needs Box2D; the
requirements install `gymnasium[box2d]`. If Box2D is unavailable on your platform, CartPole and the
built-in pixel game still work.

Train the first agent (built-in Target Chase, a small 5×5 pixel puzzle with one obstacle and a yellow direction beam; no external window is needed):

```bat
python scripts/train.py --game example_game --timesteps 50000
```

Watch it play in a live window:

```bat
python scripts/play.py --game example_game --model models/example_game.zip
```

Run a headless evaluation:

```bat
python scripts/evaluate.py --game example_game --model models/example_game.zip
```

Other ready-to-run configurations:

```bat
python scripts/train.py --game cartpole --timesteps 50000
python scripts/train.py --game lunarlander --timesteps 50000
```

### Geometry Dash (real desktop game)

Geometry Dash is supported as an experimental **desktop** integration. It captures the real game
window, sends Space/Enter, and estimates progress from the visible white progress bar. It does not
read game memory or internal state. Before running, open Geometry Dash, select a level, enable
Auto-Retry, and use fullscreen/borderless on the primary monitor. Full setup, calibration, and
limitations are in [docs/geometry_dash.md](docs/geometry_dash.md).

```bat
python scripts/train.py --config configs/geometry_dash.yaml --timesteps 50000
python scripts/play.py --config configs/geometry_dash.yaml --model models/geometry_dash.zip --episodes 3
python scripts/evaluate.py --config configs/geometry_dash.yaml --model models/geometry_dash.zip --episodes 20
```

This is Windows-focused and requires the game to remain focused/visible. It is not a promise that
50,000 steps will beat a level; measure survival time and maximum progress before drawing
conclusions.

Launch the dashboard:

```bat
streamlit run dashboard/app.py
```

### The game library dashboard

The dashboard is the main way to manage games now. Open it, choose **Geometry Dash**, select or add
a level, and press **Train**. Every session saves its model, configuration, metrics, evaluation, and
reason for ending; resume picks up the level's best evaluated state. Each game keeps its own agent,
and each level keeps its own models and history.

```bat
streamlit run dashboard/app.py
```

Training runs in a background worker, so closing the tab does not stop it. The CLI still works
exactly as documented below. Full guide: [docs/library.md](docs/library.md).

## What reinforcement learning is doing here

1. A **game adapter** exposes `reset()`, `observe()`, one control step, reward signals, termination,
   rendering, and cleanup through a Gymnasium-compatible environment.
2. Wrappers turn raw frames into the neural network's input (for example 84×84 grayscale,
   channel-first, four-frame stack) and apply action repetition.
3. Stable-Baselines3 collects experience in vectorized environments.
4. **PPO** (Proximal Policy Optimization) updates a policy network using a clipped objective. In
   plain terms: it tries changes that improve expected reward without letting one update move the
   policy too far, which tends to make training stable.
5. The trained policy is saved with a JSON sidecar describing the exact game/observation/action
   configuration. Playback recreates that environment and refuses a mismatched model.

### State observations vs screen pixels

- `observation.mode: state` uses the values the game gives the environment (for example CartPole's
  four numbers). This is easy to learn from but is not available for an arbitrary desktop game.
- `observation.mode: grayscale` or `rgb` renders/captures a frame and applies the pixel pipeline.
  The policy sees only pixels. For CnnPolicy, keep `normalize: false`: Stable-Baselines3's image
  policy expects uint8 0..255 channel-first pixels and normalizes internally. Grayscale frame
  stacking is the standard choice for motion-sensitive pixel control.

GameMind never silently swaps pixel mode for a privileged state vector. If you configure pixels,
the adapter must provide pixel frames.

### Rewards

Reward components are additive and are evaluated on every step. Start with the smallest signal
that actually tells the agent whether it is doing better:

```yaml
reward:
  components:
    - type: progress       # reward getting closer to the target (distance decreases)
      key: distance
      mode: decrease
      reset_on: target_collected
      scale: 0.5
    - type: signal
      key: target_collected
      scale: 10.0
    - type: constant
      value: -0.01         # small time penalty discourages dawdling
```

A badly shaped reward can be "solved" in an unintended way. Watch the agent, inspect
`episodes.jsonl`/TensorBoard, and change the reward deliberately. See
[docs/training.md](docs/training.md) for every component and field.

## Architecture

```text
configs/*.yaml
      │
      ▼
GameConfig validation ──► GameAdapter (Gymnasium env)
      │                        │  connect / reset / observe / step / calculate_reward ...
      │                        ▼
      │              PixelObservationWrapper  (resize, crop, RGB/grayscale)
      │                        ▼
      │              ActionRepeat             (optional frame skip)
      │                        ▼
      │              FrameStackChannels       (optional motion stack)
      │                        ▼
      │              ObservationValidator     (fail loudly on contract violations)
      │                        ▼
      │              Stable-Baselines3 VecEnv ──► PPO / A2C
      ▼                        │
 training config ──────────────┴──► checkpoints, TensorBoard, status, episodes.jsonl
                                             │
                                             ▼
                              model .zip + .metadata.json
                                             │
                              evaluate.py / play.py recreate exactly the same env
```

### Adapter contract

Every adapter subclasses `GameAdapter`, which is also a Gymnasium environment, and implements:

| Method | Purpose |
| --- | --- |
| `connect()` | open/attach to the game, window, emulator, or underlying Gym env |
| `_reset(...)` | restart game state and return the first raw observation |
| `observe()` | return the current raw RGB frame or native state |
| `_apply_action(action)` | advance one game tick with a validated action |
| `calculate_reward()` | evaluate configured reward components (base implementation) |
| `is_episode_over()` | game-specific death/win/termination signal |
| `render()` | show a window and/or return an RGB frame |
| `close()` | release held keys, windows, capture handles, and processes |
| `get_action_space()` / `get_observation_space()` | declared Gymnasium spaces |

`reset()` and `step()` provide the standard Gymnasium five-value API. The shared engine lives in
`gamemind/`; game-specific code lives in `games/`.

### Project layout

```text
GameMind-AI/
├── README.md, requirements.txt, pyproject.toml, LICENSE, .gitignore
├── configs/
│   ├── default.yaml
│   ├── cartpole.yaml
│   ├── lunarlander.yaml
│   ├── example_game.yaml
│   ├── example_game_random.yaml
│   ├── geometry_dash.yaml
│   └── desktop_game.example.yaml
├── gamemind/
│   ├── library/         game/level registry, sessions, skills, transfer, queues, worker control
│   ├── adapters/        base contract, Gymnasium, example game, desktop, Geometry Dash, registry
│   ├── observations/    pixel processor and Gymnasium wrappers
│   ├── actions/         discrete binding helpers and PyAutoGUI input backend
│   ├── rewards/         reward components, custom functions, template termination
│   ├── algorithms/      PPO/A2C/MaskablePPO selection and policy inference
│   ├── training/        trainer, callbacks, status files, model persistence
│   ├── evaluation/      non-training episode runner and live playback
│   ├── envs/            single and vectorized environment construction
│   └── utils/           errors, logging, paths, seeding, vision, serialization
├── games/example_game/  Target Chase game and renderer
├── scripts/             train.py, play.py, evaluate.py, test_setup.py, library_worker.py
├── dashboard/app.py     Streamlit dashboard (games, levels, training, models, knowledge)
├── library/             game/level profiles, session records, skills (runtime data Git-ignored)
├── tests/               pytest suite, including Geometry Dash fake-backend and dashboard tests
├── models/              generated models (Git-ignored)
├── checkpoints/         generated checkpoints (Git-ignored)
├── logs/                status, metrics, TensorBoard, monitor logs (Git-ignored)
└── docs/                installation, training, connecting games, Geometry Dash, troubleshooting
```

## Ready-made games

| Config | What the agent sees | Works immediately? | Notes |
| --- | --- | --- | --- |
| `cartpole` | native four-value state | Yes | Fast smoke test; PPO commonly learns in tens of thousands of steps, but results vary. |
| `lunarlander` | native eight-value state | Yes after Box2D installs | Harder; 50k is a starting point, not mastery. |
| `example_game` | 4 stacked grayscale 84×84 frames | Yes | Fixed 5×5 Target Chase starting puzzle: one obstacle, one target, terminal success. The included 50k run learned this layout; see the measured result below. |
| `example_game_random` | 4 stacked grayscale 84×84 frames | Yes, but harder | Random target placement, respawning targets, 60-step limit. Treat 50k as a starting experiment and evaluate deterministic play before claiming skill. |
| `geometry_dash` | 4 stacked grayscale 84×84 frames from the real game | **Requires setup** | Windows, Geometry Dash, Auto-Retry, fullscreen/borderless, focus. Progress is an approximate bar reading; see [docs/geometry_dash.md](docs/geometry_dash.md). |
| `desktop_game.example` | captured pixels | **No, needs editing** | Real keyboard/mouse control. You must set the window title, capture region, reset keys, rewards, and possibly a game-over detector. |

### The built-in Target Chase game

Target Chase is deliberately small and self-contained: a 5×5 grid, a white agent square, a green
target ring, grey obstacles, and an optional yellow beam pointing along the dominant direction to the
target. The beam is **drawn into the frame**; the policy still receives only 84×84 pixels. It is a
visual affordance, not privileged coordinates, and can be disabled with `show_compass: false`.

The shipped `example_game.yaml` uses a fixed first puzzle (`start: [2, 4]`, `target: [2, 0]`, one
obstacle at `[2, 2]`) and `terminate_on_collect: true`. That makes the first training run a
navigation puzzle with a clear success signal instead of a random-target search. It is not a claim
that the policy generalizes to other layouts. To make a randomized version, remove `start`, `target`,
and `obstacles`, set `terminate_on_collect: false`, and expect a longer run. `example_game_random.yaml`
is the included harder variant.

On the tested Windows CPU setup, the included 50,176-step `example_game` run reached the target in
**20/20 deterministic evaluation episodes** (mean reward 13.94, mean episode length 6.0). That is a
success on this fixed training puzzle, not evidence of generalization to new layouts or games.

Available game parameters (all optional):

| Parameter | Meaning |
| --- | --- |
| `grid_size`, `cell_size` | Grid dimensions and pixels per cell; the rendered frame is square. |
| `n_obstacles` | Number of randomly placed obstacles when `obstacles` is not supplied. |
| `max_steps` | Game-side step limit; also used when `episode.max_steps` is omitted. |
| `show_compass` | Draw the yellow direction beam into the pixels (default `true`). |
| `start`, `target` | Fixed `[x, y]` cells for reproducible layouts. Omit for random placement. |
| `obstacles` | Fixed list of `[x, y]` cells; overrides random obstacle placement. |
| `terminate_on_collect` | End the episode as a success when the target is reached (default `false`). |

## Configuring training

The main controls are in YAML (see [configs/default.yaml](configs/default.yaml) for a commented
template):

```yaml
game:
  name: example_game
  adapter: example_game
  params:
    grid_size: 5
    cell_size: 64
    n_obstacles: 1
    max_steps: 30
    show_compass: true
    start: [2, 4]
    target: [2, 0]
    obstacles: [[2, 2]]
    terminate_on_collect: true
observation:
  mode: grayscale
  width: 84
  height: 84
  stack: 4
  normalize: false
  # crop: [top, left, bottom, right] fractional crop before resize
actions:
  frame_skip: 1
  duration: 0.0
  bindings:
    - {name: noop, params: {move: [0, 0]}}
    # ...
episode:
  max_steps: 30
training:
  algorithm: ppo
  timesteps: 50000
  n_envs: 4
  checkpoint_freq: 10000
  eval_freq: 2500
  save_best: true
```

You can override common values from the CLI, for example:

```bat
python scripts/train.py --game example_game --timesteps 100000 --width 96 --height 96 --frame-stack 4 --algorithm ppo --seed 7 --run-name my_run
```

`--timesteps` means **additional timesteps in this invocation**. When resuming from a model, the
number is added to the model's existing step count. See [docs/training.md](docs/training.md) for
all fields, reward components, checkpoint behavior, and PPO guidance.

## Outputs and monitoring

A training run creates:

```text
models/example_game.zip                    final model
models/example_game.metadata.json          config + spaces + versions + training metrics
checkpoints/example_game/<run-id>/         periodic and stop checkpoints + best/
logs/runs/<run-id>/status.json             live machine-readable status
logs/runs/<run-id>/episodes.jsonl          one record per completed episode
logs/runs/<run-id>/monitor.csv             Stable-Baselines3 episode monitor
logs/runs/<run-id>/tensorboard/            TensorBoard event files
logs/runs/<run-id>/train.log               full training log
logs/runs/<run-id>/config.resolved.yaml    exact config used
```

TensorBoard:

```bat
tensorboard --logdir logs/runs
```

### Pause, stop, and resume

- **Ctrl+C** stops training and saves `models/<game>_interrupted.zip` when possible.
- **Dashboard Stop** writes `STOP` in the active run directory. The training callback saves a
  `stopped_<steps>_steps.zip` checkpoint and exits gracefully. The trainer also saves
  `models/<game>_stopped.zip` for convenience.
- **Dashboard Pause** writes `PAUSE`. Training pauses at the next callback boundary and resumes
  when the file is removed. The dashboard Resume button removes it.
- From the CLI, create/remove the same files. On Windows PowerShell:

```powershell
New-Item logs\runs\<run-id>\PAUSE -ItemType File
Remove-Item logs\runs\<run-id>\PAUSE
New-Item logs\runs\<run-id>\STOP -ItemType File
```

- **Resume / continue training**:

```bat
python scripts/train.py --game example_game --model models/example_game.zip --timesteps 50000
```

This loads the algorithm recorded in the metadata, validates observation/action spaces, and trains
for 50,000 additional steps. Use `--output models/example_game_continued.zip` if you want to keep
the original file.

Periodic evaluation uses a separate evaluation environment and can save the best model under
`checkpoints/<game>/<run-id>/best/best_model.zip` with its own metadata. Periodic evaluation is
disabled for the desktop adapter because a second environment would send input to the same window.

## Watch an agent play

```bat
python scripts/play.py --game example_game --model models/example_game.zip
python scripts/play.py --game example_game --model models/example_game.zip --episodes 3 --fps 12
python scripts/evaluate.py --game example_game --model models/example_game.zip --episodes 10 --output logs/eval.json
```

`play.py` uses deterministic actions by default, recreates the environment from the model metadata,
and never trains. Close the game window, press Escape, or press Ctrl+C to stop cleanly. Use
`--stochastic` to sample actions, `--fps 0` for unpaced playback, or `--no-render` for a headless
run. For generic desktop games, the adapter can additionally display captured frames. The Geometry
Dash adapter intentionally does not open that duplicate viewer: watch the real game window instead.

## Connect another game

Start with [docs/connecting_games.md](docs/connecting_games.md). The shortest paths are:

1. **A Gymnasium environment already exists.** Copy `configs/cartpole.yaml`, change
   `game.params.env_id`, and choose `state` or pixel mode. No code needed.
2. **The game has a Python API.** Write a `GameAdapter` subclass in `games/`, set
   `game.adapter: "games.my_game.adapter:MyGameAdapter"`, and configure observations, actions, and
   rewards in YAML. Use [games/example_game](games/example_game) and
   [gamemind/adapters/example_game_adapter.py](gamemind/adapters/example_game_adapter.py) as
   concrete references. The adapter can expose native signals for reward components.
3. **The game accepts ordinary desktop keyboard/mouse input.** Copy
   [configs/desktop_game.example.yaml](configs/desktop_game.example.yaml), edit the window title,
   capture region, reset keys, action bindings, and rewards, then:

```bat
python scripts/train.py --config configs/my_desktop_game.yaml --timesteps 50000
```

   The desktop adapter is deliberately limited: screenshots do not reveal score, lives, or game over
   unless you configure a template detector or write an adapter with access to a real API/emulator.

A compatibility matrix, wiring guide, mouse examples, custom reward functions, and safety notes are
in [docs/connecting_games.md](docs/connecting_games.md). There is no promise of universal
compatibility: some games need a custom observation, reward, or termination source.

## Supported systems and integration methods

- **Tested:** Windows 11, Python 3.13.14, CPU PyTorch; CartPole, LunarLander (Box2D), Target Chase,
  the desktop adapter logic (fake backends in tests), and a live Geometry Dash adapter/PPO/playback
  smoke test.
- **Expected:** Python 3.11–3.13 on macOS/Linux for Gymnasium and the built-in game. Live Pygame
  playback needs an SDL display. Desktop capture (mss) and input (PyAutoGUI) depend on the OS:
  macOS requires Accessibility/Screen Recording permissions; Linux needs an X11-compatible session
  (Wayland is not reliably supported by these tools). Verify with `test_setup.py` and a manual
  window test.
- **Headless servers:** training for Gymnasium/Target Chase works headless; live windows and desktop
  control do not.

## Testing

```bat
python -m pytest -q
python scripts/test_setup.py
```

Current checkout: **73 tests passed** and **28 setup checks passed**. On a machine with Geometry Dash
running, `test_setup.py` also performs a read-only window lookup and one-frame capture check.

The test suite covers config parsing/validation, observation shapes/dtypes/bounds, wrappers, action
bindings, reward components and template termination, the custom game's Gymnasium API compliance,
desktop adapter input cleanup/focus/template termination with fake backends, a real small PPO
training run, checkpoint metadata, model/environment compatibility, evaluation, and resume training.
Geometry Dash tests cover window lookup, bar reading/calibration, death/reset, completion, jump
release, and a fake-backend PPO smoke run. The live smoke run also exercised a real window,
real screenshots, real keys, model saving, evaluation, and playback; details are in
[docs/geometry_dash.md](docs/geometry_dash.md).
The verification details and environment-specific limitations are recorded in
[docs/troubleshooting.md](docs/troubleshooting.md).

## Limitations and next steps

- No single pretrained model plays every game. Each experiment is game-specific.
- Pixel RL is sample-hungry. 50k steps is a demonstration; expect longer runs, reward iteration, and
  experimentation for meaningful play.
- Desktop screenshot control cannot read hidden state. Use template matching for a fixed game-over
  or success banner, or implement a custom adapter using an emulator/API for scores, lives, and
  termination.
- Desktop training is restricted to `n_envs: 1` and periodic evaluation is disabled, because both
  would control the same window.
- Geometry Dash progress is estimated from the progress bar's coloured fill, not OCR or game memory.
  It can be off by a percentage point or two and depends on the bar remaining visible and unstyled
  differently by a mod. Auto-Retry is required for the current death detector. See
  [docs/geometry_dash.md](docs/geometry_dash.md).
- Template matching is sensitive to resolution/UI changes. It is not OCR or general vision.
- The built-in policies support Box and Dict observation spaces and Discrete/Box action spaces.
  Custom policies/feature extractors can be passed through `training.extra_params.policy_kwargs`;
  continuous control is available to custom adapters with a Box action space.
- MaskablePPO is optional and only appropriate when an adapter exposes genuine `action_masks()`.
- Audio observations, multiplayer, emulator-specific process control, OCR score detection, and a
  polished web frontend are future work; the extension points are documented rather than pretended
  to work.

## License

MIT. See [LICENSE](LICENSE).
