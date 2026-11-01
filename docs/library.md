# Game library and dashboard

The library is GameMind's memory. It records which games are installed, which levels belong to each
game, what every training session did, which model came out of it, and what the evaluation measured.
The dashboard is a thin interface over those records.

Two rules shape everything here:

1. **Each game has its own agent.** Games never share weights, checkpoints, optimizer state, reward
   configuration, or history. The framework code is shared; the learning state is not.
2. **Records beat recollection.** A level's status comes from stored sessions and evaluations, not
   from a chat log, a folder you remember, or a reward value. You never have to reconstruct what a
   previous run did.

## Start the app

From the project root, with the virtual environment active:

```bat
streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>). The first launch creates
`library/` and seeds a Geometry Dash profile with one level, `Stereo Madness`.

The CLI still works exactly as before (`scripts/train.py`, `scripts/play.py`, `scripts/evaluate.py`).
The dashboard drives the same training engine and the same configuration files.

## What the dashboard contains

| Section | What it is for |
| --- | --- |
| **Home** | Current training, overall counts, quick start for the selected level, recent sessions. |
| **Games** | One card per game, with levels, best recorded result, last session, Add game, and the selected game's level list. |
| **Training** | Live status, pause/resume/stop, level queues, worker log. |
| **Models** | Models and evaluations for the selected level, plus preserved artifacts from earlier runs. |
| **Knowledge** | Per-game skills with evidence, confidence, and applicability to the selected level. |
| **Settings** | Routine defaults, advanced YAML editors, and diagnostics. |

## Folder layout

Everything the library owns lives under `library/`, separate from source code:

```text
library/
├── settings.json                     defaults used by the UI
├── games/
│   └── geometry_dash/
│       ├── game.json                 game profile (name, adapter, notes)
│       ├── game.yaml                 default configuration for new levels
│       └── levels/
│           └── stereo_madness/
│               ├── level.json        level profile and completion settings
│               ├── config.yaml        the exact configuration training uses
│               ├── validation.json   last validation report
│               ├── sessions/         one JSON file per training session
│               ├── evaluations/      one JSON file per evaluation run
│               ├── models/           .zip models + .metadata.json sidecars
│               ├── checkpoints/      periodic and stop checkpoints
│               └── runs/             status.json, episodes.jsonl, train.log, TensorBoard
├── knowledge/skills/                 per-game skills with evidence
├── queues/                           multi-level queue plans
└── state/                            worker id, job files, control flags (not tracked by Git)
```

Profiles, configurations, and validation reports are small and safe to version. Sessions,
evaluations, models, checkpoints, runs, queues, and worker state are runtime data and are ignored by
Git. The project-wide `models/`, `checkpoints/`, and `logs/` roots still hold anything produced by
the CLI, and are listed in the Models page as preserved, unassigned artifacts.

## First run with Geometry Dash

1. Open Geometry Dash, select the level, and enable **Auto-Retry**.
2. Keep the game fullscreen or borderless on the primary monitor, and focused.
3. In GameMind open **Games → Geometry Dash**.
4. Select the level, or add one with **Add level**.
5. Press **Validate configuration**, then **Validate with the game open**.
6. Press **Train**. A worker starts in the background.

Closing the browser tab does not stop training. The worker keeps writing status, and the session is
saved when it ends. Return to **Training** to pause, resume, stop, or read the log.

### Why completion is off by default

`Stereo Madness` is created with completion detection **disabled** and initialization set to
**scratch**. The earlier 20,000-step checkpoint was trained before the library existed, so the level
it was trained on was never recorded. Attaching it automatically would be a guess.

Enable completion only for a level whose completion signal you have watched succeed:

* mode **bar**: the adapter's progress-bar detector must reach the threshold for the configured
  number of consecutive frames,
* and the profile's threshold is written into the level configuration as `completion_pct` /
  `completion_frames`, with `allow_bar_completion: true`.

A reward, a timestep budget, or a high progress estimate **never** marks a level complete. If the
detector is disabled, a level can never show as completed, which is intentional.

Bar completion is only implemented for Geometry Dash. Adding it to another adapter raises a clear
error instead of offering a switch that would do nothing.

## Adding a game

**Games → Add game** lists the adapters this installation actually ships:

| Adapter | Ready from the form? | What it needs |
| --- | --- | --- |
| `geometry_dash` | Yes | Geometry Dash installed, Auto-Retry, focused window. |
| `example_game` | Yes | Nothing; the built-in Target Chase pixel game. |
| `gymnasium` | Yes, with an environment id | A Gymnasium id that imports in this venv. |
| `desktop` | No | A YAML configuration with your window title, actions, rewards, and reset keys. |

Adding an arbitrary name does **not** create support. The profile is refused with an explanation
that a new adapter class is required. That is deliberate: a card that cannot train is worse than an
honest error.

A new game gets its own `library/games/<id>/` directory, its own configuration, and therefore its
own models and history. Nothing about Geometry Dash is touched.

## Adding a level

**Games → (open a game) → Add level** asks for:

* **Level name** — display only. Renaming later keeps the same id, paths, and history.
* **Level type** — official or selected/custom. Documentation only; it changes no behaviour.
* **How GameMind reaches the level** — `manual` by default. GameMind does not navigate game menus.
* **Completion detection** — disabled, or the progress bar with a threshold and frame count.
* **Initialization** — where the next session starts:
  * *Continue this level's best evaluated model* (default),
  * *Start from a fresh policy*,
  * *Reuse another level's compatible policy when possible*.
* **Notes** — free text kept with the profile.

Every level gets its own `config.yaml`. Per-level settings (jump timings, crop, observation size,
progress-region calibration, episode length) live there and are **not** overwritten when you rename
the level, archive it, or change its completion mode. Editing the game's default YAML affects new
levels only.

## Validation

Two checks, with different evidence:

* **Validate configuration** parses the level's YAML and verifies: the adapter is available, the
  configuration uses the game's adapter, desktop games learn from pixels with one environment, the
  window title and focus requirement are set, and the completion setting matches the profile.
* **Validate with the game open** also connects to the game, resets the attempt, captures one frame,
  reads the progress bar, and sends one no-op action. It reports the observed bar state.

The live check cannot prove death detection, completion detection, reset synchronization over many
attempts, or long-run stability. It says so explicitly in the report.

## Sessions: what is saved and when

A session record is written **before** training starts and rewritten when it ends, so an interrupted
run is still visible. At the end, GameMind stores:

* the final model (`.zip` + `.metadata.json`) and any best-evaluation model,
* periodic and stop checkpoints,
* training metrics, attempts, timesteps, and the reason training ended,
* the evaluation result (when enabled) with per-attempt progress,
* the configuration snapshot and the observation/action metadata,
* the initialization source (own checkpoint, fresh, or transferred),
* verified-completion evidence, if the level's detector fired,
* whether the new result regressed against the previous best, and which model remains the best.

Session states are `finished`, `stopped`, `interrupted`, `error`, and `running`. A previous best
model is never overwritten by a worse session: models are per-session files, and the "best" pointer
is chosen by recorded evaluation results.

### Resume

**Resume** (or **Train**) uses the level's initialization policy:

1. the level's best model with a recorded evaluation,
2. otherwise the level's most recent completed checkpoint,
3. otherwise a compatible policy from another level in the same game (only when the policy says so),
4. otherwise a fresh policy.

A model is refused if its observation preprocessing, action bindings, frame skip, or episode length
no longer match the level. Transfer between levels is explicit, is logged as a transfer (not a
continuation), and never copies a progress percentage from the source level.

## Queues

**Training → Level queue** trains several levels in order:

* one session per level, each with its own record and checkpoint,
* evaluation after each session,
* a level with verified completion is skipped on later passes unless **Re-train completed levels**
  is on,
* progress is saved after every level, and an interrupted queue resumes from its first unfinished
  entry (the interrupted session's record stays visible, marked as interrupted),
* a session error stops the queue instead of hiding it.

The queue shares the single worker slot with normal training. Starting one while another runs is
refused.

## Progress states

| Status | Meaning |
| --- | --- |
| Not started | No session and no validation. |
| Configured | Configuration validated, or trained without a comparable evaluation. |
| Training | A session is running right now. |
| Evaluated | One evaluation, or a newer one that did not improve. |
| Improving | The latest evaluation beat the previous one. |
| Stagnating | Three consecutive evaluated sessions did not beat the earlier best. |
| Completed | The level's own completion detector fired and the evidence is stored. |
| Needs attention | The last session failed, or an interruption needs a look. |

For Geometry Dash the comparison metric is estimated maximum progress from the bar. For other games
it is mean episode reward, which is only comparable inside one game and reward configuration. The
dashboard labels which metric it used.

When a level is stagnating, the **Diagnostics** panel suggests checks — validate the level, confirm
the bar reading matches the game, inspect action counts in `episodes.jsonl`, run a deterministic
5-attempt evaluation — and does not start another long run by itself.

## Skills

Skills are per-game notes with evidence, conditions, and confidence. A skill records what was
observed (for example a jump hold duration that registered) and the environment it applies to. The
Knowledge page shows which skills match the selected level's observation mode, resolution, frame
skip, and action bindings, and why others do not.

Skills are **not** weights and are never turned into a progress claim. Transferring a policy is a
separate, checked operation; transferring a note is just documentation.

## Data safety

* All records are written atomically (temporary file, then replace).
* Removing a level archives its models, checkpoints, and run directories to
  `library/archive/<game>/<level>-<timestamp>/` before deleting the profile. A mis-click cannot
  destroy the only copy of a policy.
* Removing a game refuses while levels exist unless removal is explicitly confirmed.
* One worker at a time, enforced with an exclusive lock file plus a PID liveness check. A stale lock
  from a crashed process is detected and cleared.
* A worker that dies without writing its final state is marked `lost` on the next dashboard load;
  the session record keeps whatever was saved.
* Secrets never enter the library. There are none to store: nothing here reads API keys.

## Command-line equivalents

The dashboard is optional. The library is a normal Python package:

```python
from gamemind.library import GameLibrary, SessionRequest, TrainingController, run_session

library = GameLibrary()            # or GameLibrary("D:/GameMind/library")
library.ensure_migrated()
game = library.get_game("geometry_dash")
level = library.list_levels(game.id)[0]

# Train in the foreground of this process:
run_session(library, SessionRequest(game_id=game.id, level_id=level.id, timesteps=5_000))

# Or start the same background worker the dashboard uses:
TrainingController(library).start_session(level, timesteps=5_000)
```

A job can also be run directly with the worker script the dashboard launches:

```bat
python scripts/library_worker.py --job library/state/jobs/<job_id>.json
```

## Tests

`tests/test_library.py` covers profile creation and editing, per-level isolation, migration,
session persistence, completion rules, resume selection, transfer compatibility, queue execution,
queue recovery, error handling, worker duplicate prevention, pause/resume/stop files, and a real
worker subprocess. `tests/test_dashboard.py` runs the actual Streamlit app with Streamlit's test
harness and checks every page, the Train button, the pause/resume buttons, and the Add level form.

Both use temporary libraries; neither touches a real game.

## Honest limitations

* **Only Geometry Dash has been exercised against a real game** in this project, and its completion
  detection remains unverified. Bar progress is an estimate (roughly a percentage point or two) and
  has shown implausible readings before.
* Transfer between levels is implemented and compatibility-checked, but has **not** been shown to
  improve play on a second real level. It is an experiment with a guard rail, not a proven
  curriculum.
* Automatic progression stops at levels whose completion detector has not fired. No practice
  segments or level-select automation exist.
* The dashboard is covered by headless Streamlit tests, not by a long manual soak test. Restarting
  the UI, closing the browser, and reopening the library are covered by tests; a multi-hour run
  while interacting with the page has not been soak-tested here.
* Queue scheduling is ordered only. There is no priority or round-robin policy.
