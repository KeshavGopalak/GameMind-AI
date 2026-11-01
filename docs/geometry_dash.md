# Geometry Dash integration

GameMind AI can control a real Geometry Dash installation on Windows through ordinary keyboard
input and screen capture. The agent learns from the same pixels you see. It does **not** read the
game's memory, level file, collision data, attempt counter, or internal position.

**Safety and validation status:** The 2026-10-29 run logged implausible 100% readings
within seconds, so its progress and completion statistics are not a reliable evaluation.
A short follow-up on the installed game tested resets, OS-level key timing and five-attempt
checkpoint evaluations. Actual game-frame timing, completion detection and generalization remain
unverified; the short training experiment regressed. Do not start a long run on this evidence.

This is an experimental desktop integration. The first goal is a reliable loop:

1. detect the real game window,
2. capture the visible game,
3. press jump,
4. estimate progress from the on-screen bar,
5. detect a death/restart,
6. reset to a playable attempt,
7. produce a model you can watch and evaluate.

It is **not** a claim that 50,000 timesteps will beat a level. Geometry Dash needs precise timing;
progress depends on the level, reward design, capture rate, and how much you train.

## Requirements and supported surface

- Windows desktop session (not a headless CI machine).
- Geometry Dash running in a visible window.
- The game's normal **Auto-Retry** setting enabled.
- `mss` and `pyautogui` installed from `requirements.txt`.
- Primary monitor, fullscreen or borderless. The default config captures the whole primary
  monitor and detects the progress bar automatically.
- One GameMind process controlling the game at a time. `n_envs` must be 1; parallel copies would
  send conflicting keys to one window.

The adapter refuses to run unless `require_window_focus: true`. It focuses the exact
`window_title` before every reset and action, and it releases held keys in a `finally` block and on
close. PyAutoGUI's mouse-corner failsafe remains enabled.

## Dashboard workflow

The dashboard manages this game as a profile with per-level configurations, sessions, and
checkpoints. Open it with `streamlit run dashboard/app.py`, choose **Games → Geometry Dash**, select
or add a level, validate, then train. See [library.md](library.md) for the library layout, level
fields, completion rules, and queue behaviour. The commands below remain the CLI equivalents.

## Manual setup

1. Start Geometry Dash.
2. Open a level and start an attempt. Stereo Madness is a reasonable first level.
3. In Geometry Dash settings, enable **Auto-Retry** (the setting may be called Auto-Retry or
   Auto-Restart depending on version).
4. Use fullscreen or borderless mode. Do not let Discord, Steam, a browser, or a recording overlay
   cover the game. GameMind can focus the game, but it cannot remove another window drawn on top.
5. Run the setup check from the repository root:

   ```bat
   python scripts/test_setup.py
   ```

6. Start training. Leave the game focused; the adapter counts down for `start_delay` seconds:

   ```bat
   python scripts/train.py --config configs/geometry_dash.yaml --timesteps 50000
   ```

   Stop safely with Ctrl+C or by moving the real mouse to a screen corner.

The adapter will attempt to restart the current level at each episode reset. It does not navigate
the main menu or choose levels for you; select a level manually first. If it cannot confirm the
progress bar after reset, it raises a clear error instead of sending keys blindly into a menu.

## How observations and actions work

`configs/geometry_dash.yaml` uses:

- 84×84 grayscale pixels,
- four stacked frames (motion),
- the top 4% cropped out of the policy view (the raw frame is still used by the detector),
- discrete actions: `noop`, `jump`, `hold_jump`, `release_jump`,
- Space as the default jump key,
- configurable jump duration (`duration` on the action binding),
- `frame_skip: 1` by default; raise it to 2–4 if your capture rate is lower than the game's pace.

Every action releases all held keys. `release_jump` is an explicit panic/unstick action; it is
mostly redundant because the adapter never leaves a key held between steps. An optional mouse-click
jump binding is included as a comment in the config. Mouse clicks are sent only after the game
window is focused.

The policy receives only pixels. It never receives `progress_pct`, attempt counts, or other
detector values as observations. Those values are used only for reward/termination and logging.

## Visual progress detection

Geometry Dash shows a white outlined horizontal progress bar near the top of the screen. The
adapter finds that bar automatically by looking for a long, short, high-aspect white component near
the top of the captured frame. It then measures how far the coloured fill extends from the left.

This means:

- progress is an **estimate**, typically within a percentage point or two; it is not OCR and not
  the game's exact percentage,
- a still bar with no fill is progress 0,
- a missing bar means "not confidently in gameplay" (`bar_found: false`), not "progress 0",
- the adapter keeps the last reliable value when the bar is briefly hidden instead of inventing a
  reward,
- implausible progress jumps and changed bar locations are treated as uncertain; rate limits
  are conservative configurable heuristics, not proof that a reading is correct,
- bar-only completion is disabled by default (`allow_bar_completion: false`); enabling it
  awards completion based on the bar alone and requires independent real-game validation.

The progress fill normally uses the player's colour, so it can be orange, blue, green, or another
colour. At reset, the adapter learns the empty bar's colour from the right-hand side and detects
the fill by colour contrast. This is why the shipped config does not assume an orange bar. If a
mod's fill is too close to the empty-bar colour, adjust `fill_color_distance`; the HSV ranges
`progress_fill_hsv_low` / `progress_fill_hsv_high` are a fallback. If the HUD is scaled or moved,
set `progress_region` to the bar's bounding box **in captured-image pixels**, for example
`[608, 12, 703, 30]` for a 1920×1080 full-monitor capture. Set it back to `null` for automatic
detection.

Useful detector fields in `info` and episode logs:

- `progress_pct`: latest reliable estimate,
- `max_progress_pct`: best estimate during the episode,
- `survival_seconds`: wall-clock time since reset,
- `bar_found`: whether the outlined bar was visible,
- `bar_confidence`: currently 1.0 for a detected bar, 0.0 otherwise.

## Death and reset logic

With Auto-Retry enabled, a death makes the game restart the attempt and the progress bar returns to
0%. The adapter treats a drop from at least `death_min_progress` to at most
`reset_progress_max`, sustained for `death_confirm_frames` captures, as a likely death.
Deaths before detectable progress may be missed. A death ends the episode with a negative reward; the next `reset()`
restarts the attempt.

If the bar disappears for `lost_gameplay_frames` consecutive captures (pause menu, focus loss, or a
menu), the episode terminates as `gameplay_lost` rather than spinning forever. Reset then tries the
configured `reset_keys` (Enter) and, if needed, `resume_keys` (Space).

The adapter does not automate every menu. If the game is on a menu or the progress bar cannot be
found, reset fails with an actionable error. That is deliberate: blindly pressing Enter and Space
inside menus is less reliable and less safe.

## Reward shaping

The shipped reward has five components:

| Component | Signal | Value | Purpose |
| --- | --- | --- | --- |
| `level_progress` | `max_progress_pct` | +1 per new percentage point | reward estimated new highs only |
| `death_penalty` | `died` | −5 | discourage dying |
| `level_complete` | `completed` | +50 | disabled until bar-only completion is explicitly enabled |
| `lost_gameplay` | `gameplay_lost` | −1 | discourage losing the playable screen |
| `time_penalty` | constant | −0.01/step | discourage stalling; may also incentivize early termination |

The old survival bonus was removed because it exceeded the time penalty even with no
progress. Progress uses an approximate visual value and can still be wrong; the rate limit
must be calibrated against real recordings. Rewards have not been empirically tuned.
If you change the reward, re-run a short experiment first. A large completion bonus with no
survival/progress shaping often produces a policy that simply waits for rare luck.

## Commands

Train (the config name is `geometry_dash`; `--game geometry_dash` also works):

```bat
python scripts/train.py --config configs/geometry_dash.yaml --timesteps 50000
```

Watch a saved model in the real game window:

```bat
python scripts/play.py --config configs/geometry_dash.yaml --model models/geometry_dash.zip --episodes 3
```

Evaluate without training updates:

```bat
python scripts/evaluate.py --config configs/geometry_dash.yaml --model models/geometry_dash.zip --episodes 20
```

The model metadata sidecar stores the observation/action settings. `play.py` and `evaluate.py`
validate the loaded model against the recreated environment. GameMind does not open a second
screenshot viewer for this adapter, so you watch the actual Geometry Dash window. `--no-render`
still exercises the environment and policy but does not redraw anything; the game itself remains
visible.

On a 20–30 Hz effective loop, 50,000 environment steps can take tens of minutes to a few hours on
CPU, plus PPO update time. Measure your own machine with a short run before starting a long one.

## Dashboard

Select `geometry_dash.yaml` in the Streamlit dashboard. The dashboard exposes:

- observation mode/resolution/stack,
- frame skip,
- jump key, short-press duration, and hold duration (these are written to a temporary run config),
- timesteps, seed, checkpoint/continuation model,
- start/pause/resume/stop,
- reward and episode-length charts,
- `max_progress_pct` and `survival_seconds` charts after episodes complete,
- the training log and evaluation/playback launch.

The dashboard still launches a separate training process, exactly like the CLI. Periodic in-training
evaluation is disabled for this adapter: a second environment would send keys to the same game
window. Use the dashboard's **Evaluate headlessly** button or `evaluate.py` after training stops.

## Experimental settings and measurements

`observation.mode: canny` selects a one-channel edge map; `canny_low` and `canny_high`
(default 50 and 150) configure thresholds. It has not been compared on the real game and
may lose important visual details. In one recorded 84×84 frame, a visible spike was no
longer clearly triangular in either grayscale or Canny; the player was not visible in that
frame, so its preservation could not be assessed. This is not an evaluation of other scenes.
Do not reuse a grayscale checkpoint for Canny, even though their observation shapes match.

For new action spaces, `params: {hold_frames: 4}` (or 5, 6) converts the requested count
using `game_frame_rate` (default 60 Hz). It is a wall-clock approximation: there is no
synchronization with actual game frames, input backend delays add time, and registration
has not been verified. `frame_skip` repeats complete actions, including releases; it does
not hold a button continuously and does not represent actual game frames.

The original run logged 24,769 steps, 935 episodes over 4,049 s: 6.12 steps/s and
13.86 attempts/min. Reported median maximum progress was 3.35%, mean 10.60%, maximum
100%; the latter two are distorted by suspect readings (24 completions in about six
seconds). 911 episodes were labelled deaths; neither true death frequency nor obstacle
locations are validated. First/last 200 reported medians were 3.20%/3.35%, insufficient
evidence of improvement. TensorBoard contained no scalar metrics for PPO entropy.

A desktop-only benchmark without Geometry Dash (30 samples except 20 inferences) measured
MSS RGB capture median/p95 28.79/31.98 ms, grayscale processing 6.97/7.73 ms, bar
search on the desktop 0.84/1.10 ms, and checkpoint inference on zeros 1.13/1.25 ms.
Later live probes measured median capture 22.88 ms and roughly 49 ms per no-op step.
Three consecutive strict resets were read at 0% and no-op attempts terminated near 1.46%;
these limited checks do not establish reliability on later sections. OS keyboard sampling
showed a requested 67 ms/100 ms hold lasting approximately 86/119 ms with the old pause,
then 66.49/99.19 ms with the Geometry Dash key-call pause removed. Both were released;
OS state does not prove how many game frames saw the press or that a jump occurred. [DXcam](https://github.com/ra1nty/DXcam) is a possible Windows capture
alternative, but has not been benchmarked on this installation; no replacement is claimed.

A deterministic five-attempt evaluation of the 20,000-step checkpoint using the revised
configuration gave estimated mean/best maximum progress 3.84%/6.40%. A separate 64-step
smoke experiment initialized from its policy then scored 1.46%/1.46% over five attempts.
This is a regression, not evidence of learning. Evaluation used a 160-step limit; estimates
are not independently verified. Detailed local records are in `logs/gd_baseline_eval_1793357289.json`
and `logs/gd_revised_eval_1793357332.json`; the original model is unchanged.

Episode logs now include summed reward components and status includes recent progress and
survival summaries. Geometry Dash periodic evaluation remains disabled because a second
environment would control the same window. Compare checkpoints in separate, deterministic,
multiple-attempt runs using the same level and action configuration. Never treat the SB3
reward-based best checkpoint as a best-progress checkpoint. Existing Geometry Dash model
outputs are protected from overwrite; use a fresh `--output` for an experiment.

No selectable practice segments or curriculum transfer has been verified. The only supported
stage remains the manually selected real level; no simulated curriculum is claimed.

## Performance notes and limitations

- Capture is desktop-based (`mss`), so it includes whatever is visibly on screen. Other windows or
  overlays that cover the game become part of the observation.
- `mss` capture and OpenCV preprocessing add latency. The policy acts on the most recent completed
  capture; this is asynchronous relative to the game's frame, not a locked game-thread loop.
- Action duration, capture time, and policy inference determine the effective control rate. The
  adapter does not artificially slow the game, but it cannot guarantee a fixed rate.
- The game is fast and timing-sensitive. Frame skipping and a larger action duration reduce decision
  frequency but can make precise jumps harder.
- Auto-Retry is required for the current death detector. Without it, a death screen may not reset
  the progress bar and the adapter may only notice via `gameplay_lost`.
- The progress bar colour/placement can change with levels, mods, or UI scale. Calibrate
  `progress_region` and HSV ranges if `bar_found` is false or progress is obviously wrong.
- The adapter does not support anti-cheat bypass, memory reading, or protected games. Use it only
  with games you are allowed to automate locally.
- 50,000 timesteps is a starting point. Do not expect a skilled Geometry Dash player from the
  default run; measure whether survival time and max progress improve, then train longer or tune
  rewards.

## Automated versus live verification

The repository tests use fake capture/input/window backends so CI never presses keys on a real
machine. Those tests cover window lookup, capture validation, bar reading/calibration, death/reset,
completion, rewards, action release, Gymnasium compliance, and a short PPO run.

A historical manual Windows check used the actual installed Geometry Dash (performed before
these safeguards; its estimated progress figures are unvalidated and are not evidence of skill):

- real window found at **1919×1080**; desktop capture **1920×1080**;
- live frame test detected both an orange and a blue (player-colour) fill after reset calibration;
- real Enter reset, Space resume, jump keys, death detection, and safe key release;
- a **256-step PPO smoke run** (real keys, `CnnPolicy`, 84×84, four-frame stack) completed and saved
  a model;
- 9 real episodes were recorded. Best episode in that tiny run reached **23.58% estimated progress (unverified and potentially spurious)**
  and **5.28s** survival; recent mean episode reward was **−0.50** and mean length **24.6**. This was a
  smoke run with an almost-untrained policy, not evidence of skill.
- deterministic evaluation of that smoke model over 2 episodes reached only **1.46%** max progress
  and **1.89s** mean survival; playback with the real game window also ran without a training update.

Those numbers are intentionally included rather than hidden: 256 steps is far too little to claim
learning. The 50,000-step config is the starting experiment. Re-run evaluation after a longer run and
compare `max_progress_pct`/`survival_seconds` against this baseline before drawing conclusions.
