# Training guide

This document describes the actual configuration fields and behavior implemented by GameMind AI.
For a first run, start with [README.md](../README.md) and `configs/example_game.yaml`.

## What a training command does

```bat
python scripts/train.py --game example_game --timesteps 50000
```

1. Loads `configs/example_game.yaml`, applies CLI overrides, and validates every field.
2. Creates a run under `logs/runs/<run-id>/`, writes a resolved config snapshot, and creates a
   TensorBoard logger.
3. Builds the configured adapter and wrapper stack. For `example_game`, the policy receives four
   stacked 84×84 grayscale frames, not grid coordinates.
4. Creates PPO with one or more vectorized environments.
5. Trains for the requested number of **additional** timesteps, while checkpointing, evaluating, and
   writing status/metric files.
6. Saves `models/<game>.zip` and `models/<game>.metadata.json` (or the path passed with `--output`).

`--timesteps` is not a hardcoded global. It can be changed per run, and on resume it is the number of
additional steps to train.

## Observation settings

```yaml
observation:
  mode: grayscale       # state | rgb | grayscale
  width: 84             # 32..1024, output width
  height: 84            # 32..1024, output height
  stack: 4              # 1..16; state mode requires 1
  normalize: false      # false for CnnPolicy (uint8 pixels)
  crop: null            # optional [top, left, bottom, right] fractions
```

- **state** uses the adapter's native observation. It must be a Box or Dict space supported by the
  selected policy. It intentionally cannot be combined with frame stacking.
- **grayscale** converts RGB frames to one channel, then resizes. `stack: 4` produces
  `(4, height, width)`, the standard Atari-style input.
- **rgb** keeps three channels. With `stack: 4` the resulting `(12, height, width)` image still
  works with Stable-Baselines3's NatureCNN; grayscale stacking is usually more sample-efficient.
- **normalize: true** outputs float32 in `[0, 1]`. The default is uint8 `[0, 255]` because
  Stable-Baselines3's CnnPolicy expects that and normalizes internally. If you use a custom feature
  extractor, normalization may be useful.
- **crop** is applied before resizing. Fractions are removed from each edge, e.g.
  `[0.1, 0.0, 0.15, 0.0]` removes 10% from the top and 15% from the bottom (useful for HUDs).
  Each fraction is at most 0.45; the remaining region must be at least 2×2.

All pixel observations are validated for dtype, channel count, shape, finite values, and declared
range. The wrapper stack produces the same shape on reset and step; a mismatch is an error, not a
silent fallback.

### Frame skipping vs action duration

- `actions.frame_skip: N` repeats the selected game action for N internal ticks and sums rewards. It
  works for Gymnasium and the example game. For desktop games, each internal tick also holds the
  input for `duration` seconds, so total control time is approximately `N × duration`.
- `actions.duration` is the number of seconds desktop keys are held for one action. It has no effect
  on Gymnasium or `example_game`, which advance one game tick per action.
- The generic action-repeat wrapper stops early if an inner step terminates or truncates, and records
  `info["action_repeats"]`.

## Action configuration

Discrete bindings are indexed in the order listed. Each binding can include keyboard keys, a mouse
click, relative mouse movement, a per-action duration, and arbitrary adapter-specific params.

```yaml
actions:
  frame_skip: 1
  duration: 0.08
  max_duration: 0.5
  bindings:
    - name: noop
      keys: []
    - name: left
      keys: ["a"]
    - name: right_jump
      keys: ["d", "space"]
    - name: interact
      keys: ["e"]
    # optional mouse examples
    - name: click
      mouse_click: ["left"]
    - name: look_right
      mouse_move: [80, 0]
```

- Action names must be unique; a binding cannot list the same key twice.
- Duration must be between 0 and `max_duration`.
- PyAutoGUI key names are validated when the desktop adapter connects (`a`, `space`, `left`,
  `enter`, `f1`, ...).
- For `example_game`, each binding supplies `params: {move: [dx, dy]}`. The default config lists
  five moves; you can change the order and therefore the policy's action indices.
- For Gymnasium adapters, the underlying environment's action space is authoritative; bindings are
  not needed. The built-in PPO/A2C configuration supports Discrete and Box action spaces.
- Custom adapters can use a Box action space for continuous control. The framework does not impose
  the discrete binding system on them. Document the meaning of each continuous dimension in your
  adapter and choose a compatible policy.

## Reward configuration

Reward components are additive. Each component is evaluated every step and the final total is
optionally clipped by `reward.clip` (`null`, a positive number for `[-n, n]`, or `[low, high]`).

| Type | Required fields | Purpose |
| --- | --- | --- |
| `native` | `scale` (default 1) | Use the Gymnasium environment's returned reward. |
| `constant` | `value` | Fixed per-step reward, usually negative for a time penalty. |
| `signal` | `key`; optional `scale`, `default` | Multiply a named numeric adapter signal. |
| `event` | `key`; optional `value` | Fire `value` when a signal is positive on that step. |
| `progress` | `key`; optional `scale`, `mode`, `reset_on` | Reward movement of a signal. `mode: decrease` rewards a decreasing distance. |
| `survival` | optional `value`, `key` | Reward each step while a signal (default `alive`) is positive, or if the signal is absent. |
| `template` | `template`; optional region/threshold/scale/once_per_episode/cooldown | Reward a visual template match in the raw frame. |
| `custom` | `path: module:function`; optional `params` | Call user code with `(context, **params)` and return a number. |

`name` is optional for every component; it appears in `info["reward_breakdown"]`.

### Signals provided by the built-in example game

- `distance`: current Manhattan distance to the target.
- `distance_delta`: old distance minus new distance for the step.
- `target_collected`: 1 on the collection step, otherwise 0.
- `wall_collision`: 1 when a wall blocked movement, otherwise 0.
- `targets_collected`, `wall_hits`: episode counters.
- `alive`: always 1.

`info["success"]` is also set to true when `terminate_on_collect` ends the episode successfully; it
is informational rather than a reward signal.

The shipped reward uses progress toward the target, a collection bonus, a wall penalty, and a small
time penalty. The `reset_on: target_collected` rule avoids rewarding the artificial jump in distance
when a target is collected and a new one appears.

The `example_game` preset is a fixed 5×5 first puzzle: `start: [2, 4]`, `target: [2, 0]`, one
obstacle at `[2, 2]`, and `terminate_on_collect: true`. The yellow beam is rendered into the frame,
not passed as state. `show_compass: false` removes it for a harder visual task. Omitting `start`,
`target`, and `obstacles` gives random placement; `example_game_random.yaml` is the included harder
variant. Fixed layouts are training puzzles, not evidence that a policy generalizes to new levels.

### Reward shaping advice

- Start sparse and honest: success/failure plus a small time cost. Add progress shaping only when the
  agent would otherwise never see a success.
- Test the reward before a long run. Use `info["reward_breakdown"]`, `episodes.jsonl`, and short
  manual play.
- Reward scale matters. Huge shaping terms drown out the real objective; tiny terms vanish among
  penalties.
- Penalizing "alive" time and rewarding survival can cancel out. Know what each component sums to.
- Reward hacking is real. An agent may learn to spin, collect one easy item, or exploit a visual
  detector. Watch it play and change the configuration.

### Custom reward functions

```yaml
reward:
  components:
    - type: custom
      path: "my_game.rewards:score_bonus"
      params: {threshold: 10.0}
```

```python
def score_bonus(context, threshold):
    score = context.signal("score")  # raises if the adapter did not supply it
    return 1.0 if score >= threshold else 0.0
```

The function must return a finite number. Exceptions are reported with the component name.

## Optional template-based termination

Screenshot games rarely expose a reliable game-over signal. You can stop an episode after a fixed
template appears consecutively:

```yaml
episode:
  max_steps: 300
  terminate_on:
    type: template
    template: assets/game_over.png
    region: [280, 120, 240, 80]   # optional, absolute pixels in the raw captured frame
    threshold: 0.88
    consecutive: 2
```

This is normalized template matching, not OCR. It requires a stable screenshot of the exact banner
at the same resolution and UI state. A missing template file fails before training. A time limit is
still required: no generic screenshot adapter can know when an unknown game ends.

## Episode limits

`episode.max_steps` maps to truncation, not termination. Terminal rewards and bootstrapping are
handled according to Stable-Baselines3's standard vector-env behavior. The desktop adapter requires a
finite limit so an experiment can never send keyboard input indefinitely. `example_game` also uses
its own `max_steps` when no config limit is set. With `terminate_on_collect: true`, reaching the
target is a real success termination and is reported as `info["success"]`; the time limit remains a
truncation.

## Algorithms and policies

- `algorithm: ppo` is the default and recommended starting point.
- `algorithm: a2c` is available for lighter experiments. PPO-only fields such as epochs and clipping
  are ignored by A2C where its constructor does not accept them.
- `algorithm: maskable_ppo` is available only when `sb3-contrib` is installed and the adapter exposes
  `action_masks()`.
- The policy is selected automatically: CnnPolicy for pixel/3D image observations, MlpPolicy for
  vectors, MultiInputPolicy for Dict observations. You can set `training.policy` explicitly; invalid
  combinations fail with an explanation.

PPO requires `batch_size` to divide `n_steps × n_envs`. GameMind validates this before constructing
the model. For example, `n_steps: 256`, `n_envs: 4`, `batch_size: 128` is valid (1024 / 128 = 8).

Useful PPO fields:

| Field | Meaning | Default |
| --- | --- | --- |
| `learning_rate` | optimizer step size | 3e-4 |
| `gamma` | discount factor; higher values care more about future reward | 0.99 |
| `n_steps` | rollout steps per environment before each update | 512 |
| `batch_size` | minibatch size for the update | 128 |
| `n_epochs` | passes over each rollout | 10 |
| `gae_lambda` | GAE smoothing | 0.95 |
| `clip_range` | PPO policy-change clipping | 0.2 |
| `ent_coef` | entropy bonus; small positive values encourage exploration | 0.0 |
| `n_envs` | vectorized environment replicas | 4 |
| `device` | `auto`, `cpu`, `cuda`, or `mps` | auto |

Advanced constructor arguments can go under `training.extra_params`, for example:

```yaml
training:
  extra_params:
    target_kl: 0.03
    policy_kwargs:
      net_arch: [128, 128]
```

GameMind validates that each extra key matches the selected algorithm's constructor and refuses to
override managed fields.

## Vectorization

`n_envs > 1` uses a `DummyVecEnv` by default (in-process). `subproc: true` can use
`SubprocVecEnv` for picklable Gymnasium/example-game environments on platforms with a proper
`if __name__ == "__main__"` entry point. The desktop adapter is restricted to one environment, and
periodic evaluation is disabled for it, because parallel copies would send conflicting input to the
same game window.

## Runs, checkpoints, and evaluation

- `checkpoint_freq` saves a model + metadata every N environment steps. `0` disables periodic
  checkpoints. Stop checkpoints are always attempted when a stop is requested.
- `eval_freq` is approximately the number of environment steps between evaluations; GameMind divides
  it by `n_envs` when configuring the Stable-Baselines3 callback. `0` disables evaluation.
- `n_eval_episodes` controls evaluation episodes. Evaluation is deterministic by default.
- `save_best: true` writes `checkpoints/<game>/<run-id>/best/best_model.zip` and its metadata.
- Evaluation uses a separate environment; it never trains the model.
- `save_best` has no effect if `eval_freq: 0`.

All paths are under the project unless absolute paths are supplied. Generated models/checkpoints/logs
are Git-ignored.

## Pause, graceful stop, and resume

The training callback checks `PAUSE` and `STOP` files in the run directory. Pause is cooperative: it
takes effect at the next callback boundary. Stop saves a checkpoint and returns from `learn`.
`Ctrl+C` is handled separately and attempts to save an interrupted bundle.

Resume from a normal model or checkpoint:

```bat
python scripts/train.py --game example_game --model checkpoints/example_game/<run-id>/checkpoint_20000_steps.zip --timesteps 30000 --output models/example_game_continued.zip
```

The sidecar metadata is required for automatic config recreation during playback, but resume also
accepts an explicit `--config`. The model's observation and action spaces are checked against the
new environment before training continues.

## Monitoring

- `logs/runs/<run-id>/status.json`: current state, timesteps, progress, recent reward/length, FPS,
  error, checkpoint/best-model paths.
- `episodes.jsonl`: one JSON object per finished episode.
- `monitor.csv`: Stable-Baselines3's episode monitor.
- TensorBoard: full algorithm metrics. Use `tensorboard --logdir logs/runs`.
- `train.log`: console/file log.
- Dashboard: reads these files; it does not invent metrics.

A reward number only makes sense within the same game and reward configuration. Comparing CartPole
reward (episode length) with LunarLander reward (landing score) is meaningless.

## Time expectations

- CartPole on CPU: often tens of seconds for 50k steps, depending on machine and thread settings.
- Target Chase with a CNN on CPU: roughly several minutes for 50k steps in the tested environment,
  excluding evaluation overhead. The fixed 5×5 first puzzle converged well before 50k steps in the
  tested run; a random-target version needs more experimentation.
- LunarLander: 50k is a short starting run and may produce a poor lander. Try hundreds of thousands
  of steps and tune rewards/hyperparameters.
- External desktop games: much slower per step because each action waits for a real hold duration.
  Long training runs are expected, and sample efficiency is limited by screenshot resolution and
  reward quality.
