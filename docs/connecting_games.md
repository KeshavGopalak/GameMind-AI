# Connecting a new game

The framework is designed so the training engine never needs to know game-specific details. A game
is added in three places:

1. a configuration under `configs/`,
2. an adapter (unless an existing adapter already fits),
3. optionally, game-specific reward signals or visual detectors.

Do not modify `gamemind/training/` to add a game. If you find yourself doing that, the adapter
contract is usually missing a signal or a lifecycle hook.

## Choose the easiest integration method

| Game situation | Integration | Code needed? | What works immediately |
| --- | --- | --- | --- |
| A Gymnasium environment exists (CartPole, Atari, custom Gym env) | `adapter: gymnasium` | No, just YAML | State or rendered pixels; native reward |
| A Python game/API can expose state and step functions | Custom `GameAdapter` | Yes, small class | Any observation/reward the API exposes |
| An emulator has a Python API (e.g. a project-specific wrapper) | Custom adapter | Yes | RAM/state observations if the API exposes them |
| A game has a socket or HTTP control API | Custom adapter | Yes | Depending on the API's observations and score |
| The game only accepts desktop keyboard/mouse input and screenshots | `adapter: desktop` | Config + reward/termination design; sometimes custom code | Pixels, configured controls, time limits, template detectors |
| Geometry Dash (specialised real-window integration) | `adapter: geometry_dash` | Use the shipped config; optional bar calibration | Pixels, jump keys, approximate visual progress, death/reset with Auto-Retry |
| The game needs memory access, anti-cheat bypass, or a proprietary SDK | Out of scope | — | Not supported; do not attempt to defeat anti-cheat or access-protect a game. |

The desktop adapter is intentionally not advertised as universal. It can send ordinary input and
capture pixels. It cannot read a score, lives, level, or game-over state unless the user supplies a
detector or a custom adapter with access to real game state.

## Path 1: Gymnasium environment (no code)

Copy a config and replace the environment ID:

```yaml
version: 1
game:
  name: my_gym_game
  adapter: gymnasium
  params:
    env_id: MyGame-v0
    kwargs: {}
    max_episode_steps: null
  seed: 42
observation:
  mode: state           # or grayscale/rgb to render pixels
  width: 84
  height: 84
  stack: 1
reward:
  components:
    - type: native
training:
  algorithm: ppo
  timesteps: 50000
  n_envs: 4
  n_steps: 512
  batch_size: 128
```

Then:

```bat
python scripts/train.py --config configs/my_gym_game.yaml --timesteps 50000
```

Notes:

- The environment must follow the Gymnasium five-value step API. Legacy four-value Gym APIs are not
  supported.
- If `game.params.kwargs` contains constructor keyword arguments that conflict with render/max-step
  handling, prefer the dedicated `max_episode_steps` field.
- Pixel mode calls `env.render()` and requires it to return an RGB uint8 frame. If the env does not
  support `render_mode="rgb_array"`, use state observations or a custom adapter.
- The native environment reward is used only if you configure a `native` reward component. GameMind
  does not silently add it.

## Path 2: Python API adapter

Create a package under `games/`:

```text
games/my_game/
├── __init__.py
├── adapter.py
└── game_api.py       # your existing game/API code
```

A minimal real adapter looks like this (replace the imaginary API calls with your game's methods):

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamemind.adapters.base import GameAdapter


class MyGameAdapter(GameAdapter):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config, render_mode=None):
        super().__init__(config, render_mode=render_mode)
        self.action_space = spaces.Discrete(4)          # e.g. no-op, left, right, jump
        self._declare_observation_space(
            spaces.Box(0, 255, (240, 320, 3), dtype=np.uint8)
        )
        self._game = None

    def connect(self):
        if self._connected:
            return
        import my_game  # your API/library
        self._game = my_game.launch()                   # open window/process
        self._connected = True

    def _reset(self, *, seed=None, options=None):
        self._game.restart(seed=seed)
        return self._game.screenshot(), {"level": self._game.level}

    def _apply_action(self, action):
        self._game.send_action(int(action))

    def observe(self):
        return self._game.screenshot()                    # RGB uint8 HxWx3

    def is_episode_over(self):
        return bool(self._game.is_dead_or_won)           # real API state, not a guess

    def _reward_signals(self):
        return {
            "score": float(self._game.score),
            "collected": 1.0 if self._game.collected_this_step else 0.0,
            "alive": 0.0 if self._game.is_dead_or_won else 1.0,
        }

    def render(self):
        frame = self._game.screenshot()
        if self.render_mode == "human":
            self._game.present_window()                  # or use FrameViewer
        return frame

    def _close_resources(self):
        if self._game is not None:
            self._game.close()
            self._game = None
        self._connected = False
```

Important details:

- `_reset` must return `(raw_observation, info)` and set up all per-episode game state.
- `_apply_action` advances exactly one game tick/action. Do not try to implement the outer frame
  skip yourself; use `actions.frame_skip` in YAML and let the wrapper repeat the action.
- `observe()` must return only what you selected. If `observation.mode` is pixel, return an RGB uint8
  array. Do not fall back to a hidden state vector when pixels were requested.
- `is_episode_over()` should represent a real terminal condition (death, win, game-reported loss).
  Time limits belong in `episode.max_steps` and are reported as truncation.
- `_reward_signals()` should expose only legitimate measurements/events your game API provides.
  Reward components can use them; the policy does not see them unless they are in the observation.
- Release resources in `_close_resources`; the base `close()` calls it exactly once and is safe to
  call repeatedly.
- If you need a custom reset gesture or game launch, do it in `connect()`/`_reset`, not in the
  trainer.

Point YAML at the class with a module path:

```yaml
game:
  name: my_game
  adapter: "games.my_game.adapter:MyGameAdapter"
  params: {}
observation:
  mode: grayscale
  width: 84
  height: 84
  stack: 4
actions:
  frame_skip: 1
  bindings: []          # this adapter uses its own Discrete(4) space
reward:
  components:
    - type: signal
      key: collected
      scale: 10.0
    - type: signal
      key: score
      scale: 0.1
    - type: survival
      key: alive
      value: 0.01
```

Run from the project root so `games.my_game` is importable:

```bat
python scripts/train.py --config configs/my_game.yaml --timesteps 1000
```

For a first smoke test, use `n_envs: 1`, a tiny timestep count, and `eval_freq: 0`. Confirm the
adapter reset/steps and that a frame/state has the declared shape before training for hours.

## Path 3: Desktop screenshot/keyboard adapter

Copy the template:

```bat
copy configs\desktop_game.example.yaml configs\my_desktop_game.yaml
```

Edit these fields:

1. **`window_title`** and keep **`require_window_focus: true`**. Input is refused until a window with
   that title is active. This prevents keys from going to the wrong application.
2. **`region`** is `[x, y, width, height]` in physical screen pixels. It must contain the game's
   playable area, not the desktop or an unrelated window. `monitor: 1` means the primary monitor when
   `region` is null.
3. **`reset_keys`** should restart the game (for example a menu shortcut). If the game cannot be
   reset by keys, you must reset it yourself between episodes or write a custom adapter. There is no
   universal restart key.
4. **`actions.bindings`** must list only controls the game supports. A no-op is strongly recommended.
   `duration` controls how long keys are held; start around 0.05–0.15 seconds and observe the game's
   response.
5. **`episode.max_steps`** is mandatory. It is the only guaranteed episode boundary for a generic
   desktop game.
6. **Rewards** must encode actual progress. A constant time penalty alone teaches nothing.
7. **`episode.terminate_on`** optionally detects a fixed game-over screen.

Example reward for a game that shows a stable "level complete" badge:

```yaml
reward:
  components:
    - type: constant
      value: -0.01
    - type: template
      name: level_complete
      template: assets/level_complete.png
      region: [250, 80, 300, 80]
      threshold: 0.86
      scale: 10.0
      once_per_episode: true
```

### Screenshot limitations (read this before expecting learning)

- A score number in a HUD is not a reward signal unless you detect/read it. The built-in template
  matcher does not perform OCR.
- A fixed "Game Over" banner can be detected by template matching. Animated banners, resolution
  changes, lighting, UI scaling, and localisation can break it.
- A static screen may mean game over, a pause menu, or simply nothing happening. Do not assume.
- A game that continues after "death" without a menu may not have any observable terminal state.
- If the game/emulator exposes memory, a Python API, a socket, or a score file, use a custom adapter
  and read the real value. That is more reliable than pixel guessing.
- Mouse actions are relative movements and clicks on the current cursor position. There is no
  absolute targeting support in the generic adapter; implement that in a custom adapter if needed.

### Safety behavior

- Held keys are released after each action and on close.
- If configured focus is lost, the adapter raises rather than sending keys to an unknown window.
- PyAutoGUI's corner failsafe is enabled; moving the real mouse to a screen corner aborts input.
- `start_delay` gives you time to focus/close the game before the first input.
- The desktop adapter runs one environment only and periodic evaluation is disabled, because two
  environments would fight over the same window.
- Always test in a windowed game with a finite `max_steps` and watch the first episode.

### Capturing and testing a region

A safe capture-only check (no inputs):

```python
import mss, numpy as np
with mss.mss() as sct:
    shot = sct.grab({"left": 100, "top": 100, "width": 800, "height": 600})
    print(np.asarray(shot).shape)   # height, width, 4 (BGRA), about 600x800
```

The adapter converts BGRA to RGB. Adjust the region for display scaling (Windows DPI, Retina, etc.).

## Geometry Dash

Geometry Dash uses a dedicated adapter rather than the generic desktop template. It supplies
window lookup, a configurable capture region, progress-bar vision, death/reset logic, and jump
actions. Read [geometry_dash.md](geometry_dash.md) before changing its parameters. The general
desktop safety rules still apply: focus is required, keys are released, Auto-Retry is required for
death detection, and only one environment may control the window.

## Template termination and reward detection

Template matching uses normalized pixel similarity on a grayscale frame. Create the template by
cropping a screenshot to the exact banner/badge, and save it under `assets/` or another directory.
Keep the resolution and UI scale fixed. It is suitable for a static, high-contrast banner, not for
reading dynamic text.

If detection is unreliable, a custom adapter can implement a reliable detector from the game API,
emulator memory, or OCR in user code. GameMind does not silently simulate success.

## Custom reward functions and action masks

- Custom reward: `path: "module:function"`, signature `function(context, **params)`. Use
  `context.signals`, `context.frame`, and `context.info`.
- Custom termination: the built-in detector is only `template`. Override `is_episode_over()` in a
  custom adapter using a real game signal.
- Action masks: implement `action_masks()` on the adapter (and/or expose it through wrappers) only if
  some actions are genuinely invalid. Install `sb3-contrib` and choose `maskable_ppo`; otherwise use
  standard PPO. Masking all actions as valid is not a meaningful use of MaskablePPO.

## Testing a new integration

```bat
python scripts/train.py --config configs/my_game.yaml --timesteps 256 --n-envs 1 --eval-freq 0 --no-tensorboard
python scripts/evaluate.py --config configs/my_game.yaml --model models/my_game.zip --episodes 1
```

Evaluation is headless by default; add `--render` if you want a window. If the adapter is state/pixel
based and Gymnasium-compliant, its raw adapter can be checked with:

```python
from gymnasium.utils.env_checker import check_env
from gamemind.adapters.registry import create_adapter
from gamemind.config import load_game_config

adapter = create_adapter(load_game_config("configs/my_game.yaml"))
adapter.connect()
check_env(adapter, skip_render_check=True)
adapter.close()
```

If `check_env` complains about wrappers, check the raw adapter rather than the wrapped environment.
The pixel game's tests demonstrate this.

## Universal compatibility is not claimed

Some games are not practical with this framework without substantial engineering: games with hidden
state, anti-cheat protection, dynamic UIs, high-resolution 3D visual requirements, or no legal
automation interface. Prefer official APIs or emulator interfaces. Do not use GameMind to bypass
access controls, anti-cheat, or terms of service.
