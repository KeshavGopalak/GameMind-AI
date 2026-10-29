# Installation

GameMind AI is a source checkout, not a system-wide package. Run the scripts from the repository
root so `gamemind/` and `games/` are importable.

## Supported versions

- Python **3.11, 3.12, or 3.13**. This checkout was tested with **Python 3.13.14 on Windows 11**.
- CPU training works out of the box. CUDA works if your installed PyTorch build supports your GPU.
- A display (SDL/Pygame) is required for live playback. Headless training works for Gymnasium and
  the built-in Target Chase game.
- The desktop adapter additionally needs a desktop session, screen-capture access, and input
  permissions.

The pinned versions in [requirements.txt](../requirements.txt) were installed and tested together.
Newer versions may work, but they are not what this checkout verified.

## Windows

```bat
cd "path\to\GameMind-AI"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/test_setup.py
```

Useful notes:

- If `python` opens the Microsoft Store or resolves to the wrong interpreter, use the full path to
  Python 3.11–3.13 and create the venv with that interpreter.
- `pygame` may print its "Hello from the pygame community" banner. That is harmless.
- If you have an NVIDIA GPU, install the matching PyTorch build from
  <https://pytorch.org/get-started/locally/> after `requirements.txt` if you want CUDA instead of
  the default CPU wheel.
- Long paths and spaces in the checkout path are supported by the scripts because they resolve paths
  relative to their own location. In a terminal, quote paths that contain spaces.

## macOS / Linux

```bash
cd path/to/GameMind-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/test_setup.py
```

Linux desktop input:

- PyAutoGUI/mss generally need an X11 session. Wayland may block synthetic input and screen capture.
- A pure headless server can train and evaluate Gymnasium/Target Chase (`evaluate.py` is headless by
  default; `play.py --no-render` also works), but it cannot open a real game window or reliably send
  desktop input.
- If you use a server without a display, do not run `play.py` with rendering.

macOS desktop input:

- Grant **Accessibility** and **Screen Recording** permissions to the terminal or Python process
  that runs GameMind.
- The game window must be visible; some games ignore synthetic events unless they are frontmost.

## LunarLander / Box2D

`requirements.txt` installs `gymnasium[box2d]`. If Box2D fails to install on your platform, install
the platform build prerequisites and retry:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

CartPole and `example_game` do not need Box2D. If LunarLander is unavailable, the rest of the
framework still works.

## Optional: MaskablePPO

Action masking is only genuinely useful when an adapter exposes invalid actions (for example, illegal
build orders or moves). It is not enabled by default:

```bash
pip install sb3-contrib
```

Then set `training.algorithm: maskable_ppo` and implement `action_masks()` on the adapter. Without
that method, GameMind stops with a clear error rather than pretending masking works.

## Optional: TensorBoard

TensorBoard is included in `requirements.txt`. Start it from the project root:

```bat
tensorboard --logdir logs/runs
```

Open the URL it prints (normally <http://localhost:6006>).

## Verify the installation

```bat
python scripts/test_setup.py
```

The check performs real imports, parses every shipped config, resets and steps CartPole and
LunarLander, renders a Pygame surface off-screen, builds the pixel game environment, tests write
access to `models/`, `checkpoints/`, and `logs/`, and reports the PyTorch device. It does not send
desktop input.

## Common installation failures

| Symptom | Likely cause / fix |
| --- | --- |
| `ModuleNotFoundError: gamemind` | You ran a script from another directory or used a different Python. Run `python scripts/train.py ...` from the project root with the venv active. |
| `ModuleNotFoundError: box2d` | Install `gymnasium[box2d]` (and build tools/SWIG on platforms without a wheel), or use CartPole/example_game. |
| `pygame.error: No available video device` | You are on a headless server. Train headless; live windows require a display/SDL. |
| `pyautogui` import or focus errors | Desktop session/permissions are missing. The desktop adapter is optional; Gymnasium and the built-in game do not need PyAutoGUI. |
| Torch install is slow or huge | The CPU wheel is still large. CUDA builds are larger. Use the CPU wheel for the first run. |
| Streamlit port already in use | `streamlit run dashboard/app.py --server.port 8502` |
