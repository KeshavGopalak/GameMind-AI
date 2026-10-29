#!/usr/bin/env python
"""Load a trained model and watch it play in a live window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamemind import __version__  # noqa: E402
from gamemind.config import find_config_for_game, load_game_config  # noqa: E402
from gamemind.evaluation import run_episodes  # noqa: E402
from gamemind.training.model_io import config_for_model  # noqa: E402
from gamemind.utils.errors import GameMindError  # noqa: E402
from gamemind.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a trained GameMind AI model in a live visual window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to a trained .zip model.")
    parser.add_argument("--game", default=None, help="Optional game name for validation.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch.")
    parser.add_argument("--seed", type=int, default=None, help="Episode seed; omit for random episodes.")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Playback frame rate; 0 means as fast as possible; omit for adapter default.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic inference.",
    )
    parser.add_argument("--no-render", action="store_true", help="Run without opening a window.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--version", action="version", version=f"GameMind AI {__version__}")
    return parser


def _load_config(args: argparse.Namespace):
    try:
        return config_for_model(
            args.model, config_path=args.config, expected_game=args.game
        )[0]
    except GameMindError:
        # A model created outside GameMind has no sidecar. If the user supplied a
        # game/config, fall back explicitly rather than guessing an environment.
        if args.config:
            config = load_game_config(args.config)
        elif args.game:
            config = load_game_config(find_config_for_game(args.game))
        else:
            raise
        if args.game and config.name != args.game:
            print(
                f"Warning: --game {args.game!r} does not match config game.name {config.name!r}.",
                file=sys.stderr,
            )
        return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.fps is not None and args.fps < 0:
        parser.error("--fps must be >= 0")
    configure_logging(1)
    try:
        config = _load_config(args)
        print(f"GameMind AI — watching {config.name}")
        print(f"Model: {args.model}")
        print("Close the game window or press Escape to stop. Ctrl+C also exits cleanly.")
        result = run_episodes(
            config,
            args.model,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            render=not args.no_render,
            seed=args.seed,
            fps=args.fps,
            device=args.device,
        )
        print("\n" + "=" * 70)
        print(f"Watched {len(result.episodes)} episode(s)")
        print(f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
        print(f"Mean episode length: {result.mean_length:.1f}")
        print(f"Best episode reward: {result.best_reward:.2f}")
        print("=" * 70)
        return 0
    except KeyboardInterrupt:
        print("\nPlayback stopped.", file=sys.stderr)
        return 130
    except GameMindError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        import traceback

        print(f"\nUnexpected failure: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
