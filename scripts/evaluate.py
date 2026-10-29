#!/usr/bin/env python
"""Evaluate a trained GameMind AI model over several episodes without training."""

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
        description="Evaluate a trained model without updating its weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to a trained .zip model.")
    parser.add_argument("--game", default=None, help="Optional game name for validation.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123, help="Base seed for reproducible episodes.")
    parser.add_argument("--fps", type=int, default=None, help="Playback FPS when --render is used.")
    parser.add_argument("--render", action="store_true", help="Show a live window while evaluating.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic.")
    parser.add_argument("--output", default=None, help="Optional JSON file for per-episode results.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--version", action="version", version=f"GameMind AI {__version__}")
    return parser


def _load_config(args: argparse.Namespace):
    try:
        return config_for_model(args.model, config_path=args.config, expected_game=args.game)[0]
    except GameMindError:
        if args.config:
            return load_game_config(args.config)
        if args.game:
            return load_game_config(find_config_for_game(args.game))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    configure_logging(1)
    try:
        config = _load_config(args)
        if args.output:
            Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        result = run_episodes(
            config,
            args.model,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            render=args.render,
            seed=args.seed,
            fps=args.fps,
            device=args.device,
            output_path=Path(args.output).expanduser() if args.output else None,
        )
        print("\n" + "=" * 70)
        print(f"Game: {result.game}")
        print(f"Model: {result.model_path}")
        print(f"Algorithm: {result.algorithm.upper()}")
        print(f"Episodes: {len(result.episodes)}")
        print(f"Mean reward: {result.mean_reward:.3f} ± {result.std_reward:.3f}")
        print(f"Mean length: {result.mean_length:.1f}")
        print(f"Best reward: {result.best_reward:.3f}")
        print("=" * 70)
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation stopped.", file=sys.stderr)
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
