#!/usr/bin/env python
"""Train a GameMind AI agent from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamemind import __version__  # noqa: E402
from gamemind.config import (  # noqa: E402
    apply_config_overrides,
    find_config_for_game,
    load_game_config,
)
from gamemind.training import Trainer  # noqa: E402
from gamemind.utils.errors import GameMindError  # noqa: E402
from gamemind.utils.paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, safe_name  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PPO/A2C agent on a configured GameMind environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--game", help="Game config name (looks in configs/<name>.yaml).")
    source.add_argument("--config", help="Path to a YAML experiment config.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Additional environment timesteps for this invocation (default from config).",
    )
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "maskable_ppo"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy", default=None, help="MlpPolicy, CnnPolicy, MultiInputPolicy, or null for auto.")
    parser.add_argument("--n-envs", type=int, default=None, dest="n_envs")
    parser.add_argument("--eval-freq", type=int, default=None, help="Approx. env steps between evaluations; 0 disables.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)

    parser.add_argument("--observation-mode", choices=["state", "rgb", "grayscale"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=None, help="Normalize pixel observations to float32 [0,1].")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--action-duration", type=float, default=None)

    parser.add_argument(
        "--model",
        "--resume-from",
        dest="resume_from",
        default=None,
        help="Existing .zip checkpoint to continue training from.",
    )
    parser.add_argument("--output", default=None, help="Destination .zip for the trained model.")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint root directory.")
    parser.add_argument("--log-dir", default=None, help="Log root directory.")
    parser.add_argument("--run-name", default=None, help="Unique run name; its status/logs go under logs/runs/<name>.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")
    parser.add_argument("--version", action="version", version=f"GameMind AI {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.game and not args.config:
        parser.error("Provide --game NAME or --config path/to/game.yaml.")
    try:
        if args.config:
            config = load_game_config(args.config)
            if args.game and safe_name(args.game) != safe_name(config.name):
                parser.error(
                    f"--game {args.game!r} does not match config game.name {config.name!r}. "
                    "Use only --config or make the names consistent."
                )
        else:
            config = load_game_config(find_config_for_game(args.game))
        config = apply_config_overrides(
            config,
            timesteps=args.timesteps,
            algorithm=args.algorithm,
            seed=args.seed,
            observation_mode=args.observation_mode,
            width=args.width,
            height=args.height,
            stack=args.frame_stack,
            normalize=args.normalize,
            frame_skip=args.frame_skip,
            action_duration=args.action_duration,
            policy=args.policy,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            device=args.device,
        )
        if args.no_tensorboard:
            from dataclasses import replace

            config = replace(config, training=replace(config.training, tensorboard=False))
        if args.quiet:
            from dataclasses import replace

            config = replace(config, training=replace(config.training, verbose=0))

        model_path = (
            Path(args.output).expanduser()
            if args.output
            else DEFAULT_MODEL_DIR / f"{safe_name(config.name)}.zip"
        )
        trainer = Trainer(
            config,
            model_path=model_path,
            checkpoint_dir=args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR,
            log_dir=args.log_dir or DEFAULT_LOG_DIR,
            run_name=args.run_name,
            resume_from=args.resume_from,
        )
        print(f"GameMind AI — training {config.name} ({config.training.algorithm.upper()})")
        print(f"Config: {config.source_path or '(in-memory)'}")
        print(f"Run id: {trainer.run_id}")
        print(f"Status: {trainer.status_path}")
        result = trainer.run()
        print("\n" + "=" * 70)
        if result.interrupted:
            print("Training interrupted by Ctrl+C.")
        elif result.stopped:
            print("Training stopped early (STOP file). A checkpoint was saved.")
        elif result.completed:
            print("Training completed.")
        else:
            print(f"Training finished with state: {result.status_path}")
        print(f"Final timesteps: {result.final_timesteps:,}")
        if result.model_path:
            print(f"Model: {result.model_path}")
        if result.best_model_path:
            print(f"Best eval model: {result.best_model_path}")
        metrics = result.metrics or {}
        if metrics.get("ep_rew_mean") is not None:
            print(
                f"Recent episode reward: {metrics['ep_rew_mean']:.3f} | "
                f"recent length: {metrics.get('ep_len_mean', float('nan')):.1f}"
            )
        print(f"Checkpoints: {result.checkpoint_dir}")
        print(f"TensorBoard: tensorboard --logdir {result.run_dir / 'tensorboard'}")
        print("Watch it play:")
        print(f"  python scripts/play.py --model {result.model_path} --episodes 5" if result.model_path else "")
        print("=" * 70)
        if result.interrupted:
            return 130
        if result.stopped:
            return 3
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted before training finished.", file=sys.stderr)
        return 130
    except GameMindError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # unexpected programming/OS error: preserve traceback
        import traceback

        print(f"\nUnexpected failure: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
