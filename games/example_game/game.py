"""Target Chase: a tiny grid game rendered as pixels.

The agent moves on a grid, collects green targets, and bumps into grey walls.
The game exposes legitimate state signals (distance, collection events, wall
collisions) for reward shaping; the policy itself only receives rendered frames.

You can run a random-action demo without GameMind's RL stack:

    python -m games.example_game.game
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gamemind.utils.errors import ConfigError
from gamemind.utils.logging import get_logger

logger = get_logger("example_game")

BACKGROUND = (14, 20, 28)
GRID_LINE = (30, 43, 55)
WALL = (92, 107, 118)
TARGET = (72, 213, 151)
TARGET_RING = (174, 245, 211)
AGENT = (242, 247, 255)
AGENT_INNER = (77, 163, 255)
COMPASS = (255, 214, 0)


@dataclass
class StepSignals:
    """Events and measurements produced by one game tick."""

    distance: int
    distance_delta: int
    target_collected: int
    wall_collision: int

    def as_dict(self) -> dict[str, float]:
        return {
            "distance": float(self.distance),
            "distance_delta": float(self.distance_delta),
            "target_collected": float(self.target_collected),
            "wall_collision": float(self.wall_collision),
        }


class TargetChaseGame:
    """Small, deterministic-when-seeded pixel game with no external assets."""

    def __init__(
        self,
        *,
        grid_size: int = 10,
        cell_size: int = 32,
        n_obstacles: int = 7,
        max_steps: int = 120,
        seed: int = 42,
        fps: int = 15,
        show_compass: bool = True,
        fixed_start: tuple[int, int] | None = None,
        fixed_target: tuple[int, int] | None = None,
        fixed_obstacles: list[tuple[int, int]] | None = None,
        terminate_on_collect: bool = False,
    ) -> None:
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 5 <= grid_size <= 40:
            raise ConfigError("example_game.params.grid_size must be an integer between 5 and 40.")
        if isinstance(cell_size, bool) or not isinstance(cell_size, int) or not 8 <= cell_size <= 64:
            raise ConfigError("example_game.params.cell_size must be an integer between 8 and 64.")
        if isinstance(n_obstacles, bool) or not isinstance(n_obstacles, int) or n_obstacles < 0:
            raise ConfigError("example_game.params.n_obstacles must be a non-negative integer.")
        if n_obstacles > grid_size * grid_size - 2:
            raise ConfigError(
                "There must be at least two free cells for the agent and a target: "
                "reduce n_obstacles or increase grid_size."
            )
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ConfigError("example_game.params.max_steps must be a positive integer.")
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 120:
            raise ConfigError("example_game.params.fps must be an integer between 1 and 120.")
        if not isinstance(show_compass, bool):
            raise ConfigError("example_game.params.show_compass must be true or false.")
        if not isinstance(terminate_on_collect, bool):
            raise ConfigError("example_game.params.terminate_on_collect must be true or false.")
        self.grid_size = grid_size
        self.fixed_start = self._validate_cell(fixed_start, "start")
        self.fixed_target = self._validate_cell(fixed_target, "target")
        if self.fixed_start is not None and self.fixed_start == self.fixed_target:
            raise ConfigError("example_game.params.start and target must be different cells.")
        self.fixed_obstacles = None
        if fixed_obstacles is not None:
            if not isinstance(fixed_obstacles, (list, tuple)):
                raise ConfigError("example_game.params.obstacles must be a list of [x, y] cells.")
            cells = [self._validate_cell(cell, "obstacles") for cell in fixed_obstacles]
            if any(cell is None for cell in cells):
                raise ConfigError("example_game.params.obstacles must not contain null cells.")
            unique = set(cells)
            if len(unique) != len(cells):
                raise ConfigError("example_game.params.obstacles contains duplicate cells.")
            if self.fixed_start is not None and self.fixed_start in unique:
                raise ConfigError("The fixed start cell cannot also be a fixed obstacle.")
            if self.fixed_target is not None and self.fixed_target in unique:
                raise ConfigError("The fixed target cell cannot also be a fixed obstacle.")
            self.fixed_obstacles = list(cells)  # type: ignore[arg-type]
        self.show_compass = show_compass
        self.terminate_on_collect = terminate_on_collect
        self.success = False
        self.cell_size = cell_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.fps = fps
        self._rng = np.random.default_rng(seed)
        self.agent: tuple[int, int] = (0, 0)
        self.target: tuple[int, int] = (0, 0)
        self.obstacles: set[tuple[int, int]] = set()
        self.steps = 0
        self.targets_collected = 0
        self.wall_hits = 0
        self.success = False
        self._target_armed = True
        self.frame: np.ndarray | None = None
        self._surface: Any = None
        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self.window_closed = False
        self._open_display = False
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Game mechanics
    # ------------------------------------------------------------------
    def _validate_cell(self, value: Any, name: str) -> tuple[int, int] | None:
        if value is None:
            return None
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        ):
            raise ConfigError(
                f"example_game.params.{name} must be [x, y] integer grid coordinates or null."
            )
        x, y = int(value[0]), int(value[1])
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ConfigError(
                f"example_game.params.{name} cell ({x}, {y}) is outside the "
                f"{self.grid_size}x{self.grid_size} grid."
            )
        return x, y

    def _all_cells(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)]

    def _place_new_target(self) -> None:
        if self.fixed_target is not None:
            self.target = self.fixed_target
            return
        blocked = set(self.obstacles)
        blocked.add(self.agent)
        candidates = [cell for cell in self._all_cells() if cell not in blocked]
        if not candidates:
            raise RuntimeError("No free cell remains for a target; reduce n_obstacles.")
        self.target = candidates[int(self._rng.integers(0, len(candidates)))]

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset the world and return the initial RGB frame."""
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self._rng = np.random.default_rng(seed)
        cells = self._all_cells()
        blocked_for_start = set(self.fixed_obstacles or ())
        if self.fixed_target is not None:
            blocked_for_start.add(self.fixed_target)
        start_candidates = [cell for cell in cells if cell not in blocked_for_start]
        if not start_candidates:
            raise RuntimeError("No free cell remains for the agent; reduce obstacles.")
        if self.fixed_start is not None:
            self.agent = self.fixed_start
        else:
            self.agent = start_candidates[int(self._rng.integers(0, len(start_candidates)))]
        if self.fixed_obstacles is not None:
            self.obstacles = set(self.fixed_obstacles)
        else:
            remaining = [
                cell
                for cell in cells
                if cell != self.agent and cell != self.fixed_target
            ]
            obstacle_count = min(self.n_obstacles, max(0, len(remaining) - 1))
            chosen = self._rng.choice(len(remaining), size=obstacle_count, replace=False)
            self.obstacles = {remaining[int(index)] for index in np.asarray(chosen).reshape(-1)}
        self._place_new_target()
        self.steps = 0
        self.targets_collected = 0
        self.wall_hits = 0
        self.success = False
        self._target_armed = True
        self.window_closed = False
        self.frame = self.render_frame()
        return self.frame.copy()

    def move(self, dx: int, dy: int) -> StepSignals:
        """Move the agent by one grid cell in each axis (values in -1..1)."""
        if self.frame is None:
            raise RuntimeError("Call reset() before move().")
        if isinstance(dx, bool) or isinstance(dy, bool) or not isinstance(dx, int) or not isinstance(dy, int):
            raise ValueError(f"dx and dy must be integers, got {dx!r}, {dy!r}.")
        if abs(dx) > 1 or abs(dy) > 1:
            raise ValueError(f"dx and dy must be in -1..1, got {dx}, {dy}.")
        old_distance = self.distance_to_target()
        old = self.agent
        target_cell = (old[0] + dx, old[1] + dy)
        wall_collision = 0
        if not (0 <= target_cell[0] < self.grid_size and 0 <= target_cell[1] < self.grid_size):
            wall_collision = 1
        elif target_cell in self.obstacles:
            wall_collision = 1
        else:
            self.agent = target_cell

        target_collected = 0
        if self.agent != self.target:
            # Leaving a fixed target re-arms it; entering it again can collect.
            self._target_armed = True
        elif self._target_armed:
            target_collected = 1
            self._target_armed = False
            self.targets_collected += 1
            if self.terminate_on_collect:
                self.success = True
            else:
                self._place_new_target()

        self.steps += 1
        if wall_collision:
            self.wall_hits += 1
        distance = self.distance_to_target()
        self.frame = self.render_frame()
        return StepSignals(
            distance=distance,
            distance_delta=old_distance - distance,
            target_collected=target_collected,
            wall_collision=wall_collision,
        )

    def distance_to_target(self) -> int:
        return abs(self.agent[0] - self.target[0]) + abs(self.agent[1] - self.target[1])

    @property
    def done(self) -> bool:
        return self.success or self.steps >= self.max_steps

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _get_pygame(self):
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:  # pragma: no cover - installation dependent
                raise ConfigError(
                    "The example game needs pygame. Install it with: pip install pygame"
                ) from exc
            self._pygame = pygame
        return self._pygame

    def render_frame(self) -> np.ndarray:
        """Draw the game to an off-screen RGB array (safe for headless training)."""
        pygame = self._get_pygame()
        size = self.grid_size * self.cell_size
        if self._surface is None:
            self._surface = pygame.Surface((size, size))
        surface = self._surface
        surface.fill(BACKGROUND)
        cell = self.cell_size

        for index in range(self.grid_size + 1):
            coordinate = min(size - 1, index * cell)
            pygame.draw.line(surface, GRID_LINE, (coordinate, 0), (coordinate, size), 1)
            pygame.draw.line(surface, GRID_LINE, (0, coordinate), (size, coordinate), 1)

        for x, y in self.obstacles:
            pygame.draw.rect(
                surface,
                WALL,
                pygame.Rect(x * cell + 5, y * cell + 5, cell - 10, cell - 10),
                border_radius=max(2, cell // 8),
            )
            pygame.draw.rect(
                surface,
                (63, 76, 86),
                pygame.Rect(x * cell + 10, y * cell + 10, cell - 20, max(2, cell - 20)),
                width=1,
                border_radius=max(1, cell // 10),
            )

        center = (self.target[0] * cell + cell // 2, self.target[1] * cell + cell // 2)
        # Keep the target and agent visually large after downscaling to the policy input.
        pygame.draw.circle(surface, TARGET_RING, center, max(4, int(cell * 0.38)))
        pygame.draw.circle(surface, TARGET, center, max(2, int(cell * 0.24)))

        inset = max(2, cell // 10)
        agent_rect = pygame.Rect(
            self.agent[0] * cell + inset,
            self.agent[1] * cell + inset,
            cell - 2 * inset,
            cell - 2 * inset,
        )
        pygame.draw.rect(surface, AGENT, agent_rect, border_radius=max(2, cell // 6))
        inner = agent_rect.inflate(-max(2, cell // 3), -max(2, cell // 3))
        pygame.draw.rect(surface, AGENT_INNER, inner, border_radius=max(1, cell // 8))

        if self.show_compass and (self.agent != self.target):
            # A deliberate visual HUD affordance: a yellow arrow points along the
            # dominant axis toward the target. It is still only pixels to the agent;
            # no hidden coordinates are added to the observation.
            dx = self.target[0] - self.agent[0]
            dy = self.target[1] - self.agent[1]
            if abs(dx) >= abs(dy) and dx != 0:
                direction = (1.0 if dx > 0 else -1.0, 0.0)
            elif dy != 0:
                direction = (0.0, 1.0 if dy > 0 else -1.0)
            else:
                direction = (0.0, 0.0)
            if direction != (0.0, 0.0):
                center_float = (
                    self.agent[0] * cell + cell / 2,
                    self.agent[1] * cell + cell / 2,
                )
                vx, vy = direction
                px, py = -vy, vx
                # A large, high-contrast beam. It has to survive the downscale to
                # 84x84: a tiny arrow becomes a couple of pixels and the CNN may
                # never learn to use it.
                tip = (center_float[0] + vx * cell * 0.92, center_float[1] + vy * cell * 0.92)
                base = (center_float[0] + vx * cell * 0.18, center_float[1] + vy * cell * 0.18)
                left = (base[0] + px * cell * 0.17, base[1] + py * cell * 0.17)
                right = (base[0] - px * cell * 0.17, base[1] - py * cell * 0.17)
                pygame.draw.line(
                    surface,
                    COMPASS,
                    (int(center_float[0]), int(center_float[1])),
                    (int(base[0]), int(base[1])),
                    width=max(3, cell // 8),
                )
                pygame.draw.polygon(surface, COMPASS, [tip, left, right])
                pygame.draw.circle(surface, COMPASS, (int(center_float[0]), int(center_float[1])), max(2, cell // 12))

        raw = pygame.image.tostring(surface, "RGB")
        return np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3).copy()

    def display_frame(self, frame: np.ndarray | None = None) -> bool:
        """Show the current frame in a pygame window. Returns False when closed."""
        pygame = self._get_pygame()
        if frame is None:
            frame = self.frame
        if frame is None:
            raise RuntimeError("No frame is available to display.")
        if not self._open_display:
            pygame.display.init()
            pygame.display.set_caption("GameMind AI — Target Chase")
            self._screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            self._clock = pygame.time.Clock()
            self._open_display = True
        surface = self._surface
        if surface is None:
            surface = pygame.Surface((frame.shape[1], frame.shape[0]))
            self._surface = surface
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.window_closed = True
        if self.window_closed:
            return False
        if self._screen is not None:
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
        return True

    def close(self) -> None:
        if self._open_display and self._pygame is not None:
            try:
                self._pygame.display.quit()
                self._pygame.quit()
            except Exception:  # pragma: no cover - shutdown best effort
                logger.debug("pygame shutdown failed", exc_info=True)
        self._open_display = False
        self._screen = None
        self._clock = None
        self.window_closed = True


def _random_demo() -> None:  # pragma: no cover - interactive demo
    """Run the game with random actions so you can inspect it without training."""
    import time

    game = TargetChaseGame(fps=20)
    actions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    print("Target Chase demo — random actions. Press Escape or close the window to stop.")
    rng = np.random.default_rng()
    while True:
        dx, dy = actions[int(rng.integers(0, len(actions)))]
        signals = game.move(dx, dy)
        if signals.target_collected:
            print(f"Target collected! total={game.targets_collected}")
        if not game.display_frame():
            break
        if game.done:
            game.reset()
        time.sleep(0.01)
    game.close()


if __name__ == "__main__":  # pragma: no cover
    _random_demo()
