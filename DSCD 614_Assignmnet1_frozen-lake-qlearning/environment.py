"""Custom Frozen Lake environment implemented from first principles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


Action = int
State = int
Position = Tuple[int, int]


DEFAULT_MAP: List[str] = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]


ACTION_NAMES: Dict[Action, str] = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
}

ACTION_DELTAS: Dict[Action, Position] = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping used by the custom environment.

    The terminal rewards preserve the main task objective: reaching the goal
    is strongly positive and falling into a hole is strongly negative. The
    small step penalty discourages wandering and helps the tabular learner
    distinguish shorter successful paths from longer ones.
    """

    goal: float = 1.0
    hole: float = -1.0
    step: float = -0.01
    boundary: float = -0.05


class FrozenLakeEnv:
    """Deterministic 8x8 Frozen Lake grid world without external RL libraries."""

    def __init__(
        self,
        lake_map: List[str] | None = None,
        rewards: RewardConfig | None = None,
        max_steps: int = 200,
    ) -> None:
        self.map = lake_map or DEFAULT_MAP
        self.rewards = rewards or RewardConfig()
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTION_DELTAS)
        self.max_steps = max_steps
        self.start_position = self._find_tile("S")
        self.goal_position = self._find_tile("G")
        self.agent_position = self.start_position
        self.steps_taken = 0

    def reset(self) -> State:
        """Reset the environment and return the starting state index."""
        self.agent_position = self.start_position
        self.steps_taken = 0
        return self.get_state()

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, object]]:
        """Apply an action and return next_state, reward, done, and info."""
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Expected one of {list(ACTION_DELTAS)}.")

        self.steps_taken += 1
        row, col = self.agent_position
        d_row, d_col = ACTION_DELTAS[action]
        candidate = (row + d_row, col + d_col)

        hit_boundary = not self._is_inside(candidate)
        if hit_boundary:
            next_position = self.agent_position
            reward = self.rewards.boundary
        else:
            next_position = candidate
            reward = self.rewards.step

        self.agent_position = next_position
        tile = self._tile_at(self.agent_position)
        done = self.is_terminal() or self.steps_taken >= self.max_steps

        if tile == "H":
            reward = self.rewards.hole
        elif tile == "G":
            reward = self.rewards.goal

        info: Dict[str, object] = {
            "position": self.agent_position,
            "tile": tile,
            "hit_boundary": hit_boundary,
            "steps": self.steps_taken,
            "success": tile == "G",
            "truncated": self.steps_taken >= self.max_steps and not self.is_terminal(),
        }
        return self.get_state(), reward, done, info

    def render(self) -> str:
        """Return an ASCII rendering of the current grid."""
        lines: List[str] = []
        for row_idx, row in enumerate(self.map):
            cells = []
            for col_idx, tile in enumerate(row):
                cells.append("A" if (row_idx, col_idx) == self.agent_position else tile)
            lines.append(" ".join(cells))
        rendering = "\n".join(lines)
        print(rendering)
        return rendering

    def get_state(self) -> State:
        """Return the integer state corresponding to the current position."""
        row, col = self.agent_position
        return self.position_to_state((row, col))

    def is_terminal(self) -> bool:
        """Return True when the agent is on a hole or the goal."""
        return self._tile_at(self.agent_position) in {"H", "G"}

    def position_to_state(self, position: Position) -> State:
        """Convert a row-column position to a flattened state index."""
        row, col = position
        return row * self.cols + col

    def state_to_position(self, state: State) -> Position:
        """Convert a flattened state index to a row-column position."""
        return divmod(state, self.cols)

    def terminal_states(self) -> set[State]:
        """Return all hole and goal states."""
        states: set[State] = set()
        for row_idx, row in enumerate(self.map):
            for col_idx, tile in enumerate(row):
                if tile in {"H", "G"}:
                    states.add(self.position_to_state((row_idx, col_idx)))
        return states

    def tile_for_state(self, state: State) -> str:
        """Return the map tile for a state index."""
        return self._tile_at(self.state_to_position(state))

    def _find_tile(self, target: str) -> Position:
        for row_idx, row in enumerate(self.map):
            for col_idx, tile in enumerate(row):
                if tile == target:
                    return row_idx, col_idx
        raise ValueError(f"Map does not contain required tile {target!r}.")

    def _is_inside(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _tile_at(self, position: Position) -> str:
        row, col = position
        return self.map[row][col]
