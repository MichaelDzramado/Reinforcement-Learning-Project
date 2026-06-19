"""Tabular Q-Learning agent for the custom Frozen Lake environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class AgentConfig:
    learning_rate: float = 0.12
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995
    seed: int = 42


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states: int, n_actions: int, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions), dtype=float)
        self.rng = np.random.default_rng(self.config.seed)

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    def select_action(self, state: int, training: bool = True) -> int:
        """Select an action using epsilon-greedy exploration during training."""
        if training and self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Apply Q(s,a) <- Q(s,a) + alpha[r + gamma max Q(s',a') - Q(s,a)]."""
        best_next_value = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.config.discount_factor * best_next_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration while preserving a small non-zero exploration floor."""
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )

    def greedy_policy(self, terminal_states: Iterable[int] | None = None) -> np.ndarray:
        """Return the greedy action for each state."""
        policy = np.argmax(self.q_table, axis=1).astype(int)
        for state in terminal_states or []:
            policy[state] = -1
        return policy

    def state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s,a) for each state."""
        return np.max(self.q_table, axis=1)

    def save_q_table(self, path: str | Path) -> None:
        """Export the learned Q-table as CSV."""
        np.savetxt(path, self.q_table, delimiter=",", fmt="%.8f")

    def load_q_table(self, path: str | Path) -> None:
        """Load a Q-table from CSV."""
        loaded = np.loadtxt(path, delimiter=",")
        if loaded.shape != self.q_table.shape:
            raise ValueError(f"Expected Q-table shape {self.q_table.shape}, got {loaded.shape}.")
        self.q_table = loaded
