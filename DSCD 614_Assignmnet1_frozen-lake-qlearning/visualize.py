"""Visualization utilities for Frozen Lake Q-Learning experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

_mpl_config_dir = Path("results") / "mplconfig"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from environment import ACTION_NAMES, FrozenLakeEnv


ARROWS = {
    0: "<",
    1: "v",
    2: ">",
    3: "^",
    -1: "",
}

UNICODE_ARROWS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
    -1: "",
}


def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """Compute a trailing moving average with a shorter prefix for early episodes."""
    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        return values_array
    output = np.empty_like(values_array)
    for idx in range(len(values_array)):
        start = max(0, idx - window + 1)
        output[idx] = np.mean(values_array[start : idx + 1])
    return output


def success_rate_trend(successes: Sequence[int], window: int = 100) -> np.ndarray:
    """Return rolling success rate in percent."""
    return moving_average(successes, window=window) * 100.0


def policy_grid(env: FrozenLakeEnv, policy: np.ndarray, unicode: bool = True) -> list[list[str]]:
    """Convert a vector of greedy actions into a readable 8x8 policy grid."""
    symbols = UNICODE_ARROWS if unicode else ARROWS
    grid: list[list[str]] = []
    for row_idx, row in enumerate(env.map):
        grid_row: list[str] = []
        for col_idx, tile in enumerate(row):
            state = env.position_to_state((row_idx, col_idx))
            if tile == "H":
                grid_row.append("H")
            elif tile == "G":
                grid_row.append("G")
            else:
                grid_row.append(symbols[int(policy[state])])
        grid.append(grid_row)
    return grid


def print_policy_grid(env: FrozenLakeEnv, policy: np.ndarray) -> str:
    """Return and print the learned policy as a formatted grid."""
    rendered = "\n".join(" ".join(row) for row in policy_grid(env, policy))
    safe_rendered = "\n".join(" ".join(row) for row in policy_grid(env, policy, unicode=False))
    print(safe_rendered)
    return safe_rendered


def plot_training_curves(metrics: dict[str, list[float]], output_dir: str | Path) -> None:
    """Create reward, moving average, success trend, and epsilon plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(metrics["episode_reward"]) + 1)

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes, metrics["episode_reward"], linewidth=0.8, color="#2f6f9f")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward vs Episode")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "reward_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes, metrics["moving_average_reward"], linewidth=1.2, color="#197278")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Moving Average Reward")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "moving_average_reward.png")
    plt.close()

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes, success_rate_trend(metrics["success"], window=100), color="#6a994e")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success Rate (%)")
    plt.title("Success Rate Trend")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "success_rate.png")
    plt.close()

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes, metrics["epsilon"], color="#bc4749")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "epsilon_decay.png")
    plt.close()


def plot_value_heatmap(env: FrozenLakeEnv, state_values: np.ndarray, output_dir: str | Path) -> None:
    """Create a heatmap of learned state values."""
    output = Path(output_dir)
    values = state_values.reshape(env.rows, env.cols)
    plt.figure(figsize=(7, 6), dpi=150)
    image = plt.imshow(values, cmap="viridis")
    plt.colorbar(image, label="State value V(s)")
    plt.title("Learned State Value Heatmap")
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    for row_idx in range(env.rows):
        for col_idx in range(env.cols):
            tile = env.map[row_idx][col_idx]
            label = tile if tile in {"S", "H", "G"} else f"{values[row_idx, col_idx]:.2f}"
            plt.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "value_heatmap.png")
    plt.close()


def plot_policy(env: FrozenLakeEnv, policy: np.ndarray, output_dir: str | Path) -> None:
    """Create a grid visualization of the learned policy."""
    output = Path(output_dir)
    grid = policy_grid(env, policy, unicode=False)
    plt.figure(figsize=(7, 7), dpi=150)
    plt.imshow(np.zeros((env.rows, env.cols)), cmap="Greys", vmin=0, vmax=1)
    plt.title("Learned Greedy Policy")
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(which="major", color="black", linewidth=0.8)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx, row in enumerate(env.map):
        for col_idx, tile in enumerate(row):
            if tile == "H":
                color = "#8f2d56"
                text = "H"
            elif tile == "G":
                color = "#2a9d8f"
                text = "G"
            elif tile == "S":
                color = "#f4a261"
                text = grid[row_idx][col_idx]
            else:
                color = "#edf6f9"
                text = grid[row_idx][col_idx]
            ax.add_patch(plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, color=color, alpha=0.9))
            plt.text(col_idx, row_idx, text, ha="center", va="center", fontsize=16, weight="bold")
    legend_text = "Actions: 0=Left, 1=Down, 2=Right, 3=Up | " + ", ".join(
        f"{idx}={name}" for idx, name in ACTION_NAMES.items()
    )
    plt.xlabel(legend_text)
    plt.tight_layout()
    plt.savefig(output / "policy.png")
    plt.close()
