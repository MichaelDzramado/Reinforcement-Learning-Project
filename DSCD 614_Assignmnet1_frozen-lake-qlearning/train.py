"""Training pipeline for Q-Learning on the custom Frozen Lake environment."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from agent import AgentConfig, QLearningAgent
from environment import FrozenLakeEnv
from visualize import moving_average, plot_policy, plot_training_curves, plot_value_heatmap, print_policy_grid


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 20_000
    max_steps_per_episode: int = 200
    moving_average_window: int = 100
    output_dir: str = "results"
    seed: int = 42


def train(config: TrainingConfig | None = None, agent_config: AgentConfig | None = None) -> QLearningAgent:
    """Train a Q-Learning agent and export all experiment artifacts."""
    training_config = config or TrainingConfig()
    np.random.seed(training_config.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    env = FrozenLakeEnv(max_steps=training_config.max_steps_per_episode)
    agent = QLearningAgent(env.n_states, env.n_actions, agent_config or AgentConfig(seed=training_config.seed))
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, List[float]] = {
        "episode": [],
        "episode_reward": [],
        "success": [],
        "cumulative_reward": [],
        "moving_average_reward": [],
        "epsilon": [],
        "episode_length": [],
    }

    cumulative_reward = 0.0
    rewards: List[float] = []
    rolling_reward_sum = 0.0

    for episode in range(1, training_config.episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        success = 0

        for step in range(1, training_config.max_steps_per_episode + 1):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                success = int(bool(info["success"]))
                break

        agent.decay_epsilon()
        cumulative_reward += episode_reward
        rewards.append(episode_reward)
        rolling_reward_sum += episode_reward
        if len(rewards) > training_config.moving_average_window:
            rolling_reward_sum -= rewards[-training_config.moving_average_window - 1]
        current_window = min(len(rewards), training_config.moving_average_window)
        current_moving_average = rolling_reward_sum / current_window

        metrics["episode"].append(float(episode))
        metrics["episode_reward"].append(float(episode_reward))
        metrics["success"].append(float(success))
        metrics["cumulative_reward"].append(float(cumulative_reward))
        metrics["moving_average_reward"].append(current_moving_average)
        metrics["epsilon"].append(float(agent.epsilon))
        metrics["episode_length"].append(float(step))

        if episode % 1000 == 0:
            recent_success = np.mean(metrics["success"][-100:]) * 100.0
            logging.info(
                "Episode %s/%s | reward %.3f | recent success %.1f%% | epsilon %.3f",
                episode,
                training_config.episodes,
                episode_reward,
                recent_success,
                agent.epsilon,
            )

    _write_metrics_csv(metrics, output_dir / "training_metrics.csv")
    agent.save_q_table(output_dir / "qtable.csv")

    policy = agent.greedy_policy(env.terminal_states())
    _write_policy_txt(env, policy, output_dir / "policy.txt")
    plot_training_curves(metrics, output_dir)
    plot_value_heatmap(env, agent.state_values(), output_dir)
    plot_policy(env, policy, output_dir)

    logging.info("Final learned policy:\n%s", print_policy_grid(env, policy))
    return agent


def _write_metrics_csv(metrics: Dict[str, List[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        headers = list(metrics.keys())
        writer.writerow(headers)
        for row in zip(*(metrics[header] for header in headers)):
            writer.writerow(row)


def _write_policy_txt(env: FrozenLakeEnv, policy: np.ndarray, path: Path) -> None:
    from visualize import policy_grid

    grid = policy_grid(env, policy, unicode=True)
    path.write_text("\n".join(" ".join(row) for row in grid), encoding="utf-8")


if __name__ == "__main__":
    train()
