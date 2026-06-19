"""Evaluation script for a trained Q-Learning agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 100
    max_steps_per_episode: int = 200
    qtable_path: str = "results/qtable.csv"


def evaluate(config: EvaluationConfig | None = None) -> Dict[str, float]:
    """Evaluate the greedy policy over a fixed number of episodes."""
    evaluation_config = config or EvaluationConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    env = FrozenLakeEnv(max_steps=evaluation_config.max_steps_per_episode)
    agent = QLearningAgent(env.n_states, env.n_actions)
    agent.load_q_table(evaluation_config.qtable_path)

    successes = 0
    failures = 0
    rewards: list[float] = []
    successful_steps: list[int] = []

    for _ in range(evaluation_config.episodes):
        state = env.reset()
        episode_reward = 0.0
        reached_goal = False

        for step in range(1, evaluation_config.max_steps_per_episode + 1):
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                reached_goal = bool(info["success"])
                break

        rewards.append(episode_reward)
        if reached_goal:
            successes += 1
            successful_steps.append(step)
        else:
            failures += 1

    results = {
        "success_rate_percent": successes / evaluation_config.episodes * 100.0,
        "average_reward": float(np.mean(rewards)),
        "failures": float(failures),
        "successful_runs": float(successes),
        "average_steps_to_goal": float(np.mean(successful_steps)) if successful_steps else float("nan"),
    }

    result_path = Path("results/evaluation_summary.txt")
    result_path.write_text(
        "\n".join(f"{key}: {value:.4f}" for key, value in results.items()),
        encoding="utf-8",
    )
    for key, value in results.items():
        logging.info("%s: %.4f", key, value)
    return results


if __name__ == "__main__":
    evaluate()
