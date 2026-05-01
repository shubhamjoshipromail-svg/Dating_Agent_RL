"""Core tabular RL logic for the Dating Agent demo.

The LLM is not trained in this project. The trained model is the Q-table.
The LLM handles language; this module handles discrete states, actions, the
toy handcoded simulator, and Q-learning.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np


INTERESTS = ["low", "medium", "high"]
STAGES = ["opener", "chat", "date"]
TONES = ["cold", "neutral", "warm"]

ACTIONS = [
    "ask_question",
    "give_compliment",
    "share_story",
    "be_playful",
    "be_direct",
    "suggest_date",
    "slow_down",
    "end_chat",
]

ACTION_GUIDANCE = {
    "ask_question": "Ask an open-ended question to keep the conversation moving.",
    "give_compliment": "Give a specific, light compliment without overdoing it.",
    "share_story": "Share a short personal story that builds rapport.",
    "be_playful": "Use playful banter or gentle teasing.",
    "be_direct": "Use direct but warm escalation.",
    "suggest_date": "Suggest a simple, low-pressure plan to meet.",
    "slow_down": "Reduce pressure and rebuild comfort.",
    "end_chat": "End the conversation politely.",
}

N_STATES = len(INTERESTS) * len(STAGES) * len(TONES)
N_ACTIONS = len(ACTIONS)


def _label_to_id(value: str | int, labels: list[str], name: str) -> int:
    if isinstance(value, str):
        if value not in labels:
            raise ValueError(f"Invalid {name} label {value!r}. Expected one of {labels}.")
        return labels.index(value)

    value = int(value)
    if not 0 <= value < len(labels):
        raise ValueError(f"Invalid {name} index {value}. Expected 0 to {len(labels) - 1}.")
    return value


def state_to_id(interest: str | int, stage: str | int, tone: str | int) -> int:
    interest_id = _label_to_id(interest, INTERESTS, "interest")
    stage_id = _label_to_id(stage, STAGES, "stage")
    tone_id = _label_to_id(tone, TONES, "tone")
    return interest_id * len(STAGES) * len(TONES) + stage_id * len(TONES) + tone_id


def id_to_state(state_id: int) -> tuple[int, int, int]:
    state_id = int(state_id)
    if not 0 <= state_id < N_STATES:
        raise ValueError(f"Invalid state_id {state_id}. Expected 0 to {N_STATES - 1}.")

    interest = state_id // (len(STAGES) * len(TONES))
    remainder = state_id % (len(STAGES) * len(TONES))
    stage = remainder // len(TONES)
    tone = remainder % len(TONES)
    return interest, stage, tone


def describe_state(state_id: int) -> str:
    interest, stage, tone = id_to_state(state_id)
    return f"interest={INTERESTS[interest]}, stage={STAGES[stage]}, tone={TONES[tone]}"


def action_to_id(action_name: str) -> int:
    if action_name not in ACTIONS:
        raise ValueError(f"Invalid action {action_name!r}. Expected one of {ACTIONS}.")
    return ACTIONS.index(action_name)


def compute_reward(next_state_id: int, outcome: str | None = None) -> float:
    """Score a transition from its next state and terminal outcome."""
    if outcome == "date_success":
        return 30.0
    if outcome in {"date_fail", "date_failed", "ghosted", "ended", "ended_by_agent"}:
        return -5.0

    interest, stage, tone = id_to_state(next_state_id)

    interest_score = [-0.40, 0.20, 0.80][interest]
    stage_score = [0.00, 0.30, 0.80][stage]
    tone_score = [-0.30, 0.10, 0.40][tone]
    step_penalty = -0.10

    return float(interest_score + stage_score + tone_score + step_penalty)


def _clamp(value: int, low: int = 0, high: int = 2) -> int:
    return max(low, min(high, int(value)))


def simulator_step_hardcoded(state_id: int, action_id: int) -> tuple[int, float, bool, str | None]:
    """Fast toy simulator. No LLM calls happen here."""
    interest, stage, tone = id_to_state(state_id)
    action_name = ACTIONS[int(action_id)]
    done = False
    outcome = None

    if action_name == "ask_question":
        interest += np.random.choice([0, 1], p=[0.55, 0.45])
        tone += np.random.choice([0, 1], p=[0.65, 0.35])
        if interest >= 1 and tone >= 1:
            stage += np.random.choice([0, 1], p=[0.70, 0.30])

    elif action_name == "give_compliment":
        if tone == 0:
            interest -= 1
            tone -= 1
        else:
            interest += np.random.choice([0, 1], p=[0.45, 0.55])
            tone += np.random.choice([0, 1], p=[0.55, 0.45])

    elif action_name == "share_story":
        if interest >= 1:
            stage += np.random.choice([0, 1], p=[0.50, 0.50])
            tone += np.random.choice([0, 1], p=[0.60, 0.40])
        else:
            interest += np.random.choice([0, 1], p=[0.75, 0.25])

    elif action_name == "be_playful":
        if tone >= 1 and interest >= 1:
            interest += np.random.choice([0, 1], p=[0.55, 0.45])
            tone += 1
        else:
            interest -= 1
            tone -= 1

    elif action_name == "be_direct":
        if interest >= 2 and tone >= 1:
            stage += 1
            tone += np.random.choice([0, 1], p=[0.65, 0.35])
        elif interest >= 1:
            stage += np.random.choice([0, 1], p=[0.70, 0.30])
        else:
            interest -= 1
            tone -= 1

    elif action_name == "suggest_date":
        done = True
        if interest == 2 and stage == 2 and tone == 2:
            outcome = "date_success"
        elif interest == 2 and stage >= 1 and tone >= 1 and random.random() < 0.35:
            outcome = "date_success"
        else:
            outcome = "date_fail"

    elif action_name == "slow_down":
        tone += np.random.choice([0, 1], p=[0.55, 0.45])
        if interest == 0:
            interest += np.random.choice([0, 1], p=[0.70, 0.30])

    elif action_name == "end_chat":
        done = True
        outcome = "ended"

    interest = _clamp(interest)
    stage = _clamp(stage)
    tone = _clamp(tone)

    if not done and interest == 0 and tone == 0 and random.random() < 0.08:
        done = True
        outcome = "ghosted"

    next_state_id = state_to_id(interest, stage, tone)
    reward = compute_reward(next_state_id, outcome=outcome)
    return next_state_id, reward, done, outcome


def reset_episode() -> int:
    interest = np.random.choice([0, 1], p=[0.35, 0.65])
    stage = 0
    tone = np.random.choice([0, 1], p=[0.25, 0.75])
    return state_to_id(interest, stage, tone)


def choose_action(Q: np.ndarray, state_id: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    return int(np.argmax(Q[state_id]))


def greedy_action(Q: np.ndarray, state_id: int) -> int:
    return int(np.argmax(Q[state_id]))


def run_episode_q_learning(
    env_step: Callable[[int, int], tuple[int, float, bool, str | None]],
    Q: np.ndarray,
    epsilon: float,
    reset_fn: Callable[[], int] = reset_episode,
    alpha: float = 0.1,
    gamma: float = 0.95,
    max_steps: int = 25,
) -> tuple[float, str | None]:
    state_id = reset_fn()
    total_reward = 0.0
    outcome = "max_steps"

    for _ in range(max_steps):
        action_id = choose_action(Q, state_id, epsilon)
        next_state_id, reward, done, outcome = env_step(state_id, action_id)
        total_reward += reward

        best_next = np.max(Q[next_state_id])
        target = reward + (0.0 if done else gamma * best_next)
        Q[state_id, action_id] += alpha * (target - Q[state_id, action_id])

        state_id = next_state_id
        if done:
            break

    return float(total_reward), outcome


def train_q_learning(
    env_step: Callable[[int, int], tuple[int, float, bool, str | None]],
    episodes: int = 8000,
    reset_fn: Callable[[], int] = reset_episode,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_fraction: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, list[float], list[str | None]]:
    """Train a tabular Q-table on the supplied environment step function."""
    random.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards = []
    outcomes = []

    for episode in range(episodes):
        frac = min(1.0, episode / max(1, int(episodes * epsilon_decay_fraction)))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        reward, outcome = run_episode_q_learning(
            env_step=env_step,
            Q=Q,
            epsilon=epsilon,
            reset_fn=reset_fn,
            alpha=alpha,
            gamma=gamma,
        )

        rewards.append(reward)
        outcomes.append(outcome)

    return Q, rewards, outcomes


def make_greedy_q_policy(Q: np.ndarray) -> Callable[[int], int]:
    return lambda state_id: greedy_action(Q, state_id)
