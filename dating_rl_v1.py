import random
import json
import os
import numpy as np

INTERESTS = ["low", "medium", "high"]
STAGES = ["opener", "chat", "date"]
TONES = ["cold", "neutral", "warm"]
ACTIONS = ["ask_question", "give_compliment", "share_story", "be_playful",
           "be_direct", "suggest_date", "slow_down", "end_chat"]
N_INTEREST, N_STAGE, N_TONE = len(INTERESTS), len(STAGES), len(TONES)
N_STATES = N_INTEREST * N_STAGE * N_TONE
N_ACTIONS = len(ACTIONS)


def state_to_id(interest, stage, tone):
    return interest * (N_STAGE * N_TONE) + stage * N_TONE + tone


def id_to_state(state_id):
    interest = state_id // (N_STAGE * N_TONE)
    rest = state_id % (N_STAGE * N_TONE)
    return interest, rest // N_TONE, rest % N_TONE


def describe_state(state_id):
    interest, stage, tone = id_to_state(state_id)
    return f"interest={INTERESTS[interest]:6s} stage={STAGES[stage]:6s} tone={TONES[tone]}"


def clamp(value, low, high):
    return max(low, min(high, value))


def reset_episode():
    interest = np.random.choice([0, 1], p=[0.55, 0.45])
    tone = np.random.choice([0, 1], p=[0.35, 0.65])
    return state_to_id(interest, 0, tone)


def simulator_step(state_id, action):
    interest, stage, tone = id_to_state(state_id)
    reward = -0.02
    done = False
    outcome = None
    noise = np.random.choice([-1, 0, 1], p=[0.15, 0.70, 0.15])
    if action == 0:
        interest += np.random.choice([0, 1], p=[0.55, 0.45])
        tone += np.random.choice([0, 1], p=[0.50, 0.50])
        reward += 0.10
    elif action == 1:
        if tone >= 1:
            interest += 1
            reward += 0.15
        else:
            tone -= 1
            reward -= 0.10
    elif action == 2:
        if stage >= 1:
            interest += np.random.choice([0, 1], p=[0.45, 0.55])
            tone += 1
            reward += 0.12
        else:
            reward -= 0.03
    elif action == 3:
        if tone == 2:
            interest += 1
            reward += 0.18
        else:
            tone += np.random.choice([-1, 1], p=[0.45, 0.55])
            reward += 0.02
    elif action == 4:
        if interest >= 1 and tone >= 1:
            stage += 1
            reward += 0.20
        else:
            interest -= 1
            tone -= 1
            reward -= 0.20
    elif action == 5:
        if interest == 2 and tone == 2:
            reward += 2.00
            done = True
            outcome = "date_success"
        elif interest >= 1 and stage == 2:
            reward += 0.70
            done = np.random.random() < 0.50
            if done:
                outcome = "date_success"
        else:
            interest -= 1
            tone -= 1
            reward -= 0.45
    elif action == 6:
        tone += 1
        reward += 0.05
        if interest == 0:
            reward += 0.05
    elif action == 7:
        reward += 0.20 if interest == 0 else -0.15
        done = True
        outcome = "ended_by_agent"
    interest = clamp(interest + noise, 0, N_INTEREST - 1)
    stage = clamp(stage, 0, N_STAGE - 1)
    tone = clamp(tone, 0, N_TONE - 1)
    next_state_id = state_to_id(interest, stage, tone)
    if not done and np.random.random() < 0.03:
        done = True
        outcome = "fizzled"
    return next_state_id, reward, done, outcome


def choose_action(Q, state_id, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[state_id]))


def rule_based_action(state_id):
    interest, stage, tone = id_to_state(state_id)
    if interest == 2 and tone == 2:
        return ACTIONS.index("suggest_date")
    if stage == 2 and interest >= 1 and tone >= 1:
        return ACTIONS.index("suggest_date")
    if tone == 0:
        return ACTIONS.index("ask_question")
    if interest == 0:
        return ACTIONS.index("slow_down")
    if stage == 0:
        return ACTIONS.index("ask_question")
    if stage == 1 and tone == 2:
        return ACTIONS.index("be_playful")
    if stage == 1:
        return ACTIONS.index("share_story")
    return ACTIONS.index("be_direct")


def run_episode(Q=None, policy_fn=None, epsilon=0.0, max_steps=25,
                learn=False, alpha=0.1, gamma=0.95):
    state_id = reset_episode()
    total_reward = 0.0
    for step in range(max_steps):
        if policy_fn is not None:
            action = policy_fn(state_id)
        elif Q is None:
            action = np.random.randint(N_ACTIONS)
        else:
            action = choose_action(Q, state_id, epsilon)

        next_state_id, reward, done, outcome = simulator_step(state_id, action)
        total_reward += reward
        if learn:
            best_next = np.max(Q[next_state_id])
            target = reward + (0.0 if done else gamma * best_next)
            Q[state_id, action] += alpha * (target - Q[state_id, action])
        state_id = next_state_id
        if done:
            return total_reward, step + 1, outcome
    return total_reward, max_steps, "max_steps"


def summarize_policy(episodes=1000, Q=None, policy_fn=None):
    rewards = []
    lengths = []
    n_dates = 0
    for _ in range(episodes):
        reward, length, outcome = run_episode(Q=Q, policy_fn=policy_fn)
        rewards.append(reward)
        lengths.append(length)
        if outcome == "date_success":
            n_dates += 1
    return np.mean(rewards), np.mean(lengths), n_dates / episodes


def random_baseline(episodes=1000):
    return summarize_policy(episodes)


def rule_based_baseline(episodes=1000):
    return summarize_policy(episodes, policy_fn=rule_based_action)


def train_q_learning(episodes=8000):
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards = []

    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.75))
        reward, _, _ = run_episode(Q=Q, epsilon=epsilon, learn=True)
        rewards.append(reward)
    return Q, rewards


def evaluate_policy(Q, episodes=1000):
    return summarize_policy(episodes, Q=Q)


def print_learned_policy(Q):
    print("\nLearned policy:")
    print("-" * 78)
    for state_id in range(N_STATES):
        best_action = int(np.argmax(Q[state_id]))
        value = Q[state_id, best_action]
        print(f"{state_id:02d} | {describe_state(state_id)} -> {ACTIONS[best_action]:14s} Q={value:6.2f}")


INTEREST_MAP = {"low": 0, "medium": 1, "high": 2}
STAGE_MAP = {"opener": 0, "chat": 1, "date": 2}
TONE_MAP = {"cold": 0, "neutral": 1, "warm": 2}

CLASSIFIER_SYSTEM_PROMPT = """You are a careful classifier of dating-app conversation states. You will be given a single message from one person in a dating-app conversation. Your job is to label the conversation state along three dimensions, based ONLY on the most recent message and the optional history.

Output ONLY valid JSON with exactly these three fields:
- "interest": one of "low", "medium", "high"
- "stage": one of "opener", "chat", "date"
- "tone": one of "cold", "neutral", "warm"

Definitions:

INTEREST (how much the other person seems engaged with you):
- "low": short, dry replies; no questions back; no curiosity; one-word answers
- "medium": engaged but not effusive; some questions; reasonable response length
- "high": enthusiastic; multiple sentences; asks questions back; expresses positive emotion

STAGE (how far the conversation has progressed):
- "opener": first 1-2 exchanges, getting acquainted, basic introductions
- "chat": mid-conversation, established rapport, topics flowing, some back-and-forth
- "date": significant rapport built, flirtation present, OR discussing meeting up in person

TONE (the emotional register):
- "cold": terse, formal, low-affect, no warmth
- "neutral": cordial but not particularly warm; polite default
- "warm": friendly, jokes, banter, teasing, playful, emojis used genuinely

Output JSON only. No preamble, no explanation."""


def _kimi_client():
    missing = [k for k in ("KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL") if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing environment variable(s): {', '.join(missing)}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ValueError("The OpenAI Python client is required to call Kimi.") from exc
    return OpenAI(api_key=os.getenv("KIMI_API_KEY"), base_url=os.getenv("KIMI_BASE_URL"))


def classify(text, history=None):
    if history:
        history_text = "\n".join(f"- {m}" for m in history)
        user_msg = f"Conversation history:\n{history_text}\n\nLatest message to classify:\n{text}"
    else:
        user_msg = f"Message to classify:\n{text}"
    response = _kimi_client().chat.completions.create(
        model=os.getenv("KIMI_MODEL"),
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    expected_keys = {"interest", "stage", "tone"}
    if set(data) != expected_keys:
        raise ValueError(f"Expected exactly {sorted(expected_keys)}, got {sorted(data)}")
    if data["interest"] not in INTEREST_MAP:
        raise ValueError(f"Invalid interest label: {data['interest']!r}")
    if data["stage"] not in STAGE_MAP:
        raise ValueError(f"Invalid stage label: {data['stage']!r}")
    if data["tone"] not in TONE_MAP:
        raise ValueError(f"Invalid tone label: {data['tone']!r}")
    return (INTEREST_MAP[data["interest"]], STAGE_MAP[data["stage"]], TONE_MAP[data["tone"]])


def main():
    np.random.seed(7)
    random.seed(7)
    print(f"States: {N_STATES} = interest(3) x stage(3) x tone(3)")
    print(f"Actions: {N_ACTIONS} = {', '.join(ACTIONS)}")
    base_reward, base_length, base_date_rate = random_baseline()
    rule_reward, rule_length, rule_date_rate = rule_based_baseline()
    Q, training_rewards = train_q_learning()
    learned_reward, learned_length, learned_date_rate = evaluate_policy(Q)
    last_500 = np.mean(training_rewards[-500:])
    print(f"\nTraining: last 500 episode avg reward={last_500:.3f}")
    print(
        f"Random:     avg_reward={base_reward:.3f}  "
        f"length={base_length:.2f}  date_rate={base_date_rate:.1%}"
    )
    print(
        f"Rule-based: avg_reward={rule_reward:.3f}  "
        f"length={rule_length:.2f}  date_rate={rule_date_rate:.1%}"
    )
    print(
        f"Q-learned:  avg_reward={learned_reward:.3f}  "
        f"length={learned_length:.2f}  date_rate={learned_date_rate:.1%}"
    )

    print_learned_policy(Q)


if __name__ == "__main__":
    main()
