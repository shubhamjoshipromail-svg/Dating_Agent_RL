from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from llm_utils import (
    KIMI_BASE_URL,
    KIMI_MODEL,
    classify_message_with_llm,
    generate_recommended_reply,
    llm_available,
    simulate_potential_date,
)
from rl_core import (
    ACTION_GUIDANCE,
    ACTIONS,
    INTERESTS,
    STAGES,
    TONES,
    N_ACTIONS,
    N_STATES,
    compute_reward,
    describe_state,
    greedy_action,
    simulator_step_hardcoded,
    state_to_id,
    train_q_learning,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

Q_TABLE_PATH = DATA_DIR / "Q_hardcoded.npy"
Q_REPLAY_PATH = DATA_DIR / "Q_conversation_replay.npy"
TURN_CACHE_PATH = DATA_DIR / "conversation_turns.jsonl"
TRANSITION_CACHE_PATH = DATA_DIR / "conversation_transitions.jsonl"

OUTCOME_OPTIONS = ["continued", "date_success", "date_fail", "ghosted", "ended"]
TERMINAL_OUTCOMES = {"date_success", "date_fail", "ghosted", "ended"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def cache_counts() -> tuple[int, int]:
    return len(load_jsonl(TURN_CACHE_PATH)), len(load_jsonl(TRANSITION_CACHE_PATH))


@st.cache_resource
def load_or_train_q_table() -> tuple[np.ndarray, str]:
    """Load replay-trained Q if available, otherwise load/train the handcoded baseline."""
    expected_shape = (N_STATES, N_ACTIONS)

    if Q_REPLAY_PATH.exists():
        Q = np.load(Q_REPLAY_PATH)
        if Q.shape == expected_shape:
            return Q, f"Loaded replay-trained Q-table from {Q_REPLAY_PATH}"

    if Q_TABLE_PATH.exists():
        Q = np.load(Q_TABLE_PATH)
        if Q.shape == expected_shape:
            return Q, f"Loaded saved handcoded Q-table from {Q_TABLE_PATH}"

    Q, _, _ = train_q_learning(env_step=simulator_step_hardcoded, episodes=8000)
    np.save(Q_TABLE_PATH, Q)
    return Q, f"Trained handcoded Q-table at startup and saved to {Q_TABLE_PATH}"


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_debug", None)
    st.session_state.setdefault("suggested_reply", None)
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("last_turn_record", None)


def reset_conversation() -> None:
    st.session_state.chat_history = []
    st.session_state.last_debug = None
    st.session_state.suggested_reply = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_turn_record = None


def q_values_for_state(Q: np.ndarray, state_id: int) -> pd.DataFrame:
    rows = [
        {
            "rank": rank + 1,
            "action": ACTIONS[action_id],
            "q_value": float(Q[state_id, action_id]),
        }
        for rank, action_id in enumerate(np.argsort(Q[state_id])[::-1])
    ]
    return pd.DataFrame(rows)


def build_transition_from_previous(
    previous_turn: dict | None,
    next_state_id: int,
    outcome_since_previous: str,
) -> dict | None:
    if previous_turn is None:
        return None

    outcome = None if outcome_since_previous == "continued" else outcome_since_previous
    done = outcome_since_previous in TERMINAL_OUTCOMES
    reward = compute_reward(next_state_id, outcome=outcome)

    return {
        "transition_id": str(uuid.uuid4()),
        "session_id": previous_turn["session_id"],
        "source_turn_id": previous_turn["turn_id"],
        "state_id": int(previous_turn["state_id"]),
        "state": previous_turn["state"],
        "action_id": int(previous_turn["action_id"]),
        "action": previous_turn["action"],
        "next_state_id": int(next_state_id),
        "next_state": describe_state(next_state_id),
        "reward": float(reward),
        "done": bool(done),
        "outcome": outcome,
        "outcome_label": outcome_since_previous,
        "created_at": now_iso(),
    }


def cache_turn_and_transition(
    *,
    mode: str,
    user_message: str,
    other_reply: str,
    classification: dict[str, str],
    state_id: int,
    action_id: int,
    suggested_reply: str,
    outcome_since_previous: str,
) -> tuple[dict, dict | None]:
    transition = build_transition_from_previous(
        st.session_state.last_turn_record,
        next_state_id=state_id,
        outcome_since_previous=outcome_since_previous,
    )
    if transition is not None:
        append_jsonl(transition, TRANSITION_CACHE_PATH)

    turn = {
        "turn_id": str(uuid.uuid4()),
        "session_id": st.session_state.session_id,
        "mode": mode,
        "user_message": user_message,
        "other_reply": other_reply,
        "classified_state": classification,
        "state_id": int(state_id),
        "state": describe_state(state_id),
        "action_id": int(action_id),
        "action": ACTIONS[action_id],
        "suggested_reply": suggested_reply,
        "outcome_since_previous": outcome_since_previous,
        "timestamp": now_iso(),
    }
    append_jsonl(turn, TURN_CACHE_PATH)
    st.session_state.last_turn_record = turn
    return turn, transition


def replay_train_from_cached_transitions(
    base_Q: np.ndarray,
    transitions: list[dict],
    epochs: int = 30,
    alpha: float = 0.08,
    gamma: float = 0.95,
) -> np.ndarray:
    """Train from cached real/simulated conversation transitions without LLM calls."""
    Q = np.array(base_Q, copy=True)

    for _ in range(epochs):
        for record in transitions:
            state_id = int(record["state_id"])
            action_id = int(record["action_id"])
            next_state_id = int(record["next_state_id"])
            reward = float(record["reward"])
            done = bool(record["done"])

            target = reward + (0.0 if done else gamma * np.max(Q[next_state_id]))
            Q[state_id, action_id] += alpha * (target - Q[state_id, action_id])

    return Q


def classify_state_and_recommend(
    *,
    mode: str,
    user_message: str,
    other_reply: str,
    outcome_since_previous: str,
    Q: np.ndarray,
) -> None:
    classification = classify_message_with_llm(
        other_reply,
        conversation_context=st.session_state.chat_history,
    )
    state_id = state_to_id(
        classification["interest"],
        classification["stage"],
        classification["tone"],
    )
    action_id = greedy_action(Q, state_id)
    action = ACTIONS[action_id]

    suggested_reply = generate_recommended_reply(
        action=action,
        state=classification,
        chat_history=st.session_state.chat_history,
    )

    turn, transition = cache_turn_and_transition(
        mode=mode,
        user_message=user_message,
        other_reply=other_reply,
        classification=classification,
        state_id=state_id,
        action_id=action_id,
        suggested_reply=suggested_reply,
        outcome_since_previous=outcome_since_previous,
    )

    st.session_state.last_debug = {
        "classification": classification,
        "state_id": state_id,
        "state_description": describe_state(state_id),
        "action_id": action_id,
        "action": action,
        "q_values": q_values_for_state(Q, state_id),
        "cached_turn": turn,
        "cached_transition": transition,
    }
    st.session_state.suggested_reply = suggested_reply


def process_llm_demo_turn(user_message: str, outcome_since_previous: str, Q: np.ndarray) -> None:
    chat_before_reply = list(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "you", "content": user_message})

    other_reply = simulate_potential_date(
        user_message=user_message,
        chat_history=chat_before_reply,
    )
    st.session_state.chat_history.append({"role": "potential_date", "content": other_reply})

    classify_state_and_recommend(
        mode="llm_demo",
        user_message=user_message,
        other_reply=other_reply,
        outcome_since_previous=outcome_since_previous,
        Q=Q,
    )


def process_human_input_turn(
    user_message: str,
    other_reply: str,
    outcome_since_previous: str,
    Q: np.ndarray,
) -> None:
    if user_message:
        st.session_state.chat_history.append({"role": "you", "content": user_message})
    st.session_state.chat_history.append({"role": "potential_date", "content": other_reply})

    classify_state_and_recommend(
        mode="human_input",
        user_message=user_message,
        other_reply=other_reply,
        outcome_since_previous=outcome_since_previous,
        Q=Q,
    )


def render_chat() -> None:
    st.subheader("Conversation")

    if not st.session_state.chat_history:
        st.info("Start in LLM demo mode or paste a real reply in human input mode.")
        return

    for message in st.session_state.chat_history:
        if message["role"] == "you":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    if st.session_state.suggested_reply:
        st.markdown("#### Suggested reply")
        st.success(st.session_state.suggested_reply)


def render_debug_panel(Q_status: str) -> None:
    st.subheader("RL Debug Panel")
    st.caption("The LLM handles language. The trained Q-table chooses the abstract strategy.")

    st.write("**Q-table status:**")
    st.write(Q_status)

    turns, transitions = cache_counts()
    c1, c2 = st.columns(2)
    c1.metric("Cached turns", turns)
    c2.metric("Cached transitions", transitions)

    if not st.session_state.last_debug:
        st.info("No state/action debug data yet. Submit a turn first.")
        return

    debug = st.session_state.last_debug
    classification = debug["classification"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Interest", classification["interest"])
    c2.metric("Stage", classification["stage"])
    c3.metric("Tone", classification["tone"])

    st.write(f"**state_id:** `{debug['state_id']}`")
    st.write(f"**state:** `{debug['state_description']}`")
    st.write(f"**recommended action:** `{debug['action']}`")

    if debug.get("cached_transition"):
        st.success("A transition from the previous recommendation was cached.")
    else:
        st.info("This turn was cached. A training transition appears after the next observed reply.")

    st.markdown("#### Top Q-values for current state")
    st.dataframe(debug["q_values"], use_container_width=True, hide_index=True)


def render_replay_training_panel(Q: np.ndarray) -> None:
    st.subheader("Replay Training")
    st.caption("Uses cached transitions only. No LLM calls happen here.")

    transitions = load_jsonl(TRANSITION_CACHE_PATH)
    st.write(f"Cached transition records: `{len(transitions)}`")

    epochs = st.number_input("Replay epochs", min_value=1, max_value=200, value=30, step=5)
    train = st.button(
        "Train Q-table from cached transitions",
        disabled=not transitions,
        help="This updates a saved replay Q-table from cached conversation transitions.",
    )

    if train:
        Q_replay = replay_train_from_cached_transitions(Q, transitions, epochs=int(epochs))
        np.save(Q_REPLAY_PATH, Q_replay)
        st.success(f"Saved replay-trained Q-table to {Q_REPLAY_PATH}")
        load_or_train_q_table.clear()
        st.rerun()

    with st.expander("Cached files", expanded=False):
        st.write(f"Turns: `{TURN_CACHE_PATH}`")
        st.write(f"Transitions: `{TRANSITION_CACHE_PATH}`")
        st.write(f"Replay Q-table: `{Q_REPLAY_PATH}`")


def main() -> None:
    st.set_page_config(page_title="Dating Agent RL Demo", layout="wide")
    init_session_state()

    Q, Q_status = load_or_train_q_table()

    st.title("Dating Agent RL Demo")
    st.markdown(
        """
        Choose **LLM demo mode** to test against a simulated conversation partner, or
        **human input mode** to paste real replies. Every turn is cached, and observed
        transitions can later replay-train the tabular Q-table.
        """
    )

    with st.expander("Architecture note", expanded=False):
        st.write(
            """
            The LLM is not trained. The Q-table is the trained RL model.
            The LLM handles language; the RL agent chooses the strategy/action.
            Cached conversations are used for deliberate replay training, not instant live self-training.
            """
        )

    if not llm_available():
        st.error(
            "LLM API key is missing. Add KIMI_API_KEY to your .env file or Railway variables."
        )
        st.caption(f"Configured base URL: {KIMI_BASE_URL}")
        st.caption(f"Configured model: {KIMI_MODEL}")

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        mode = st.radio(
            "Testing mode",
            ["LLM demo", "Human / real input"],
            horizontal=True,
            help="LLM demo simulates the other person. Human input lets you paste a real reply.",
        )

        render_chat()

        has_previous_turn = st.session_state.last_turn_record is not None
        with st.form("message_form", clear_on_submit=True):
            outcome_since_previous = st.selectbox(
                "Outcome since previous recommendation",
                OUTCOME_OPTIONS,
                index=0,
                disabled=not has_previous_turn,
                help=(
                    "This labels what happened after the last recommended action. "
                    "It becomes the reward signal for replay training."
                ),
            )

            if mode == "LLM demo":
                user_message = st.text_area(
                    "Your message",
                    placeholder="Type a message you might send...",
                    disabled=not llm_available(),
                    height=100,
                )
                other_reply = ""
            else:
                user_message = st.text_area(
                    "Your message or context (optional)",
                    placeholder="Paste what you sent, or leave blank if you only have their reply.",
                    disabled=not llm_available(),
                    height=80,
                )
                other_reply = st.text_area(
                    "Other person's latest reply",
                    placeholder="Paste the real reply you want the agent to evaluate...",
                    disabled=not llm_available(),
                    height=120,
                )

            send = st.form_submit_button("Analyze turn", disabled=not llm_available())

        reset = st.button("Reset conversation")
        if reset:
            reset_conversation()
            st.rerun()

        if send:
            cleaned_user_message = user_message.strip()
            cleaned_other_reply = other_reply.strip()

            if mode == "LLM demo" and not cleaned_user_message:
                st.warning("Type a message first.")
            elif mode != "LLM demo" and not cleaned_other_reply:
                st.warning("Paste the other person's reply first.")
            else:
                with st.spinner("Classifying state, choosing RL action, caching turn..."):
                    try:
                        if mode == "LLM demo":
                            process_llm_demo_turn(cleaned_user_message, outcome_since_previous, Q)
                        else:
                            process_human_input_turn(
                                cleaned_user_message,
                                cleaned_other_reply,
                                outcome_since_previous,
                                Q,
                            )
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")
                    else:
                        st.rerun()

    with right:
        render_debug_panel(Q_status)
        render_replay_training_panel(Q)

        with st.expander("State/action space", expanded=False):
            st.write("**Interest labels:**", ", ".join(INTERESTS))
            st.write("**Stage labels:**", ", ".join(STAGES))
            st.write("**Tone labels:**", ", ".join(TONES))
            st.write("**Actions:**", ", ".join(ACTIONS))
            st.write("**Action guidance:**")
            st.json(ACTION_GUIDANCE)


if __name__ == "__main__":
    main()
