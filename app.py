from __future__ import annotations

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
    ACTIONS,
    INTERESTS,
    STAGES,
    TONES,
    N_ACTIONS,
    N_STATES,
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


@st.cache_resource
def load_or_train_q_table() -> tuple[np.ndarray, str]:
    """Load a saved Q-table, or train and save the handcoded baseline."""
    if Q_TABLE_PATH.exists():
        Q = np.load(Q_TABLE_PATH)
        expected_shape = (N_STATES, N_ACTIONS)
        if Q.shape == expected_shape:
            return Q, f"Loaded saved Q-table from {Q_TABLE_PATH}"

    Q, _, _ = train_q_learning(env_step=simulator_step_hardcoded, episodes=8000)
    np.save(Q_TABLE_PATH, Q)
    return Q, f"Trained Q-table at startup and saved to {Q_TABLE_PATH}"


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_debug", None)
    st.session_state.setdefault("suggested_reply", None)


def reset_conversation() -> None:
    st.session_state.chat_history = []
    st.session_state.last_debug = None
    st.session_state.suggested_reply = None


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


def process_user_message(user_message: str, Q: np.ndarray) -> None:
    chat_before_reply = list(st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "you", "content": user_message})

    potential_date_reply = simulate_potential_date(
        user_message=user_message,
        chat_history=chat_before_reply,
    )
    st.session_state.chat_history.append(
        {"role": "potential_date", "content": potential_date_reply}
    )

    classification = classify_message_with_llm(
        potential_date_reply,
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

    st.session_state.last_debug = {
        "classification": classification,
        "state_id": state_id,
        "state_description": describe_state(state_id),
        "action_id": action_id,
        "action": action,
        "q_values": q_values_for_state(Q, state_id),
    }
    st.session_state.suggested_reply = suggested_reply


def render_chat() -> None:
    st.subheader("Chat")

    if not st.session_state.chat_history:
        st.info("Send a first message to start the demo conversation.")
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

    if not st.session_state.last_debug:
        st.info("No state/action debug data yet. Send a message first.")
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

    st.markdown("#### Top Q-values for current state")
    st.dataframe(debug["q_values"], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Dating Agent RL Demo", layout="wide")
    init_session_state()

    Q, Q_status = load_or_train_q_table()

    st.title("Dating Agent RL Demo")
    st.markdown(
        """
        Type a message as yourself. The LLM roleplays the potential date, another
        LLM call classifies their reply into the tabular RL state, and the Q-table
        chooses the next abstract strategy. The final LLM call turns that strategy
        into a respectful suggested reply.
        """
    )

    with st.expander("Architecture note", expanded=False):
        st.write(
            """
            The LLM is not trained. The Q-table is the trained RL model.
            The LLM handles language; the RL agent chooses the strategy/action.
            This demo uses `Q_hardcoded` by default.
            """
        )

    if not llm_available():
        st.error(
            "LLM API key is missing. Add KIMI_API_KEY to your .env file, then restart Streamlit."
        )
        st.caption(f"Configured base URL: {KIMI_BASE_URL}")
        st.caption(f"Configured model: {KIMI_MODEL}")

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        render_chat()

        with st.form("message_form", clear_on_submit=True):
            user_message = st.text_area(
                "Your message",
                placeholder="Type a message you might send...",
                disabled=not llm_available(),
                height=100,
            )
            send = st.form_submit_button("Send", disabled=not llm_available())

        reset = st.button("Reset conversation")
        if reset:
            reset_conversation()
            st.rerun()

        if send:
            cleaned_message = user_message.strip()
            if not cleaned_message:
                st.warning("Type a message first.")
            else:
                with st.spinner("Running LLM roleplay, classifier, RL policy, and reply generator..."):
                    try:
                        process_user_message(cleaned_message, Q)
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")
                    else:
                        st.rerun()

    with right:
        render_debug_panel(Q_status)

        with st.expander("State/action space", expanded=False):
            st.write("**Interest labels:**", ", ".join(INTERESTS))
            st.write("**Stage labels:**", ", ".join(STAGES))
            st.write("**Tone labels:**", ", ".join(TONES))
            st.write("**Actions:**", ", ".join(ACTIONS))


if __name__ == "__main__":
    main()
