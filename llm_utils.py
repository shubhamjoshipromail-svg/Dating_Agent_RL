"""LLM helpers for the Streamlit Dating Agent RL demo.

The LLM handles language only:
- roleplay the potential date
- classify text into the tabular RL state
- turn an abstract RL action into a respectful suggested reply
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rl_core import ACTION_GUIDANCE, ACTIONS, INTERESTS, STAGES, TONES


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-8k")


def llm_available() -> bool:
    return bool(KIMI_API_KEY and not KIMI_API_KEY.startswith("your_"))


@lru_cache(maxsize=1)
def get_client() -> OpenAI | None:
    if not llm_available():
        return None
    return OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)


def _history_as_text(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in chat_history[-12:]:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _chat_completion(messages: list[dict[str, str]], temperature: float = 0.4) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("LLM API key is missing. Set KIMI_API_KEY in .env.")

    response = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def simulate_potential_date(user_message: str, chat_history: list[dict[str, str]]) -> str:
    """Roleplay the potential date's next message."""
    history_text = _history_as_text(chat_history)
    system_prompt = """You are roleplaying the other person in a casual dating-app chat.

Keep the response natural, brief, and respectful. Do not be explicit, sexual, manipulative, coercive, creepy, insulting, or overly intense. You can be warm, neutral, curious, busy, playful, or mildly uninterested depending on the conversation.

Return only the potential date's message. Do not explain your reasoning."""

    return _chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{history_text}\n\n"
                    f"The user just sent:\n{user_message}\n\n"
                    "Write the potential date's next reply."
                ),
            },
        ],
        temperature=0.7,
    )


def classify_message_with_llm(
    message: str,
    conversation_context: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    """Classify a message into the tabular RL state labels."""
    context_text = _history_as_text(conversation_context or [])
    system_prompt = f"""You are a careful classifier for a toy dating-app RL state space.
Return ONLY valid JSON with exactly these fields:
- "interest": one of {INTERESTS}
- "stage": one of {STAGES}
- "tone": one of {TONES}

Definitions:
- low interest: short, dry, no questions back, no curiosity
- medium interest: engaged but not effusive, some response effort
- high interest: enthusiastic, asks questions back, positive emotion
- opener stage: first 1-2 exchanges or clear first message
- chat stage: ongoing back-and-forth or a reply without clear opener context
- date stage: discussing meeting up or strong rapport/flirtation
- cold tone: terse, formal, low warmth
- neutral tone: cordial but not especially warm
- warm tone: friendly, playful, joking, genuinely enthusiastic

Important: if the message is just "k" and no context says it is the first message, classify it as low/chat/cold.
JSON only. No explanation."""

    client = get_client()
    if client is None:
        raise RuntimeError("LLM API key is missing. Set KIMI_API_KEY in .env.")

    response = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Conversation context:\n{context_text}\n\nMessage to classify:\n{message}",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier returned invalid JSON: {raw}") from exc

    _validate_state_labels(payload)
    return payload


def _validate_state_labels(payload: dict[str, str]) -> None:
    expected = {"interest", "stage", "tone"}
    if set(payload) != expected:
        raise ValueError(f"Expected keys {expected}, got {payload}")
    if payload["interest"] not in INTERESTS:
        raise ValueError(f"Invalid interest label: {payload}")
    if payload["stage"] not in STAGES:
        raise ValueError(f"Invalid stage label: {payload}")
    if payload["tone"] not in TONES:
        raise ValueError(f"Invalid tone label: {payload}")


def generate_recommended_reply(
    action: str,
    state: dict[str, str],
    chat_history: list[dict[str, str]],
) -> str:
    """Turn the RL action into a concrete suggested message for the user."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}.")

    history_text = _history_as_text(chat_history)
    action_guidance = ACTION_GUIDANCE[action]

    system_prompt = """You write suggested dating-app replies for the user.

Safety rules:
- Be casual, respectful, and low-pressure.
- Do not be manipulative, sexual, coercive, creepy, insulting, or guilt-trippy.
- Do not pretend to know private facts.
- If the action is end_chat, write a polite closing message.
- Return only the suggested reply, with no explanation."""

    return _chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{history_text}\n\n"
                    f"Detected state: {json.dumps(state)}\n"
                    f"RL recommended abstract action: {action}\n"
                    f"Action guidance: {action_guidance}\n\n"
                    "Write one message the user could send next."
                ),
            },
        ],
        temperature=0.5,
    )
