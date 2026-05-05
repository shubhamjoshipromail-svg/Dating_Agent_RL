# Dating Agent RL

This is a toy reinforcement learning project for learning tabular RL workflow design.

The project trains a small Q-learning / SARSA agent over a simplified conversation state space:

interest × stage × tone

The agent does not generate natural language directly. It chooses an abstract strategy/action such as:

- ask_question
- give_compliment
- share_story
- be_playful
- be_direct
- suggest_date
- slow_down
- end_chat

## Important distinction

The trained model is the Q-table or SARSA table.

The LLM is not trained.

The LLM can optionally be used for two separate roles:

1. Classifier/interface:
   real message → interest/stage/tone

2. World model/data generator:
   state + action → simulated next state + response + outcome

## Current working notebook

Open:

notebooks/Dating_Agent_RL_current.ipynb

## Recommended workflow

1. Train and evaluate the handcoded simulator baseline.
2. Optionally use the LLM to generate transition records.
3. Save those transitions into `data/llm_transition_cache.jsonl`.
4. Train from the cached LLM environment without repeated API calls.
5. Use the LLM classifier only for demo/interface.

## Streamlit frontend modes

The deployed frontend supports two testing modes:

1. **LLM demo mode**
   You type your message, the LLM roleplays the other person, the classifier maps
   that reply into the discrete RL state, and the Q-table recommends the next
   abstract action.

2. **Human / real input mode**
   You paste the real other person's latest reply. The classifier maps that text
   into the discrete RL state, and the Q-table recommends the next abstract
   action.

The app caches turns and observed outcomes in:

- `data/conversation_turns.jsonl`
- `data/conversation_transitions.jsonl`

Replay training from cached transitions is deliberate: press the replay-training
button in the frontend to create `data/Q_conversation_replay.npy`. The app does
not silently self-train on every message.

## Setup

Create a `.env` file based on `.env.example`:

KIMI_API_KEY=your_key_here
KIMI_BASE_URL=https://api.moonshot.ai/v1
KIMI_MODEL=moonshot-v1-8k

Install dependencies:

pip install -r requirements.txt

## Railway deployment

This repo includes `railway.json` with the Streamlit start command:

streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false

In Railway, set these environment variables:

KIMI_API_KEY=your_key_here
KIMI_BASE_URL=https://api.moonshot.ai/v1
KIMI_MODEL=moonshot-v1-8k

The local `.env` file is ignored by git and is not deployed.

## Warning

Do not hardcode API keys in notebooks.

Do not repeatedly run live LLM training loops. They are slow and expensive.

The preferred workflow is:

LLM generates transitions once → save cache → train RL from cached transitions.

## Limitations

This is a toy RL learning environment, not a real dating advice system.

The state space, rewards, and simulator are simplified. LLM-generated transitions can be noisy or biased.
