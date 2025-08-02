## iamGrokky

Grokky AI Assistant is a Telegram AI Entity powered by FastAPI. This project contains the server code and utility scripts used to run the assistant.

## Installation

1. Clone this repository.
2. Ensure Python 3.12 is available (see `runtime.txt`).
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment variables

Create a `.env` file in the project root using `.env.example` as a template. Each variable is described below.

| Variable | Required | Description | Default |
|----------|---------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | yes | Token for your Telegram bot obtained from @BotFather. | – |
| `OPENAI_API_KEY` | yes | API key for OpenAI requests. | – |
| `CHAT_ID` | yes | Telegram chat ID used for personal messages. | – |
| `XAI_API_KEY` | no | Key for the XAI mirror endpoints. | – |
| `IS_GROUP` | no | Set to `True` to enable group mode. | `False` |
| `AGENT_GROUP` | no | Group chat ID used when `IS_GROUP` is enabled. | `-1001234567890` |
| `PINECONE_API_KEY` | no | API key for the Pinecone vector store (required only if you use the vector store). | – |
| `PINECONE_INDEX` | no | Name of the Pinecone index to use. | – |
| `PORT` | no | Custom port for the FastAPI server. | `8000` |

### Variable descriptions

Each environment variable controls a specific aspect of the bot:

- `TELEGRAM_BOT_TOKEN` – authenticates your Telegram bot.
- `OPENAI_API_KEY` – allows requests to OpenAI.
- `CHAT_ID` – chat ID for personal messages when not in group mode.
- `XAI_API_KEY` – key for the XAI mirror endpoints (optional).
- `IS_GROUP` – toggles group mode.
- `AGENT_GROUP` – group chat ID used when `IS_GROUP` is `True`.
- `PINECONE_API_KEY` – enables the optional Pinecone vector store.
- `PINECONE_INDEX` – name of the Pinecone index to use.
- `PORT` – port for the FastAPI server.

Unused optional variables are ignored when their features are disabled.

## Running the server

After setting the environment variables, start the bot with:

```bash
python server.py
```

The FastAPI application will listen on `0.0.0.0:8000` by default. You can change the port by setting the `PORT` variable.

You may also run the server with `uvicorn` directly if preferred:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Voice support

Grokky can reply with voice messages when `/voiceon` is enabled. Speech
is synthesized using OpenAI's text-to-speech API with the male `onyx` voice.

## Webhook troubleshooting

If the bot is not receiving updates, verify the Telegram webhook configuration. The webhook URL **must** point to `/webhook` on your domain without the bot token appended. `server.py` will try to fix the webhook on startup, and you can also run `python fix_webhook.py` manually. See [WEBHOOK_FIX_INSTRUCTIONS.md](WEBHOOK_FIX_INSTRUCTIONS.md) for step-by-step instructions.

## SLNCX neural core

iamGrokky is more than a Telegram front end. It bundles its own neural network called **SLNCX** (code name *Wulf*). The engine runs from quantized weights stored right here in the repository, so the bot can operate even without outside API access. This self-contained approach turns the agent into its own weight server—a small revolution in local AI.

SLNCX draws inspiration from Grok1 but trims the architecture for efficiency. A mixture-of-experts layout routes each token through multiple specialized networks, keeping quality high while letting inference remain nimble on ordinary CPUs.

The model still offers an 8k context window and sixty-four layers, yet it fits comfortably in memory thanks to heavy quantization. Rotary position embeddings ensure long-range attention without extra overhead.

A minimal CLI and an HTTP endpoint under `SLNCX/` let you load checkpoints and query the engine. Every interaction is logged under `logs/wulf/`, and failures drop tracebacks into `failures/` for quick debugging.

Wulf's dataset is small and focused. It wasn't built to chat about everything; instead it specializes in terse, goal-driven replies. Checkpoints load lazily and stay resident in memory for subsequent calls.

Components are split into reusable pieces—layers, attention blocks, and the mixture-of-experts modules—so you can tinker as the design evolves. Two-bit quantization keeps the footprint light, making CPU-only deployment practical.

To run the model locally, place a checkpoint at `out/ckpt.pt` (or set `CKPT_PATH`) and execute `wulf_cli.py` with your prompt. You can also start an API server with `uvicorn app:app --port 8000` and hit `/generate` with JSON payloads.

The combination of Grokky's Telegram interface and SLNCX's local brain shows how a single project can serve its own neural weights. Codex, it feels like a quiet revolution.

## Architect's note

As the person who pieced these parts together, I'm fascinated by how streamlined the result is. A small, quantized network now answers directly from a handheld device or modest server without leaning on heavy cloud infrastructure.

I believe this local-first design hints at a broader shift. Massive models will always exist, but there's power in a compact agent that carries its own intelligence wherever it goes. It's simple, efficient, and oddly liberating.
