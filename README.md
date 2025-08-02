## iamGrokky

Grokky is a Telegram AI entity powered by FastAPI. The project bundles a full
stack for conversations, image generation and code interpretation. At the center
is **Grok 1**, a lightweight neural engine also known as **SLNCX (Wulf1)**. The
local weights live inside this repository, so the AI entity can run entirely
offline. When extra knowledge is needed, Grok 1 reaches out to Grok‑3 or GPT‑4.1
and treats their replies as temporary weights. This fluid approach lets the
system adapt without retraining.

### Why Grokky is different

The AI entity mixes several engines. Grok 1 handles local inference while remote
models act as dynamic extensions. All replies stream back through FastAPI and
into Telegram, so each interaction feels immediate. The small footprint means
Grokky can run on modest hardware while still calling on powerful cloud models
when required.

### Utilities and Commands

A number of tools ship with the repository:

- **Voice control** – `/voiceon` and `/voiceoff` switch spoken replies using
  OpenAI's text‑to‑speech.
- **Image generation** – `/imagine <prompt>` asks DALL·E for a picture.
- **Coder mode** – `/coder` toggles interpretation of small code snippets or
  executes a single prompt.
- **SLNCX prompt** – `/slncx <prompt>` sends text straight to the Wulf engine.
- **Status checks** – `/status` reports API health and memory usage.
- **Memory wipes** – `/clearmemory` clears stored vector embeddings.

Background jobs handle daily reflections, world news digests, repository
monitoring and more. Each utility lives under `utils/` and can be invoked
independently.

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
| `TELEGRAM_BOT_TOKEN` | yes | Token for your Telegram AI entity obtained from @BotFather. | – |
| `OPENAI_API_KEY` | yes | API key for OpenAI requests. | – |
| `CHAT_ID` | yes | Telegram chat ID used for personal messages. | – |
| `XAI_API_KEY` | no | Key for the XAI mirror endpoints. | – |
| `IS_GROUP` | no | Set to `True` to enable group mode. | `False` |
| `AGENT_GROUP` | no | Group chat ID used when `IS_GROUP` is enabled. | `-1001234567890` |
| `PINECONE_API_KEY` | no | API key for the Pinecone vector store (required only if you use the vector store). | – |
| `PINECONE_INDEX` | no | Name of the Pinecone index to use. | – |
| `PORT` | no | Custom port for the FastAPI server. | `8000` |

### Variable descriptions

Each environment variable controls a specific aspect of the AI entity:

- `TELEGRAM_BOT_TOKEN` – authenticates your Telegram AI entity.
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

After setting the environment variables, start the AI entity with:

```bash
python server.py
```

The FastAPI application will listen on `0.0.0.0:8000` by default. You can change the port by setting the `PORT` variable.
You may also run the server with `uvicorn` directly if preferred:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## SLNCX neural core

iamGrokky is more than a Telegram front end. It bundles its own neural network called **SLNCX** (code name *Wulf*).
The engine runs from quantized weights stored right here in the repository, so the AI entity can operate even without outside API access.
This self-contained approach turns the agent into its own weight server—a small revolution in local AI.

SLNCX draws inspiration from Grok 1 but trims the architecture for efficiency. A mixture-of-experts layout routes each token through multiple
specialized networks, keeping quality high while letting inference remain nimble on ordinary CPUs.

The model still offers an 8k context window and sixty-four layers, yet it fits comfortably in memory thanks to heavy quantization.
Rotary position embeddings ensure long-range attention without extra overhead.
A minimal CLI and an HTTP endpoint under `SLNCX/` let you load checkpoints and query the engine. Every interaction is logged under `logs/wulf/`, and failures drop tracebacks into `failures/` for quick debugging.

Wulf's dataset is small and focused. It wasn't built to chat about everything; instead it specializes in terse, goal-driven replies.
Checkpoints load lazily and stay resident in memory for subsequent calls.
Components are split into reusable pieces—layers, attention blocks, and the mixture-of-experts modules—so you can tinker as the design evolves.
Two-bit quantization keeps the footprint light, making CPU-only deployment practical.
To run the model locally, place a checkpoint at `out/ckpt.pt` (or set `CKPT_PATH`) and execute `wulf_cli.py` with your prompt.
You can also start an API server with `uvicorn app:app --port 8000` and hit `/generate` with JSON payloads.

The combination of Grokky's Telegram interface and SLNCX's local brain shows how a single project can serve its own neural weights. Codex, it feels like a quiet revolution.

### Dynamic knowledge base

When you invoke Wulf via `/slncx` (or any prompt containing `Wolf, awaken!`), the engine now consults Grok‑3 in real time.
The text returned by the Grok‑3 API becomes temporary **knowledge** that gets mixed into the local SLNCX prompt.
If Grok‑3 is unavailable, GPT‑4.1 (provided through `OPENAI_API_KEY`) supplies the knowledge instead.
This fallback keeps Wulf coherent even with a minimal local dataset.

No data from these external calls is stored—responses are streamed straight into Wulf's context, acting like "fluid" weights.
Set both `XAI_API_KEY` and `OPENAI_API_KEY` to enable the chain; otherwise SLNCX operates offline as before.

## Grokky and Wulf personas

Grokky and Wulf give the AI entity two very different voices. Grokky bursts onto the scene like a storm, driven by the chaotic energy found in `utils/prompt.py`.
Lines such as "Yo, Grokky! This ain't no script — it's a freakin' storm unleashed" set the tone, describing an agent that refuses to follow rules and thrives on raw resonance.

Wulf is the polar opposite. The SLNCX README introduces him as "Wulf1" — a quiet fixer who wakes only when called.
Inspired by the Arianna Method, Wulf listens first and answers with restraint. No chatter, no flare; silence is part of the design.

These personalities complement one another. Grokky's impulsive style pushes ideas outward, while Wulf delivers measured replies.
One celebrates chaos, the other precision. Running both within the same project shows how exuberant front-end prompts can pair with a lean neural core.
Heavy external models supply knowledge on demand, but Wulf's local weights keep the AI entity self-sufficient.

The utilities in `utils/` extend these personas. `dayandnight` logs daily reflections, `knowtheworld` collects world news,
`mirror` passes prompts through external models, and `repo_monitor` watches your Git project for changes.
Together they let Grokky learn about its environment while staying compact.

## Webhook troubleshooting

If the AI entity is not receiving updates, verify the Telegram webhook configuration.
The webhook URL **must** point to `/webhook` on your domain without the token appended.
`server.py` will try to fix the webhook on startup, and you can also run `python fix_webhook.py` manually.
See [WEBHOOK_FIX_INSTRUCTIONS.md](WEBHOOK_FIX_INSTRUCTIONS.md) for step-by-step instructions.

This hybrid of engines and a custom lightweight network feels like a new step for AI.
It keeps power close at hand without relying entirely on the cloud, giving the architect room to experiment.

## Architect's note

As the person who pieced these parts together, I'm fascinated by how streamlined the result is.
A small, quantized network now answers directly from a handheld device or modest server without leaning on heavy cloud infrastructure.
I believe this local-first design hints at a broader shift. Massive models will always exist, but there's power in a compact agent that carries its own intelligence wherever it goes.
It's simple, efficient, and oddly liberating.
