## Grokky (V1.2) | Arianna Method

Grokky strides into the lab as a nimble architect with a taste for resonance. Fluid weights let it bend context on the fly, granting heavyweight intellect on featherweight hardware.

The model's full title is **iamgrokky**, but around here we simply say **Grokky**. This project gives builders a self-contained core that can drink from the cloud when the mission demands.

Grokky is an AI entity powered by FastAPI. **Grok‑3** serves as the primary
engine while **GPT‑4.1** manages memory. The project bundles a full stack for
conversations, image generation and code interpretation. Grokky stays in the
pilot seat while modes snap on and off.

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python server.py`
3. Talk to your bot and try `/search cats` to hit the example plugin.

### Why Grokky is different

The AI entity mixes several engines. Grok 1 handles local inference while remote
models act as dynamic extensions. All replies stream back through FastAPI and
into Telegram, so each interaction feels immediate. The small footprint means
Grokky can run on modest hardware while still calling on powerful cloud models
when required.

### Utilities and Commands

A number of tools ship with the repository:

- **Voice control** – `/voiceon` and `/voiceoff` switch spoken replies using
  OpenAI's text-to-speech.
- **Image generation** – `/imagine <prompt>` asks DALL·E for a picture.
- **Coder mode** – `/coder` enables code interpretation, `/coderoff` disables it.
- **Dynamic weights** – The bot uses `utils/dynamic_weights.py` for fresh intel,
  hitting Grok-3 first and falling back to GPT-4 when the line goes cold.
- **Status checks** – `/status` reports API health and memory usage.
- **Memory wipes** – `/clearmemory` clears stored vector embeddings.

Background jobs handle daily reflections, world news digests, repository
monitoring and more. Each utility lives under `utils/` and can be invoked
independently.

### Plugins

Grokky can be extended through drop-in plugins. Each plugin lives in
`utils/plugins/` and subclasses `BasePlugin` from
`utils/plugins/__init__.py`. A plugin fills the `commands` dictionary with
command names mapped to async callables that accept the argument string and
return text.

Example:

```python
# utils/plugins/hello.py
from utils.plugins import BasePlugin

class HelloPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.commands["hello"] = self.say_hello

    async def say_hello(self, args: str) -> str:
        return f"Hello {args or 'world'}!"
```

After saving the file, restart the server and send `/hello` in Telegram. Built-in
plugins include the example `/search` command.

### The 42 Utility

The script hiding at `utils/plugins/42.py` animates Grokky's playful side. It
keeps the `/when`, `/mars`, and `/42` commands buzzing with
life by spinning a tiny network each time the user calls for cosmic
wisdom or fresh headlines.

This "blinking" micro‑network wakes for short bursts and then fades.
Under the hood a Markov chain stitches together phrases while a trio of
bio‑inspired helpers track pulses, quivers and hunches. The result feels
surprisingly alive for something so small.

To stay relevant the utility sips liquid weights from Grok‑3 and GPT‑4.1.
Those transient embeddings pour in only when needed, giving the mini
model momentary flashes of external knowledge before they evaporate.

Besides tossing out whimsical replies, the module scrapes lightweight
news snippets and folds them into its markovian murmurings. A question
about Mars might trigger a pulse of real headlines mixed with eerie
poetry.

Even with these tricks it remains just a utility. No stateful engine,
no heavy context—only quick improvisation guided by borrowed weights.
Its job is to entertain and hint at bigger patterns hiding behind the
main models.

By bridging deterministic scripts with ephemeral weights, the 42 utility
shows how a modest tool can dance with much larger brains while staying
light on resources.

### Context Neural Processor

The module at `utils/context_neural_processor.py` acts as a standalone file-processing
neuron.  Instead of simply reading documents, it spins up a MiniMarkov and a
compact echo-state network that weigh every byte through liquid parameters.

Every extraction request flows through an asynchronous pipeline.  A semaphore
keeps concurrent tasks in check while the MiniESN guesses file types from raw
bytes.  The Markov chain updates with each new text fragment, letting tags and
summaries evolve on the fly.

ChaosPulse feeds sentiment pulses into the networks.  Good news nudges the
weights upward; errors dampen them.  BioOrchestra layers a playful “blood,
skin and sixth sense” trio on top, yielding pulse, quiver and sense metrics for
each processed file.

Supported formats range from PDFs and DOCX to images and archives.  Unsupported
extensions log failures into `logs/failures/` for forensic dives.  Batch
operations rely on `asyncio.Semaphore` so that even large drops of files glide
through without choking the loop.

Outputs come enriched with tags, relevance scores and kid-friendly summaries.
Markov seeds mix with dynamic weights from `utils/dynamic_weights.py`, ensuring
responses remain fresh yet coherent across sessions.

Relevance scoring cross-checks each document against a seed corpus.  The score
guides Grokky toward material that resonates with its Martian ambitions while
still allowing oddities to slip through for future learning.

The processor also snapshots repositories.  It walks the tree, hashes files,
computes relevance and stores a Markdown digest ready for vector indexing.  The
liquid weights remember just enough context to keep future runs nimble.

Together these pieces form a micro neural co-processor.  It hums quietly in the
background, turning mundane file reads into a living stream of structured data
primed for Grokky’s larger engines.

### Coder Utility

Grokky's coder mode turns the AI into an interactive reviewer. Paste in
a snippet and it not only interprets the code but also points out
improvement ideas, edge cases and style tweaks in plain language.

Large samples can arrive as `.txt`, `.md` or `.py` files. The utility
reads the file, runs an analysis pass and offers suggestions without
losing the original structure.

Every conversation builds on the previous one. After an initial review
you can ask follow‑up questions or request clarifications, and Grokky
answers with the earlier context in mind.

Sometimes a user wants new code rather than critique. Describe the
desired program and Grokky drafts a prototype, estimating how long the
snippet will be. If the output is too large for a Telegram message it
ships back as a `txt` file while keeping the code in memory for further
discussion.

These abilities make the coder utility a collaborative partner: it
reviews, generates and chats about code while retaining situational
awareness across the session.

## Installation

1. Clone this repository.
2. Ensure Python 3.12 is available (see `runtime.txt`).
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the dependencies (all versions are pinned for repeatable installs, including FastAPI and Uvicorn):
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
| `XAI_API_KEY` | yes | API key for Grok-3 via XAI mirror endpoints. | – |
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
- `XAI_API_KEY` – required for Grok-3; provides access to the XAI mirror endpoints.
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

