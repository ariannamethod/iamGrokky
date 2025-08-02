## iamGrokky

Grokky is a Telegram-first AI assistant built with FastAPI. The project began as a straightforward bot interface, but it now ships with its own compact neural model called **SLNCX (Wulf)**. The agent loads its quantized weights directly and acts as a self-contained inference engine. No external GPU services are required. This setup keeps conversations fast and private.

### Why SLNCX?

SLNCX stands for *Silencieux*. It wakes only when called, like the fixer from "Pulp Fiction"—quiet, efficient and precise. The design borrows ideas from Grok1 but pares them down. Two-bit quantization and mixture-of-experts routing let the model run smoothly on standard CPUs. A large context window and rotary embeddings keep it comfortable in long chats.

The Arianna Method shapes each exchange. Wulf listens first, then answers with restraint. Silence is part of the philosophy. This approach avoids the overhead of constant chatter while still providing thoughtful replies when needed.

### Core Features

- **Mixture of Experts**: eight experts per layer with two chosen per token.
- **Large Context Window**: up to 8,192 tokens for continuity.
- **Layer Chaos**: 64 stacked blocks for diverse routing.
- **Rotary Position Embeddings**: stable long-range attention.
- **2-bit Quantization**: fits the model in memory for CPU-only inference.

The CLI loads a checkpoint and responds immediately. `wulf_cli.py` offers the fastest way to query the model locally. The same network also powers the `/generate` API endpoint for simple HTTP requests.

### Installation

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

### Environment Variables

Create a `.env` file in the project root using `.env.example` as a template. Each variable controls a specific aspect of the bot.

| Variable | Required | Description | Default |
|----------|---------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | yes | Telegram bot token. | – |
| `OPENAI_API_KEY` | yes | Key for optional OpenAI requests. | – |
| `CHAT_ID` | yes | Telegram chat ID for personal messages. | – |
| `XAI_API_KEY` | no | Key for the XAI mirror endpoints. | – |
| `IS_GROUP` | no | Set to `True` for group mode. | `False` |
| `AGENT_GROUP` | no | Group chat ID when `IS_GROUP` is `True`. | `-1001234567890` |
| `PINECONE_API_KEY` | no | Pinecone vector store key. | – |
| `PINECONE_INDEX` | no | Name of the Pinecone index. | – |
| `PORT` | no | Port for the FastAPI server. | `8000` |

Unused optional variables are ignored when their features are disabled.

### Running the Server

After configuring the environment, start the bot with:

```bash
python server.py
```

The application listens on `0.0.0.0:8000` by default. You can change the port with the `PORT` variable or run the server via `uvicorn`:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Logging and Memory

All requests leave JSON entries in `logs/wulf/`. Failures capture tracebacks under `failures/`. Checkpoints load lazily and remain in memory for subsequent calls. The dataset in `SLNCX/datasets` keeps responses sharp without growing unwieldy.

### Development

Run `pytest` to execute the test suite and `ruff .` to lint the code. The `scripts` directory contains helpers such as `session_logger.py` and `read_session_logs.py`.

### Deployment on Railway

1. Create a Railway project and point it at this repository.
2. Set the start command to `python app.py`.
3. Upload your `out/ckpt.pt` file as a deployment asset or volume.
4. Deploy and query the `/generate` endpoint with JSON like:

```json
{
  "user": "alice",
  "prompt": "Hello"
}
```

### Voice Support

Grokky replies with voice when `/voiceon` is enabled, synthesizing speech using OpenAI's `onyx` voice.

### Webhook Troubleshooting

If the bot is not receiving updates, verify the Telegram webhook configuration. The webhook URL **must** point to `/webhook` on your domain without the bot token appended. Run `python fix_webhook.py` manually if needed. See `WEBHOOK_FIX_INSTRUCTIONS.md` for details.

---

## Architect's Note

This project blends a lightweight neural network with the practicality of a Telegram interface. By keeping the weights close at hand, Grokky stays responsive even without GPU access. It is not perfect, but it proves that small, well-routed models can still deliver. 

I believe AI agents should respect silence as much as speech. Wulf embodies that principle: it listens, answers, and then steps aside. In an era of noisy bots, a quiet helper feels surprisingly fresh.

