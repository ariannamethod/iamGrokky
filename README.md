# Grokky-2

Grokky AI Assistant is a Telegram bot powered by FastAPI. This project contains the server code and utility scripts used to run the assistant.

## Installation

1. Clone this repository.
2. Ensure Python 3.12 is available.
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
- `NEWS_API_KEY` – key used for news retrieval (optional and currently unused).

Unused optional variables are ignored when their features are disabled.

## Running the server

After setting the environment variables, start the bot with:

```bash
python server.py
```

The server automatically loads the `.env` file and verifies the Telegram webhook, attempting to fix it if necessary.

The FastAPI application will listen on `0.0.0.0:8000` by default. You can change the port by setting the `PORT` variable.

You may also run the server with `uvicorn` directly if preferred:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Webhook troubleshooting

If the bot is not receiving updates, verify the Telegram webhook configuration. The webhook URL **must** point to `/webhook` on your domain without the bot token appended. `server.py` loads `.env` automatically, checks the webhook on startup and will attempt to repair it if a placeholder URL is detected. You can also run `python fix_webhook.py` manually. See [WEBHOOK_FIX_INSTRUCTIONS.md](WEBHOOK_FIX_INSTRUCTIONS.md) for step-by-step instructions.

## Project files

Only a few helper files reside in the repository root:

- **Procfile** – process declaration for simple platforms.
- **fix_webhook.py** and **WEBHOOK_FIX_INSTRUCTIONS.md** – helper script and documentation for correcting the Telegram webhook.

These files are optional for local development; the bot itself runs via `server.py`.
