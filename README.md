# Grokky-2

Grokky AI Assistant is a Telegram bot built with **Aiogram** and `aiohttp`. It contains the server and utility scripts needed to run the assistant.

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
| `XAI_API_KEY` | no | Key for the XAI mirror endpoints. | – |
| `IS_GROUP` | no | Set to `True` to enable group mode. | `False` |
| `AGENT_GROUP` | no | Group chat ID used when `IS_GROUP` is enabled. | `-1001234567890` |
| `PORT` | no | Custom port for the server. | `8000` |

### Variable descriptions

Each environment variable controls a specific aspect of the bot:

- `TELEGRAM_BOT_TOKEN` – authenticates your Telegram bot.
- `OPENAI_API_KEY` – allows requests to OpenAI.
- `XAI_API_KEY` – key for the XAI mirror endpoints (optional).
- `IS_GROUP` – toggles group mode.
- `AGENT_GROUP` – group chat ID used when `IS_GROUP` is `True`.
- `PORT` – port for the server.

Unused optional variables are ignored when their features are disabled.

## Running the server

After setting the environment variables, start the bot with:

```bash
python server.py
```

The server listens on `0.0.0.0:8000` by default. Change the port with the `PORT` variable if needed.

## Webhook troubleshooting

If the bot is not receiving updates, ensure the Telegram webhook URL is set to `https://<your-domain>/webhook` (without appending the bot token). You can set it via [BotFather](https://t.me/BotFather) or the Telegram API.
