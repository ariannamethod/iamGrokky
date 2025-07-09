# Grokky-2

Grokky AI Assistant is a Telegram bot powered by FastAPI. This project contains the server code and utility scripts used to run the assistant.

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
| `NEWS_API_KEY` | no | Key for retrieving news headlines. | – |
| `PORT` | no | Custom port for the FastAPI server. | `8000` |

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

## Fixing the Telegram webhook

Telegram requires a valid webhook URL before it can deliver messages to your bot.
Run the helper script after setting up your `.env` file:

```bash
python fix_webhook.py
```

The script checks that `TELEGRAM_BOT_TOKEN` is real, removes any old webhook and
sets a new one using the public URL of your deployment.

**Common pitfalls**

- Keeping the placeholder token (`your_telegram_bot_token_here`) causes Telegram
  to hit `/webhook` with an invalid token, leading to `404` errors.
- Ensure the application URL you provide is reachable from the internet so that
  Telegram can access `<app-url>/webhook`.

## Running the tests

This repository contains self‑contained scripts that demonstrate key features.
Execute them from the project root after installing the dependencies:

```bash
python test_group_functionality.py      # group triggers and document handling
python test_memory_fix.py               # unified memory tests
python test_server_integration.py       # server integration checks
```

The tests rely only on the installed dependencies and do not require actual
Telegram or OpenAI credentials.
