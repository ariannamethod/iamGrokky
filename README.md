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

Create a `.env` file in the project root using `.env.example` as a template. The following variables are required:

- `TELEGRAM_BOT_TOKEN` – token from @BotFather.
- `OPENAI_API_KEY` – OpenAI API key.
- `CHAT_ID` – Telegram chat ID for direct messages.

Optional variables enable extra features:

- `XAI_API_KEY` – for the XAI mirror endpoints.
- `IS_GROUP` and `AGENT_GROUP` – enable group mode.
- `PINECONE_API_KEY` and `PINECONE_INDEX` – use the Pinecone vector store.
- `NEWS_API_KEY` – API key for news retrieval.
- `PORT` – custom server port (defaults to 8000).

Refer to `.env.example` for the complete list of supported variables and default values.

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
