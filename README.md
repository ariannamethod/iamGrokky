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

### Feature notes

Optional variables activate additional functionality:

- `XAI_API_KEY` enables the XAI mirror endpoints.
- `IS_GROUP` and `AGENT_GROUP` enable group mode.
- `PINECONE_API_KEY` and `PINECONE_INDEX` enable the Pinecone vector store.
- `NEWS_API_KEY` is required for news retrieval (can be ignored if that feature is disabled).

Unused optional variables are ignored when their features are disabled. Currently `NEWS_API_KEY` is unused unless you enable news retrieval.

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
