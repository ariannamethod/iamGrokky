import os
import requests
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
chat_id_env = os.getenv("CHAT_ID")
is_group_env = os.getenv("IS_GROUP", "False").lower() == "true"
agent_group_env = os.getenv("AGENT_GROUP", "-1001234567890")

system_prompt = build_system_prompt(
    chat_id=chat_id_env,
    is_group=is_group_env,
    AGENT_GROUP=agent_group_env
)

def query_grok(user_message, chat_context=None, author_name=None, attachments=None):
    url = "https://api.x.ai/v1/chat/completions"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]

    # Try to parse for raw function call (JSON) and route to utility if needed
    try:
        import json as pyjson
        # Look for function_call pattern in reply, strictly at start of string
        if reply.strip().startswith('{'):
            data = pyjson.loads(reply)
            if "function_call" in data:
                fn = data["function_call"]["name"]
                args = data["function_call"]["arguments"]
                if fn == "genesis2_handler":
                    return handle_genesis2(args)
                elif fn == "vision_handler":
                    return handle_vision(args)
                elif fn == "impress_handler":
                    return handle_impress(args)
                else:
                    # Unknown function, fallback to plain text
                    return f"Grokky raw: {reply}"
    except Exception as e:
        # If not JSON, just return text
        pass

    return reply

def handle_genesis2(args):
    ping = args.get("ping")
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = genesis2_handler(
        ping=ping,
        group_history=group_history,
        personal_history=personal_history,
        is_group=is_group,
        author_name=author_name,
        raw=raw
    )
    # Always reply in raw JSON for full chaos
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

def handle_vision(args):
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = vision_handler(
        image_bytes_or_url=image,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

def handle_impress(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = impress_handler(
        prompt=prompt,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text")
    chat_id = message.get("chat", {}).get("id")
    author_name = message.get("from", {}).get("first_name") or "anon"
    # Handle photo/image if present
    attachments = []
    if "photo" in message and message["photo"]:
        # Take the largest (last) photo
        file_id = message["photo"][-1]["file_id"]
        # Get file path via Telegram API
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        ).json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)
    reply_text = ""
    if attachments:
        # If photo, always trigger vision_handler, with chat context
        reply_text = handle_vision({
            "image": attachments[0],
            "chat_context": user_text or "",
            "author_name": author_name,
            "raw": True
        })
    elif user_text:
        reply_text = query_grok(user_text, chat_context=None, author_name=author_name)
    else:
        reply_text = "Grokky got nothing to say to static void."
    send_telegram_message(chat_id, reply_text)
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "Grokky alive and wild!"}
