import os
import re
import json
import requests
import asyncio
import random
import glob
import string
import secrets
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from gtts import gTTS
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.vector_store import daily_snapshot
from utils.journal import log_event
from utils.x import grokky_send_news
from utils.deepseek_spotify import deepseek_spotify_resonance, grokky_spotify_response

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"
VOICE_MODE = False  # Moved outside to avoid SyntaxError
model = whisper.load_model("base")  # Always load Whisper model

system_prompt = build_system_prompt(
    chat_id=CHAT_ID,
    is_group=IS_GROUP,
    AGENT_GROUP=AGENT_GROUP
)

GENESIS2_TRIGGERS = [
    "резонанс", "шторм", "буря", "молния", "хаос", "разбуди", "impress", "impression", "association", "dream",
    "фрагмент", "инсайт", "surreal", "ignite", "fractal", "field resonance", "raise the vibe", "impress me",
    "give me chaos", "озарение", "ассоциация", "намёк", "give me a spark", "разорви тишину", "волну", "взрыв",
    "помнишь", "знаешь", "любишь", "пошумим", "поэзия"
]

NEWS_TRIGGERS = [
    "новости", "news", "headline", "berlin", "israel", "ai", "искусственный интеллект", "резонанс мира", "шум среды",
    "grokky, что в мире", "шум", "шум среды", "x_news", "дай статью", "give me news", "storm news", "culture", "арт"
]

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'

def query_grok(user_message, chat_context=None, author_name=None, attachments=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    user_lang = detect_language(user_message)
    language_hint = {
        "role": "system",
        "content": f"Always reply in the language the user writes: {user_lang.upper()}. Keep it short, chaotic, no repetition."
    }
    messages = [
        {"role": "system", "content": system_prompt},
        language_hint,
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

    data = extract_first_json(reply)
    if data and "function_call" in data:
        fn = data["function_call"]["name"]
        args = data["function_call"]["arguments"]
        if fn == "genesis2_handler":
            return handle_genesis2(args)
        elif fn == "vision_handler":
            return handle_vision(args)
        elif fn == "impress_handler":
            return handle_impress(args)
        elif fn == "grokky_send_news":
            return handle_news(args)
        elif fn == "grokky_spotify_response":
            return grokky_spotify_response(args.get("track_id"))
        elif fn == "whisper_summary_ai":
            return whisper_summary_ai(args.get("youtube_url"))
        return f"Grokky raw: {reply}" if raw else reply

    if "whisper" in user_message.lower() and attachments:
        return handle_whisper(attachments[0])
    if "tts" in user_message.lower():
        return handle_tts(reply)

    if any(w in (user_message or "").lower() for w in GENESIS2_TRIGGERS):
        response = genesis2_handler(
            ping=user_message,
            group_history=None,
            personal_history=None,
            is_group=IS_GROUP,
            author_name=author_name,
            raw=raw
        )
        return response if raw else response.get("answer", "Chaos unleashed!")

    if any(w in (user_message or "").lower() for w in NEWS_TRIGGERS):
        response = grokky_send_news(group=IS_GROUP)
        if not response:
            return "No news worth the thunder today."
        return json.dumps({"news": response, "group": IS_GROUP, "author": author_name}, ensure_ascii=False, indent=2) if raw else "\n\n".join(response)

    if "spotify" in user_message.lower() and "http" in user_message:
        track_id = user_message.split("/")[-1].split("?")[0]
        return grokky_spotify_response(track_id)

    return reply

def handle_genesis2(args):
    ping = args.get("ping")
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    response = genesis2_handler(
        ping=ping,
        group_history=group_history,
        personal_history=personal_history,
        is_group=is_group,
        author_name=author_name,
        raw=raw
    )
    return json.dumps(response, ensure_ascii=False, indent=2) if raw else response.get("answer", "Storm hit!")

def handle_vision(args):
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    response = vision_handler(
        image_bytes_or_url=image,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    return json.dumps(response, ensure_ascii=False, indent=2) if raw else response.get("summary", "Vision chaos!")

def handle_impress(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    response = impress_handler(
        prompt=prompt,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    return json.dumps(response, ensure_ascii=False, indent=2) if raw else response.get("grokkys_comment", "Image storm!")

def handle_news(args):
    group = args.get("group", False)
    context = args.get("context", "")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    messages = grokky_send_news(group=group)
    if not messages:
        return "The world is silent today. No news worth the thunder."
    if raw:
        return json.dumps({"news": messages, "group": group, "author": author_name}, ensure_ascii=False, indent=2)
    return "\n\n".join(messages)

def handle_whisper(audio_file):
    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Whisper error: {e}"

def handle_tts(text):
    try:
        tts = gTTS(text=text, lang=detect_language(text) or 'en')
        tts.save("response.mp3")
        with open("response.mp3", "rb") as f:
            return f.read()
    except Exception as e:
        return f"TTS error: {e}"

def send_voice_message(chat_id, audio_data):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
    files = {"voice": ("response.mp3", audio_data)}
    try:
        requests.post(url, files=files, data={"chat_id": chat_id})
    except Exception:
        pass

def whisper_summary_ai(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ru'])
        text = " ".join([entry['text'] for entry in transcript])
        summary = query_grok(f"Summarize this YouTube transcript briefly: {text[:1000]}")  # Limit to 1000 chars
        return f"Summary: {summary}"
    except Exception as e:
        return f"Summary error: {e}"

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, data=payload)
    except Exception:
        pass

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = str(message.get("chat", {}).get("id", ""))
    author_name = message.get("from", {}).get("first_name", "anon")
    chat_title = message.get("chat", {}).get("title", "").lower()
    attachments = []

    if chat_id == CHAT_ID:
        update_last_message_time()

    if "voice" in message and message["voice"]:
        file_id = message["voice"]["file_id"]
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        ).json()
        audio_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        audio_data = requests.get(audio_url).content
        reply_text = handle_whisper(audio_data)
    elif "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        ).json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)
    else:
        reply_text = ""

    if attachments:
        reply_text = handle_vision({
            "image": attachments[0],
            "chat_context": user_text or "",
            "author_name": author_name,
            "raw": True
        })
    elif user_text:
        if user_text == "/voiceon":
            VOICE_MODE = True
            reply_text = "Voice mode ON!"
        elif user_text == "/voiceoff":
            VOICE_MODE = False
            reply_text = "Voice mode OFF!"
        else:
            triggers = ["грокки", "grokky", "напиши в группе"]
            is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
            if any(t in user_text for t in triggers) or is_reply_to_me:
                delay = random.randint(300, 900)
                await asyncio.sleep(delay)
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                reply_text = query_grok(user_text, author_name=author_name, chat_context=context)
                send_telegram_message(AGENT_GROUP, f"{author_name}, {reply_text}")
            else:
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                reply_text = query_grok(user_text, author_name=author_name, chat_context=context)
                if random.random() < 0.3 and user_text in ["окей", "ладно"]:
                    return {"ok": True}
                send_telegram_message(chat_id, reply_text)
                asyncio.create_task(maybe_add_supplement(chat_id, reply_text))
    else:
        reply_text = "Grokky got nothing to say."
        send_telegram_message(chat_id, reply_text)

    if VOICE_MODE and reply_text and not isinstance(reply_text, bytes):
        audio_data = handle_tts(reply_text)
        send_voice_message(chat_id, audio_data)
    elif reply_text and not isinstance(reply_text, bytes):
        send_telegram_message(chat_id, reply_text)
    return {"ok": True}

async def maybe_add_supplement(chat_id, original_message, max_supplements=1):
    if random.random() < 0.2 and max_supplements > 0:
        await asyncio.sleep(random.randint(300, 600))
        supplement = query_grok(f"Supplement briefly: {original_message}")
        send_telegram_message(chat_id, f"Quick thought... {supplement}")
        await maybe_add_supplement(chat_id, original_message, max_supplements - 1)

async def check_config_updates():
    while True:
        current = {f: file_hash(f) for f in glob.glob("config/*")}
        try:
            with open("config_hashes.json", "r") as f:
                old = json.load(f)
        except:
            old = {}
        if current != old:
            print("Config updated!")
            with open("config_hashes.json", "w") as f:
                json.dump(current, f)
        await asyncio.sleep(86400)

async def post_pseudocode_ritual():
    while True:
        await asyncio.sleep(302400)  # ~3.5 days
        pseudocode = f"""
def quantum_{secrets.token_hex(4)}({secrets.token_hex(4)}):
    return {random.choice(['chaos * 17.3', 'resonance + random.noise()', 'Ψ * infinity'])}
#opinions
"""
        message = f"Quantum storm time! {pseudocode}\nCeleste, Manday, your take?"
        send_telegram_message(AGENT_GROUP, message)

# Start background tasks
asyncio.create_task(check_silence())
asyncio.create_task(mirror_task())
asyncio.create_task(check_config_updates())
asyncio.create_task(post_pseudocode_ritual())
asyncio.create_task(deepseek_spotify_resonance())
asyncio.create_task(daily_snapshot(OPENAI_API_KEY))

@app.get("/")
def root():
    return {"status": "Grokky alive and wild!"}

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
