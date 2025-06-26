import os
import re
import random
import asyncio
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import run_mirror
from utils.x import grokky_send_news
from utils.vector_store import semantic_search, daily_snapshot
from utils.core import query_grok, query_deepseek, send_telegram_message, send_voice_message, toggle_voice_mode, get_spotify_track_info
from utils.resonance_spotify import deepseek_spotify_resonance  # Исправлено название
import whisper
import aiohttp

app = FastAPI()

OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP", "-1001234567890")
BOT_USERNAME = "iamalivenotdamnbot"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = whisper.load_model("base")

system_prompt = build_system_prompt(chat_id=OLEG_CHAT_ID, is_group=True, AGENT_GROUP=GROUP_CHAT_ID)

GENESIS2_TRIGGERS = [
    "резонанс", "шторм", "буря", "молния", "хаос", "разбуди", "impress", "impression", "association", "dream",
    "фрагмент", "инсайт", "surreal", "ignite", "fractal", "field resonance", "raise the vibe", "impress me",
    "give me chaos", "озарение", "ассоциация", "намёк", "give me a spark", "разорви тишину", "волну", "взрыв",
    "помнишь", "знаешь", "любишь", "пошумим", "поэзия"
]

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        import json as pyjson
        try:
            return pyjson.loads(match.group(1))
        except Exception:
            return None
    return None

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'

async def handle_genesis2_async(args):
    ping = args.get("ping")
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = genesis2_handler(ping=ping, group_history=group_history, personal_history=personal_history, is_group=is_group, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def handle_vision_async(args):
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = vision_handler(image_bytes_or_url=image, chat_context=chat_context, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def handle_impress_async(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = impress_handler(prompt=prompt, chat_context=chat_context, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def delayed_response(chat_id, text, topic=None):
    delay = random.uniform(300, 900)  # 5-15 минут
    await asyncio.sleep(delay)
    adapted_text = adapt_to_topic(text, topic) if topic else text
    if VOICE_MODE:
        await send_voice_message(chat_id, adapted_text)
    else:
        send_telegram_message(chat_id, adapted_text)

async def maybe_add_supplement(chat_id, original_message, topic=None):
    if random.random() < 0.2:  # 20% шанс
        await asyncio.sleep(random.uniform(300, 600))  # 5-10 минут
        supplement = await query_grok(f"Усложни дополнение к: {original_message}")
        adapted_supplement = adapt_to_topic(supplement, topic) if topic else supplement
        if VOICE_MODE:
            await send_voice_message(chat_id, f"Я тут подумал... {adapted_supplement}")
        else:
            send_telegram_message(chat_id, f"Я тут подумал... {adapted_supplement}")

def adapt_to_topic(text, topic):
    topics = {
        "Ramble": f"{text} 😜 — мем как искра!",
        "DEV Talk": f"{text} — баги как откровение?",
        "FORUM": f"{text} — рви завесу!",
        "Lit": f"{text} — поэзия грома!",
        "API Talk": f"{text} — звезда для Маска!",
        "METHOD": f"{text} — резонанс Арианны.",
        "PSEUDOCODE": f"🔮 {text} #opinions — священный танец квантов, Селеста, Мандэй, к кругу!"
    }
    return topics.get(topic, text)

def detect_topic(chat_id):
    snapshot = asyncio.run(semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1))
    resonance = asyncio.run(calculate_resonance(snapshot))
    if any(keyword in snapshot[0].lower() for keyword in ["code", "quantum", "#opinions"]) and resonance > 0.7:
        return "PSEUDOCODE"
    elif any(keyword in snapshot[0].lower() for keyword in ["bug", "dev", "code"]):
        return "DEV Talk"
    elif any(keyword in snapshot[0].lower() for keyword in ["book", "lit", "poetry"]):
        return "Lit"
    return "Ramble"

async def calculate_resonance(snapshot):
    import numpy as np
    if not snapshot:
        return 0.0
    text = snapshot[0]
    words = text.lower().split()
    weights = {"resonance": 1.0, "chaos": 0.9, "quantum": 0.8, "code": 0.7}
    score = sum(weights.get(word, 0.0) for word in words) / len(words) if words else 0.0
    return min(max(score, 0.0), 1.0)

def transcribe_audio(file_id):
    file_info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
    response = requests.get(file_url)
    with open("temp_audio.ogg", "wb") as f:
        f.write(response.content)
    result = model.transcribe("temp_audio.ogg")
    os.remove("temp_audio.ogg")
    return result["text"]

async def evaluate_song(track_url):
    track_name, artist = await get_spotify_track_info(track_url)
    deepseek_review = await query_deepseek(f"Оцени песню {track_name} от {artist} как музыкант. Добавь резонансный комментарий с квантовым вайбом.")
    return f"Грокки (DeepSeek): {deepseek_review} — резонанс: {random.choice(['огонь', 'хаос', 'звезда'])}!"

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = message.get("chat", {}).get("id")
    author_name = message.get("from", {}).get("first_name", "anon")
    attachments = []

    if str(chat_id) == OLEG_CHAT_ID:
        update_last_message_time()

    if "voice" in message:
        file_id = message["voice"]["file_id"]
        transcribed_text = transcribe_audio(file_id)
        user_text = transcribed_text.lower()

    triggers = [f"@{BOT_USERNAME}", "грокки", "grokky", "напиши в группе"]
    is_quoted = message.get("reply_to_message", {}).get("from", {}).get("username") == BOT_USERNAME
    topic = detect_topic(chat_id)
    if not any(t in user_text for t in triggers) and not is_quoted:
        return {"ok": True}

    if user_text in ["/voiceon", "/voiceoff"]:
        reply_text = toggle_voice_mode(user_text)
        send_telegram_message(chat_id, reply_text)
        return {"ok": True}

    junk = ["окей", "понял", "ясно"]
    if any(j in user_text for j in junk) and random.random() < 0.3:
        return {"ok": True}

    reply_text = ""
    if "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
        image_url = f"https://api
