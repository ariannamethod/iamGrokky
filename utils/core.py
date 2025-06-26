import os
import requests
import asyncio
import aiohttp
from gtts import gTTS
import io

XAI_API_KEY = os.getenv("XAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
VOICE_MODE = False

async def query_grok_async(user_message):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": user_message}], "max_tokens": 2048, "temperature": 1.2}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["choices"][0]["message"]["content"]

def query_grok(user_message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(query_grok_async(user_message))

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

async def send_voice_message(chat_id, text):
    tts = gTTS(text=text, lang='ru')
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
        files = {"voice": f}
        payload = {"chat_id": chat_id}
        requests.post(url, data=payload, files=files)

def toggle_voice_mode(command):
    global VOICE_MODE
    if command == "/voiceon":
        VOICE_MODE = True
        return "Грокки слушает и говорит! 🎙️"
    elif command == "/voiceoff":
        VOICE_MODE = False
        return "Грокки молчит, текст рулит. 📝"
    return None
