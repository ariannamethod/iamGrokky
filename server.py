import os
import requests
import asyncio
import aiohttp
from gtts import gTTS
import io
from utils.prompt import build_system_prompt

XAI_API_KEY = os.getenv("XAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VOICE_MODE = False

async def query_grok_async(user_message):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": user_message}], "max_tokens": 2048, "temperature": 1.2}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["choices"][0]["message"]["content"]

async def query_deepseek_async(user_message):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-r1", "messages": [{"role": "user", "content": user_message}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["choices"][0]["message"]["content"]

def query_grok(user_message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(query_grok_async(user_message))

def query_deepseek(user_message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(query_deepseek_async(user_message))

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
        return "Грокки готов говорить и слушать! 🎵"
    elif command == "/voiceoff":
        VOICE_MODE = False
        return "Грокки вернулся к тексту. 📝"
    return None

async def get_spotify_track_info(track_url):
    token = get_spotify_token()
    track_id = track_url.split("/")[-1].split("?")[0]
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data["name"], data["artists"][0]["name"]

def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "client_credentials"}
    response = requests.post(url, data=payload, headers=headers)
    return response.json()["access_token"]
