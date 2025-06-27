import os
import random
import asyncio
import aiohttp
import base64
import requests
from utils.vector_store import semantic_search
from utils.journal import log_event

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

async def get_deepseek_poem(mood):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": f"Напиши стихотворение о настроении группы: {mood}"}]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["choices"][0]["message"]["content"]

def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "client_credentials"}
    response = requests.post(url, data=payload, headers=headers)
    return response.json()["access_token"]

async def get_spotify_playlist(mood):
    token = get_spotify_token()
    if not token:
        return []
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_genres": mood.lower(), "limit": 5}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            tracks = (await response.json())["tracks"]
            return [track["name"] for track in tracks]

async def deepseek_spotify_resonance():
    while True:
        await asyncio.sleep(86400)  # Once a day
        snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
        mood = analyze_mood(snapshot)
        if mood["score"] > 0.5:
            poem = await get_deepseek_poem(mood["label"])
            playlist = await get_spotify_playlist(mood["label"])
            message = f"DeepSeek-Spotify: Настроение группы: {mood['label']} ({mood['score']:.2f})\n\n{poem}\n\nПлейлист: {', '.join(playlist)}"
            if GROUP_CHAT_ID:
                await send_telegram_message(GROUP_CHAT_ID, message)
            if CHAT_ID:
                await send_telegram_message(CHAT_ID, message)  # Добавлена отправка в личку
            log_event({"type": "deepseek_spotify", "message": message})

def analyze_mood(snapshot):
    return {"label": "хаос", "score": random.uniform(0, 1)}  # Локализовал

def grokky_spotify_response(track_id):
    token = get_spotify_token()
    if not token:
        return "Не удалось получить токен Spotify"
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        track_data = r.json()
        analysis = requests.post(
            "https://api.deepseek.com/v1/analyze",  # Проверь эндпоинт
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"text": track_data["name"]}
        ).json()
        return f"Вайбы Грокки: {analysis.get('analysis', 'Нет анализа')} для {track_data['name']}"
    except Exception as e:
        return f"Ошибка анализа Spotify: {e}"

async def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                response.raise_for_status()
    except Exception:
        pass
