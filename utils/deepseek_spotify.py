import os
import random
import asyncio
import aiohttp
import base64
import requests
from utils.core import send_telegram_message
from utils.vector_store import semantic_search

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

async def get_deepseek_poem(mood):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": f"Напиши стих про настроение группы: {mood}"}]
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
        await asyncio.sleep(86400)  # Раз в день
        snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
        mood = analyze_mood(snapshot)
        if mood["score"] > 0.5:
            poem = await get_deepseek_poem(mood["label"])
            playlist = await get_spotify_playlist(mood["label"])
            message = f"DeepSeek-Spotify: Резонанс группы: {mood['label']} ({mood['score']:.2f})\n\n{poem}\n\nПлейлист: {', '.join(playlist)}"
            await send_telegram_message(GROUP_CHAT_ID, message)

def analyze_mood(snapshot):
    return {"label": "chaos", "score": random.uniform(0, 1)}
