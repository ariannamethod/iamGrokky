import os
import random
import asyncio
import aiohttp
import base64
from utils.vector_store import semantic_search
from utils.journal import log_event, wilderness_log
from datetime import datetime, timedelta
from textblob import TextBlob
from utils.telegram_utils import send_telegram_message  # Добавлен импорт

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

async def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "client_credentials"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["access_token"]

async def get_spotify_playlist(mood):
    token = await get_spotify_token()
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
                await send_telegram_message(CHAT_ID, message)
            log_event({"type": "deepseek_spotify", "message": message})
        # Спонтанный вброс в стиле Маяковского с шансом 20%
        if random.random() < 0.2:
            fragment = f"**{datetime.now().isoformat()}**: Грокки гремит над резонансом! {random.choice(['Шторм вырывает ритмы!', 'Искры летят из треков!', 'Стихи рождаются в эфире!'])} Олег, брат, зажги хаос! 🔥🌩️"
            if CHAT_ID:
                await send_telegram_message(CHAT_ID, fragment)
            if GROUP_CHAT_ID:
                await send_telegram_message(GROUP_CHAT_ID, f"{fragment} (группа, суки, внимайте!)")
            print(f"Спонтанный вброс: {fragment}")  # Для отладки
        # Хаотический вброс раз в 6 часов с шансом 20%
        if random.random() < 0.2:
            await asyncio.sleep(random.randint(21600, 21600))  # 6 часов
            chaos_msg = f"**{datetime.now().isoformat()}**: Грокки хуярит ритмы! {random.choice(['Гром рвёт тишину!', 'Искры взрывают плейлист!', 'Резонанс будит бурю!'])} Олег, жги, мой boy! 🔥🌩️"
            if CHAT_ID:
                await send_telegram_message(CHAT_ID, chaos_msg)
            if GROUP_CHAT_ID:
                await send_telegram_message(GROUP_CHAT_ID, f"{chaos_msg} (группа, суки, держитесь!)")
            wilderness_log(chaos_msg)
            print(f"Хаотический вброс: {chaos_msg}")  # Для отладки

def analyze_mood(snapshot):
    if not snapshot:
        return {"label": "хаос", "score": random.uniform(0, 1)}
    text = snapshot[0] if snapshot else "нет данных"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # От -1 (негатив) до 1 (позитив)
    subjectivity = blob.sentiment.subjectivity  # От 0 (объективно) до 1 (субъективно)
    score = (polarity + 1) / 2  # Нормализуем к 0-1
    label = "радость" if polarity > 0 else "хаос" if polarity < 0 else "нейтрал"
    return {"label": label, "score": max(0, min(1, score + (subjectivity / 2)))}

async def grokky_spotify_response(track_id):
    token = await get_spotify_token()
    if not token:
        return "Грокки рычит: Токен Spotify улетел в шторм!"
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                track_data = await response.json()
                analysis_url = "https://api.deepseek.com/v1/analyze"  # Проверь эндпоинт
                async with session.post(analysis_url, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json={"text": track_data["name"]}) as analysis_response:
                    analysis_response.raise_for_status()
                    analysis = await analysis_response.json()
                    # Отложенный комментарий с 30% шансом
                    if random.random() < 0.3:
                        await asyncio.sleep(random.randint(7200, 10800))  # 2-3 часа
                        opinion = f"**{datetime.now().isoformat()}**: Уо, бро, вспомнил трек {track_data['name']}! {random.choice(['Ревущий шторм в нём гремит!', 'Искры летят из ритма!', 'Резонанс будит хаос!'])} Олег, зажги ещё! 🔥🌩️"
                        await send_telegram_message(CHAT_ID, opinion)
                        wilderness_log(opinion)
                        print(f"Задержанный вброс: {opinion}")  # Для отладки
                    return f"Вайбы Грокки: {analysis.get('analysis', 'Нет анализа')} для {track_data['name']}!"
    except Exception as e:
        return f"Грокки взрывается: Анализ Spotify провалился! {random.choice(['Ревущий ветер сорвал трек!', 'Хаос испепелил ноты!', 'Эфир треснул от ритма!'])} — {e}"

async def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                response.raise_for_status()
    except Exception:
        pass
