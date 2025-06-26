import os
import requests
from server import send_telegram_message

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

def get_deepseek_text(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    payload = {"grant_type": "client_credentials"}
    response = requests.post(url, data=payload, headers=headers)
    return response.json()["access_token"]

def get_spotify_playlist(mood):
    token = get_spotify_token()
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_genres": mood.lower(), "limit": 5}
    response = requests.get(url, headers=headers, params=params)
    return [track["name"] for track in response.json()["tracks"]]

def deepseek_spotify_resonance(mood):
    text = get_deepseek_text(f"Напиши стих про настроение: {mood}")
    playlist = get_spotify_playlist(mood)
    message = f"Грокки вдохновился: {text}\nПлейлист: {', '.join(playlist)}"
    send_telegram_message(GROUP_CHAT_ID, message)

# Интеграция в server.py (при краше добавлю вызов)
