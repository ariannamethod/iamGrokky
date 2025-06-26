import os
import requests
from gtts import gTTS
import io
from utils.prompt import build_system_prompt

XAI_API_KEY = os.getenv("XAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
VOICE_MODE = False

def query_grok(user_message):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": user_message}], "max_tokens": 2048, "temperature": 1.0}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

def send_voice_message(chat_id, text):
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

def get_spotify_track_info(track_url):
    import base64
    url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "client_credentials"}
    token_response = requests.post(url, data=payload, headers=headers)
    token = token_response.json()["access_token"]
    track_id = track_url.split("/")[-1].split("?")[0]
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    return response.json()["name"], response.json()["artists"][0]["name"]
