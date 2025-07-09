
"""
Grokky AI Assistant - Telegram Utilities
Утилиты для работы с Telegram API
"""

import os
import requests
import asyncio
import aiohttp
from typing import Optional

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_telegram_message(chat_id: str, text: str, parse_mode: str = None):
    """Синхронная отправка сообщения в Telegram"""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print("Telegram токен или chat_id не настроены")
        return False
    
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4096],  # Лимит Telegram
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Ошибка отправки сообщения: {e}")
        return False

async def send_telegram_message_async(chat_id: str, text: str, parse_mode: str = None):
    """Асинхронная отправка сообщения в Telegram"""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print("Telegram токен или chat_id не настроены")
        return False
    
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": parse_mode
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                response.raise_for_status()
                return True
    except Exception as e:
        print(f"Ошибка отправки сообщения: {e}")
        return False

async def send_voice_message(chat_id: str, audio_data: bytes):
    """Отправка голосового сообщения"""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    
    url = f"{BASE_URL}/sendVoice"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('chat_id', chat_id)
            data.add_field('voice', audio_data, filename='voice.ogg', content_type='audio/ogg')
            
            async with session.post(url, data=data, timeout=30) as response:
                response.raise_for_status()
                return True
    except Exception as e:
        print(f"Ошибка отправки голосового сообщения: {e}")
        return False

async def get_file_url(file_id: str):
    """Получает URL файла по file_id"""
    if not TELEGRAM_BOT_TOKEN:
        return None
    
    url = f"{BASE_URL}/getFile"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"file_id": file_id}) as response:
                response.raise_for_status()
                data = await response.json()
                file_path = data["result"]["file_path"]
                return f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    except Exception as e:
        print(f"Ошибка получения URL файла: {e}")
        return None

async def download_file(file_url: str):
    """Скачивает файл по URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                response.raise_for_status()
                return await response.read()
    except Exception as e:
        print(f"Ошибка скачивания файла: {e}")
        return None

def split_message(text: str, max_length: int = 4000):
    """Разбивает длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]
    
    result = []
    while len(text) > max_length:
        # Ищем последний перенос строки в пределах лимита
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        
        result.append(text[:split_pos])
        text = text[split_pos:].lstrip('\n')
    
    if text:
        result.append(text)
    
    return result if result else ["Пустое сообщение"]
