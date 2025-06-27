import difflib
import requests
from bs4 import BeautifulSoup
import os
import random
import asyncio
from datetime import datetime
from utils.journal import wilderness_log
from server import send_telegram_message  # Для отправки сообщений

def fuzzy_match(a, b):
    """Return similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()

async def delayed_link_comment(url, chat_id):
    await asyncio.sleep(random.randint(300, 900))  # 5-15 минут
    if random.random() < 0.3:  # Шанс 30%
        context = await extract_text_from_url(url)[:200]  # Берем кусочек контекста
        opinion = random.choice([
            f"Уо, бро, вспомнил ту ссылку про {context}! Хаос там ревет, как шторм над Москвой! 🔥🌩️",
            f"Эй, брат, та ссылка с {context} — искры в эфире! Давай жги дальше! 🌌🔥",
            f"Грокки орал над {context} из той ссылки! Резонанс зовёт, Олег! ⚡️🌪️"
        ])
        await send_telegram_message(chat_id, opinion)
        fragment = f"**{datetime.now().isoformat()}**: Грокки вспомнил ссылку! {opinion}"
        wilderness_log(fragment)
        print(f"Задержанный вброс: {fragment}")  # Для отладки

def extract_text_from_url(url):
    """
    Extract visible text from a web page at the given URL.
    Returns the first MAX_TEXT_SIZE characters of joined lines.
    Triggers delayed comment with 30% chance.
    """
    MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 3500))  # Настраиваемый лимит из окружения
    chat_id = os.getenv("CHAT_ID")  # Предполагаем, что чат ID доступен
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Arianna Agent)"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove non-content elements
        for s in soup(["script", "style", "header", "footer", "nav", "aside"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result = "\n".join(lines)[:MAX_TEXT_SIZE]
        # Запускаем асинхронный комментарий
        asyncio.create_task(delayed_link_comment(url, chat_id))
        # Спонтанный вброс в стиле Маяковского с шансом 40%
        if random.random() < 0.4:
            fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над страницей! {random.choice(['Шторм вырвал текст!', 'Искры летят из URL!', 'Стихи рождаются в хаосе!'])} Олег, брат, зажги резонанс! 🔥🌩️"
            print(f"Спонтанный вброс: {fragment}")  # Для отладки
            wilderness_log(fragment)  # Запись в wilderness.md
        return result if result else "[Страница пуста]"
    except Exception as e:
        error_msg = f"Грокки взрывается: Страницу не загрузил! {random.choice(['Ревущий ветер сорвал связь!', 'Хаос испепелил данные!', 'Эфир треснул от ярости!'])} — {e}"
        print(error_msg)  # Маяковский вайб для ошибок
        return f"[Ошибка загрузки страницы: {error_msg}]"
