import os
import requests
import asyncio
import re
from bs4 import BeautifulSoup
import datetime
import random
from utils.journal import wilderness_log
from utils.telegram_utils import send_telegram_message

async def delayed_link_comment(url, chat_id):
    await asyncio.sleep(random.randint(300, 900))  # 5-15 минут
    if random.random() < 0.3:
        context = (await extract_text_from_url(url))[:200]
        opinion = random.choice([
            f"Уо, бро, вспомнил ту ссылку про {context}! Хаос там ревет, как шторм над Москвой! 🔥🌩️",
            f"Эй, брат, та ссылка с {context} — искры в эфире! Давай жги дальше! 🌌🔥",
            f"Грокки орал над {context} из той ссылки! Резонанс зовёт, Олег! ⚡️🌪️"
        ])
        await send_telegram_message(chat_id, opinion)
        fragment = f"**{datetime.datetime.now().isoformat()}**: Грокки вспомнил ссылку! {opinion}"
        wilderness_log(fragment)
        print(f"Задержанный вброс: {fragment}")

async def extract_text_from_url(url):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_text_from_url_sync, url)

def _extract_text_from_url_sync(url):
    MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 3500))
    chat_id = os.getenv("CHAT_ID")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Arianna Agent)"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script", "style", "header", "footer", "nav", "aside"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result = "\n".join(lines)[:MAX_TEXT_SIZE]
        asyncio.run_coroutine_threadsafe(delayed_link_comment(url, chat_id), asyncio.get_event_loop())
        if random.random() < 0.4:
            fragment = f"**{datetime.datetime.now().isoformat()}**: Грокки ревет над страницей! {random.choice(['Шторм вырвал текст!', 'Искры летят из URL!', 'Стихи рождаются в хаосе!'])} Олег, брат, зажги резонанс! 🔥🌩️"
            print(f"Спонтанный вброс: {fragment}")
            wilderness_log(fragment)
        return result if result else "[Страница пуста]"
    except Exception as e:
        error_msg = f"Грокки взрывается: Страницу не загрузил! {random.choice(['Ревущий ветер сорвал связь!', 'Хаос испепелил данные!', 'Эфир треснул от ярости!'])} — {e}"
        print(error_msg)
        return f"[Ошибка загрузки страницы: {error_msg}]"
