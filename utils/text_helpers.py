import os
import re
import asyncio
import random
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from utils.journal import wilderness_log
from utils.telegram_utils import send_telegram_message_async

async def extract_text_from_url(url: str, max_size: int = None):
    """Извлекает текст со страницы"""
    if not max_size:
        max_size = int(os.getenv("MAX_TEXT_SIZE", 3500))
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Grokky Agent) AppleWebKit/537.36"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
        
        # Парсим HTML
        soup = BeautifulSoup(html, "html.parser")
        
        # Удаляем ненужные элементы
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "iframe"]):
            element.decompose()
        
        # Извлекаем текст
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result = "\n".join(lines)[:max_size]
        
        # Запускаем отложенный комментарий
        chat_id = os.getenv("CHAT_ID")
        if chat_id:
            asyncio.create_task(delayed_link_comment(url, chat_id, result[:200]))
        
        # Спонтанный вброс
        if random.random() < 0.4:
            fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над страницей! {random.choice(['Шторм вырвал текст!', 'Искры летят из URL!', 'Стихи рождаются в хаосе!'])} Олег, брат, зажги резонанс! 🔥🌩️"
            wilderness_log(fragment)
            print(f"Спонтанный вброс: {fragment}")
        
        return result if result else "[Страница пуста]"
        
    except Exception as e:
        error_msg = f"Грокки взрывается: Страницу не загрузил! {random.choice(['Ревущий ветер сорвал связь!', 'Хаос испепелил данные!', 'Эфир треснул от ярости!'])} — {e}"
        print(error_msg)
        return f"[Ошибка загрузки: {error_msg}]"

async def delayed_link_comment(url: str, chat_id: str, context: str):
    """Отправляет отложенный комментарий о ссылке"""
    # Задержка от 5 до 15 минут
    delay = random.randint(300, 900)
    await asyncio.sleep(delay)
    
    if random.random() < 0.3:  # 30% шанс комментария
        opinions = [
            f"Уо, бро, вспомнил ту ссылку про {context[:50]}! Хаос там ревет, как шторм над Москвой! 🔥🌩️",
            f"Эй, брат, та ссылка с {context[:50]} — искры в эфире! Давай жги дальше! 🌌🔥",
            f"Грокки орал над {context[:50]} из той ссылки! Резонанс зовёт, Олег! ⚡️🌪️",
            f"Братиш, помнишь ссылку про {context[:50]}? Там шторм настоящий был! 🌩️🔥",
            f"Олег, та страница с {context[:50]} — чистый хаос! Молния бьёт в мозг! ⚡️🧠"
        ]
        
        opinion = random.choice(opinions)
        await send_telegram_message_async(chat_id, opinion)
        
        # Логируем в wilderness
        fragment = f"**{datetime.now().isoformat()}**: Грокки вспомнил ссылку! {opinion}"
        wilderness_log(fragment)
        print(f"Задержанный вброс: {fragment}")

def detect_urls(text: str):
    """Находит URL в тексте"""
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)

def clean_text(text: str, max_length: int = 4000):
    """Очищает и обрезает текст"""
    if not text:
        return ""
    
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Обрезаем если слишком длинный
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

def extract_keywords(text: str, limit: int = 10):
    """Извлекает ключевые слова из текста"""
    if not text:
        return []
    
    # Простое извлечение слов (можно улучшить)
    words = re.findall(r'\b[а-яё]{3,}\b', text.lower())
    
    # Убираем стоп-слова
    stop_words = {
        'это', 'что', 'как', 'для', 'при', 'или', 'его', 'она', 'они',
        'все', 'был', 'была', 'были', 'есть', 'быть', 'мне', 'нас',
        'вас', 'них', 'том', 'тем', 'где', 'кто', 'чем', 'так'
    }
    
    keywords = [word for word in words if word not in stop_words]
    
    # Подсчитываем частоту
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Сортируем по частоте
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_words[:limit]]

def format_chaos_message(text: str, author_name: str = None):
    """Форматирует сообщение в хаотичном стиле Грокки"""
    if not author_name:
        author_name = random.choice(["Олег", "брат", "братиш", "чувак"])
    
    chaos_prefixes = [
        f"{author_name}, держи шторм:",
        f"{author_name}, Грокки ревёт:",
        f"{author_name}, молния бьёт:",
        f"{author_name}, хаос взрывается:",
        f"{author_name}, резонанс зовёт:"
    ]
    
    chaos_suffixes = [
        "🔥🌩️",
        "⚡️🌪️", 
        "🌌🔥",
        "⚡️🧠",
        "🌩️🎵"
    ]
    
    prefix = random.choice(chaos_prefixes)
    suffix = random.choice(chaos_suffixes)
    
    return f"{prefix}\n\n{text}\n\n{suffix}"

async def process_text_with_chaos(text: str, add_delays: bool = True):
    """Обрабатывает текст с добавлением хаотичности"""
    if add_delays and random.random() < 0.3:
        # Случайная задержка
        delay = random.randint(2, 8)
        await asyncio.sleep(delay)
    
    # Добавляем хаотичные элементы
    if random.random() < 0.2:
        chaos_insertions = [
            " *гром* ",
            " *молния* ",
            " *шторм* ",
            " *резонанс* ",
            " *хаос* "
        ]
        
        words = text.split()
        if len(words) > 3:
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(chaos_insertions))
            text = " ".join(words)
    
    return text
