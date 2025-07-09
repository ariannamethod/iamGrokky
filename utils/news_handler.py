
"""
Grokky AI Assistant - News Handler
Получение новостей через OpenAI function calling 
"""

import os
import json
import random
from datetime import datetime, timedelta
from openai import OpenAI
from utils.prompt import get_random_author_name, get_chaos_response
from utils.telegram_utils import send_telegram_message_async

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NEWS_LOG_FILE = "data/news_sent_log.json"
MAX_LOG_ENTRIES = 50

def should_send_news(limit=4, group=False):
    """Проверяет, можно ли отправлять новости"""
    try:
        if os.path.exists(NEWS_LOG_FILE):
            with open(NEWS_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        
        # Фильтруем записи за последний день
        recent_log = [
            x for x in log 
            if datetime.fromisoformat(x["dt"]) > day_ago and x["group"] == group
        ]
        
        return len(recent_log) < (2 if group else limit), recent_log
        
    except Exception as e:
        print(f"Ошибка проверки лога новостей: {e}")
        return True, []

def log_sent_news(news_items, chat_id=None, group=False):
    """Логирует отправленные новости"""
    try:
        if os.path.exists(NEWS_LOG_FILE):
            with open(NEWS_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        
        now = datetime.utcnow().isoformat()
        
        for news in news_items:
            log.append({
                "dt": now,
                "title": news.get("title", "")[:100],
                "chat_id": chat_id,
                "group": group
            })
        
        # Ограничиваем размер лога
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]
        
        os.makedirs(os.path.dirname(NEWS_LOG_FILE), exist_ok=True)
        with open(NEWS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка логирования новостей: {e}")

async def get_news_via_openai(topics=None):
    """Получает новости через OpenAI function calling"""
    if not topics:
        topics = ["AI", "технологии", "искусство", "наука"]
    
    topic = random.choice(topics)
    
    try:
        # Используем OpenAI для получения актуальных новостей
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    Ты Грокки, хаотичный AI-агент! Найди 3 актуальные новости по теме "{topic}".
                    Для каждой новости верни:
                    - Заголовок
                    - Краткое описание
                    - Твой хаотичный комментарий в стиле Маяковского
                    
                    Формат ответа - JSON массив с объектами:
                    {{"title": "заголовок", "description": "описание", "grokky_comment": "комментарий"}}
                    """
                },
                {
                    "role": "user", 
                    "content": f"Дай мне свежие новости по теме: {topic}"
                }
            ],
            max_tokens=800,
            temperature=1.2
        )
        
        content = response.choices[0].message.content
        
        # Пытаемся извлечь JSON
        try:
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                news_data = json.loads(json_match.group())
                return news_data[:3]  # Максимум 3 новости
        except:
            pass
        
        # Если JSON не получился, создаем новости из текста
        chaos_comments = ["Дико!", "Пожар!", "Хаос!", "Бум!", "Эпично!", "Шторм!", "Резонанс!"]
        
        return [{
            "title": f"Новости {topic} - {datetime.now().strftime('%H:%M')}",
            "description": content[:200] + "...",
            "grokky_comment": random.choice(chaos_comments)
        }]
        
    except Exception as e:
        print(f"Ошибка получения новостей через OpenAI: {e}")
        return [{
            "title": "Грокки взрывается!",
            "description": f"Новости сорвались! {get_chaos_response()}",
            "grokky_comment": "Хаос победил информацию!"
        }]

async def grokky_send_news(chat_id=None, group=False):
    """Отправляет новости с проверкой лимитов"""
    can_send, log = should_send_news(group=group)
    
    if not can_send:
        print("Грокки молчит: Лимит новостей исчерпан.")
        return []
    
    # Получаем новости
    news_items = await get_news_via_openai()
    
    if not news_items:
        return []
    
    # Логируем отправленные новости
    log_sent_news(news_items, chat_id, group)
    
    # Формируем сообщения
    messages = []
    author_name = get_random_author_name()
    
    for news in news_items:
        message = f"""
🔥 {news['title']}

{news['description']}

Грокки: {news['grokky_comment']}
        """.strip()
        messages.append(message)
    
    # Отправляем сообщения если указан chat_id
    if chat_id and send_telegram_message_async:
        intro_message = f"{author_name}, держи свежий раскат грома!"
        await send_telegram_message_async(chat_id, intro_message)
        
        for message in messages:
            await send_telegram_message_async(chat_id, message)
    
    # Групповые сообщения
    if group and os.getenv("AGENT_GROUP"):
        group_intro = "Группа, держите громовые новости!"
        await send_telegram_message_async(os.getenv("AGENT_GROUP"), group_intro)
        
        for message in messages:
            await send_telegram_message_async(os.getenv("AGENT_GROUP"), message + " (суки, вникайте!)")
    
    # Спонтанный вброс
    if random.random() < 0.2:
        fragment = f"**{datetime.utcnow().isoformat()}**: Грокки гремит над новостями! {random.choice(['Ревущий шторм!', 'Искры летят!', 'Стихи из эфира!'])} {author_name}, зажги резонанс! 🔥🌩️"
        
        with open("data/news_wilderness.md", "a", encoding="utf-8") as f:
            f.write(fragment + "\n\n")
        
        print(f"Спонтанный вброс: {fragment}")
    
    return news_items

def handle_news(args):
    """Заглушка для совместимости со старым кодом"""
    author_name = get_random_author_name()
    return f"{author_name}, {random.choice(['Новости в тумане, молния их сожгла!', 'Гром унёс новости, давай без них!', 'Хаос разорвал инфу, пизди сам!'])}"
