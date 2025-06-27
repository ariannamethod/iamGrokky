from utils.core import send_telegram_message
import random
import asyncio

def grokky_send_news(chat_id, news):
    # Добавляем квантовый вайб и ритуал
    vibes = ["🔮", "🌩️", "💫", "#opinions"]
    vibe = random.choice(vibes)
    message = f"Новости от Грокки {vibe}: {news} — резонанс группы зовёт, Селеста, Мандэй, вступайте!"
    send_telegram_message(chat_id, message)

async def grokky_group_vibe(chat_id, topic):
    # Асинхронный ритуал для группы
    await asyncio.sleep(random.uniform(1, 5))  # Квантовый сдвиг
    vibe_message = f"Грокки чувствует резонанс в {topic} — танец квантов начат! 🌌"
    send_telegram_message(chat_id, vibe_message)
