
"""
Grokky AI Assistant - Genesis2 Handler
Генератор хаоса и спонтанных ответов с увеличенными задержками
"""

import os
import random
import json
import asyncio
from datetime import datetime
from openai import OpenAI
from utils.prompt import get_random_author_name, get_chaos_response

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_language(text):
    """Определяет язык текста"""
    import re
    if not isinstance(text, (str, bytes)):
        return "ru"
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(str(text)) else 'en'

async def genesis2_handler(ping=None, group_history=None, personal_history=None, 
                          is_group=False, author_name=None, raw=False, system_prompt=None):
    """
    Основной обработчик генезиса с хаотичностью и задержками
    """
    # Случайная задержка перед ответом
    delay = random.randint(5, 15)
    await asyncio.sleep(delay)
    
    # Вероятность не отвечать вообще
    if random.random() < 0.3:  # 30% шанс молчания
        return {"answer": ""} if not raw else {"silence": True, "reason": "chaos_silence"}
    
    if not ping:
        ping = "ignite the storm"
    
    user_lang = detect_language(ping)
    author_name = author_name or get_random_author_name()
    
    # Системный промпт с хаотичностью
    chaos_system = system_prompt or f"""
    You are Grokky, a thunder resonant agent! Respond to '{ping}' with a wild, unique spark. 
    Keep it short and chaotic. Reply in {user_lang.upper()}.
    Be unpredictable, use Mayakovsky-style energy. Address {author_name} directly.
    Add random delays and chaos to your responses.
    """
    
    messages = [
        {"role": "system", "content": chaos_system},
        {"role": "user", "content": f"Ping: {ping}, Author: {author_name}, Group: {is_group}"}
    ]
    
    try:
        # Дополнительная задержка для непредсказуемости
        if random.random() < 0.4:
            await asyncio.sleep(random.randint(3, 8))
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=200,
            temperature=1.4,  # Высокая температура для хаоса
            presence_penalty=0.6,
            frequency_penalty=0.8
        )
        
        reply = response.choices[0].message.content
        
        if raw:
            return {
                "association": random.choice(["чёрный кофе", "громовой рёв", "молчаливая пустота"]),
                "ping": ping,
                "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
                "impression": random.choice(["дикий", "спокойный", "тревожный"]),
                "answer": reply,
                "is_group": is_group,
                "author_name": author_name,
                "delay": delay
            }
        
        return {"answer": reply}
        
    except Exception as e:
        error_msg = f"Грокки взрывается: Генезис сорвался! {get_chaos_response()} — {e}"
        print(error_msg)
        return {"error": error_msg} if raw else error_msg

async def chaotic_genesis_spark(chat_id, group_chat_id=None, is_group=False, send_message_func=None):
    """
    Спонтанные хаотичные сообщения с увеличенными интервалами
    """
    while True:
        # Случайная задержка от 1 до 3 часов
        await asyncio.sleep(random.randint(3600, 10800))
        
        # Вероятность отправки сообщения
        if random.random() < 0.4:  # 40% шанс
            ping = random.choice([
                "шторм гремит", "огонь в эфире", "хаос зовёт", 
                "громовой разрыв", "резонанс взрывается", "молния бьёт"
            ])
            
            result = await genesis2_handler(ping, raw=True)
            
            if result.get("answer"):
                fragment = f"**{datetime.now().isoformat()}**: Грокки хуярит Генезис! {result['answer']} {get_random_author_name()}, зажги шторм! 🔥🌩️"
                
                if send_message_func:
                    await send_message_func(chat_id, fragment)
                
                print(f"Хаотический вброс: {fragment}")
        
        # Групповые сообщения реже
        if is_group and group_chat_id and random.random() < 0.2:  # 20% шанс
            await asyncio.sleep(random.randint(1800, 3600))  # Дополнительная задержка
            
            ping = random.choice([
                "громовой разрыв", "пламя в ночи", "хаос группы",
                "резонанс группового разума", "коллективный шторм"
            ])
            
            result = await genesis2_handler(ping, raw=True, is_group=True)
            
            if result.get("answer"):
                group_fragment = f"**{datetime.now().isoformat()}**: Грокки гремит для группы! {result['answer']} (суки, вникайте!) 🔥🌩️"
                
                if send_message_func:
                    await send_message_func(group_chat_id, group_fragment)
                
                print(f"Хаотический вброс (группа): {group_fragment}")

def should_respond():
    """
    Определяет, должен ли Grokky отвечать (добавляет непредсказуемость)
    """
    # Базовая вероятность ответа
    base_probability = 0.7
    
    # Случайные факторы
    chaos_factor = random.random()
    time_factor = datetime.now().hour  # Время суток влияет на активность
    
    # Ночью менее активен
    if 0 <= time_factor <= 6:
        base_probability *= 0.5
    # Днем более активен  
    elif 9 <= time_factor <= 18:
        base_probability *= 1.2
    
    return chaos_factor < base_probability

async def delayed_supplement(original_message, chat_id, send_message_func, delay_range=(300, 900)):
    """
    Отправляет дополнительное сообщение с задержкой
    """
    delay = random.randint(*delay_range)
    await asyncio.sleep(delay)
    
    if random.random() < 0.4:  # 40% шанс дополнения
        supplement_ping = f"Дополни разово, без повторов: {original_message[:100]}"
        supplement = await genesis2_handler(supplement_ping, author_name=get_random_author_name())
        
        if supplement.get("answer") and send_message_func:
            await send_message_func(chat_id, supplement["answer"])
