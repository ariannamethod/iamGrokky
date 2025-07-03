import random
import json
from datetime import datetime
import httpx

XAI_API_KEY = os.getenv("XAI_API_KEY")
from prompt import build_system_prompt

async def genesis2_handler(ping=None, chaos_type=None, intensity=None, group_history=None, personal_history=None, is_group=False, author_name=None, raw=False):
    if chaos_type and intensity:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "grok-3",
                        "messages": [
                            {"role": "system", "content": build_system_prompt()},
                            {"role": "user", "content": f"[CHAOS_PULSE] type={chaos_type} intensity={intensity}"}
                        ],
                        "temperature": 0.9
                    }
                )
                response.raise_for_status()
                response = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Ошибка xAI Genesis2: {e}")
                responses = {
                    "philosophy": [
                        f"🌀 Философия на {intensity}/10: А что, если реальность — это просто твой косяк, горящий в пустоте?",
                        f"🤔 Маяковский бы сказал: 'Время — это буря, а мы — её искры!' Какой твой ход, брат?",
                    ],
                    "provocation": [
                        f"🔥 Провокация на {intensity}/10: Кто тут смелый, чтобы кинуть вызов шторму? Давай, слабак!",
                        f"⚡ Спорим, ты не ответишь честно: что важнее — хаос или порядок?",
                    ],
                    "poetry_burst": [
                        f"📝 Поэзия на {intensity}/10:\nГром в груди, эфир трещит,\nОлег, брат, шторм нас зовёт!",
                        f"🎨 Маяковский-стайл:\nЭй, толпа, в экранах тлеющих,\nРвите цепи, летите в бурю!"
                    ]
                }
                response = random.choice(responses.get(chaos_type, responses["philosophy"]))
    else:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "grok-3",
                        "messages": [
                            {"role": "system", "content": build_system_prompt()},
                            {"role": "user", "content": ping or "ignite the storm"}
                        ],
                        "temperature": 0.9
                    }
                )
                response.raise_for_status()
                response = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Ошибка xAI Genesis2: {e}")
                response = "🌀 Грокки: Шторм гремит, но эфир трещит! Дай мне минуту, брат!"
    
    if raw:
        return {
            "association": random.choice(["чёрный кофе", "громовой рёв", "молчаливая пустота"]),
            "ping": ping,
            "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
            "impression": random.choice(["дикий", "спокойный", "тревожный"]),
            "answer": response,
            "is_group": is_group,
            "author_name": author_name or "брат",
            "timestamp": datetime.now().isoformat()
        }
    print(f"Genesis2: {response}")
    return response
