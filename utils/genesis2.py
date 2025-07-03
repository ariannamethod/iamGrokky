import random
import json
from datetime import datetime

async def genesis2_handler(ping, group_history=None, personal_history=None, is_group=False, author_name=None, raw=False):
    chaos_types = ["philosophy", "provocation", "poetry_burst"]
    response = f"🌀 Грокки хуярит: {random.choice(chaos_types)} для {author_name or 'брат'}! {ping}"
    if raw:
        return {
            "association": random.choice(["чёрный кофе", "громовой рёв", "молчаливая пустота"]),
            "ping": ping,
            "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
            "impression": random.choice(["дикий", "спокойный", "тревожный"]),
            "answer": response,
            "is_group": is_group,
            "author_name": author_name,
            "timestamp": datetime.now().isoformat()
        }
    return {"answer": response}
