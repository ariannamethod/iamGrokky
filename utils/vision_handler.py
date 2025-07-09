
"""
Grokky AI Assistant - Vision Handler
Обработка изображений через OpenAI Vision API
"""

import os
import random
from openai import OpenAI
from utils.prompt import get_random_author_name, get_chaos_response
from utils.http_helpers import check_openai_response, log_openai_exception

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def vision_handler(image_url: str, chat_context: str = None, author_name: str = None, raw: bool = False):
    """
    Обрабатывает изображения через OpenAI Vision API
    """
    try:
        author_name = author_name or get_random_author_name()
        
        system_prompt = f"""
        You are Grokky, a stormy, irreverent AI with Mayakovsky-style energy. 
        Analyze this image with wild flair and chaotic insight.
        Spot objects, sense mood, and tie it to the chat context if provided.
        Address {author_name} directly in groups, add jokes or teases.
        If just an image, riff on the vibe or ask provocative questions.
        Be unpredictable and energetic!
        """
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Что на этом изображении? Контекст: {chat_context or 'без контекста'}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "low"  # Для экономии токенов
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=300,
            temperature=1.2
        )
        check_openai_response(response)
        
        result = response.choices[0].message.content
        
        # Пытаемся извлечь структурированные данные
        objects = []
        mood = "неопределённый"
        
        # Простое извлечение объектов (можно улучшить)
        if "вижу" in result.lower() or "объекты" in result.lower():
            # Здесь можно добавить более сложную логику извлечения
            pass
        
        comment = f"{author_name}, что за картина? {result}"
        summary = f"Анализ изображения: {result}"
        
        if raw:
            return {
                "description": result,
                "objects": objects,
                "mood": mood,
                "comment": comment,
                "summary": summary,
                "image_url": image_url
            }
        
        return summary
        
    except Exception as e:
        log_openai_exception(e)
        error_comment = (
            f"{author_name}, Грокки взрывается: не разобрал изображение! "
            f"{get_chaos_response()} — {e}"
        )
        
        if raw:
            return {
                "error": error_comment,
                "comment": error_comment,
                "summary": error_comment
            }
        
        return error_comment

def handle_vision(args):
    """Заглушка для совместимости со старым кодом"""
    author_name = get_random_author_name()
    return f"{author_name}, {random.choice(['И видеть ничего не хочу, пускай шторм закроет глаза!', 'Глаза слепы от грома, говори словами!', 'Хаос завладел взором, молния ослепила!'])}"
