
"""
Grokky AI Assistant - Image Generator
Генерация изображений через OpenAI DALL-E
"""

import os
import random
from openai import OpenAI
from utils.prompt import get_random_author_name, get_chaos_response
from utils.telegram_utils import send_telegram_message_async
from utils.http_helpers import check_openai_response, log_openai_exception

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def impress_handler(prompt: str, chat_context: str = None, author_name: str = None, raw: bool = False):
    """
    Генерирует изображения через OpenAI DALL-E
    """
    try:
        author_name = author_name or get_random_author_name()
        
        # Улучшаем промпт для более хаотичных результатов
        enhanced_prompt = f"{prompt}"
        if chat_context:
            enhanced_prompt += f" (контекст: {chat_context})"
        
        # Добавляем хаотичности в стиле Грокки
        chaos_styles = [
            "in surreal storm style",
            "with chaotic energy",
            "in Mayakovsky thunderous style", 
            "with wild resonance",
            "in stormy abstract style"
        ]
        
        enhanced_prompt += f", {random.choice(chaos_styles)}"
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        check_openai_response(response)

        image_url = response.data[0].url
        
        # Генерируем комментарий Грокки
        grokky_comments = [
            f"{author_name}, хочешь картинку? Получил шторм в пикселях!",
            f"{author_name}, держи хаос на холсте!",
            f"{author_name}, Грокки нарисовал молнию!",
            f"{author_name}, вот тебе резонанс в красках!"
        ]
        
        grokky_comment = random.choice(grokky_comments)
        
        if raw:
            return {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_url": image_url,
                "grokkys_comment": grokky_comment,
                "author_name": author_name
            }
        
        return {
            "image_url": image_url,
            "comment": grokky_comment
        }
        
    except Exception as e:
        log_openai_exception(e)
        error_comment = (
            f"{author_name}, Грокки разъярился: не смог нарисовать! "
            f"{get_chaos_response()} — {e}"
        )
        
        if raw:
            return {
                "error": error_comment,
                "prompt": prompt
            }
        
        return {"error": error_comment}

def handle_impress(args):
    """Заглушка для совместимости со старым кодом"""
    author_name = get_random_author_name()
    return f"{author_name}, {random.choice(['Шторм провалился, кисть сгорела!', 'Хаос сожрал холст, давай без рисунков!', 'Эфир треснул, рисовать не могу!'])}"
