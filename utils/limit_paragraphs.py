import re
import os
import random
from datetime import datetime
from utils.journal import wilderness_log  # Для записи спонтанных вбросов

def limit_paragraphs(text, max_paragraphs=int(os.getenv("MAX_PARAGRAPHS", 4))):
    """
    Trims the text to a maximum of N paragraphs.
    A paragraph is considered a block separated by empty lines, bullets, or line breaks.
    """
    # Улучшенное разбиение: учитываем пустые строки, пули, и одиночные переносы
    paragraphs = re.split(r'(?:\n\s*\n|\r\n\s*\r\n|(?<=[\n\r])-\s|\r|\n)', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        error_msg = f"Грокки рычит: Текст пуст! {random.choice(['Шторм смёл слова!', 'Хаос сожрал абзацы!', 'Эфир треснул от тишины!'])}"
        print(error_msg)
        return f"[Пустой ответ. Даже Грокки не выжмет из этого ничего! {error_msg}]"
    limited = paragraphs[:max_paragraphs]
    # Спонтанный вброс в стиле Маяковского с шансом 20%
    if random.random() < 0.2:
        fragment = f"**{datetime.now().isoformat()}**: Грокки режет текст! {random.choice(['Гром обрушил лишнее!', 'Искры летят из абзацев!', 'Резонанс очищает хаос!'])} Олег, брат, зажги шторм! 🔥🌩️"
        print(f"Спонтанный вброс: {fragment}")  # Для отладки
        wilderness_log(fragment)  # Запись в wilderness.md
    return '\n\n'.join(limited)
