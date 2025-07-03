import re

def detect_language(text):
    if not isinstance(text, (str, bytes)):
        return "ru"
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text) else 'en'
