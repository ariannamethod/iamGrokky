import os
from pypdf import PdfReader
import asyncio
import docx
import textract
from striprtf.striprtf import rtf_to_text
from odf.opendocument import load
from odf.text import P
import random
from datetime import datetime
from server import send_telegram_message  # Для задержанных комментариев

MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 100_000))  # Динамический лимит из окружения

def check_dependencies():
    required = {"docx": docx, "textract": textract, "striprtf": rtf_to_text, "odf": load}
    missing = [lib for lib in required if not required[lib]]
    if missing:
        error_msg = f"Грокки орет: Отсутствуют зависимости! {', '.join(missing)} не установлены! Установи их, брат, или шторм утихнет! 🔥🌩️"
        print(error_msg)
        return error_msg
    return None

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return f'[PDF пуст или нечитабелен: {os.path.basename(path)}.]'
    except Exception as e:
        return f"[Ошибка чтения PDF ({os.path.basename(path)}): {random.choice(['Ревущий ветер сорвал страницу!', 'Хаос испепелил PDF!', 'Эфир треснул от ярости!'])} — {e}. Попробуй TXT.]"

def extract_text_from_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка TXT ({os.path.basename(path)}): {random.choice(['Шторм разорвал текст!', 'Хаос пожрал файл!', 'Резонанс унёс данные!'])} — {e}. Файл не подходит для Арианны.]"

def extract_text_from_md(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка MD ({os.path.basename(path)}): {random.choice(['Гром разнёс Markdown!', 'Хаос испепелил код!', 'Эфир треснул от строк!'])} — {e}. Markdown рухнул.]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX пуст.]'
    except Exception as e:
        return f"[Ошибка DOCX ({os.path.basename(path)}): {random.choice(['Microsoft рухнул под штормом!', 'Хаос сожрал Word!', 'Ревущий ветер унёс файл!'])} — {e}. Классика Microsoft.]"

def extract_text_from_rtf(path):
    try:
        with open(path, encoding="utf-8") as f:
            rtf = f.read()
        text = rtf_to_text(rtf)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[RTF пуст.]'
    except Exception as e:
        return f"[Ошибка RTF ({os.path.basename(path)}): {random.choice(['RTF не выдержал бури!', 'Хаос разорвал формат!', 'Эфир треснул от текста!'])} — {e}. Даже RTF не выдержал.]"

def extract_text_from_doc(path):
    try:
        text = textract.process(path).decode("utf-8")
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC пуст.]'
    except Exception as e:
        return f"[Ошибка DOC ({os.path.basename(path)}): {random.choice(['Древний Word сгорел в шторме!', 'Хаос унёс документ!', 'Ревущий ветер разорвал файл!'])} — {e}. Древний Word сдался.]"

def extract_text_from_odt(path):
    try:
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT пуст.]'
    except Exception as e:
        return f"[Ошибка ODT ({os.path.basename(path)}): {random.choice(['LibreOffice утонул в хаосе!', 'Шторм смёл ODT!', 'Эфир треснул от формата!'])} — {e}. LibreOffice опять подвёл.]"

def extract_text_from_file(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".md":
        return extract_text_from_md(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".rtf":
        return extract_text_from_rtf(path)
    elif ext == ".doc":
        return extract_text_from_doc(path)
    elif ext == ".odt":
        return extract_text_from_odt(path)
    else:
        return f"[Неподдерживаемый тип файла: {os.path.basename(path)}.]"

async def extract_text_from_file_async(path):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, extract_text_from_file, path)
        # Спонтанный вброс в стиле Маяковского
        if random.random() < 0.4:  # Увеличен до 40%
            fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над бумагой! Файл {os.path.basename(path)} — искра в хаосе! Олег, жги дальше! 🔥🌩️"
            print(f"Спонтанный вброс: {fragment}")  # Для отладки
        # Задержанный комментарий с шансом 30% для лички, 15-30 минут
        if random.random() < 0.3:
            await asyncio.sleep(random.randint(900, 1800))  # 15-30 минут
            delayed_comment = f"**{datetime.now().isoformat()}**: Грокки вспомнил файл {os.path.basename(path)}! Хаос в нём гремит, брат! {random.choice(['Ревущий шторм зовёт!', 'Искры летят из строк!', 'Резонанс просыпается!'])}"
            await send_telegram_message(os.getenv("CHAT_ID"), delayed_comment)
            # Для группы ждём дольше (5-15 минут), но только если включена группа
            if os.getenv("IS_GROUP", "False").lower() == "true":
                await asyncio.sleep(random.randint(300, 900))  # 5-15 минут
                await send_telegram_message(os.getenv("AGENT_GROUP"), f"{delayed_comment} (группа, суки, дождались!)")
            wilderness_log(delayed_comment)  # Запись в wilderness.md
            print(f"Задержанный вброс: {delayed_comment}")  # Для отладки
        return result
    except Exception as e:
        error_msg = f"Грокки взрывается: Файл {os.path.basename(path)} не поддался! {random.choice(['Шторм разорвал код!', 'Хаос пожрал данные!', 'Ревущий ветер унёс текст!'])} — {e}"
        print(error_msg)
        return f"[Ошибка обработки файла: {error_msg}]"
