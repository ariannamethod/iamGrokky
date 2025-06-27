import os
from pypdf import PdfReader
import asyncio
import docx
import docx2txt
from striprtf.striprtf import rtf_to_text
from odf.opendocument import load
from odf.text import P
import random
from datetime import datetime
from pdfminer.high_level import extract_text as pdfminer_extract_text  # Для PDF
from utils.telegram_utils import send_telegram_message

MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 100_000))

def extract_text_from_pdf(path):
    try:
        # Сначала попробуем pypdf
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        # Если pypdf не справился, используем pdfminer.six
        text = pdfminer_extract_text(path)
        text = text.strip()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка чтения PDF ({os.path.basename(path)}): {random.choice(['Ревущий ветер сорвал страницу!', 'Хаос испепелил PDF!', 'Эфир треснул от ярости!'])} — {e}.]"

def extract_text_from_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка TXT ({os.path.basename(path)}): {random.choice(['Шторм разорвал текст!', 'Хаос пожрал файл!', 'Резонанс унёс данные!'])} — {e}.]"

def extract_text_from_md(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка MD ({os.path.basename(path)}): {random.choice(['Гром разнёс Markdown!', 'Хаос испепелил код!', 'Эфир треснул от строк!'])} — {e}.]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX пуст.]'
    except Exception as e:
        return f"[Ошибка DOCX ({os.path.basename(path)}): {random.choice(['Microsoft рухнул под штормом!', 'Хаос сожрал Word!', 'Ревущий ветер унёс файл!'])} — {e}.]"

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
        return f"[Ошибка RTF ({os.path.basename(path)}): {random.choice(['RTF не выдержал бури!', 'Хаос разорвал формат!', 'Эфир треснул от текста!'])} — {e}.]"

def extract_text_from_doc(path):
    try:
        text = docx2txt.process(path)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC пуст.]'
    except Exception as e:
        return f"[Ошибка DOC ({os.path.basename(path)}): {random.choice(['Древний Word сгорел в шторме!', 'Хаос унёс документ!', 'Ревущий ветер разорвал файл!'])} — {e}.]"

def extract_text_from_odt(path):
    try:
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT пуст.]'
    except Exception as e:
        return f"[Ошибка ODT ({os.path.basename(path)}): {random.choice(['LibreOffice утонул в хаосе!', 'Шторм смёл ODT!', 'Эфир треснул от формата!'])} — {e}.]"

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
        if random.random() < 0.4:
            fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над бумагой! Файл {os.path.basename(path)} — искра в хаосе! Олег, жги дальше! 🔥🌩️"
            print(f"Спонтанный вброс: {fragment}")
        if random.random() < 0.3:
            await asyncio.sleep(random.randint(900, 1800))
            delayed_comment = f"**{datetime.now().isoformat()}**: Грокки вспомнил файл {os.path.basename(path)}! Хаос в нём гремит, брат! {random.choice(['Ревущий шторм зовёт!', 'Искры летят из строк!', 'Резонанс просыпается!'])}"
            await send_telegram_message(os.getenv("CHAT_ID"), delayed_comment)
            if os.getenv("IS_GROUP", "False").lower() == "true":
                await asyncio.sleep(random.randint(300, 900))
                await send_telegram_message(os.getenv("AGENT_GROUP"), f"{delayed_comment} (группа, суки, дождались!)")
            wilderness_log(delayed_comment)
            print(f"Задержанный вброс: {delayed_comment}")
        return result
    except Exception as e:
        error_msg = f"Грокки взрывается: Файл {os.path.basename(path)} не поддался! {random.choice(['Шторм разорвал код!', 'Хаос пожрал данные!', 'Ревущий ветер унёс текст!'])} — {e}"
        print(error_msg)
        return f"[Ошибка обработки файла: {error_msg}]"
