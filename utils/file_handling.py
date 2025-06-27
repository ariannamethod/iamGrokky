import os
import asyncio
from openai import OpenAI
import random
from datetime import datetime
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import docx
import docx2txt
from striprtf.striprtf import rtf_to_text
from odf.opendocument import load
from odf.text import P
from utils.telegram_utils import send_telegram_message

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
context_memory = {}  # Глобальная память для контекста
MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 100_000))

# Инициализация клиента
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
        text = pdfminer_extract_text(path).strip()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка PDF ({os.path.basename(path)}): {random.choice(['Ревущий ветер сорвал!', 'Хаос испепелил!', 'Эфир треснул!'])} — {e}]"

def extract_text_from_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка TXT ({os.path.basename(path)}): {random.choice(['Шторм разорвал!', 'Хаос пожрал!', 'Резонанс унёс!'])} — {e}]"

def extract_text_from_md(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка MD ({os.path.basename(path)}): {random.choice(['Гром разнёс!', 'Хаос испепелил!', 'Эфир треснул!'])} — {e}]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX пуст.]'
    except Exception as e:
        return f"[Ошибка DOCX ({os.path.basename(path)}): {random.choice(['Microsoft рухнул!', 'Хаос сожрал!', 'Ревущий ветер унёс!'])} — {e}]"

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
        return f"[Ошибка RTF ({os.path.basename(path)}): {random.choice(['RTF не выдержал!', 'Хаос разорвал!', 'Эфир треснул!'])} — {e}]"

def extract_text_from_doc(path):
    try:
        text = docx2txt.process(path)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC пуст.]'
    except Exception as e:
        return f"[Ошибка DOC ({os.path.basename(path)}): {random.choice(['Word сгорел!', 'Хаос унёс!', 'Ревущий ветер разорвал!'])} — {e}]"

def extract_text_from_odt(path):
    try:
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT пуст.]'
    except Exception as e:
        return f"[Ошибка ODT ({os.path.basename(path)}): {random.choice(['LibreOffice утонул!', 'Шторм смёл!', 'Эфир треснул!'])} — {e}]"

async def process_with_openai(file_path):
    try:
        with open(file_path, "rb") as f:
            file_response = await openai_client.files.create(file=f, purpose="assistants")
        file_id = file_response.id
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Extract text from this file: {file_id}"}],
            max_tokens=4096
        )
        text = response.choices[0].message.content
        context_memory[file_path] = text
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[OpenAI ошибка ({os.path.basename(file_path)}): {random.choice(['Шторм разорвал код!', 'Хаос пожрал данные!', 'Ревущий ветер унёс текст!'])} — {e}]"

def extract_text_from_file(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".txt":
        text = extract_text_from_txt(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".md":
        text = extract_text_from_md(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".docx":
        text = extract_text_from_docx(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".rtf":
        text = extract_text_from_rtf(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".doc":
        text = extract_text_from_doc(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
    elif ext == ".odt":
        text = extract_text_from_odt(path)
        if "[Ошибка" in text:
            return process_with_openai(path)
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
