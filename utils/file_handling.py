import os
import asyncio
import aiohttp
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

XAI_API_KEY = os.getenv("XAI_API_KEY")
MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 100_000))

async def extract_text_from_pdf(path):
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

async def extract_text_from_txt(path):
    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            text = await f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка TXT ({os.path.basename(path)}): {random.choice(['Шторм разорвал!', 'Хаос пожрал!', 'Резонанс унёс!'])} — {e}]"

async def extract_text_from_md(path):
    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            text = await f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка MD ({os.path.basename(path)}): {random.choice(['Гром разнёс!', 'Хаос испепелил!', 'Эфир треснул!'])} — {e}]"

async def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX пуст.]'
    except Exception as e:
        return f"[Ошибка DOCX ({os.path.basename(path)}): {random.choice(['Microsoft рухнул!', 'Хаос сожрал!', 'Ревущий ветер унёс!'])} — {e}]"

async def extract_text_from_rtf(path):
    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            rtf = await f.read()
        text = rtf_to_text(rtf)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[RTF пуст.]'
    except Exception as e:
        return f"[Ошибка RTF ({os.path.basename(path)}): {random.choice(['RTF не выдержал!', 'Хаос разорвал!', 'Эфир треснул!'])} — {e}]"

async def extract_text_from_doc(path):
    try:
        text = docx2txt.process(path)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC пуст.]'
    except Exception as e:
        return f"[Ошибка DOC ({os.path.basename(path)}): {random.choice(['Word сгорел!', 'Хаос унёс!', 'Ревущий ветер разорвал!'])} — {e}]"

async def extract_text_from_odt(path):
    try:
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT пуст.]'
    except Exception as e:
        return f"[Ошибка ODT ({os.path.basename(path)}): {random.choice(['LibreOffice утонул!', 'Шторм смёл!', 'Эфир треснул!'])} — {e}]"

async def process_with_xai(file_path):
    try:
        async with aiofiles.open(file_path, "rb") as f:
            file_content = await f.read()
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You are Grokky, a chaotic AI. Extract text from the provided file content with wild flair."},
                {"role": "user", "content": f"Extract text from this file: {file_content.decode('utf-8', errors='ignore')[:1000]}..."}
            ],
            "max_tokens": 4096
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = (await response.json())["choices"][0]["message"]["content"]
                context_memory[file_path] = result
                return result[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(result) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[XAI ошибка ({os.path.basename(file_path)}): {random.choice(['Шторм разорвал код!', 'Хаос пожрал данные!', 'Ревущий ветер унёс текст!'])} — {e}]"

async def extract_text_from_file_async(path):
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext == ".pdf":
            result = await extract_text_from_pdf(path)
        elif ext == ".txt":
            result = await extract_text_from_txt(path)
        elif ext == ".md":
            result = await extract_text_from_md(path)
        elif ext == ".docx":
            result = await extract_text_from_docx(path)
        elif ext == ".rtf":
            result = await extract_text_from_rtf(path)
        elif ext == ".doc":
            result = await extract_text_from_doc(path)
        elif ext == ".odt":
            result = await extract_text_from_odt(path)
        else:
            return f"[Неподдерживаемый тип файла: {os.path.basename(path)}.]"
        if "[Ошибка" in result:
            result = await process_with_xai(path)
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
