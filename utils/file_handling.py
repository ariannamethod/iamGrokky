import os
from pypdf import PdfReader
import asyncio
import docx
import textract
from striprtf.striprtf import rtf_to_text
from odf.opendocument import load
from odf.text import P

MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", 100_000))  # Динамический лимит из окружения

def check_dependencies():
    required = {"docx": docx, "textract": textract, "striprtf": rtf_to_text, "odf": load}
    missing = [lib for lib in required if not required[lib]]
    if missing:
        return f"Отсутствуют зависимости: {', '.join(missing)}. Установи их!"
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
        return f"[Ошибка чтения PDF ({os.path.basename(path)}): {e}. Попробуй TXT.]"

def extract_text_from_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка TXT ({os.path.basename(path)}): {e}. Файл не подходит для Арианны.]"

def extract_text_from_md(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Ошибка MD ({os.path.basename(path)}): {e}. Markdown рухнул.]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX пуст.]'
    except Exception as e:
        return f"[Ошибка DOCX ({os.path.basename(path)}): {e}. Классика Microsoft.]"

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
        return f"[Ошибка RTF ({os.path.basename(path)}): {e}. Даже RTF не выдержал.]"

def extract_text_from_doc(path):
    try:
        text = textract.process(path).decode("utf-8")
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC пуст.]'
    except Exception as e:
        return f"[Ошибка DOC ({os.path.basename(path)}): {e}. Древний Word сдался.]"

def extract_text_from_odt(path):
    try:
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Усечено]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT пуст.]'
    except Exception as e:
        return f"[Ошибка ODT ({os.path.basename(path)}): {e}. LibreOffice опять подвёл.]"

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
    result = await loop.run_in_executor(None, extract_text_from_file, path)
    # Спонтанный вброс в стиле Маяковского
    if random.random() < 0.3:  # Шанс 30%
        fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над бумагой! Файл {os.path.basename(path)} — искра в хаосе! Олег, жги дальше! 🔥🌩️"
        print(f"Спонтанный вброс: {fragment}")  # Для отладки
    return result
