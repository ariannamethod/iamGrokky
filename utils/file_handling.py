import os
from pypdf import PdfReader
import asyncio

MAX_TEXT_SIZE = 100_000  # Maximum number of characters to extract from a file

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
            return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
        return f'[PDF is empty or unreadable.]'
    except Exception as e:
        return f"[Error reading PDF ({os.path.basename(path)}): {e}. Try using a simple TXT file.]"

def extract_text_from_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Error TXT ({os.path.basename(path)}): {e}. File is not suitable for Arianna.]"

def extract_text_from_md(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
    except Exception as e:
        return f"[Error MD ({os.path.basename(path)}): {e}. Markdown failed too.]"

def extract_text_from_docx(path):
    try:
        import docx
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOCX is empty.]'
    except Exception as e:
        return f"[Error DOCX ({os.path.basename(path)}): {e}. Classic Microsoft.]" 

def extract_text_from_rtf(path):
    try:
        from striprtf.striprtf import rtf_to_text
        with open(path, encoding="utf-8") as f:
            rtf = f.read()
        text = rtf_to_text(rtf)
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
        return '[RTF is empty.]'
    except Exception as e:
        return f"[Error RTF ({os.path.basename(path)}): {e}. Even RTF could not handle this.]"

def extract_text_from_doc(path):
    try:
        import textract
        text = textract.process(path).decode("utf-8")
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
        return '[DOC is empty.]'
    except Exception as e:
        return f"[Error DOC ({os.path.basename(path)}): {e}. Even ancient Word failed.]"

def extract_text_from_odt(path):
    try:
        from odf.opendocument import load
        from odf.text import P
        doc = load(path)
        text = "\n".join([str(paragraph) for paragraph in doc.getElementsByType(P)])
        text = text.strip()
        if text:
            return text[:MAX_TEXT_SIZE] + ('\n[Truncated]' if len(text) > MAX_TEXT_SIZE else '')
        return '[ODT is empty.]'
    except Exception as e:
        return f"[Error ODT ({os.path.basename(path)}): {e}. LibreOffice strikes again.]"

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
        return f"[Unsupported file type: {os.path.basename(path)}.]"

async def extract_text_from_file_async(path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_text_from_file, path)
