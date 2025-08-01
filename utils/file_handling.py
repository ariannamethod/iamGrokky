"""Utilities for extracting text from various file types."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tarfile
import tempfile
import zipfile
from typing import Callable, Dict

from pypdf import PdfReader

DEFAULT_MAX_TEXT_SIZE = 100_000
REPO_SNAPSHOT_PATH = "config/repo_snapshot.md"


class FileHandler:
    """Flexible text extraction utility supporting many formats."""

    def __init__(self, max_text_size: int = DEFAULT_MAX_TEXT_SIZE) -> None:
        self.max_text_size = max_text_size
        self._extractors: Dict[str, Callable[[str], str]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._extractors.update(
            {
                ".pdf": self._extract_pdf,
                ".txt": self._extract_txt,
                ".md": self._extract_txt,
                ".docx": self._extract_docx,
                ".rtf": self._extract_rtf,
                ".doc": self._extract_doc,
                ".odt": self._extract_odt,
                ".zip": self._extract_zip,
                ".tar": self._extract_tar,
                ".tar.gz": self._extract_tar,
                ".tgz": self._extract_tar,
                ".png": self._extract_image,
                ".jpg": self._extract_image,
                ".jpeg": self._extract_image,
                ".gif": self._extract_image,
                ".bmp": self._extract_image,
                ".webp": self._extract_image,
            }
        )

    def register_extractor(self, ext: str, func: Callable[[str], str]) -> None:
        """Add a custom extractor for the given extension."""

        self._extractors[ext.lower()] = func

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) > self.max_text_size:
            return text[: self.max_text_size] + "\n[Обрезано]"
        return text

    def _detect_extension(self, path: str) -> str:
        path_lower = path.lower()
        for ext in self._extractors:
            if path_lower.endswith(ext):
                return ext
        return os.path.splitext(path_lower)[-1]

    def _extract_pdf(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return self._truncate(text) if text.strip() else "[PDF пуст.]"
        except Exception as e:
            return f"[Ошибка чтения PDF ({os.path.basename(path)}): {e}]"

    def _extract_txt(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return self._truncate(f.read())
        except Exception as e:
            return f"[Ошибка TXT ({os.path.basename(path)}): {e}]"

    def _extract_docx(self, path: str) -> str:
        try:
            import docx

            doc = docx.Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
            return self._truncate(text) if text.strip() else "[DOCX пуст.]"
        except Exception as e:
            return f"[Ошибка DOCX ({os.path.basename(path)}): {e}]"

    def _extract_rtf(self, path: str) -> str:
        try:
            from striprtf.striprtf import rtf_to_text

            with open(path, encoding="utf-8") as f:
                text = rtf_to_text(f.read())
            return self._truncate(text) if text.strip() else "[RTF пуст.]"
        except Exception as e:
            return f"[Ошибка RTF ({os.path.basename(path)}): {e}]"

    def _extract_doc(self, path: str) -> str:
        try:
            import textract

            text = textract.process(path).decode("utf-8")
            return self._truncate(text) if text.strip() else "[DOC пуст.]"
        except Exception as e:
            return f"[Ошибка DOC ({os.path.basename(path)}): {e}]"

    def _extract_odt(self, path: str) -> str:
        try:
            from odf.opendocument import load
            from odf.text import P

            doc = load(path)
            text = "\n".join(str(p) for p in doc.getElementsByType(P))
            return self._truncate(text) if text.strip() else "[ODT пуст.]"
        except Exception as e:
            return f"[Ошибка ODT ({os.path.basename(path)}): {e}]"

    def _extract_image(self, path: str) -> str:
        try:
            from PIL import Image

            with Image.open(path) as img:
                info = f"{img.format} {img.width}x{img.height} mode={img.mode}"
            return info
        except Exception as e:
            return f"[Ошибка чтения изображения ({os.path.basename(path)}): {e}]"

    def _extract_zip(self, path: str) -> str:
        try:
            texts = []
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    try:
                        data = zf.read(name)
                        ext = self._detect_extension(name)
                        extractor = self._extractors.get(ext)
                        if extractor and extractor not in {
                            self._extract_zip,
                            self._extract_tar,
                        }:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=ext
                            ) as tmp:
                                tmp.write(data)
                                tmp.flush()
                                text = extractor(tmp.name)
                            os.unlink(tmp.name)
                        else:
                            try:
                                text = data.decode("utf-8")
                            except UnicodeDecodeError:
                                text = data.decode("latin1", "ignore")
                        texts.append(text)
                    except Exception:
                        continue
            combined = "\n".join(texts)
            return self._truncate(combined) if combined.strip() else "[ZIP пуст.]"
        except Exception as e:
            return f"[Ошибка ZIP ({os.path.basename(path)}): {e}]"

    def _extract_tar(self, path: str) -> str:
        try:
            texts = []
            with tarfile.open(path, "r:*") as tf:
                for member in tf.getmembers():
                    if member.isdir():
                        continue
                    try:
                        file_obj = tf.extractfile(member)
                        if not file_obj:
                            continue
                        data = file_obj.read()
                        ext = self._detect_extension(member.name)
                        extractor = self._extractors.get(ext)
                        if extractor and extractor not in {
                            self._extract_zip,
                            self._extract_tar,
                        }:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=ext
                            ) as tmp:
                                tmp.write(data)
                                tmp.flush()
                                text = extractor(tmp.name)
                            os.unlink(tmp.name)
                        else:
                            try:
                                text = data.decode("utf-8")
                            except UnicodeDecodeError:
                                text = data.decode("latin1", "ignore")
                        texts.append(text)
                    except Exception:
                        continue
            combined = "\n".join(texts)
            return self._truncate(combined) if combined.strip() else "[TAR пуст.]"
        except Exception as e:
            return f"[Ошибка TAR ({os.path.basename(path)}): {e}]"

    def extract(self, path: str) -> str:
        ext = self._detect_extension(path)
        extractor = self._extractors.get(ext)
        if not extractor:
            return f"[Неподдерживаемый тип файла: {os.path.basename(path)}]"
        return extractor(path)

    async def extract_async(self, path: str) -> str:
        return await asyncio.to_thread(self.extract, path)


async def parse_and_store_file(
    path: str,
    *,
    handler: FileHandler | None = None,
    engine: "VectorGrokkyEngine" | None = None,
) -> str:
    """Extract text and store a short summary in vector memory."""

    from utils.vector_engine import VectorGrokkyEngine

    handler = handler or FileHandler()
    text = await handler.extract_async(path)

    engine = engine or VectorGrokkyEngine()
    summary = ""
    try:
        summary = await engine.generate_with_xai(
            [
                {
                    "role": "user",
                    "content": f"Сжато опиши файл {os.path.basename(path)}: {text}",
                }
            ]
        )
    except Exception:
        pass

    try:
        content = f"FILE {os.path.basename(path)} SUMMARY: {summary}\n{text}"
        await engine.add_memory("document", content, role="journal")
    except Exception:
        pass

    return text


def create_repo_snapshot(base_path: str = ".", out_path: str = REPO_SNAPSHOT_PATH) -> None:
    """Write a simple listing of the repository to ``out_path``."""

    lines = []
    for root, _, files in os.walk(base_path):
        if ".git" in root.split(os.sep):
            continue
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, base_path)
            try:
                with open(p, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()[:8]
                size = os.path.getsize(p)
                line = f"{rel} ({size}b {h})"
                if name.endswith(".py"):
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        snippet = " ".join(f.readline().strip() for _ in range(3))
                    line += f" -> {snippet}"
                lines.append(line)
            except Exception:
                continue
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines)))

