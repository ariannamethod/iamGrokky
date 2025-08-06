import re


def detect_language(text: str) -> str:
    """Detect language from ``text`` using simple heuristics.

    Returns ``"ru"`` if any Cyrillic characters are present, otherwise
    returns ``"en"``. This lightweight detector avoids heavyweight
    dependencies while enabling language-aware responses.
    """
    return "ru" if re.search(r"[А-Яа-я]", text) else "en"
