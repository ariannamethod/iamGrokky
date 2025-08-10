"""Utilities for detecting text language."""

from langdetect import DetectorFactory, LangDetectException, detect

# Ensure deterministic results from langdetect
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """Detect the language of ``text``.

    Uses the :mod:`langdetect` library to return an ISO 639-1 language code.
    Falls back to ``"en"`` if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "en"


__all__ = ["detect_language"]
