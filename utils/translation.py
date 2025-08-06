"""Utility for translating text via OpenAI."""
from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def translate(text: str, target_lang: str) -> str:
    """Translate ``text`` into ``target_lang`` using OpenAI."""
    prompt = f"Translate the following text to {target_lang}: {text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network
        logger.error("Translation error: %s", exc)
        return text


__all__ = ["translate"]
