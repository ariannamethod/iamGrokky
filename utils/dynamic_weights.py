import os
import time
import math
from typing import Optional, Sequence, List

import requests


def query_grok3(prompt: str, api_key: Optional[str] = None) -> str:
    """Call the Grok-3 API as a dynamic knowledge base."""
    api_key = api_key or os.getenv("XAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"prompt": prompt, "max_tokens": 500}
    try:
        res = requests.post(
            "https://api.xai.org/grok-3/generate", json=payload, headers=headers
        )
        res.raise_for_status()
        return res.json().get("text", "")
    except Exception as exc:  # pragma: no cover - network
        try:
            os.makedirs("failures", exist_ok=True)
            with open(
                f"failures/{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8"
            ) as f:
                f.write(f"{time.time()}: Grok-3 API failed - {exc}\n")
        except OSError:
            pass
        return "Grok-3 offline"


def query_gpt4(prompt: str, api_key: Optional[str] = None, model: str = "gpt-4o") -> str:
    """Call the GPT-4 API as a secondary knowledge base."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - network
        try:
            os.makedirs("failures", exist_ok=True)
            with open(
                f"failures/{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8"
            ) as f:
                f.write(f"{time.time()}: GPT-4 API failed - {exc}\n")
        except OSError:
            pass
        return "GPT-4 offline"


def get_dynamic_knowledge(prompt: str, api_key: Optional[str] = None) -> str:
    """Fetch knowledge from Grok-3 with GPT-4 fallback."""
    knowledge = query_grok3(prompt, api_key)
    if knowledge.startswith("Grok-3 offline"):
        knowledge = query_gpt4(prompt, api_key)
    return knowledge


def apply_pulse(weights: Sequence[float], pulse: float) -> List[float]:
    """Scale ``weights`` by ``pulse`` using a softmax normalisation.

    ``pulse`` is expected to be between ``0`` and ``1``.  The function first
    scales the weights by ``1 + pulse * 0.7`` and then applies a numerically
    stable softmax.  The returned list sums to ``1`` and can be used directly
    as probabilities.
    """

    scaled = [w * (1 + pulse * 0.7) for w in weights]
    if not scaled:
        return []
    max_w = max(scaled)
    exps = [math.exp(w - max_w) for w in scaled]
    total = sum(exps) or 1.0
    return [e / total for e in exps]
