import os
import time
import math
import json
from typing import Optional, Sequence, List

import httpx


def query_grok3(prompt: str, api_key: Optional[str] = None) -> str:
    """Call the Grok-3 API as a dynamic knowledge base."""
    api_key = api_key or os.getenv("XAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"prompt": prompt, "max_tokens": 500}
    try:
        res = httpx.post(
            "https://api.xai.org/grok-3/generate",
            json=payload,
            headers=headers,
            timeout=30,
        )
        res.raise_for_status()
        return res.json().get("text", "")
    except Exception as exc:  # pragma: no cover - network
        try:
            os.makedirs("failures", exist_ok=True)
            with open(
                f"failures/{time.strftime('%Y-%m-%d')}.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(f"{time.time()}: Grok-3 API failed - {exc}\n")
        except OSError:
            pass
        return "Grok-3 offline"


def query_gpt4(
    prompt: str, api_key: Optional[str] = None, model: str = "gpt-4.1"
) -> str:
    """Call the GPT-4 API as a secondary knowledge base.

    The default model is set to ``gpt-4.1`` to align with SLNCX's requirement
    for dynamic weighting based on the latest GPT-4 series."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    try:
        res = httpx.post(
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
                f"failures/{time.strftime('%Y-%m-%d')}.log",
                "a",
                encoding="utf-8",
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

    ``pulse`` is expected to be between ``0`` and ``1``. The function first
    scales the weights by ``1 + pulse * 0.7`` and then applies a numerically
    stable softmax. The returned list sums to ``1`` and can be used directly as
    probabilities.
    """

    scaled = [w * (1 + pulse * 0.7) for w in weights]
    if not scaled:
        return []
    max_w = max(scaled)
    exps = [math.exp(w - max_w) for w in scaled]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _log_feedback(
    prompt: str, reward: float, weights: Sequence[float]
) -> None:
    """Append feedback info to ``data/feedback/log.jsonl``."""

    try:
        os.makedirs("data/feedback", exist_ok=True)
        with open(
            "data/feedback/log.jsonl", "a", encoding="utf-8"
        ) as f:
            entry = {
                "timestamp": time.time(),
                "prompt": prompt,
                "reward": reward,
                "weights": list(weights),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:  # pragma: no cover - best effort
        pass


_GLOBAL_BASE: List[float] = [1.0]


class DynamicWeights:
    """Utility for producing fluid, context-aware weight coefficients.

    The class fetches dynamic knowledge for a given ``prompt`` and converts the
    amount of retrieved information into a ``pulse`` value between ``0`` and
    ``1``. The ``pulse`` then modulates a base sequence of weights via
    :func:`apply_pulse` to yield a normalised list of coefficients.
    """

    def __init__(self, base: Optional[Sequence[float]] = None) -> None:
        global _GLOBAL_BASE
        if base is not None:
            _GLOBAL_BASE = list(base)
        self.base = _GLOBAL_BASE

    def pulse_from_prompt(
        self, prompt: str, api_key: Optional[str] = None
    ) -> float:
        """Derive a pulse value from external knowledge."""

        knowledge = get_dynamic_knowledge(prompt, api_key)
        # Simple heuristic: longer knowledge implies a stronger pulse.
        # The denominator is tuned to 300 so that relatively small snippets of
        # knowledge can already yield noticeable pulse values, enabling the
        # lightweight SLNCX model to vary its responses.
        pulse = min(len(knowledge) / 300.0, 1.0)
        return max(pulse, 0.0)

    def weights_for_prompt(
        self,
        prompt: str,
        reward: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> List[float]:
        """Return softmax-normalised weights for ``prompt``.

        If ``reward`` is provided, the pair is logged to
        ``data/feedback/log.jsonl`` together with the computed weights.
        """

        pulse = self.pulse_from_prompt(prompt, api_key)
        n = len(self.base)
        if n == 0:
            return []
        if n == 1:
            weights = [1.0]
            if reward is not None:
                _log_feedback(prompt, reward, weights)
            return weights

        # Distribute weight mass across ``self.base`` depending on how strong
        # the pulse is.  For ``n`` responses, we treat their indices as evenly
        # spaced points in ``[0, 1]`` and assign each response a triangular
        # weighting centred on its position.  This makes the first response
        # dominate when the pulse is low and gradually shifts preference toward
        # later responses as the pulse grows.
        positions = [i / (n - 1) for i in range(n)]
        shaped = [
            self.base[i] * max(1.0 - abs(pulse - pos) * 2, 0.0)
            for i, pos in enumerate(positions)
        ]
        weights = apply_pulse(shaped, pulse)
        if reward is not None:
            _log_feedback(prompt, reward, weights)
        return weights

    def generate_response(
        self, prompt: str, api_key: Optional[str] = None
    ) -> str:
        """Return text influenced by dynamic weighting.

        The method derives a ``pulse`` from ``prompt`` and appends a style hint
        before querying :func:`get_dynamic_knowledge` again. Low pulse values
        yield terse answers while higher pulses encourage more elaborate
        replies. This enables SLNCX to "speak" without relying on static
        templates.
        """

        pulse = self.pulse_from_prompt(prompt, api_key)
        if pulse < 0.3:
            style = "Answer briefly."
        elif pulse < 0.7:
            style = "Respond in a balanced tone."
        else:
            style = "Provide a detailed, expressive answer."
        enriched = f"{prompt}\n\n{style}"
        return get_dynamic_knowledge(enriched, api_key)
