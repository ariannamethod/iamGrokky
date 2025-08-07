import os
import time
import math
import asyncio
from typing import Optional, Sequence, List

import httpx


async def query_grok3(prompt: str, api_key: Optional[str] = None) -> str:
    """Call the Grok-3 API as a dynamic knowledge base."""
    api_key = api_key or os.getenv("XAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"prompt": prompt, "max_tokens": 500}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                "https://api.xai.org/grok-3/generate",
                json=payload,
                headers=headers,
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


async def query_gpt4(
    prompt: str, api_key: Optional[str] = None, model: str = "gpt-3.5"
) -> str:
    """Call the GPT-3.5 API as a secondary knowledge base.

    The default model is set to ``gpt-3.5`` to align with SLNCX's requirement
    for dynamic weighting based on the GPT-3.5 series."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
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
                f.write(f"{time.time()}: GPT-3.5 API failed - {exc}\n")
        except OSError:
            pass
        return "GPT-3.5 offline"


async def aget_dynamic_knowledge(
    prompt: str, api_key: Optional[str] = None
) -> str:
    """Fetch knowledge from Grok-3 with GPT-3.5 fallback asynchronously."""
    grok_task = query_grok3(prompt, api_key)
    gpt_task = query_gpt4(prompt, api_key)
    grok_res, gpt_res = await asyncio.gather(grok_task, gpt_task)
    return grok_res if not grok_res.startswith("Grok-3 offline") else gpt_res


def get_dynamic_knowledge(prompt: str, api_key: Optional[str] = None) -> str:
    """Synchronous wrapper for :func:`aget_dynamic_knowledge`."""
    return asyncio.run(aget_dynamic_knowledge(prompt, api_key))


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


class DynamicWeights:
    """Utility for producing fluid, context-aware weight coefficients.

    The class fetches dynamic knowledge for a given ``prompt`` and converts the
    amount of retrieved information into a ``pulse`` value between ``0`` and
    ``1``. The ``pulse`` then modulates a base sequence of weights via
    :func:`apply_pulse` to yield a normalised list of coefficients.
    """

    def __init__(self, base: Optional[Sequence[float]] = None) -> None:
        self.base = list(base or [1.0])

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
        api_key: Optional[str] = None,
        reward: float = 0.0,
    ) -> List[float]:
        """Return softmax-normalised weights for ``prompt``.

        ``reward`` is a feedback signal in ``[-1, 1]`` that nudges the
        resulting ``pulse`` up or down before weights are calculated.  A
        positive reward strengthens later responses while a negative value
        shifts preference toward earlier ones.  The value is clamped to keep
        the effective pulse within ``[0, 1]``.
        """

        pulse = self.pulse_from_prompt(prompt, api_key)
        pulse = max(0.0, min(1.0, pulse + reward))
        n = len(self.base)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

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
        return apply_pulse(shaped, pulse)

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
