from typing import Optional, List, Dict, AsyncGenerator

import anyio

from utils.dynamic_weights import DynamicWeights
from utils.language import detect_language

# A tiny placeholder model that selects a canned response based on
# dynamically computed weights. This keeps the inference fully local while
# still deriving its behaviour from external knowledge sources via
# :class:`DynamicWeights`.

_RESPONSES: Dict[str, List[str]] = {
    "en": [
        "⚡ SLNCX: Task acknowledged.",
        "⚡ SLNCX: Clarify your request.",
        "⚡ SLNCX: No action required.",
        "⚡ SLNCX: Resonance confirmed.",
        "⚡ SLNCX: Awaiting next directive.",
    ],
    "ru": [
        "⚡ SLNCX: Задача принята.",
        "⚡ SLNCX: Уточните запрос.",
        "⚡ SLNCX: Действий не требуется.",
        "⚡ SLNCX: Резонанс подтвержден.",
        "⚡ SLNCX: Ожидаю дальнейших указаний.",
    ],
}


async def generate(
    prompt: str, ckpt_path: str = "out/ckpt.pt", api_key: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Yield a response from the lightweight SLNCX model token by token.

    The model itself is a minimal local network: it selects one of a few
    predefined responses. The selection is driven by dynamic weights computed
    from ``prompt`` using external knowledge fetched by ``DynamicWeights``.
    This allows SLNCX to remain offline while still being influenced by
    GPT‑4.1 or Grok‑3.  To support streaming, the chosen response is yielded
    as an async generator one token at a time.
    """

    lang = detect_language(prompt)
    responses = _RESPONSES.get(lang, _RESPONSES["en"])
    controller = DynamicWeights([1.0] * len(responses))
    weights = controller.weights_for_prompt(prompt, api_key)
    if not weights:
        response = responses[0]
    else:
        index = max(range(len(weights)), key=lambda i: weights[i])
        response = responses[index]

    for token in response.split():
        # Yield each token followed by a space to mimic streaming behaviour.
        yield token + " "
        await anyio.sleep(0)

