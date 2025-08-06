from typing import Optional, List

from utils.dynamic_weights import DynamicWeights

# A tiny placeholder model that selects a canned response based on
# dynamically computed weights. This keeps the inference fully local while
# still deriving its behaviour from external knowledge sources via
# :class:`DynamicWeights`.
_RESPONSES: List[str] = [
    "⚡ SLNCX: Task acknowledged.",
    "⚡ SLNCX: Clarify your request.",
    "⚡ SLNCX: No action required."
]


def generate(prompt: str, ckpt_path: str = "out/ckpt.pt", api_key: Optional[str] = None) -> str:
    """Return a response from the lightweight SLNCX model.

    The model itself is a minimal local network: it selects one of a few
    predefined responses. The selection is driven by dynamic weights computed
    from ``prompt`` using external knowledge fetched by ``DynamicWeights``.
    This allows SLNCX to remain offline while still being influenced by
    GPT‑4.1 or Grok‑3.
    """

    controller = DynamicWeights([1.0] * len(_RESPONSES))
    weights = controller.weights_for_prompt(prompt, api_key)
    if not weights:
        return _RESPONSES[0]
    index = max(range(len(weights)), key=lambda i: weights[i])
    return _RESPONSES[index]
