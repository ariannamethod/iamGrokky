from typing import Optional
import importlib


def generate(
    prompt: str,
    ckpt_path: str = "out/ckpt.pt",
    api_key: Optional[str] = None,
) -> str:
    """Proxy to :func:`model.generate` for backward compatibility."""

    try:
        model_module = importlib.import_module(".model", __package__)
        return model_module.generate(prompt, ckpt_path, api_key)
    except ModuleNotFoundError:
        from utils.dynamic_weights import DynamicWeights

        controller = DynamicWeights()
        return controller.generate_response(prompt, api_key)
