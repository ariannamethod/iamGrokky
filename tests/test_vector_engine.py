import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import asyncio

import pytest

sys.modules.setdefault("httpx", types.SimpleNamespace())

spec = importlib.util.spec_from_file_location(
    "vector_engine", Path(__file__).resolve().parents[1] / "utils" / "vector_engine.py"
)
vector_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_engine)
VectorGrokkyEngine = vector_engine.VectorGrokkyEngine


def old_generate_embedding(text: str, dimension: int):
    hash_obj = hashlib.sha256(text.encode())
    hash_digest = hash_obj.digest()
    vector = []
    for i in range(dimension):
        byte_index = i % len(hash_digest)
        vector.append((hash_digest[byte_index] / 255.0) * 2 - 1)
    return vector


def test_generate_embedding_length_and_range():
    engine = VectorGrokkyEngine()
    text = "test text"

    new_vector = asyncio.run(engine.generate_embedding(text))
    old_vector = old_generate_embedding(text, engine.vector_dimension)

    assert len(new_vector) == len(old_vector)
    assert max(new_vector) == pytest.approx(max(old_vector))
    assert min(new_vector) == pytest.approx(min(old_vector))

