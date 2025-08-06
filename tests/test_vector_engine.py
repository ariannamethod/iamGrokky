import importlib.util
from pathlib import Path
import sys
import types
import asyncio
import numpy as np

import pytest

# Stub out httpx to avoid network calls during import
sys.modules.setdefault("httpx", types.SimpleNamespace())

spec = importlib.util.spec_from_file_location(
    "vector_engine", Path(__file__).resolve().parents[1] / "utils" / "vector_engine.py"
)
vector_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_engine)
VectorGrokkyEngine = vector_engine.VectorGrokkyEngine


class DummyIndex:
    """Простейший in-memory индекс для тестов."""

    def __init__(self):
        self.records = []

    def upsert(self, vectors):
        for record_id, vector, metadata in vectors:
            self.records.append((record_id, np.array(vector), metadata))

    def query(self, *, vector, filter=None, top_k=5, include_metadata=True):
        vector = np.array(vector)
        matches = []
        for record_id, vec, metadata in self.records:
            if filter and filter.get("user_id") != metadata.get("user_id"):
                continue
            if np.linalg.norm(vec) == 0 or np.linalg.norm(vector) == 0:
                score = 0.0
            else:
                score = float(np.dot(vec, vector) / (np.linalg.norm(vec) * np.linalg.norm(vector)))
            matches.append({"id": record_id, "score": score, "metadata": metadata})
        matches.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": matches[:top_k]}


def test_generate_embedding_shape_and_determinism():
    engine = VectorGrokkyEngine()
    text = "test text"

    vec1 = asyncio.run(engine.generate_embedding(text))
    vec2 = asyncio.run(engine.generate_embedding(text))

    assert len(vec1) == engine.vector_dimension
    assert vec1 == pytest.approx(vec2)


def test_search_memory_returns_relevant_context():
    engine = VectorGrokkyEngine()
    engine.index = DummyIndex()

    async def setup():
        await engine.add_memory("user1", "The cat sits on the mat", role="user")
        await engine.add_memory("user1", "Dogs are friendly animals", role="user")

    asyncio.run(setup())

    result = asyncio.run(engine.search_memory("user1", "The cat sits on the mat"))
    assert "cat" in result.lower()
    assert "dogs" not in result.lower()
