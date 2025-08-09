import asyncio
import logging
import time
from typing import Dict, List

logger = logging.getLogger(__name__)


class ImprovedMemoryManager:
    """Store conversation history with optional Pinecone backend."""

    def __init__(self, api_key: str | None, index: str | None) -> None:
        self.api_key = api_key
        self.index_name = index
        self.local_memory: Dict[str, List[str]] = {}
        self.index = None
        if api_key and index:
            try:
                from pinecone import Pinecone

                pc = Pinecone(api_key=api_key)
                self.index = pc.Index(index)
            except Exception as e:  # pragma: no cover - optional
                logger.warning("Pinecone init failed: %s", e)
                self.index = None

    async def save(self, user_id: str, content: str, role: str = "user") -> None:
        if self.index:
            try:  # pragma: no cover - network
                await asyncio.to_thread(
                    self.index.upsert,
                    vectors=[(
                        f"{user_id}-{int(time.time()*1000)}",
                        [0.0] * 8,
                        {"text": content, "role": role, "user_id": user_id},
                    )],
                )
            except Exception as e:
                logger.warning("Pinecone save failed: %s", e)
        messages = self.local_memory.setdefault(user_id, [])
        messages.append(f"{role}: {content}")
        if len(messages) > 50:
            self.local_memory[user_id] = messages[-50:]

    async def retrieve(self, user_id: str, _query: str) -> str:
        if self.index:
            try:  # pragma: no cover - network
                res = await asyncio.to_thread(
                    self.index.query,
                    vector=[0.0] * 8,
                    filter={"user_id": user_id},
                    top_k=5,
                    include_metadata=True,
                )
                texts = [
                    m.get("metadata", {}).get("text", "") for m in res.get("matches", [])
                ]
                if texts:
                    return "\n".join(texts)
            except Exception as e:
                logger.warning("Pinecone query failed: %s", e)
        return "\n".join(self.local_memory.get(user_id, [])[-10:])
