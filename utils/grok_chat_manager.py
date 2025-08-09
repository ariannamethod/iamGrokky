import asyncio
import logging
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)


class GrokChatManager:
    """Manage chat history and communication with xAI API."""

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.history: Dict[str, List[dict]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        messages = self.history.setdefault(session_id, [])
        messages.append({"role": role, "content": content})
        # keep last 20 messages
        if len(messages) > 20:
            self.history[session_id] = messages[-20:]

    def get_messages(self, session_id: str) -> List[dict]:
        return self.history.get(session_id, [])

    async def _request(self, payload: dict) -> str:
        backoff = 1.0
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                if res.status_code in (400, 422):
                    raise RuntimeError(f"Bad request: {res.text}")
                if res.status_code == 401:
                    raise RuntimeError("Unauthorized: check API key")
                if res.status_code == 429:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"xAI error {res.status_code}: {res.text}")
            except httpx.RequestError as e:
                logger.warning("Request error: %s", e)
                if attempt == 2:
                    raise RuntimeError(f"Request failed: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Max retries exceeded")

    async def safe_chat_completion(self, session_id: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.extend(self.get_messages(session_id))
        payload = {"model": "grok-3", "messages": messages}
        return await self._request(payload)

    async def quick_chat(self, messages: List[dict]) -> str:
        payload = {"model": "grok-3", "messages": messages}
        return await self._request(payload)
