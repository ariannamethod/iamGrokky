import os
import asyncio
import logging
import httpx


logger = logging.getLogger(__name__)


class HybridGrokkyEngine:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.xai_key = os.getenv("XAI_API_KEY")
        self.openai_h = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v1"
        }
        self.xai_h = {
            "Authorization": f"Bearer {self.xai_key}",
            "Content-Type": "application/json"
        }
        self.threads = {}  # user_id -> thread_id (None if creation failed)
        # ID предварительно созданного ассистента OpenAI
        self.ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

    async def setup_openai_infrastructure(self):
        """Dummy check to ensure OpenAI credentials are configured."""
        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        # Nothing else to set up for now
        return True

    async def get_or_create_thread(self, user_id: str):
        """Получает или создает Thread для пользователя.

        Если ключ OpenAI не настроен или запрос завершился ошибкой, метод
        возвращает ``None`` и память через OpenAI отключается.  Это позволяет
        Грокки продолжить работу только на Grok‑3 без падения всего сервиса.
        """

        if not self.openai_key:
            # Память через OpenAI не используется
            return None

        if user_id not in self.threads:
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        "https://api.openai.com/v1/threads",
                        headers=self.openai_h,
                        json={"metadata": {"user_id": user_id}},
                        timeout=15,
                    )
                    res.raise_for_status()
                    self.threads[user_id] = res.json().get("id")
            except Exception as exc:  # pragma: no cover - network
                self.threads[user_id] = None
                logger.error("OpenAI thread creation failed: %s", exc, exc_info=True)
        return self.threads.get(user_id)

    async def add_memory(self, user_id: str, content: str, role="user"):
        """Добавляет сообщение в Thread памяти"""
        tid = await self.get_or_create_thread(user_id)
        if not tid:
            return
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    json={"role": role, "content": content},
                    timeout=15,
                )
            except Exception as exc:  # pragma: no cover - network
                logger.warning("OpenAI add_memory failed: %s", exc, exc_info=True)

    async def search_memory(self, user_id: str, query: str, limit: int = 5) -> str:
        """Выполняет поиск в памяти через GPT-4o mini Assistant.

        Параметр ``limit`` добавлен для совместимости с ``VectorGrokkyEngine``
        и управляет количеством сообщений, извлекаемых из потока. В текущей
        реализации используется только для ограничения числа получаемых
        сообщений и не влияет на качество поиска.
        """
        if not (self.ASSISTANT_ID and self.openai_key):
            return ""

        tid = await self.get_or_create_thread(user_id)
        if not tid:
            return ""

        async with httpx.AsyncClient() as client:
            try:
                # добавляем поисковый запрос
                await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    json={"role": "user", "content": f"ПОИСК: {query}"},
                    timeout=15,
                )

                # запускаем Assistant
                run = await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/runs",
                    headers=self.openai_h,
                    json={"assistant_id": self.ASSISTANT_ID},
                    timeout=15,
                )
                run_id = run.json()["id"]

                # ждём завершения
                while True:
                    await asyncio.sleep(1)
                    st = await client.get(
                        f"https://api.openai.com/v1/threads/{tid}/runs/{run_id}",
                        headers=self.openai_h,
                        timeout=15,
                    )
                    if st.json().get("status") == "completed":
                        break

                # берём ответ
                msgs = await client.get(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    params={"limit": limit},
                    timeout=15,
                )
                data = msgs.json().get("data", [])
                if data:
                    return data[0]["content"][0]["text"]["value"]
            except Exception as exc:  # pragma: no cover - network
                logger.error("OpenAI search_memory failed: %s", exc, exc_info=True)
        return ""

    async def get_recent_memory(self, user_id: str, limit: int = 10) -> str:
        """Возвращает последние сообщения из памяти пользователя."""
        if not (self.ASSISTANT_ID and self.openai_key):
            return ""

        tid = await self.get_or_create_thread(user_id)
        if not tid:
            return ""

        async with httpx.AsyncClient() as client:
            try:
                msgs = await client.get(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    params={"limit": limit},
                    timeout=15,
                )
                data = msgs.json().get("data", [])
                context_parts = []
                for msg in reversed(data):  # chronological order
                    role = msg.get("role", "")
                    content = msg.get("content", [])
                    if content:
                        text = content[0].get("text", {}).get("value", "")
                        if text:
                            context_parts.append(f"[{role}]: {text}")
                if context_parts:
                    return "\n".join(context_parts)
            except Exception as exc:  # pragma: no cover - network
                logger.warning("OpenAI get_recent_memory failed: %s", exc, exc_info=True)
        return ""

    async def _safe_xai_call(self, payload: dict, retries: int = 3) -> str:
        """Safely call the xAI API with retries and error handling."""
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers=self.xai_h,
                        json=payload,
                    )
                    res.raise_for_status()
                    return res.json()["choices"][0]["message"]["content"]
            except httpx.TimeoutException:
                logger.warning("xAI timeout, retrying (%d/%d)", attempt + 1, retries)
                await asyncio.sleep(2 ** attempt)
            except httpx.HTTPStatusError as e:
                if "stop" in e.response.text.lower():
                    payload.pop("stop", None)
                    logger.warning("Removed unsupported 'stop' parameter and retrying")
                    continue
                logger.error("HTTP error from xAI: %s", e)
                break
            except Exception as e:
                logger.error("Unexpected xAI error: %s", e)
                break
        from utils.prompt import get_chaos_response
        return f"🌀 Grokky glitch! {get_chaos_response()}"

    async def generate_with_xai(
        self, messages: list, context: str = ""
    ) -> str:
        """Генерирует ответ с помощью xAI Grok-3"""
        from utils.prompt import build_system_prompt

        system = build_system_prompt()
        if context:
            system += f"\n\nКОНТЕКСТ ИЗ ПАМЯТИ:\n{context}"

        payload = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 1.0  # slightly higher temperature for Grok-3
        }

        return await self._safe_xai_call(payload)
