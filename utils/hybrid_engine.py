import os
import asyncio
import httpx


class HybridGrokkyEngine:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.xai_key = os.getenv("XAI_API_KEY")

        # Формируем заголовки только если ключи установлены, чтобы избежать
        # ошибочных запросов на OpenAI, которые приводят к 400 ошибкам.
        self.openai_h = (
            {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v1",
            }
            if self.openai_key
            else {}
        )
        self.xai_h = (
            {
                "Authorization": f"Bearer {self.xai_key}",
                "Content-Type": "application/json",
            }
            if self.xai_key
            else {}
        )

        # Локальный кэш нитей для Assistant API
        self.threads = {}  # user_id -> thread_id
        # ID предварительно созданного ассистента OpenAI
        self.ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

    async def setup_openai_infrastructure(self):
        """Проверяет наличие OpenAI API.

        Возвращает ``True``, если ключ настроен, иначе ``False``. Это
        позволяет работать без памяти GPT, не обрывая работу основного
        движка Grok-3."""
        return bool(self.openai_key)

    async def get_or_create_thread(self, user_id: str):
        """Получает или создает Thread для пользователя.

        Если OpenAI не настроен или запрос завершается ошибкой, возвращает
        ``None`` и тем самым отключает память GPT для данного пользователя."""
        if not self.openai_key:
            return None
        try:
            if user_id not in self.threads:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        "https://api.openai.com/v1/threads",
                        headers=self.openai_h,
                        json={"metadata": {"user_id": user_id}},
                        timeout=30,
                    )
                    res.raise_for_status()
                    self.threads[user_id] = res.json().get("id")
            return self.threads.get(user_id)
        except Exception:
            return None

    async def add_memory(self, user_id: str, content: str, role="user"):
        """Добавляет сообщение в Thread памяти"""
        tid = await self.get_or_create_thread(user_id)
        if not tid:
            return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    json={"role": role, "content": content},
                    timeout=30,
                )
        except Exception:
            # При ошибках записи памяти просто пропускаем
            pass

    async def search_memory(self, user_id: str, query: str) -> str:
        """Выполняет поиск в памяти через GPT-4o mini Assistant."""
        if not (self.openai_key and self.ASSISTANT_ID):
            return ""

        tid = await self.get_or_create_thread(user_id)
        if not tid:
            return ""
        try:
            async with httpx.AsyncClient() as client:
                # добавляем поисковый запрос
                await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    json={"role": "user", "content": f"ПОИСК: {query}"},
                    timeout=30,
                )

                # запускаем Assistant
                run = await client.post(
                    f"https://api.openai.com/v1/threads/{tid}/runs",
                    headers=self.openai_h,
                    json={"assistant_id": self.ASSISTANT_ID},
                    timeout=30,
                )
                run_id = run.json().get("id")

                # ждём завершения
                while True:
                    await asyncio.sleep(1)
                    st = await client.get(
                        f"https://api.openai.com/v1/threads/{tid}/runs/{run_id}",
                        headers=self.openai_h,
                        timeout=30,
                    )
                    if st.json().get("status") == "completed":
                        break

                # берём ответ
                msgs = await client.get(
                    f"https://api.openai.com/v1/threads/{tid}/messages",
                    headers=self.openai_h,
                    params={"limit": 1},
                    timeout=30,
                )
                data = msgs.json().get("data", [])
                if data:
                    return data[0]["content"][0]["text"]["value"]
        except Exception:
            pass
        return ""

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

        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=self.xai_h,
                json=payload
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
