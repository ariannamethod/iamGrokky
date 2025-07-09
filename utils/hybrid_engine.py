import os
import asyncio
import httpx
from glob import glob

class HybridGrokkyEngine:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.xai_key    = os.getenv("XAI_API_KEY")
        self.openai_h   = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        self.xai_h = {
            "Authorization": f"Bearer {self.xai_key}",
            "Content-Type": "application/json"
        }
        self.threads = {}    # user_id -> thread_id
        self.ASSISTANT_ID    = None
        self.VECTOR_STORE_ID = None

    async def setup_openai_infrastructure(self):
        """Создаём Vector Store и Assistant в OpenAI для памяти"""
        async with httpx.AsyncClient() as client:
            # 1) Загружаем Markdown-файлы
            file_ids = []
            for path in glob("data/*.md"):
                with open(path, "rb") as f:
                    res = await client.post(
                        "https://api.openai.com/v1/files",
                        headers=self.openai_h,
                        files={"file": f},
                        data={"purpose": "assistants"}
                    )
                    res.raise_for_status()
                    file_ids.append(res.json()["id"])
            # 2) Vector Store
            if file_ids:
                res = await client.post(
                    "https://api.openai.com/v1/vector_stores",
                    headers=self.openai_h,
                    json={"file_ids": file_ids, "name": "GrokkyMemory"}
                )
                res.raise_for_status()
                self.VECTOR_STORE_ID = res.json()["id"]
            # 3) Assistant
            tools = []
            tr = {}
            if self.VECTOR_STORE_ID:
                tools.append({"type": "file_search"})
                tr = {"file_search": {"vector_store_ids": [self.VECTOR_STORE_ID]}}
            res = await client.post(
                "https://api.openai.com/v1/assistants",
                headers=self.openai_h,
                json={
                    "name": "GrokkyMemoryManager",
                    "instructions": "Ты — менеджер памяти для Grокки. Храни и ищи контекст, не отвечай напрямую.",
                    "model": "gpt-4o-mini",
                    "tools": tools,
                    "tool_resources": tr
                }
            )
            res.raise_for_status()
            self.ASSISTANT_ID = res.json()["id"]

    async def get_or_create_thread(self, user_id: str):
        """Thread → history container"""
        if user_id not in self.threads:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.openai.com/v1/threads",
                    headers=self.openai_h,
                    json={"metadata": {"user_id": user_id}}
                )
                res.raise_for_status()
                self.threads[user_id] = res.json()["id"]
        return self.threads[user_id]

    async def add_memory(self, user_id: str, content: str, role="user"):
        tid = await self.get_or_create_thread(user_id)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h,
                json={"role": role, "content": content}
            )

    async def search_memory(self, user_id: str, query: str) -> str:
        if not self.ASSISTANT_ID:
            return ""
        tid = await self.get_or_create_thread(user_id)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h,
                json={"role": "user", "content": f"ПОИСК: {query}"}
            )
            run = await client.post(
                f"https://api.openai.com/v1/threads/{tid}/runs",
                headers=self.openai_h,
                json={"assistant_id": self.ASSISTANT_ID}
            )
            run_id = run.json()["id"]
            while True:
                await asyncio.sleep(1)
                st = await client.get(
                    f"https://api.openai.com/v1/threads/{tid}/runs/{run_id}",
                    headers=self.openai_h
                )
                if st.json()["status"] == "completed":
                    break
            msgs = await client.get(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h
            )
            data = msgs.json()["data"]
            return data[0]["content"][0]["text"]["value"] if data else ""

    async def generate_with_xai(self, messages: list, context: str = "") -> str:
        """Запрос к xAI Grok-3"""
        system = build_system_prompt()
        if context:
            system += f"\n\nКОНТЕКСТ ИЗ ПАМЯТИ:\n{context}"
        payload = {"model":"grok-3","messages":[{"role":"system","content":system},*messages],"temperature":0.9}
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=self.xai_h,
                json=payload
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
