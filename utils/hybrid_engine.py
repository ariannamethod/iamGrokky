import os
import asyncio
import httpx

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
        self.threads = {}  # user_id -> thread_id
        self.ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")  # Предустановленный ID ассистента
        
    async def get_or_create_thread(self, user_id: str):
        """Получает или создает Thread для пользователя"""
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
        """Добавляет сообщение в Thread памяти"""
        tid = await self.get_or_create_thread(user_id)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h,
                json={"role": role, "content": content}
            )

    async def search_memory(self, user_id: str, query: str) -> str:
        """Выполняет поиск в памяти через GPT-4o mini Assistant"""
        if not self.ASSISTANT_ID:
            return ""
            
        tid = await self.get_or_create_thread(user_id)
        async with httpx.AsyncClient() as client:
            # добавляем поисковый запрос
            await client.post(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h,
                json={"role": "user", "content": f"ПОИСК: {query}"}
            )
            
            # запускаем Assistant
            run = await client.post(
                f"https://api.openai.com/v1/threads/{tid}/runs",
                headers=self.openai_h,
                json={"assistant_id": self.ASSISTANT_ID}
            )
            run_id = run.json()["id"]
            
            # ждём завершения
            while True:
                await asyncio.sleep(1)
                st = await client.get(
                    f"https://api.openai.com/v1/threads/{tid}/runs/{run_id}",
                    headers=self.openai_h
                )
                if st.json()["status"] == "completed":
                    break
                    
            # берём ответ
            msgs = await client.get(
                f"https://api.openai.com/v1/threads/{tid}/messages",
                headers=self.openai_h,
                params={"limit": 1}
            )
            data = msgs.json()["data"]
            return data[0]["content"][0]["text"]["value"] if data else ""

    async def generate_with_xai(self, messages: list, context: str = "") -> str:
        """Генерирует ответ с помощью xAI Grok-3"""
        from utils.prompt import build_system_prompt
        
        system = build_system_prompt()
        if context:
            system += f"\n\nКОНТЕКСТ ИЗ ПАМЯТИ:\n{context}"
            
        payload = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.9
        }
        
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=self.xai_h,
                json=payload
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
