import asyncio
import os
import json
import re
import random
from datetime import datetime
import httpx
from openai import OpenAI

class HybridGrokkyEngine:
    def __init__(self):
        # Существующие настройки
        self.openai_h = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
        self.xai_h = {"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}", "Content-Type": "application/json"}
        self.memory_path = "data/memory/"
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Новые настройки для Assistants API
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant_id = os.getenv("ASSISTANT_ID")
        self.threads = {}  # chat_id -> thread_id
        self._load_thread_mapping()

    async def setup_openai_infrastructure(self):
        # Проверка существующей настройки памяти

        # Проверка и создание ассистента если нужно
        if not self.assistant_id:
            from utils.prompt import build_system_prompt
            system_prompt = build_system_prompt()
            assistant = self.openai_client.beta.assistants.create(
                name="Grokky",
                instructions=system_prompt,
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}]
            )
            self.assistant_id = assistant.id
            print(f"Создан новый ассистент с ID: {self.assistant_id}")
        return True

    async def add_memory(self, user_id, text, role="user"):
        """Добавляет сообщение в память пользователя"""
        filename = f"{self.memory_path}/{user_id}.jsonl"
        try:
            memory_item = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": text
            }
            
            # Сохраняем в файловую память
            with open(filename, "a+", encoding="utf-8") as f:
                f.write(f"{json.dumps(memory_item, ensure_ascii=False)}\n")
                
            # Добавляем в тред Assistants API, если есть
            if user_id in self.threads and self.assistant_id:
                try:
                    self.openai_client.beta.threads.messages.create(
                        thread_id=self.threads[user_id],
                        role=role,
                        content=text
                    )
                except Exception as e:
                    print(f"Ошибка добавления в тред Assistant: {e}")
            
            return True
        except Exception as e:
            print(f"Ошибка добавления в память: {e}")
            return False

    async def search_memory(self, user_id, query, limit=5):
        """Ищет в памяти пользователя похожие сообщения"""
        filename = f"{self.memory_path}/{user_id}.jsonl"
        if not os.path.exists(filename):
            return ""
        
        # Существующий код для поиска в памяти...
        
        # Если используется Assistant API, можно использовать поиск сообщений там
        if user_id in self.threads and self.assistant_id:
            try:
                # Пока используем стандартную файловую память
                pass
            except Exception as e:
                print(f"Ошибка поиска в треде Assistant: {e}")
        
        return ""  # или результат поиска

    async def generate_with_xai(self, messages, context=""):
        """Генерирует ответ через xAI Grok"""
        from utils.prompt import build_system_prompt
        
        # Существующий код...
        system_prompt = build_system_prompt()
        
        if context:
            system_prompt = f"{system_prompt}\n\nContext for answering: {context}"
            
        full_messages = [{"role": "system", "content": system_prompt}] + messages
            
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=self.xai_h,
                    json={"model": "grok-3", "messages": full_messages, "temperature": 1.0}
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Ошибка xAI: {e}")
            return "Ошибка связи с нейросетью, эфир трещит!"

    async def generate_with_assistant(self, chat_id, messages=None):
        """Генерирует ответ используя Assistants API"""
        if not self.assistant_id:
            await self.setup_openai_infrastructure()
            
        thread_id = await self.get_or_create_thread(chat_id)
        
        # Если переданы новые сообщения, добавляем их
        if messages:
            for msg in messages:
                self.openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role=msg["role"],
                    content=msg["content"]
                )
        
        # Запускаем выполнение
        run = self.openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        
        # Ждем завершения выполнения
        run = self._wait_for_run_completion(thread_id, run.id)
        
        # Получаем последнее сообщение
        messages = self.openai_client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        if not messages.data:
            return "🌀 Грокки не смог сформировать ответ"
        
        return messages.data[0].content[0].text.value
    
    async def get_or_create_thread(self, chat_id):
        """Получает существующий или создает новый тред для чата"""
        if chat_id not in self.threads:
            thread = self.openai_client.beta.threads.create()
            self.threads[chat_id] = thread.id
            self._save_thread_mapping()
            return thread.id
        return self.threads[chat_id]
    
    def _wait_for_run_completion(self, thread_id, run_id):
        """Ждет завершения выполнения Run"""
        import time
        while True:
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            if run.status == 'completed':
                return run
            elif run.status in ['failed', 'cancelled', 'expired']:
                print(f"Run failed with status: {run.status}")
                return run
            time.sleep(1)
    
    def _save_thread_mapping(self):
        """Сохраняет маппинг chat_id -> thread_id в файл"""
        with open("data/thread_mapping.json", "w") as f:
            json.dump(self.threads, f)
    
    def _load_thread_mapping(self):
        """Загружает маппинг chat_id -> thread_id из файла"""
        try:
            with open("data/thread_mapping.json", "r") as f:
                self.threads = json.load(f)
        except FileNotFoundError:
            self.threads = {}
