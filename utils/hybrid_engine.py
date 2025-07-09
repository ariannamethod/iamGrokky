"""
Grokky AI Assistant - Hybrid Memory Engine
Память через OpenAI threads и vector store с поддержкой Pinecone
"""

import os
import json
import asyncio
import hashlib
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from openai import OpenAI
import aiofiles

# Опциональный импорт Pinecone
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone не установлен, используется только OpenAI threads")

from utils.prompt import build_system_prompt

class HybridMemoryEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_id = os.getenv("CHAT_ID")
        self.agent_group = os.getenv("AGENT_GROUP")
        self.is_group = os.getenv("IS_GROUP", "False").lower() == "true"
        
        # OpenAI Threads
        self.personal_thread_id = None
        self.group_thread_id = None
        
        # Vector Store (Pinecone опционально)
        self.pinecone_client = None
        self.vector_index = None
        self.setup_vector_store()
        
        # Локальные файлы
        self.threads_file = "data/threads.json"
        self.vector_meta_file = "data/vector_meta.json"
        self.memory_log_file = "data/memory_log.json"
        
        # Константы
        self.EMBED_DIM = 1536  # OpenAI ada-002
        self.MAX_THREAD_MESSAGES = 100
        self.CHUNK_SIZE = 900
        self.CHUNK_OVERLAP = 120

    def setup_vector_store(self):
        """Настройка vector store (Pinecone если доступен)"""
        if not PINECONE_AVAILABLE:
            return
            
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX")
        
        if pinecone_api_key and pinecone_index_name:
            try:
                self.pinecone_client = Pinecone(api_key=pinecone_api_key)
                
                # Создаем индекс если не существует
                existing_indexes = [idx["name"] for idx in self.pinecone_client.list_indexes()]
                if pinecone_index_name not in existing_indexes:
                    self.pinecone_client.create_index(
                        name=pinecone_index_name,
                        dimension=self.EMBED_DIM,
                        metric="cosine"
                    )
                
                self.vector_index = self.pinecone_client.Index(pinecone_index_name)
                print("Pinecone vector store инициализирован")
                
            except Exception as e:
                print(f"Ошибка инициализации Pinecone: {e}")
                self.pinecone_client = None

    async def get_or_create_thread(self, is_group_context=False):
        """Получает или создает OpenAI thread"""
        try:
            # Загружаем существующие threads
            threads_data = await self.load_threads_data()
            
            thread_key = "group_thread_id" if is_group_context else "personal_thread_id"
            thread_id = threads_data.get(thread_key)
            
            if thread_id:
                try:
                    # Проверяем существование thread
                    thread = self.openai_client.beta.threads.retrieve(thread_id)
                    return thread_id
                except Exception:
                    # Thread не существует, создаем новый
                    pass
            
            # Создаем новый thread
            thread = self.openai_client.beta.threads.create()
            thread_id = thread.id
            
            # Сохраняем thread ID
            threads_data[thread_key] = thread_id
            await self.save_threads_data(threads_data)
            
            print(f"Создан новый {'групповой' if is_group_context else 'личный'} thread: {thread_id}")
            return thread_id
            
        except Exception as e:
            print(f"Ошибка создания thread: {e}")
            return None

    async def load_threads_data(self):
        """Загружает данные threads из файла"""
        try:
            if os.path.exists(self.threads_file):
                async with aiofiles.open(self.threads_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            print(f"Ошибка загрузки threads: {e}")
        return {}

    async def save_threads_data(self, data):
        """Сохраняет данные threads в файл"""
        try:
            os.makedirs(os.path.dirname(self.threads_file), exist_ok=True)
            async with aiofiles.open(self.threads_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Ошибка сохранения threads: {e}")

    async def add_message_to_thread(self, message: str, is_group_context=False, author_name=None):
        """Добавляет сообщение в OpenAI thread"""
        try:
            thread_id = await self.get_or_create_thread(is_group_context)
            if not thread_id:
                return False
            
            # Формируем контент сообщения
            content = f"[{author_name or 'User'}]: {message}" if author_name else message
            
            # Добавляем сообщение в thread
            self.openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=content
            )
            
            # Логируем
            await self.log_memory_event({
                "type": "thread_message_added",
                "thread_id": thread_id,
                "is_group": is_group_context,
                "author": author_name,
                "message_length": len(message)
            })
            
            return True
            
        except Exception as e:
            print(f"Ошибка добавления сообщения в thread: {e}")
            return False

    async def get_thread_context(self, is_group_context=False, limit=20):
        """Получает контекст из OpenAI thread"""
        try:
            thread_id = await self.get_or_create_thread(is_group_context)
            if not thread_id:
                return []
            
            # Получаем сообщения из thread
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit
            )
            
            context = []
            for message in reversed(messages.data):
                if hasattr(message, 'content') and message.content:
                    for content_block in message.content:
                        if hasattr(content_block, 'text'):
                            context.append({
                                "role": message.role,
                                "content": content_block.text.value,
                                "timestamp": message.created_at
                            })
            
            return context
            
        except Exception as e:
            print(f"Ошибка получения контекста thread: {e}")
            return []

    async def get_embedding(self, text: str):
        """Получает embedding через OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка получения embedding: {e}")
            return None

    def chunk_text(self, text: str):
        """Разбивает текст на чанки"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def file_hash(self, filepath: str):
        """Вычисляет хеш файла"""
        try:
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    async def vectorize_config_files(self, force=False):
        """Векторизует файлы из папки config/"""
        if not self.vector_index:
            print("Vector store недоступен")
            return {"upserted": [], "deleted": []}
        
        try:
            # Сканируем файлы
            config_files = glob.glob("config/*.md") + glob.glob("config/*.txt")
            current_files = {f: self.file_hash(f) for f in config_files if self.file_hash(f)}
            
            # Загружаем предыдущие хеши
            previous_files = await self.load_vector_meta()
            
            # Определяем изменения
            changed_files = [f for f in current_files if force or current_files[f] != previous_files.get(f)]
            removed_files = [f for f in previous_files if f not in current_files]
            
            upserted_ids = []
            
            # Обрабатываем измененные файлы
            for filepath in changed_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    chunks = self.chunk_text(content)
                    
                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{filepath}:{idx}"
                        embedding = await self.get_embedding(chunk)
                        
                        if embedding:
                            self.vector_index.upsert(vectors=[{
                                "id": chunk_id,
                                "values": embedding,
                                "metadata": {
                                    "file": filepath,
                                    "chunk": idx,
                                    "hash": current_files[filepath],
                                    "timestamp": datetime.now().isoformat()
                                }
                            }])
                            upserted_ids.append(chunk_id)
                            
                except Exception as e:
                    print(f"Ошибка векторизации {filepath}: {e}")
                    continue
            
            # Удаляем векторы удаленных файлов
            deleted_ids = []
            for filepath in removed_files:
                for idx in range(100):  # Максимум 100 чанков на файл
                    chunk_id = f"{filepath}:{idx}"
                    try:
                        self.vector_index.delete(ids=[chunk_id])
                        deleted_ids.append(chunk_id)
                    except Exception:
                        break
            
            # Сохраняем метаданные
            await self.save_vector_meta(current_files)
            
            await self.log_memory_event({
                "type": "vectorization_completed",
                "upserted": len(upserted_ids),
                "deleted": len(deleted_ids),
                "files_processed": len(changed_files)
            })
            
            return {"upserted": upserted_ids, "deleted": deleted_ids}
            
        except Exception as e:
            print(f"Ошибка векторизации: {e}")
            return {"upserted": [], "deleted": []}

    async def semantic_search(self, query: str, top_k=5):
        """Семантический поиск по vector store"""
        if not self.vector_index:
            return []
        
        try:
            embedding = await self.get_embedding(query)
            if not embedding:
                return []
            
            results = self.vector_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            chunks = []
            for match in results.matches:
                metadata = match.metadata
                filepath = metadata.get("file")
                chunk_idx = metadata.get("chunk")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_chunks = self.chunk_text(content)
                    if chunk_idx < len(file_chunks):
                        chunks.append({
                            "content": file_chunks[chunk_idx],
                            "file": filepath,
                            "score": match.score
                        })
                        
                except Exception:
                    continue
            
            return chunks
            
        except Exception as e:
            print(f"Ошибка семантического поиска: {e}")
            return []

    async def load_vector_meta(self):
        """Загружает метаданные векторов"""
        try:
            if os.path.exists(self.vector_meta_file):
                async with aiofiles.open(self.vector_meta_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception:
            pass
        return {}

    async def save_vector_meta(self, data):
        """Сохраняет метаданные векторов"""
        try:
            os.makedirs(os.path.dirname(self.vector_meta_file), exist_ok=True)
            async with aiofiles.open(self.vector_meta_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Ошибка сохранения метаданных: {e}")

    async def log_memory_event(self, event):
        """Логирует события памяти"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                **event
            }
            
            # Загружаем существующий лог
            memory_log = []
            if os.path.exists(self.memory_log_file):
                async with aiofiles.open(self.memory_log_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    memory_log = json.loads(content)
            
            # Добавляем новую запись
            memory_log.append(log_entry)
            
            # Ограничиваем размер лога
            if len(memory_log) > 1000:
                memory_log = memory_log[-1000:]
            
            # Сохраняем
            os.makedirs(os.path.dirname(self.memory_log_file), exist_ok=True)
            async with aiofiles.open(self.memory_log_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(memory_log, ensure_ascii=False, indent=2))
                
        except Exception as e:
            print(f"Ошибка логирования: {e}")

    async def get_hybrid_context(self, query: str, is_group_context=False):
        """Получает гибридный контекст из threads + vector search"""
        context = {
            "thread_context": [],
            "semantic_context": [],
            "query": query
        }
        
        # Получаем контекст из thread
        thread_context = await self.get_thread_context(is_group_context, limit=10)
        context["thread_context"] = thread_context
        
        # Получаем семантический контекст
        semantic_results = await self.semantic_search(query, top_k=3)
        context["semantic_context"] = semantic_results
        
        return context

    async def create_snapshot(self, snapshot_type="daily"):
        """Создает снимок состояния для векторизации"""
        try:
            # Получаем контекст из обоих threads
            personal_context = await self.get_thread_context(False, limit=20)
            group_context = await self.get_thread_context(True, limit=20) if self.is_group else []
            
            # Формируем снимок
            snapshot = {
                "type": snapshot_type,
                "timestamp": datetime.now().isoformat(),
                "personal_context": personal_context,
                "group_context": group_context
            }
            
            # Векторизуем снимок если доступен vector store
            if self.vector_index:
                snapshot_text = json.dumps(snapshot, ensure_ascii=False)
                embedding = await self.get_embedding(snapshot_text)
                
                if embedding:
                    snapshot_id = f"snapshot_{snapshot_type}_{datetime.now().date()}"
                    self.vector_index.upsert(vectors=[{
                        "id": snapshot_id,
                        "values": embedding,
                        "metadata": {
                            "type": "snapshot",
                            "snapshot_type": snapshot_type,
                            "timestamp": datetime.now().isoformat()
                        }
                    }])
            
            await self.log_memory_event({
                "type": "snapshot_created",
                "snapshot_type": snapshot_type,
                "personal_messages": len(personal_context),
                "group_messages": len(group_context)
            })
            
            return snapshot
            
        except Exception as e:
            print(f"Ошибка создания снимка: {e}")
            return None

# Глобальный экземпляр движка памяти
memory_engine = HybridMemoryEngine()
