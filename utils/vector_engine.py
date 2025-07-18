import os
import asyncio
import httpx
import json
import traceback
import logging
import time
from datetime import datetime
import hashlib
import numpy as np

# Настройка логгера
logger = logging.getLogger(__name__)


class VectorGrokkyEngine:
    def __init__(self):
        self.xai_key = os.getenv("XAI_API_KEY")
        self.xai_h = {
            "Authorization": f"Bearer {self.xai_key}",
            "Content-Type": "application/json",
        }

        # Настройки Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        # Дополнительная переменная для указания облака и региона Pinecone
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        self.vector_dimension = 1536  # Размерность вектора, стандартная для OpenAI
        self.pc = None
        self.index = None

        # Инициализация Pinecone
        if self.pinecone_api_key and self.pinecone_index:
            self._init_pinecone()
        else:
            logger.warning("Pinecone не настроен, функция векторной памяти отключена")

        # Снепшот состояния при старте
        self._create_memory_snapshot()

    def _init_pinecone(self):
        """Инициализирует соединение с Pinecone"""
        try:
            from pinecone import Pinecone, ServerlessSpec

            # Определяем регион и облако из переменной окружения
            if "-" in self.pinecone_environment:
                region, cloud = self.pinecone_environment.split("-", 1)
            else:
                region, cloud = self.pinecone_environment, "aws"

            self.pc = Pinecone(api_key=self.pinecone_api_key)

            # Проверяем существование индекса
            existing_indexes = self.pc.list_indexes().names()

            if self.pinecone_index not in existing_indexes:
                logger.info(f"Создаем новый индекс Pinecone: {self.pinecone_index}")
                self.pc.create_index(
                    name=self.pinecone_index,
                    dimension=self.vector_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                # Даем время на создание индекса
                time.sleep(1)

            self.index = self.pc.Index(self.pinecone_index)
            logger.info(f"Pinecone индекс '{self.pinecone_index}' подключен успешно")

            # Проверяем статистику индекса
            stats = self.index.describe_index_stats()
            logger.info(f"Статистика индекса: {stats}")

        except Exception as e:
            logger.error(f"Ошибка инициализации Pinecone: {e}")
            logger.error(traceback.format_exc())
            self.index = None

    def _create_memory_snapshot(self):
        """Создает снепшот текущего состояния памяти при старте"""
        if not self.index:
            logger.warning("Снепшот памяти не создан - Pinecone не доступен")
            return

        try:
            # Получаем статистику индекса
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)

            logger.info(f"====== СНЕПШОТ ПАМЯТИ ======")
            logger.info(f"Время: {datetime.now().isoformat()}")
            logger.info(f"Всего векторов в базе: {total_vectors}")

            # Получаем образец данных для демонстрации (до 5 записей)
            if total_vectors > 0:
                try:
                    # Fetching sample of vectors with their metadata
                    # This is a simple approach - in production we'd use proper querying
                    sample_records = []
                    # This is pseudocode as Pinecone doesn't have a direct "list all" function
                    # In a real implementation, you would use a proper query approach

                    logger.info("Образец записей в памяти:")
                    for i, record in enumerate(sample_records[:5]):
                        metadata = record.get("metadata", {})
                        logger.info(
                            f"  {i+1}. User: {metadata.get('user_id')}, Role: {metadata.get('role')}"
                        )
                        logger.info(f"     Content: {metadata.get('text', '')[:50]}...")

                except Exception as e:
                    logger.error(f"Ошибка при получении образца данных: {e}")

            logger.info(f"============================")

        except Exception as e:
            logger.error(f"Ошибка при создании снепшота памяти: {e}")
            logger.error(traceback.format_exc())

    async def generate_embedding(self, text):
        """Генерирует векторный эмбеддинг для текста"""
        # В идеале здесь должен быть вызов модели для генерации эмбеддинга
        # Пока реализуем через хеш-функцию для демо

        # Создаем детерминированный хеш текста
        hash_obj = hashlib.sha256(text.encode())
        hash_digest = hash_obj.digest()

        # Генерируем псевдо-вектор нужной размерности
        vector = []
        for i in range(self.vector_dimension):
            byte_index = i % len(hash_digest)
            vector.append(
                (hash_digest[byte_index] / 255.0) * 2 - 1
            )  # нормализуем в диапазоне [-1, 1]

        return vector

    async def add_memory(self, user_id: str, content: str, role="user"):
        """Добавляет сообщение в векторную память"""
        if not self.index:
            return True  # Пропускаем, если Pinecone не настроен

        try:
            # Генерируем эмбеддинг для текста
            embedding = await self.generate_embedding(content)

            # Создаем метаданные
            metadata = {
                "text": content[:1000],  # Ограничиваем длину текста
                "role": role,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Уникальный ID для записи
            record_id = f"{user_id}_{role}_{int(time.time()*1000)}"

            # Сохраняем в Pinecone
            self.index.upsert(vectors=[(record_id, embedding, metadata)])

            logger.info(
                f"Добавлена запись в память: user={user_id}, role={role}, id={record_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении в память: {e}")
            logger.error(traceback.format_exc())
            return False

    async def search_memory(self, user_id: str, query: str, limit=5) -> str:
        """Ищет контекст в памяти на основе векторной близости"""
        if not self.index:
            return ""  # Возвращаем пустой контекст, если Pinecone не настроен

        try:
            # Генерируем эмбеддинг для запроса
            query_embedding = await self.generate_embedding(query)

            # Ищем похожие записи для данного пользователя
            results = self.index.query(
                vector=query_embedding,
                filter={"user_id": user_id},
                top_k=limit,
                include_metadata=True,
            )

            # Формируем контекст из найденных записей
            context_parts = []

            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                score = match.get("score", 0)

                if score < 0.7:  # Игнорируем записи с низкой релевантностью
                    continue

                text = metadata.get("text", "")
                role = metadata.get("role", "")
                timestamp = metadata.get("timestamp", "")

                # Форматируем запись для контекста
                context_parts.append(f"[{role} @ {timestamp}]: {text}")

            # Собираем контекст
            if context_parts:
                context = "\n\n".join(context_parts)
                logger.info(f"Найден контекст в памяти: {len(context_parts)} записей")
                return context
            else:
                logger.info("Релевантный контекст в памяти не найден")
                return ""

        except Exception as e:
            logger.error(f"Ошибка при поиске в памяти: {e}")
            logger.error(traceback.format_exc())
            return ""

    async def generate_with_xai(self, messages: list, context: str = "") -> str:
        """Генерирует ответ с помощью xAI Grok-3"""
        from utils.prompt import build_system_prompt

        system = build_system_prompt()
        if context:
            system += f"\n\nКОНТЕКСТ ИЗ ПАМЯТИ:\n{context}"

        payload = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 1.0,  # slightly higher temperature for more creativity
        }

        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=self.xai_h,
                    json=payload,
                    timeout=30.0,
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Ошибка при генерации с xAI: {e}")
                logger.error(traceback.format_exc())
                # Возвращаем резервный ответ при ошибке
                return "🌀 Грокки в замешательстве! Электрические импульсы перегружены. Попробуй еще раз!"

    async def get_recent_memory(self, user_id: str, limit: int = 10) -> str:
        """Возвращает последние записи пользователя."""
        if not self.index:
            return ""

        try:
            zero_vec = [0.0] * self.vector_dimension
            results = self.index.query(
                vector=zero_vec,
                filter={"user_id": user_id},
                top_k=limit,
                include_metadata=True,
            )

            records = sorted(
                results.get("matches", []),
                key=lambda x: x.get("metadata", {}).get("timestamp", ""),
                reverse=True,
            )
            texts = [r.get("metadata", {}).get("text", "") for r in records]
            return "\n".join(texts)

        except Exception as e:
            logger.error("Ошибка при получении последних записей: %s", e)
            logger.error(traceback.format_exc())
            return ""
