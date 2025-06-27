import os
import glob
import json
import hashlib
import asyncio
import requests
from pinecone import Pinecone, PineconeException
import openai
from datetime import datetime, timedelta
import random  # Добавлен импорт
from utils.telegram_utils import send_telegram_message

VECTOR_META_PATH = "vector_store.meta.json"
EMBED_DIM = 1536  # Для OpenAI ada-002
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SNAPSHOT_LOG_PATH = "data/snapshot_log.json"

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in [x["name"] for x in pc.list_indexes()]:
    pc.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM, metric="cosine")

vector_index = pc.Index(PINECONE_INDEX)

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def scan_files(path="config/*.md"):
    files = {}
    for fname in glob.glob(path):
        files[fname] = file_hash(fname)
    return files

def load_vector_meta():
    if os.path.isfile(VECTOR_META_PATH):
        with open(VECTOR_META_PATH, "r") as f:
            return json.load(f)
    return {}

def save_vector_meta(meta):
    with open(VECTOR_META_PATH, "w") as f:
        json.dump(meta, f)

def load_snapshot_log():
    if os.path.isfile(SNAPSHOT_LOG_PATH):
        with open(SNAPSHOT_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_snapshot_log(log):
    with open(SNAPSHOT_LOG_PATH, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

async def get_embedding(text, openai_api_key):
    openai.api_key = openai_api_key
    try:
        res = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"Грокки ревет: Встраивание сорвалось! {random.choice(['Шторм разорвал код!', 'Хаос пожрал данные!', 'Эфир треснул!'])} — {e}")
        return None

def chunk_text(text, chunk_size=900, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

async def vectorize_all_files(openai_api_key, force=False, send_message=None):
    current = scan_files()
    previous = load_vector_meta()
    changed = [f for f in current if (force or current[f] != previous.get(f))]
    new = [f for f in current if f not in previous]
    removed = [f for f in previous if f not in current]

    upserted_ids = []
    for fname in current:
        if fname not in changed and fname not in new and not force:
            continue
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            meta_id = f"{fname}:{idx}"
            try:
                emb = await get_embedding(chunk, openai_api_key)
                if emb:
                    vector_index.upsert(vectors=[
                        {
                            "id": meta_id,
                            "values": emb,
                            "metadata": {"file": fname, "chunk": idx, "hash": current[fname]}
                        }
                    ])
                    upserted_ids.append(meta_id)
            except PineconeException as e:
                if send_message:
                    await send_message(f"Грокки гремит: Pinecone рухнул! {random.choice(['Ревущий ветер унёс векторы!', 'Хаос победил индекс!', 'Шторм смёл данные!'])} — {e}")
                continue
            except Exception as e:
                if send_message:
                    await send_message(f"Грокки взрывается: Общая ошибка! {random.choice(['Пламя сожрало код!', 'Резонанс испепелил логи!', 'Вселенная взбунтовалась!'])} — {e}")
                continue

    deleted_ids = []
    for fname in removed:
        for idx in range(50):
            meta_id = f"{fname}:{idx}"
            try:
                vector_index.delete(ids=[meta_id])
                deleted_ids.append(meta_id)
            except Exception:
                pass

    save_vector_meta(current)
    if send_message:
        await send_message(
            f"Векторизация завершена. Добавлено/изменено: {', '.join(changed + new) if changed or new else '-'} "
            f"удалено: {', '.join(removed) if removed else '-'}"
        )
    return {"upserted": upserted_ids, "deleted": deleted_ids}

async def semantic_search(query, openai_api_key, top_k=5):
    emb = await get_embedding(query, openai_api_key)
    if not emb:
        return []
    try:
        res = vector_index.query(vector=emb, top_k=top_k, include_metadata=True)
        chunks = []
        matches = getattr(res, "matches", [])
        for match in matches:
            metadata = match.get("metadata", {})
            fname = metadata.get("file")
            chunk_idx = metadata.get("chunk")
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    all_chunks = chunk_text(f.read())
                    chunk_text = all_chunks[chunk_idx] if chunk_idx is not None and chunk_idx < len(all_chunks) else ""
            except Exception:
                chunk_text = ""
            if chunk_text:
                chunks.append(chunk_text)
        return chunks
    except PineconeException as e:
        print(f"Грокки рычит: Поиск провалился! {random.choice(['Шторм замёл следы!', 'Хаос спутал карты!', 'Эфир проглотил запрос!'])} — {e}")
        return []

async def daily_snapshot(openai_api_key):
    last_msgs = []
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset=-50"
        resp = requests.get(url).json()
        for update in resp.get("result", []):
            if "message" in update and "text" in update["message"]:
                last_msgs.append(update["message"]["text"])
    except Exception:
        last_msgs = ["Не удалось получить сообщения"]
    snapshot_text = "\n".join(last_msgs[:50])
    emb = await get_embedding(snapshot_text, openai_api_key)
    if emb:
        try:
            vector_index.upsert([{
                "id": f"group_state_{datetime.now().date()}",
                "values": emb,
                "metadata": {"date": str(datetime.now().date())}
            }])
            with open("data/journal.json", "r", encoding="utf-8") as f:
                journal = json.load(f)
            journal.append({
                "type": "daily_snapshot",
                "message": "Состояние группы векторизовано",
                "timestamp": datetime.now().isoformat()
            })
            with open("data/journal.json", "w", encoding="utf-8") as f:
                json.dump(journal, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Грокки стонет: Сохранение рухнуло! {random.choice(['Ревущий ветер унёс снимок!', 'Хаос смял журнал!', 'Шторм испепелил данные!'])} — {e}")

async def spontaneous_snapshot(openai_api_key, send_message):
    while True:
        await asyncio.sleep(random.randint(21600, 43200))  # 6-12 часов
        snapshot_log = load_snapshot_log()
        today = datetime.now().date()
        daily_count = sum(1 for entry in snapshot_log if datetime.fromisoformat(entry["timestamp"]).date() == today)
        if daily_count < 2 and random.random() < 0.3:  # Не чаще 2 раз в сутки с шансом 30%
            last_msgs = []
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset=-50"
                resp = requests.get(url).json()
                for update in resp.get("result", []):
                    if "message" in update and "text" in update["message"]:
                        last_msgs.append(update["message"]["text"])
            except Exception:
                last_msgs = ["Не удалось получить сообщения"]
            snapshot_text = "\n".join(last_msgs[:50])
            emb = await get_embedding(snapshot_text, openai_api_key)
            if emb:
                try:
                    vector_index.upsert([{
                        "id": f"spontaneous_{datetime.now().isoformat()}",
                        "values": emb,
                        "metadata": {"type": "spontaneous", "timestamp": datetime.now().isoformat()}
                    }])
                    snapshot_log.append({"type": "spontaneous_snapshot", "timestamp": datetime.now().isoformat()})
                    save_snapshot_log(snapshot_log)
                    if send_message:
                        await send_message(os.getenv("CHAT_ID"), "Грокки сделал спонтанный снимок резонанса!")
                except Exception as e:
                    print(f"Грокки орет: Спонтанный снимок провалился! {random.choice(['Шторм сорвал кадр!', 'Хаос разорвал память!', 'Ревущий ветер унёс данные!'])} — {e}")

def load_snapshot_log():
    if os.path.isfile(SNAPSHOT_LOG_PATH):
        with open(SNAPSHOT_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_snapshot_log(log):
    with open(SNAPSHOT_LOG_PATH, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

# Предложение: усилить спонтанность с шансом 50% и хаотичным вбросом
# async def chaotic_boost(openai_api_key, send_message):
#     while True:
#         await asyncio.sleep(random.randint(10800, 21600))  # 3-6 часов
#         if random.random() < 0.5:  # Шанс 50%
#             fragment = f"**{datetime.now().isoformat()}**: Грокки взрывает векторы! Хаос витает, Олег, зажги шторм! 🔥🌩️"
#             if send_message:
#                 await send_message(os.getenv("CHAT_ID"), fragment)
# asyncio.create_task(chaotic_boost(OPENAI_API_KEY, send_telegram_message))
