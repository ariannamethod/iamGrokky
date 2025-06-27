import os
import glob
import json
import hashlib
import asyncio
import requests
from pinecone import Pinecone, PineconeException
import openai
from datetime import datetime

VECTOR_META_PATH = "vector_store.meta.json"
EMBED_DIM = 1536  # For OpenAI ada-002
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

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

async def get_embedding(text, openai_api_key):
    openai.api_key = openai_api_key
    try:
        res = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
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

async def vectorize_all_files(openai_api_key, force=False, on_message=None):
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
                if on_message:
                    await on_message(f"Pinecone error: {e}")
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
    if on_message:
        await on_message(
            f"Vectorization complete. Added/changed: {', '.join(changed + new) if changed or new else '-'}; removed: {', '.join(removed) if removed else '-'}"
        )
    return {"upserted": upserted_ids, "deleted": deleted_ids}

async def semantic_search(query, openai_api_key, top_k=5):
    emb = await get_embedding(query, openai_api_key)
    if not emb:
        return []
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

async def daily_snapshot(openai_api_key):
    last_msgs = []
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset=-50"
        resp = requests.get(url).json()
        for update in resp.get("result", []):
            if "message" in update and "text" in update["message"]:
                last_msgs.append(update["message"]["text"])
    except Exception:
        last_msgs = ["No messages fetched"]
    snapshot_text = "\n".join(last_msgs[:50])
    emb = await get_embedding(snapshot_text, openai_api_key)
    if emb:
        vector_index.upsert([{
            "id": f"group_state_{datetime.now().date()}",
            "values": emb,
            "metadata": {"date": str(datetime.now().date())}
        }])
        try:
            with open("data/journal.json", "r", encoding="utf-8") as f:
                journal = json.load(f)
            journal.append({
                "type": "daily_snapshot",
                "message": "Group state vectorized",
                "timestamp": datetime.now().isoformat()
            })
            with open("data/journal.json", "w", encoding="utf-8") as f:
                json.dump(journal, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
