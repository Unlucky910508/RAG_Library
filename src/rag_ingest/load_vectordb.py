"""Embed each chunk (via a local OpenAI-compatible embeddings endpoint,
not local to this machine) and load it straight into a Chroma collection.

The raw vector is never written back to the chunks jsonl - it's only
meaningful to a vector index, so it goes directly into Chroma alongside
the chunk's text and metadata. Resumability is tracked by which chunk_ids
already exist in the collection, not by a field in the jsonl.
"""

import json
import sys
import time
from pathlib import Path

import chromadb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    CHROMA_DIR,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_VERIFY_SSL,
    chroma_collection_name,
    chunks_jsonl_path,
    load_llm_api_key,
)

if not LLM_VERIFY_SSL:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

SAVE_EVERY = 20
MAX_RETRIES = 3


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_id(chunk):
    return f"{chunk['record_id']}::{chunk['chunk_type']}"


def call_embedding(text, api_key):
    response = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
        verify=LLM_VERIFY_SSL,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url}: {response.text[:500]}",
            response=response,
        )
    return response.json()["data"][0]["embedding"]


def call_embedding_with_retry(chunk, api_key):
    for attempt in range(MAX_RETRIES):
        try:
            return call_embedding(chunk["text"], api_key)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  FAILED {chunk['record_id']} ({chunk['chunk_type']}): {e}")
                return None
            time.sleep(2 ** attempt)


def find_pending_chunks(collection, chunks):
    all_ids = [chunk_id(c) for c in chunks]
    existing_ids = set()
    for i in range(0, len(all_ids), 200):
        existing_ids.update(collection.get(ids=all_ids[i:i + 200])["ids"])
    return [c for c in chunks if chunk_id(c) not in existing_ids]


def load_chunks(collection, chunks, api_key):
    pending = find_pending_chunks(collection, chunks)
    print(f"{len(chunks)} chunks, {len(pending)} need embedding + loading")

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    def flush():
        if batch_ids:
            collection.add(ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents, metadatas=batch_metadatas)
            batch_ids.clear()
            batch_embeddings.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    for i, chunk in enumerate(pending):
        embedding = call_embedding_with_retry(chunk, api_key)
        if embedding:
            batch_ids.append(chunk_id(chunk))
            batch_embeddings.append(embedding)
            batch_documents.append(chunk["text"])
            batch_metadatas.append({"record_id": chunk["record_id"], "chunk_type": chunk["chunk_type"]})
            print(f"  [{i + 1}/{len(pending)}] {chunk['record_id']} ({chunk['chunk_type']})")

        if (i + 1) % SAVE_EVERY == 0:
            flush()

    flush()


def main():
    import pycolmap

    chunks = read_jsonl(chunks_jsonl_path(pycolmap.__version__))
    api_key = load_llm_api_key()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=chroma_collection_name(pycolmap.__version__))

    load_chunks(collection, chunks, api_key)
    print(f"Done. Collection '{collection.name}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
