"""Core RAG search logic: embed a query, look it up in the Chroma
collection, and resolve each hit back to its full API record.

Kept free of any MCP dependency so it's importable by anything - an MCP
server, a CLI, a test script, a future different kind of RAG consumer.
"""

import json
import sys
from pathlib import Path

import chromadb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    CHROMA_DIR,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_VERIFY_SSL,
    api_jsonl_path,
    chroma_collection_name,
    load_llm_api_key,
)

if not LLM_VERIFY_SSL:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Chroma dedupes multiple matching chunk_types (e.g. both "explanation"
# and "signature") down to one hit per record_id, so more raw chunk
# matches than top_k are pulled to still surface top_k distinct records.
OVERFETCH_MULTIPLIER = 3


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records_by_id(version):
    return {r["name"]: r for r in read_jsonl(api_jsonl_path(version))}


def get_collection(version):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=chroma_collection_name(version),
        metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )


def embed_query(text, api_key):
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


def search(query, collection, records_by_id, api_key, top_k=5):
    query_embedding = embed_query(query, api_key)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k * OVERFETCH_MULTIPLIER)

    hits = []
    seen_record_ids = set()
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        record_id = metadata["record_id"]
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        hits.append({
            "record_id": record_id,
            "matched_chunk_type": metadata["chunk_type"],
            "distance": distance,
            "record": records_by_id.get(record_id),
        })
        if len(hits) == top_k:
            break
    return hits
