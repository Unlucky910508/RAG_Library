"""Core RAG search logic: embed a query, look it up in the Chroma
collection, and resolve each hit back to its record.

What comes back is assembled here from the matched chunk_type's
return_fields (config.CHUNK_FIELDS), not read out of the vector DB - so a
chunk can be matched on one kind of text and answered with another, and
editing a recipe takes effect on the next query without re-embedding.

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
    CHUNK_FIELDS,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_VERIFY_SSL,
    MAX_TOP_K,
    chroma_collection_name,
    load_llm_api_key,
    record_jsonl_paths,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from record_fields import build_text

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
    records_by_id = {}
    for path in record_jsonl_paths(version):
        if path.exists():
            for record in read_jsonl(path):
                records_by_id[record["name"]] = record
    return records_by_id


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


def build_return_text(record, chunk_type):
    """Render the record through the matched chunk_type's return_fields.
    Falls back to the embedding_fields if a recipe defines no return of
    its own, so a half-configured chunk_type still answers with something."""
    if record is None:
        return None
    spec = CHUNK_FIELDS.get(chunk_type, {})
    field_keys = spec.get("return_fields") or spec.get("embedding_fields", [])
    return build_text(record, field_keys)


def search(query, collection, records_by_id, api_key, top_k=MAX_TOP_K):
    # Clamped rather than rejected: callers are usually models, and an
    # over-eager top_k should still get an answer. Enforced here rather
    # than in the MCP layer so every consumer of this module is covered.
    top_k = max(1, min(top_k, MAX_TOP_K))
    query_embedding = embed_query(query, api_key)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k * OVERFETCH_MULTIPLIER)

    hits = []
    seen_record_ids = set()
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        record_id = metadata["record_id"]
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        chunk_type = metadata["chunk_type"]
        hits.append({
            "record_id": record_id,
            "matched_chunk_type": chunk_type,
            "distance": distance,
            "text": build_return_text(records_by_id.get(record_id), chunk_type),
        })
        if len(hits) == top_k:
            break
    return hits
