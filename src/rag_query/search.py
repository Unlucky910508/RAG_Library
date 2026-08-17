"""Core RAG search logic: embed a query, look it up in the Chroma
collection, and resolve each hit back to its record.

Reads a dataset a pipeline run already produced and never introspects the
library it describes, so nothing here needs that library installed.
Everything it needs is in mcp_server_config.py beside it.

What comes back is the document stored beside each vector, which
load_vectordb.py wrote from the chunk's return_text. A chunk is still
found by one text and answered with another; the difference is that the
answer was settled when the store was built, so the store is all serving
needs. Changing a return_fields recipe means rebuilding rather than
taking effect on the next question.

Kept free of any MCP dependency so it's importable by anything - an MCP
server, a CLI, a test script, a future different kind of RAG consumer.
"""

import json
import sys
from pathlib import Path

import chromadb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_server_config import (
    API_KEY,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MAX_TOP_K,
    VERIFY_SSL,
)

HERE = Path(__file__).resolve().parent


def _resolve(path):
    """Settings may be relative, and are then read relative to this
    directory rather than to wherever the server happened to be started
    from - so a copied folder works the same however it is launched."""
    path = Path(path)
    return path if path.is_absolute() else (HERE / path).resolve()

if VERIFY_SSL is False:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Chroma dedupes multiple matching chunk_types (e.g. both "explanation"
# and "signature") down to one hit per record_id, so more raw chunk
# matches than top_k are pulled to still surface top_k distinct records.
OVERFETCH_MULTIPLIER = 3


def get_collection():
    """Open the collection named in the config, and fail if it is not
    there.

    get_or_create would answer a mistyped name by making an empty
    collection and returning it, so every query would come back with
    nothing and no error, having also left a stray collection in someone
    else's store. Serving only ever reads, so it opens what exists and
    says what does if the name is wrong."""
    store = _resolve(CHROMA_PATH)
    if not store.is_dir():
        raise FileNotFoundError(f"No Chroma store at {store} - check CHROMA_PATH in mcp_server_config.py")

    client = chromadb.PersistentClient(path=str(store))
    available = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in available:
        raise LookupError(
            f"No collection {COLLECTION_NAME!r} in {store}. "
            f"It holds: {', '.join(available) or 'nothing'}. "
            "Check COLLECTION_NAME in mcp_server_config.py."
        )
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(text, api_key):
    response = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
        verify=VERIFY_SSL,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url}: {response.text[:500]}",
            response=response,
        )
    return response.json()["data"][0]["embedding"]


def search(query, collection, api_key, top_k=MAX_TOP_K):
    # Clamped rather than rejected: callers are usually models, and an
    # over-eager top_k should still get an answer. Enforced here rather
    # than in the MCP layer so every consumer of this module is covered.
    top_k = max(1, min(top_k, MAX_TOP_K))
    query_embedding = embed_query(query, api_key)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * OVERFETCH_MULTIPLIER,
        include=["metadatas", "distances", "documents"],
    )

    hits = []
    seen_record_ids = set()
    for metadata, distance, document in zip(
        results["metadatas"][0], results["distances"][0], results["documents"][0]
    ):
        record_id = metadata["record_id"]
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        hits.append({
            "record_id": record_id,
            "matched_chunk_type": metadata["chunk_type"],
            "distance": distance,
            # Written when the store was built, so answering needs nothing
            # beyond the store itself.
            "text": document,
        })
        if len(hits) == top_k:
            break
    return hits
