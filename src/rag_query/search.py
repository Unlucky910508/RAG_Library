"""Core RAG search logic: embed a query, look it up in the Chroma
collection, and resolve each hit back to its record.

Reads a dataset a pipeline run already produced and never introspects the
library it describes, so nothing here needs that library installed.
Everything it needs is in mcp_server_config.py beside it.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_fields import build_text
from mcp_server_config import (
    API_KEY,
    CHROMA_PATH,
    CHUNK_FIELDS,
    COLLECTION_NAME,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MAX_TOP_K,
    RECORDS_DIR,
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


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records_by_id():
    """Every record, keyed by name: a hit names one and is answered from
    it. Reads whatever jsonl is in RECORDS_DIR, so a dataset built with
    extra sources needs nothing declared here."""
    root = _resolve(RECORDS_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"No records directory at {root} - check RECORDS_DIR in mcp_server_config.py")
    records_by_id = {}
    for path in sorted(root.glob("*.jsonl")):
        for record in read_jsonl(path):
            records_by_id[record["name"]] = record
    if not records_by_id:
        raise FileNotFoundError(f"No .jsonl records under {root}")
    return records_by_id


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
