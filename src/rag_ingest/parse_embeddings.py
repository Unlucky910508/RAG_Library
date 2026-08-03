"""Add an embedding vector to each chunk in the chunks JSONL, via a local
OpenAI-compatible embeddings endpoint (not local to this machine - same
kind of remote LLM server parse_explanations.py calls).
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_VERIFY_SSL,
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


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def embed_chunks(chunks, api_key, path):
    pending = [c for c in chunks if not c.get("embedding")]
    print(f"{len(chunks)} chunks, {len(pending)} need an embedding")

    for i, chunk in enumerate(pending):
        embedding = call_embedding_with_retry(chunk, api_key)
        if embedding:
            chunk["embedding"] = embedding
            print(f"  [{i + 1}/{len(pending)}] {chunk['record_id']} ({chunk['chunk_type']})")

        if (i + 1) % SAVE_EVERY == 0:
            write_jsonl(chunks, path)

    write_jsonl(chunks, path)


def main():
    import pycolmap

    path = chunks_jsonl_path(pycolmap.__version__)
    api_key = load_llm_api_key()

    chunks = read_jsonl(path)
    embed_chunks(chunks, api_key, path)
    print(f"Done. Wrote embeddings to {path}")


if __name__ == "__main__":
    main()
