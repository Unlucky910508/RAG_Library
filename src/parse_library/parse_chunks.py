"""Split each enriched API record into one or more embedding chunks.

Each record gets indexed under multiple text representations (its
explanation, its signature) so a conceptual query ("how do I set the
camera model") and a precise one ("does this take a min_num_trials
argument") can each match a chunk suited to that query style. Chunks only
carry record_id + chunk_type + text - the full record already lives in the
API jsonl and is looked up by record_id after a chunk matches, so nothing
is duplicated here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from parse_config import api_jsonl_path, chunks_jsonl_path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_explanation_chunk_text(record):
    return record.get("explanation") or None


def build_signature_chunk_text(record):
    if not record.get("signatures"):
        return None
    lines = [record["name"], *record["signatures"]]
    param_names = sorted({p["name"] for overload in record.get("parameters", []) for p in overload})
    if param_names:
        lines.append("parameters: " + ", ".join(param_names))
    return "\n".join(lines)


CHUNK_BUILDERS = {
    "explanation": build_explanation_chunk_text,
    "signature": build_signature_chunk_text,
}


def build_chunks(record):
    chunks = []
    for chunk_type, build_text in CHUNK_BUILDERS.items():
        text = build_text(record)
        if text:
            chunks.append({"record_id": record["name"], "chunk_type": chunk_type, "text": text})
    return chunks


def main():
    import pycolmap

    records = read_jsonl(api_jsonl_path(pycolmap.__version__))

    chunks = []
    for record in records:
        chunks.extend(build_chunks(record))

    output_path = chunks_jsonl_path(pycolmap.__version__)
    write_jsonl(chunks, output_path)
    print(f"Wrote {len(chunks)} chunks from {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
