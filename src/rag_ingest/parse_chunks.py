"""Split each enriched API record into one or more embedding chunks.

Each record gets indexed under multiple text representations (its
explanation, its signature) so a conceptual query ("how do I set the
camera model") and a precise one ("does this take a min_num_trials
argument") can each match a chunk suited to that query style. Chunks only
carry record_id + chunk_type + text - the full record already lives in the
API jsonl and is looked up by record_id after a chunk matches, so nothing
is duplicated here.

Which fields make up which chunk_type is a data-only recipe
(CHUNK_FIELDS in config/config.py), not code - this step reads each
chunk_type's embedding_fields and renders them through the shared
vocabulary in src/common/record_fields.py. The same recipe's
return_fields are used at the other end of the pipeline by search.py,
which is why that vocabulary is shared rather than local.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import CHUNK_FIELDS, chunks_jsonl_path_for, record_jsonl_paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from record_fields import build_text, has_required_fields


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_chunks(record):
    chunks = []
    for chunk_type, spec in CHUNK_FIELDS.items():
        if not has_required_fields(record, spec["required"]):
            continue
        text = build_text(record, spec["embedding_fields"])
        if text:
            chunks.append({"record_id": record["name"], "chunk_type": chunk_type, "text": text})
    return chunks


def main():
    import pycolmap

    for record_path in record_jsonl_paths(pycolmap.__version__):
        if not record_path.exists():
            print(f"Skipping {record_path} (not generated yet)")
            continue
        records = read_jsonl(record_path)
        chunks = []
        for record in records:
            chunks.extend(build_chunks(record))
        output_path = chunks_jsonl_path_for(record_path)
        write_jsonl(chunks, output_path)
        print(f"Wrote {len(chunks)} chunks from {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
