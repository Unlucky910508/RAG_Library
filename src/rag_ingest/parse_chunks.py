"""Split each enriched API record into one or more embedding chunks.

Each record gets indexed under multiple text representations (its
explanation, its signature) so a conceptual query ("how do I set the
camera model") and a precise one ("does this take a min_num_trials
argument") can each match a chunk suited to that query style. Chunks only
carry record_id + chunk_type + text - the full record already lives in the
API jsonl and is looked up by record_id after a chunk matches, so nothing
is duplicated here.

Which fields make up which chunk_type is a data-only recipe
(CHUNK_FIELDS in config/config.py), not code: FIELD_EXTRACTORS below
is the one-function-per-JSON-field vocabulary the recipe draws from, and
build_chunk_text() just looks up and concatenates whichever ones a given
chunk_type lists. Adding or reshaping a chunk_type out of existing fields
is a config edit; only a genuinely new field needs a new extractor here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import CHUNK_FIELDS, api_jsonl_path, chunks_jsonl_path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_name(record):
    return record.get("name")


def extract_kind(record):
    return record.get("kind")


def extract_explanation(record):
    return record.get("explanation")


def extract_signatures(record):
    signatures = record.get("signatures")
    return "\n".join(signatures) if signatures else None


def extract_parameter_names(record):
    names = sorted({p["name"] for overload in record.get("parameters", []) for p in overload})
    return "parameters: " + ", ".join(names) if names else None


def extract_enum_members(record):
    members = record.get("members")
    return "members: " + ", ".join(m["name"] for m in members) if members else None


def extract_enum_of(record):
    enum_of = record.get("enum_of")
    return f"belongs to enum: {enum_of}" if enum_of else None


def extract_value(record):
    return f"value: {record['value']}" if "value" in record else None


FIELD_EXTRACTORS = {
    "name": extract_name,
    "kind": extract_kind,
    "explanation": extract_explanation,
    "signatures": extract_signatures,
    "parameter_names": extract_parameter_names,
    "enum_members": extract_enum_members,
    "enum_of": extract_enum_of,
    "value": extract_value,
}


def build_chunk_text(record, field_keys):
    parts = [FIELD_EXTRACTORS[key](record) for key in field_keys]
    parts = [p for p in parts if p]
    return "\n".join(parts) if parts else None


def build_chunks(record):
    chunks = []
    for chunk_type, spec in CHUNK_FIELDS.items():
        if any(not FIELD_EXTRACTORS[key](record) for key in spec["required"]):
            continue
        text = build_chunk_text(record, spec["fields"])
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
