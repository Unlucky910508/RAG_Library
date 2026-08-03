"""Enrich the API-name JSONL (produced by parse_api.py) with signatures.

For each record, resolves the object behind its qualified name and fills in
a "signatures" list. Pybind11 builtins don't support inspect.signature(), so
their docstrings are parsed instead; overloaded functions surface as
multiple numbered signatures in the docstring and become multiple list
entries.
"""

import inspect
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OVERLOAD_LINE_RE = re.compile(r"^\d+\.\s+(.+)$")
IMPLICIT_PARAM_NAMES = {"self", "cls"}


def resolve_object(qualified_name):
    parts = qualified_name.split(".")
    obj = __import__(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def parse_overload_signatures(doc):
    signatures = []
    for line in doc.splitlines():
        match = OVERLOAD_LINE_RE.match(line.strip())
        if match:
            signatures.append(match.group(1).strip())
    return signatures


def extract_signatures(kind, obj):
    if kind == "property":
        doc = inspect.getdoc(obj)
        return [doc] if doc else []

    if kind not in ("function", "method"):
        return []

    try:
        return [str(inspect.signature(obj))]
    except (TypeError, ValueError):
        pass

    doc = inspect.getdoc(obj)
    if not doc:
        return []

    overloads = parse_overload_signatures(doc)
    if overloads:
        return overloads

    first_line = doc.splitlines()[0].strip()
    return [first_line] if first_line else []


def split_top_level(text, separator):
    parts = []
    depth = 0
    current = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def extract_param_block(signature):
    start = signature.find("(")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(signature)):
        if signature[i] == "(":
            depth += 1
        elif signature[i] == ")":
            depth -= 1
            if depth == 0:
                return signature[start + 1:i]
    return None


def parse_parameter(param_text):
    param_text = param_text.strip()
    if param_text in ("*", "/"):
        return None

    star = ""
    if param_text.startswith("**"):
        star, param_text = "**", param_text[2:]
    elif param_text.startswith("*"):
        star, param_text = "*", param_text[1:]

    eq_parts = split_top_level(param_text, "=")
    name_and_type = eq_parts[0].strip()
    default = "=".join(eq_parts[1:]).strip() if len(eq_parts) > 1 else None

    colon_parts = split_top_level(name_and_type, ":")
    name = colon_parts[0].strip()
    type_ = ":".join(colon_parts[1:]).strip() if len(colon_parts) > 1 else None

    return {
        "name": star + name,
        "type": type_,
        "required": default is None and not star,
        "default": default,
    }


def parse_signature_parameters(signature):
    block = extract_param_block(signature)
    if not block or not block.strip():
        return []
    params = []
    for part in split_top_level(block, ","):
        part = part.strip()
        if not part:
            continue
        parsed = parse_parameter(part)
        if parsed and parsed["name"] not in IMPLICIT_PARAM_NAMES:
            params.append(parsed)
    return params


def build_parameters(kind, signatures):
    if kind not in ("function", "method"):
        return []
    return [parse_signature_parameters(sig) for sig in signatures]


def enrich_records(records):
    for record in records:
        try:
            obj = resolve_object(record["name"])
            record["signatures"] = extract_signatures(record["kind"], obj)
        except Exception:
            record["signatures"] = []
        record["parameters"] = build_parameters(record["kind"], record["signatures"])
    return records


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    import pycolmap

    path = DATA_DIR / f"pycolmap_{pycolmap.__version__}_api.jsonl"

    records = read_jsonl(path)
    enrich_records(records)
    write_jsonl(records, path)
    print(f"Updated {len(records)} records with signatures in {path}")


if __name__ == "__main__":
    main()
