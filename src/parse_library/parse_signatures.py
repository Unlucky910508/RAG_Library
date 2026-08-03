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
PROPERTY_DEFAULT_RE = re.compile(r"\(([^,()]+),\s*default:\s*(.+)\)\s*$", re.DOTALL)


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


def parse_property_default(doc):
    match = PROPERTY_DEFAULT_RE.search(doc)
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def build_class_kwargs_index(records):
    """Map a class's qualified name to its properties/attributes, since
    pybind11's `__init__(self, **kwargs)` overload sets each kwarg as one
    of these - the docstring itself never lists them individually."""
    index = {}
    for record in records:
        if record["kind"] not in ("property", "attribute"):
            continue
        owner, _, name = record["name"].rpartition(".")
        doc = record["signatures"][0] if record["signatures"] else ""
        type_, default = parse_property_default(doc)
        index.setdefault(owner, []).append({
            "name": name,
            "type": type_,
            "required": False,
            "default": default,
        })
    return index


def expand_kwargs_param(param, record_name, kwargs_index):
    if param["name"] != "**kwargs" or not record_name.endswith(".__init__"):
        return [param]
    class_name = record_name.rpartition(".")[0]
    return kwargs_index.get(class_name) or [param]


def build_parameters(record, kwargs_index):
    if record["kind"] not in ("function", "method"):
        return []
    overloads = []
    for signature in record["signatures"]:
        params = []
        for param in parse_signature_parameters(signature):
            params.extend(expand_kwargs_param(param, record["name"], kwargs_index))
        overloads.append(params)
    return overloads


def enrich_records(records):
    for record in records:
        try:
            obj = resolve_object(record["name"])
            record["signatures"] = extract_signatures(record["kind"], obj)
        except Exception:
            record["signatures"] = []

    kwargs_index = build_class_kwargs_index(records)
    for record in records:
        record["parameters"] = build_parameters(record, kwargs_index)
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
