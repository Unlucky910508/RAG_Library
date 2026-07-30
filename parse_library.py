"""Walk an importable Python library and dump every public API name to a JSONL file.

Each output line is one JSON object: {"name": "<qualified.name>", "kind": "<module|class|function|method|property|attribute>"}
"""

import argparse
import inspect
import json


def is_public(name):
    if name in ("__init__", "__call__"):
        return True
    return not name.startswith("_")


def classify(owner, member):
    if inspect.ismodule(member):
        return "module"
    if inspect.isclass(member):
        return "class"
    if isinstance(member, property):
        return "property"
    if inspect.isroutine(member):
        return "method" if inspect.isclass(owner) else "function"
    return "attribute"


def walk(obj, prefix, visited, results, max_depth=6, depth=0):
    if depth > max_depth:
        return
    for name in sorted(dir(obj)):
        if not is_public(name):
            continue
        try:
            member = getattr(obj, name)
        except Exception:
            continue

        qualified_name = f"{prefix}.{name}"
        kind = classify(obj, member)
        results.append({"name": qualified_name, "kind": kind})

        if kind in ("module", "class"):
            member_id = id(member)
            if member_id in visited:
                continue
            visited.add(member_id)
            walk(member, qualified_name, visited, results, max_depth, depth + 1)


def collect_apis(root, root_name):
    results = [{"name": root_name, "kind": "module"}]
    walk(root, root_name, {id(root)}, results)
    return results


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract all API names from a Python library into a JSONL file.")
    parser.add_argument("module", nargs="?", default="pycolmap", help="Importable module name (default: pycolmap)")
    parser.add_argument("-o", "--output", default="pycolmap_apis.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    root = __import__(args.module)
    records = collect_apis(root, args.module)

    seen = set()
    unique_records = []
    for record in records:
        if record["name"] in seen:
            continue
        seen.add(record["name"])
        unique_records.append(record)
    unique_records.sort(key=lambda r: r["name"])

    write_jsonl(unique_records, args.output)
    print(f"Wrote {len(unique_records)} API entries to {args.output}")


if __name__ == "__main__":
    main()
