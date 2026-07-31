"""Walk an importable Python library and dump every public API name to a JSONL file.

Each output line is one JSON object: {"name": "<qualified.name>", "kind": "<module|class|function|method|property|attribute>"}
"""

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


def belongs_to_package(member, kind, root_name):
    if kind == "module":
        owner_name = getattr(member, "__name__", "")
    else:
        owner_name = getattr(member, "__module__", "") or ""
    return owner_name == root_name or owner_name.startswith(root_name + ".")


def walk(obj, prefix, root_name, visited, results, depth=0):
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

        if kind in ("module", "class") and belongs_to_package(member, kind, root_name):
            member_id = id(member)
            if member_id in visited:
                continue
            visited.add(member_id)
            walk(member, qualified_name, root_name, visited, results, depth + 1)


def collect_apis(root, root_name):
    results = [{"name": root_name, "kind": "module"}]
    walk(root, root_name, root_name, {id(root)}, results)
    return results


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    # setting parsed module & import 
    parsed_module_name = "pycolmap"
    root = __import__(parsed_module_name)

    # setting output file name
    version = root.__version__
    output_file = f"pycolmap_{version}_api.jsonl"

    records = collect_apis(root, parsed_module_name)

    seen = set()
    unique_records = []
    for record in records:
        if record["name"] in seen:
            continue
        seen.add(record["name"])
        unique_records.append(record)
    unique_records.sort(key=lambda r: r["name"])

    write_jsonl(unique_records, output_file)
    print(f"Wrote {len(unique_records)} API entries to {output_file}")


if __name__ == "__main__":
    main()
