"""Fetch the official pycolmap example scripts and turn them into example
records for the RAG dataset.

Source is the colmap GitHub repo at the tag matching the installed
pycolmap version (never master), so the example code can't drift ahead of
the API records. Each top-level function/class becomes one record - the
retrieval-sized unit - plus one module-context record per file holding the
imports, module-level constants, and __main__ glue.

Every record carries the pycolmap APIs it references, extracted statically
via ast and validated against the API records parse_api.py produced:
references that don't resolve against the installed version land in
unknown_refs instead of being silently trusted. Test scaffolding
(conftest.py, *_test.py) is skipped.
"""

import ast
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    EXAMPLES_GITHUB_REPO,
    EXAMPLES_LICENSE,
    EXAMPLES_PATH_IN_REPO,
    api_jsonl_path,
    examples_jsonl_path,
)

SKIP_FILES = {"conftest.py"}
SKIP_SUFFIXES = ("_test.py",)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_example_files(tag):
    url = f"https://api.github.com/repos/{EXAMPLES_GITHUB_REPO}/contents/{EXAMPLES_PATH_IN_REPO}"
    response = requests.get(url, params={"ref": tag}, timeout=30)
    response.raise_for_status()
    files = []
    for entry in response.json():
        name = entry["name"]
        if not name.endswith(".py") or name in SKIP_FILES or name.endswith(SKIP_SUFFIXES):
            continue
        files.append((name, entry["download_url"]))
    return files


def fetch_text(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def collect_pycolmap_aliases(tree):
    """Map local names to the pycolmap path they refer to, e.g.
    `import pycolmap` -> {"pycolmap": "pycolmap"},
    `from pycolmap import logging` -> {"logging": "pycolmap.logging"}."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "pycolmap":
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        aliases["pycolmap"] = "pycolmap"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "pycolmap":
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def collect_api_refs(node, aliases, known_names):
    """Resolve every pycolmap.* reference in this subtree to the longest
    name that actually exists in the API records."""
    used, unknown = set(), set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Attribute):
            continue
        chain = attribute_chain(sub)
        if not chain or chain[0] not in aliases:
            continue
        qualified = aliases[chain[0]].split(".") + chain[1:]
        for end in range(len(qualified), 1, -1):
            candidate = ".".join(qualified[:end])
            if candidate in known_names:
                used.add(candidate)
                break
        else:
            unknown.add(".".join(qualified))

    # Nested prefixes of a longer match are noise (pycolmap.Camera when
    # pycolmap.Camera.create was matched), keep only the most specific.
    used = {u for u in used if not any(other != u and other.startswith(u + ".") for other in used)}
    return sorted(used), sorted(unknown)


def segment_start_line(node):
    if getattr(node, "decorator_list", None):
        return node.decorator_list[0].lineno
    return node.lineno


def build_file_records(filename, source_text, tag, known_names):
    tree = ast.parse(source_text)
    aliases = collect_pycolmap_aliases(tree)
    lines = source_text.splitlines()
    base_url = f"https://github.com/{EXAMPLES_GITHUB_REPO}/blob/{tag}/{EXAMPLES_PATH_IN_REPO}/{filename}"

    def make_record(name, code, refs_node, start, end):
        used, unknown = collect_api_refs(refs_node, aliases, known_names)
        record = {
            "name": name,
            "kind": "example",
            "source": f"{base_url}#L{start}-L{end}",
            "license": EXAMPLES_LICENSE,
            "apis_used": used,
            "code": code,
        }
        if unknown:
            record["unknown_refs"] = unknown
        return record

    records = []
    segments = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    covered = set()
    for node in segments:
        start, end = segment_start_line(node), node.end_lineno
        covered.update(range(start, end + 1))
        code = "\n".join(lines[start - 1:end])
        records.append(make_record(f"examples/{filename}::{node.name}", code, node, start, end))

    context_code = "\n".join(
        line for i, line in enumerate(lines, 1) if i not in covered
    ).strip()
    if context_code:
        context_nodes = ast.Module(
            body=[n for n in tree.body if segment_start_line(n) not in covered], type_ignores=[]
        )
        records.append(make_record(f"examples/{filename}", context_code, context_nodes, 1, len(lines)))
    return records


def main():
    import pycolmap

    version = pycolmap.__version__
    known_names = {r["name"] for r in read_jsonl(api_jsonl_path(version))}

    records = []
    for filename, download_url in list_example_files(version):
        source_text = fetch_text(download_url)
        file_records = build_file_records(filename, source_text, version, known_names)
        records.extend(file_records)
        print(f"  {filename}: {len(file_records)} records")

    write_jsonl(records, examples_jsonl_path(version))
    unknown_count = sum(1 for r in records if r.get("unknown_refs"))
    print(f"Wrote {len(records)} example records to {examples_jsonl_path(version)}")
    if unknown_count:
        print(f"WARNING: {unknown_count} records contain pycolmap references that don't resolve against {version}")


if __name__ == "__main__":
    main()
