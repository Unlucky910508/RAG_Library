"""Drop API records by name prefix, either turning some away or keeping
only some.

Introspection takes everything a library exposes, which for a large one
is mostly not what anyone would search for - private module trees, test
helpers, vendored dependencies. Narrowing that surface is a judgement
about the library rather than something the pipeline can work out, so it
lives in files under filter/, one directory per library and version:

    filter/pycolmap_4.1.0/exclude.py
    filter/torch_2.9.1/exclude.py

Every list of strings in such a file is read and the lists are merged, so
prefixes can be grouped by reason under whatever names read best - the
variable names are not looked at. The file is parsed, never executed, so
it cannot do anything but declare lists.

A prefix matches the record of that exact name and everything beneath it:
"torch.jit" drops torch.jit and torch.jit.trace but not torch.jitter.

    filter_api_records.py filter/pycolmap_4.1.0/exclude.py --exclude
    filter_api_records.py filter/pycolmap_4.1.0/keep.py --keep --dry-run

Meant to run straight after parse_api.py, before the enrichment steps: a
record dropped now costs nothing, while one dropped after
parse_explanations.py throws away an LLM call. It rewrites the records
file in place, which parse_api.py can rebuild at any time.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import api_jsonl_path, parsed_module_name


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prefixes(filter_path):
    """Every string in every list assigned at the top level of the file.

    Read with ast rather than by importing: a filter file has no reason to
    run code, and parsing means a stray import or typo there cannot do
    anything worse than be ignored."""
    tree = ast.parse(filter_path.read_text(encoding="utf-8"))
    prefixes, names = [], []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        values = [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        if not values:
            continue
        names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
        prefixes.extend(values)
    # Deduplicated but order kept, so what gets reported reads like the file.
    return list(dict.fromkeys(prefixes)), names


def matches(name, prefixes):
    """True when the record is the prefix itself or sits beneath it."""
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def partition(records, prefixes, mode):
    """(kept, dropped) for the chosen policy."""
    kept, dropped = [], []
    for record in records:
        hit = matches(record["name"], prefixes)
        keep = not hit if mode == "exclude" else hit
        (kept if keep else dropped).append(record)
    return kept, dropped


def summarise(dropped, prefixes, mode):
    """How many records each prefix accounted for, so a prefix that
    matched nothing is visible rather than silently doing nothing."""
    counts = {prefix: 0 for prefix in prefixes}
    for record in dropped if mode == "exclude" else []:
        for prefix in prefixes:
            if record["name"] == prefix or record["name"].startswith(prefix + "."):
                counts[prefix] += 1
                break
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("filter_file", type=Path, help="file of prefix lists under filter/")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--exclude", action="store_const", dest="mode", const="exclude",
                        help="drop records matching these prefixes")
    policy.add_argument("--keep", action="store_const", dest="mode", const="keep",
                        help="drop records NOT matching these prefixes")
    parser.add_argument("--path", type=Path, help="records file to filter (default: the API records)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.filter_file.exists():
        sys.exit(f"No such filter file: {args.filter_file}")

    version = __import__(parsed_module_name).__version__
    path = args.path or api_jsonl_path(version)
    if not path.exists():
        sys.exit(f"No records at {path} - run parse_api.py first")

    prefixes, list_names = load_prefixes(args.filter_file)
    if not prefixes:
        sys.exit(f"{args.filter_file} declares no lists of strings - nothing to filter on")

    records = read_jsonl(path)
    kept, dropped = partition(records, prefixes, args.mode)

    print(f"{args.filter_file}: {len(prefixes)} prefixes from {', '.join(list_names) or 'unnamed lists'}")
    if args.mode == "exclude":
        for prefix, count in summarise(dropped, prefixes, args.mode).items():
            print(f"  {count:5} {prefix}" + ("   (matched nothing)" if not count else ""))
    print(f"{len(records)} records -> {len(kept)} kept, {len(dropped)} dropped")

    if args.dry_run:
        print("Dry run - file untouched.")
        return
    if not dropped:
        print("Nothing to drop - file untouched.")
        return
    if not kept:
        sys.exit("Refusing to write: that would leave no records at all.")

    write_jsonl(kept, path)
    print(f"Rewrote {path}. parse_api.py rebuilds it if you want them back.")


if __name__ == "__main__":
    main()
