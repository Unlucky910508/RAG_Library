"""Find community code that uses the target library and keep what passes.

Files have to be downloaded to be judged - the deciding test reads their
source - so the ones that survive are simply written out rather than
discarded and fetched again later. They land in the same directory layout
fetch_official_example_code.py produces, so parse_python_code.py reads
them the same way.

Downloading is not ingesting. Nothing here reaches the dataset until
parse_python_code.py is run over the directory, which leaves room to read
the candidates file and delete anything unwanted first - worth doing,
since third-party code carries licence obligations and version risk a
person should weigh before an agent starts quoting it.

The chain needs no credentials:

    grep.app                  which repositories import this library
    api.github.com            licence, stars, last push (60 req/hour)
    raw.githubusercontent.com the files themselves

Note that grep.app is a third-party service with no published API, so the
response shape here was derived by observation and is parsed defensively.

Ranking deliberately does not rest on stars or reputation. Those only
narrow the field cheaply; what decides a file is the same static check
the pipeline already performs - resolving every reference against the API
records of the *installed* version. A file that reaches for names this
version does not have was written for a different one, and no amount of
popularity makes it safe to quote.

Density is measured per function rather than per file, since a function
is what parse_python_code.py turns into a record. The question a file has
to answer is whether it contains at least one function dense enough to be
worth quoting - not whether its imports touch enough of the library
somewhere across two thousand lines.
"""

import ast
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    COMMUNITY_LICENSE_ALLOWLIST,
    COMMUNITY_LICENSE_REVIEW,
    COMMUNITY_MAX_AGE_DAYS,
    COMMUNITY_MAX_UNKNOWN_REFS,
    COMMUNITY_MIN_APIS_PER_FUNCTION,
    COMMUNITY_MIN_STARS,
    EXAMPLES_GITHUB_REPO,
    EXAMPLES_PATH_IN_REPO,
    EXAMPLES_MANIFEST_NAME,
    api_jsonl_path,
    community_candidates_path,
    community_src_dir,
    load_github_token,
    parsed_module_name,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_python_code import collect_api_refs, collect_module_aliases

GREP_APP_SEARCH = "https://grep.app/api/search"
GITHUB_REPO_API = "https://api.github.com/repos"
RAW_CONTENT = "https://raw.githubusercontent.com"
USER_AGENT = "Mozilla/5.0"
SEARCH_PAGES = 10
POLITE_DELAY = 0.5


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def search_repos(module_name, pages=SEARCH_PAGES):
    """Map repo -> {branch, paths} for files importing the library."""
    found = {}
    for page in range(1, pages + 1):
        response = requests.get(
            GREP_APP_SEARCH,
            params={"q": f"import {module_name}", "page": page},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if not response.ok:
            print(f"  grep.app page {page}: HTTP {response.status_code}, stopping")
            break
        hits = response.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            repo, path = hit.get("repo"), hit.get("path")
            if not repo or not path or not path.endswith(".py"):
                continue
            entry = found.setdefault(repo, {"branch": hit.get("branch") or "HEAD", "paths": set()})
            entry["paths"].add(path)
        time.sleep(POLITE_DELAY)
    return found


def github_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_repo_metadata(repo, token=None):
    response = requests.get(f"{GITHUB_REPO_API}/{repo}", headers=github_headers(token), timeout=30)
    if not response.ok:
        return None
    data = response.json()
    return {
        "stars": data.get("stargazers_count", 0),
        "license": (data.get("license") or {}).get("spdx_id") or "NO-LICENSE",
        "pushed_at": (data.get("pushed_at") or "")[:10],
        "is_fork": bool(data.get("fork")),
        "archived": bool(data.get("archived")),
    }


def already_ingested(repo, path):
    """The official fetcher already takes this exact file, so recommending
    it again would just duplicate what the dataset holds. Scoped to that
    one directory: the rest of the upstream repo is fair game."""
    return repo == EXAMPLES_GITHUB_REPO and path.startswith(EXAMPLES_PATH_IN_REPO + "/")


def licence_verdict(license_id):
    if license_id in COMMUNITY_LICENSE_ALLOWLIST:
        return "allowed"
    if license_id in COMMUNITY_LICENSE_REVIEW:
        return "needs-review"
    return "rejected"


def repo_prefilter(meta):
    """Cheap reasons to not spend requests on a repo's files."""
    if meta["is_fork"]:
        return "fork"
    if licence_verdict(meta["license"]) == "rejected":
        return f"licence {meta['license']}"
    if meta["stars"] < COMMUNITY_MIN_STARS:
        return f"{meta['stars']} stars"
    if meta["pushed_at"]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=COMMUNITY_MAX_AGE_DAYS)
        if datetime.fromisoformat(meta["pushed_at"]).replace(tzinfo=timezone.utc) < cutoff:
            return f"stale since {meta['pushed_at']}"
    return None


def fetch_file(repo, branch, path):
    response = requests.get(f"{RAW_CONTENT}/{repo}/{branch}/{path}", timeout=30)
    return response.text if response.ok else None


def score_file(source_text, module_name, known_names):
    """Static verdict on one file: which APIs it really uses, and whether
    any reference fails to resolve against the installed version.

    None means there is nothing to review - the file does not parse, never
    imports the library, or imports it without calling into it. Files that
    merely import are common (a type annotation, a re-export, a guarded
    optional dependency) and carry no usage to learn from, so they are
    dropped here rather than written out as candidates with an empty
    apis_used for a human to wade through."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    aliases = collect_module_aliases(tree, module_name)
    if not aliases:
        return None
    used, unknown = collect_api_refs(tree, aliases, known_names)
    if not used:
        return None

    # Scored per function, matching the unit parse_python_code.py records,
    # so the file-level count stays as context while the decision rests on
    # whether any single function is dense enough to be worth quoting.
    per_function = [
        len(collect_api_refs(node, aliases, known_names)[0])
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        "apis_used": used,
        "unknown_refs": unknown,
        "lines": len(source_text.splitlines()),
        "max_apis_in_function": max(per_function, default=0),
        "qualifying_functions": sum(1 for n in per_function if n >= COMMUNITY_MIN_APIS_PER_FUNCTION),
    }


def local_path_for(repo, path):
    """Namespaced by repo, since basenames collide across projects - three
    of them here are called colmap.py."""
    return Path(repo.replace("/", "__")) / path


def evaluate_repo(repo, entry, meta, module_name, known_names, dest_dir):
    """Score each file, writing out the ones that pass. Returns
    (candidates, n_skipped); skipped files have no library usage at all."""
    candidates, skipped = [], 0
    for path in sorted(entry["paths"]):
        source_text = fetch_file(repo, entry["branch"], path)
        time.sleep(POLITE_DELAY)
        if source_text is None:
            continue
        score = score_file(source_text, module_name, known_names)
        if score is None:
            skipped += 1
            continue
        duplicate = already_ingested(repo, path)
        passes = (
            not duplicate
            and score["qualifying_functions"] >= 1
            and len(score["unknown_refs"]) <= COMMUNITY_MAX_UNKNOWN_REFS
        )
        relative = local_path_for(repo, path)
        if passes:
            out = dest_dir / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(source_text, encoding="utf-8")
        candidates.append({
            "repo": repo,
            "path": path,
            "branch": entry["branch"],
            "local_path": str(relative),
            "source": f"https://github.com/{repo}/blob/{entry['branch']}/{path}",
            "license": meta["license"],
            "license_verdict": licence_verdict(meta["license"]),
            "stars": meta["stars"],
            "pushed_at": meta["pushed_at"],
            "already_ingested": duplicate,
            "recommended": passes,
            **score,
        })
    return candidates, skipped


def main():
    version = __import__(parsed_module_name).__version__
    known_names = {r["name"] for r in read_jsonl(api_jsonl_path(version))}
    token = load_github_token()
    print(f"GitHub API: {'authenticated' if token else 'unauthenticated (60 requests/hour)'}")

    dest_dir = community_src_dir(version)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching grep.app for code importing {parsed_module_name}")
    repos = search_repos(parsed_module_name)
    print(f"  {len(repos)} repositories, {sum(len(e['paths']) for e in repos.values())} files\n")

    candidates, import_only = [], 0
    for repo, entry in sorted(repos.items()):
        meta = fetch_repo_metadata(repo, token)
        time.sleep(POLITE_DELAY)
        if meta is None:
            print(f"  skip {repo}: metadata unavailable (rate limited?)")
            continue
        reason = repo_prefilter(meta)
        if reason:
            print(f"  skip {repo}: {reason}")
            continue
        found, skipped = evaluate_repo(repo, entry, meta, parsed_module_name, known_names, dest_dir)
        import_only += skipped
        kept = sum(1 for c in found if c["recommended"])
        print(f"  {repo}: {kept}/{len(found)} files kept ({meta['license']}, {meta['stars']} stars)")
        candidates.extend(found)

    candidates.sort(key=lambda c: (not c["recommended"], -c["qualifying_functions"], -c["max_apis_in_function"]))
    write_jsonl(candidates, community_candidates_path(version))

    kept = [c for c in candidates if c["recommended"]]
    # Same manifest contract as the official fetcher, so parse_python_code
    # can attribute a record without knowing where the code came from.
    manifest = {
        "source": "community",
        "license": None,
        "files": {c["local_path"]: c["source"] for c in kept},
        "licenses": {c["local_path"]: c["license"] for c in kept},
    }
    (dest_dir / EXAMPLES_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    review = [c for c in kept if c["license_verdict"] == "needs-review"]
    print(f"\nKept {len(kept)} files in {dest_dir}, "
          f"holding {sum(c['qualifying_functions'] for c in kept)} functions worth reviewing")
    if import_only:
        print(f"  ({import_only} files never used the library and were not downloaded)")
    print(f"  decisions recorded in {community_candidates_path(version)}")
    if review:
        print(f"  {len(review)} carry an unrecognised licence - read it before keeping them")
    print("Review the directory and delete anything unwanted, then run parse_python_code.py.")


if __name__ == "__main__":
    main()
