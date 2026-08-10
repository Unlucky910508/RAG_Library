"""Inventory community code that uses the target library, for review.

Writes a candidates file and nothing else - no source enters the dataset
on the strength of this script alone. Downloading whatever it approves is
a separate, deliberate step, because code from arbitrary repositories
carries licence obligations and version risk that a person should weigh
before an agent starts quoting it as an example.

The discovery chain needs no credentials:

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
    COMMUNITY_MIN_APIS_PER_FILE,
    COMMUNITY_MIN_STARS,
    EXAMPLES_GITHUB_REPO,
    EXAMPLES_PATH_IN_REPO,
    api_jsonl_path,
    community_candidates_path,
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


def fetch_repo_metadata(repo):
    response = requests.get(f"{GITHUB_REPO_API}/{repo}", timeout=30)
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
    any reference fails to resolve against the installed version."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    aliases = collect_module_aliases(tree, module_name)
    if not aliases:
        return None
    used, unknown = collect_api_refs(tree, aliases, known_names)
    return {
        "apis_used": used,
        "unknown_refs": unknown,
        "lines": len(source_text.splitlines()),
    }


def evaluate_repo(repo, entry, meta, module_name, known_names):
    candidates = []
    for path in sorted(entry["paths"]):
        source_text = fetch_file(repo, entry["branch"], path)
        time.sleep(POLITE_DELAY)
        if source_text is None:
            continue
        score = score_file(source_text, module_name, known_names)
        if score is None:
            continue
        duplicate = already_ingested(repo, path)
        passes = (
            not duplicate
            and len(score["apis_used"]) >= COMMUNITY_MIN_APIS_PER_FILE
            and len(score["unknown_refs"]) <= COMMUNITY_MAX_UNKNOWN_REFS
        )
        candidates.append({
            "repo": repo,
            "path": path,
            "branch": entry["branch"],
            "source": f"https://github.com/{repo}/blob/{entry['branch']}/{path}",
            "license": meta["license"],
            "license_verdict": licence_verdict(meta["license"]),
            "stars": meta["stars"],
            "pushed_at": meta["pushed_at"],
            "already_ingested": duplicate,
            "recommended": passes,
            **score,
        })
    return candidates


def main():
    version = __import__(parsed_module_name).__version__
    known_names = {r["name"] for r in read_jsonl(api_jsonl_path(version))}

    print(f"Searching grep.app for code importing {parsed_module_name}")
    repos = search_repos(parsed_module_name)
    print(f"  {len(repos)} repositories, {sum(len(e['paths']) for e in repos.values())} files\n")

    candidates = []
    for repo, entry in sorted(repos.items()):
        meta = fetch_repo_metadata(repo)
        time.sleep(POLITE_DELAY)
        if meta is None:
            print(f"  skip {repo}: metadata unavailable (rate limited?)")
            continue
        reason = repo_prefilter(meta)
        if reason:
            print(f"  skip {repo}: {reason}")
            continue
        found = evaluate_repo(repo, entry, meta, parsed_module_name, known_names)
        kept = sum(1 for c in found if c["recommended"])
        print(f"  {repo}: {kept}/{len(found)} files pass ({meta['license']}, {meta['stars']} stars)")
        candidates.extend(found)

    candidates.sort(key=lambda c: (not c["recommended"], -len(c["apis_used"])))
    output_path = community_candidates_path(version)
    write_jsonl(candidates, output_path)

    recommended = [c for c in candidates if c["recommended"]]
    review = [c for c in recommended if c["license_verdict"] == "needs-review"]
    print(f"\nWrote {len(candidates)} candidates to {output_path}")
    print(f"  {len(recommended)} pass the static checks")
    if review:
        print(f"  {len(review)} of those carry an unrecognised licence - read it before using them")
    print("Nothing has been added to the dataset; review the file and decide.")


if __name__ == "__main__":
    main()
