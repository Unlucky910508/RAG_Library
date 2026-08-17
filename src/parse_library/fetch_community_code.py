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
    COMMUNITY_MAX_AGE_DAYS,
    COMMUNITY_MAX_UNKNOWN_REFS,
    COMMUNITY_MIN_APIS_PER_FUNCTION,
    COMMUNITY_MIN_STARS,
    COMMUNITY_SEARCH_PAGES,
    VERIFY_SSL,
    EXAMPLES_MANIFEST_NAME,
    api_jsonl_path,
    community_candidates_path,
    community_src_dir,
    official_src_dir,
    load_github_token,
    parsed_module_name,
    parsed_module_version,
)

if VERIFY_SSL is False:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_python_code import collect_api_refs, collect_module_aliases

GREP_APP_SEARCH = "https://grep.app/api/search"
GITHUB_REPO_API = "https://api.github.com/repos"
RAW_CONTENT = "https://raw.githubusercontent.com"
USER_AGENT = "Mozilla/5.0"
POLITE_DELAY = 0.5


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def search_repos(module_name, pages=COMMUNITY_SEARCH_PAGES):
    """Map repo -> {branch, paths} for files importing the library."""
    found = {}
    for page in range(1, pages + 1):
        response = requests.get(
            GREP_APP_SEARCH,
            params={"q": f"import {module_name}", "page": page},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            verify=VERIFY_SSL,
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
    response = requests.get(f"{GITHUB_REPO_API}/{repo}", headers=github_headers(token), timeout=30, verify=VERIFY_SSL)
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


def official_source(version):
    """The repository and directory the official fetch actually used, read
    from the manifest it left behind.

    Taken from there rather than from config, because the config settings
    naming them are optional - resolved at fetch time when left as None,
    which made comparing against them silently match nothing and let the
    official examples through as community finds. The manifest records
    what was really downloaded, so it cannot drift from it."""
    manifest_path = official_src_dir() / EXAMPLES_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo, path = manifest.get("repo"), manifest.get("path_in_repo")
    return (repo, path) if repo and path else None


def already_ingested(repo, path, official):
    """The official fetch already takes this exact file, so taking it again
    would duplicate what the dataset holds. Scoped to that one directory:
    the rest of the upstream repository is fair game."""
    if official is None:
        return False
    official_repo, official_path = official
    return repo == official_repo and path.startswith(official_path + "/")


def licence_allowed(license_id):
    """Only licences whose terms are known. GitHub's NOASSERTION means a
    licence exists but could not be identified, which is not the same as
    permissive - Tencent's HY-World agreement reads that way and carries
    territorial limits and distribution duties."""
    return license_id in COMMUNITY_LICENSE_ALLOWLIST


def repo_prefilter(meta):
    """Cheap reasons to not spend requests on a repo's files."""
    if meta["is_fork"]:
        return "fork"
    if not licence_allowed(meta["license"]):
        return f"licence {meta['license']}"
    if meta["stars"] < COMMUNITY_MIN_STARS:
        return f"{meta['stars']} stars"
    if meta["pushed_at"]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=COMMUNITY_MAX_AGE_DAYS)
        if datetime.fromisoformat(meta["pushed_at"]).replace(tzinfo=timezone.utc) < cutoff:
            return f"stale since {meta['pushed_at']}"
    return None


def fetch_file(repo, branch, path):
    response = requests.get(f"{RAW_CONTENT}/{repo}/{branch}/{path}", timeout=30, verify=VERIFY_SSL)
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


def evaluate_repo(repo, entry, meta, module_name, known_names, dest_dir, official):
    """Score each file, writing out the ones that pass. Returns
    (candidates, n_unused, n_duplicate); unused files never call the
    library, duplicates are the official fetch's own."""
    candidates, unused, duplicate_count = [], 0, 0
    for path in sorted(entry["paths"]):
        # Checked before fetching: a file the official fetch already has is
        # one we will not keep either way, so downloading it to find that
        # out is a wasted request.
        if already_ingested(repo, path, official):
            duplicate_count += 1
            continue
        source_text = fetch_file(repo, entry["branch"], path)
        time.sleep(POLITE_DELAY)
        if source_text is None:
            continue
        score = score_file(source_text, module_name, known_names)
        if score is None:
            unused += 1
            continue
        passes = (
            score["qualifying_functions"] >= 1
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
            "stars": meta["stars"],
            "pushed_at": meta["pushed_at"],
            "recommended": passes,
            **score,
        })
    return candidates, unused, duplicate_count


def main():
    version = parsed_module_version
    known_names = {r["name"] for r in read_jsonl(api_jsonl_path())}
    token = load_github_token()
    print(f"GitHub API: {'authenticated' if token else 'unauthenticated (60 requests/hour)'}")

    dest_dir = community_src_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    official = official_source(version)
    if official:
        print(f"Official fetch holds {official[1]}/ of {official[0]} - skipping those")
    else:
        print("No official fetch found - run fetch_official_example_code.py first "
              "or its files will be picked up here as community finds")

    print(f"Searching grep.app for code importing {parsed_module_name}")
    repos = search_repos(parsed_module_name)
    print(f"  {len(repos)} repositories, {sum(len(e['paths']) for e in repos.values())} files\n")

    candidates, import_only, duplicates = [], 0, 0
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
        found, unused, dupes = evaluate_repo(
            repo, entry, meta, parsed_module_name, known_names, dest_dir, official
        )
        import_only += unused
        duplicates += dupes
        kept = sum(1 for c in found if c["recommended"])
        note = f", {dupes} already held officially" if dupes else ""
        print(f"  {repo}: {kept}/{len(found)} files kept ({meta['license']}, {meta['stars']} stars){note}")
        candidates.extend(found)

    candidates.sort(key=lambda c: (not c["recommended"], -c["qualifying_functions"], -c["max_apis_in_function"]))
    write_jsonl(candidates, community_candidates_path())

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

    licenses = sorted({c["license"] for c in kept})
    print(f"\nKept {len(kept)} files in {dest_dir}, "
          f"holding {sum(c['qualifying_functions'] for c in kept)} functions worth reviewing")
    if duplicates:
        print(f"  ({duplicates} skipped without downloading - the official fetch already holds them)")
    if import_only:
        print(f"  ({import_only} downloaded but never used the library, so not kept)")
    print(f"  licences: {', '.join(licenses) if licenses else 'none'}")
    print(f"  decisions recorded in {community_candidates_path()}")
    print("Review the directory and delete anything unwanted, then run parse_python_code.py.")


if __name__ == "__main__":
    main()
