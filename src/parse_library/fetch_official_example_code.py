"""Download the target library's official example scripts from GitHub.

Where those live is worked out from the package rather than written down:
the repository and licence come from its PyPI metadata, the ref from
matching the installed version against the repository's tags, and the
directory from looking through the tree for a conventionally named one
holding .py files. So pointing the pipeline at another library stays a
change of parsed_module_name.

Anything set in config/config.py is used as given and not looked up, for
the cases resolution cannot reach - examples kept somewhere unusual, a
project not on PyPI, a ref you want pinned.

When resolution fails it says so and stops, naming the setting to fill in.
It never falls back to a default: the failure is "this cannot be worked
out automatically", and guessing would quietly fill the dataset with
another project's code, which is far worse than an error.

Fetching happens at the tag matching the installed version, never the
development branch, so example code cannot drift ahead of the API
records. Everything in the directory is taken except conftest.py, which
is pytest wiring; whether a file actually uses the library is decided by
parse_python_code.py, which reads it, rather than guessed from its name.

Downloading only; turning .py files into records is parse_python_code.py's
job, which reads this directory and never touches the network. A
_manifest.json is written alongside the sources so that step can attach
provenance without knowing about GitHub. Sources needing a different
acquisition method belong in their own fetch_* script writing the same
directory layout.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    EXAMPLES_DIR_CONVENTIONS,
    EXAMPLES_GITHUB_REPO,
    EXAMPLES_LICENSE,
    EXAMPLES_MANIFEST_NAME,
    EXAMPLES_PATH_IN_REPO,
    examples_src_dir,
    load_github_token,
    parsed_module_name,
)

PYPI_API = "https://pypi.org/pypi"
GITHUB_API = "https://api.github.com/repos"
# conftest.py holds pytest wiring and nothing a reader could learn from.
# Tests are not skipped: a test living beside the examples drives them
# end to end, so it shows real usage - custom_incremental_pipeline_test.py
# has four functions using five to seven APIs each, denser than most of
# what the community fetch keeps. The unit tests under src/ that motivated
# skipping tests are a different thing and are not in this directory.
# Files that turn out to use nothing are dropped by parse_python_code.py,
# which can see that; a filename cannot.
SKIP_FILES = {"conftest.py"}
# Ordered by how directly a key names the source, since a project may list
# several and its issue tracker is not its repository.
REPO_URL_KEYS = ("Repository", "Source", "Source Code", "Code", "GitHub", "Homepage")


class Unresolved(Exception):
    """Something could not be worked out and guessing is not acceptable.
    Carries the config setting a person should fill in instead."""

    def __init__(self, what, setting, detail=""):
        super().__init__(f"Could not work out {what}. {detail}".strip())
        self.setting = setting


def github_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def pypi_metadata(package):
    response = requests.get(f"{PYPI_API}/{package}/json", timeout=30)
    if not response.ok:
        raise Unresolved(
            f"where {package} comes from",
            "EXAMPLES_GITHUB_REPO and EXAMPLES_LICENSE",
            f"PyPI has no page for it (HTTP {response.status_code}).",
        )
    return response.json()["info"]


def github_repo_from_urls(project_urls):
    """owner/name from whichever listed URL points at GitHub."""
    for key in REPO_URL_KEYS:
        url = project_urls.get(key)
        if not url:
            continue
        parsed = urlparse(url)
        if "github.com" not in parsed.netloc:
            continue
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return None


def resolve_repo(info, package):
    repo = github_repo_from_urls(info.get("project_urls") or {})
    if repo:
        return repo
    listed = ", ".join((info.get("project_urls") or {})) or "none"
    raise Unresolved(
        f"which repository hosts {package}",
        "EXAMPLES_GITHUB_REPO",
        f"Its PyPI page lists no GitHub URL (has: {listed}).",
    )


def resolve_license(info, package):
    license_id = (info.get("license") or "").strip()
    if license_id:
        return license_id
    raise Unresolved(
        f"the licence of {package}",
        "EXAMPLES_LICENSE",
        "Its PyPI metadata does not state one.",
    )


def resolve_ref(repo, version, token):
    """The tag for this version. Projects differ on the leading v, so both
    spellings are tried before giving up."""
    response = requests.get(
        f"{GITHUB_API}/{repo}/tags", params={"per_page": 100}, headers=github_headers(token), timeout=30
    )
    tags = {t["name"] for t in response.json()} if response.ok else set()
    for candidate in (version, f"v{version}"):
        if candidate in tags:
            return candidate
    sample = ", ".join(sorted(tags)[:5]) or "none readable"
    raise Unresolved(
        f"which tag of {repo} is version {version}",
        "EXAMPLES_REF (add it to config) or install a version that is tagged",
        f"Neither {version} nor v{version} exists. Tags look like: {sample}.",
    )


def find_example_dirs(repo, ref, token):
    """Conventionally named directories holding .py files, commonest first."""
    response = requests.get(
        f"{GITHUB_API}/{repo}/git/trees/{ref}",
        params={"recursive": "1"},
        headers=github_headers(token),
        timeout=60,
    )
    if not response.ok:
        return {}
    counts = {}
    for entry in response.json().get("tree", []):
        path = entry["path"]
        if not path.endswith(".py"):
            continue
        parts = path.split("/")
        for i, part in enumerate(parts[:-1]):
            if part.lower() in EXAMPLES_DIR_CONVENTIONS:
                directory = "/".join(parts[: i + 1])
                counts[directory] = counts.get(directory, 0) + 1
    return counts


def resolve_path_in_repo(repo, ref, token):
    counts = find_example_dirs(repo, ref, token)
    if not counts:
        raise Unresolved(
            f"where {repo} keeps its examples",
            "EXAMPLES_PATH_IN_REPO",
            f"No directory named like {'/'.join(EXAMPLES_DIR_CONVENTIONS[:4])} holds .py files at {ref}. "
            "The project may not ship examples, or may keep them somewhere unconventional.",
        )
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1:
        others = ", ".join(f"{d} ({n})" for d, n in ranked[1:])
        print(f"  note: also found {others} - set EXAMPLES_PATH_IN_REPO to use one of those instead")
    return ranked[0][0]


def resolve_source(package, version, token):
    """Settle repo, ref, path and licence, taking anything config states
    as given. Also reports where each value came from, so a run makes
    plain what was decided for you and what you decided."""
    repo, license_id = EXAMPLES_GITHUB_REPO, EXAMPLES_LICENSE
    origins = {
        "repo": "config" if repo else "resolved",
        "license": "config" if license_id else "resolved",
        "path": "config" if EXAMPLES_PATH_IN_REPO else "resolved",
        "ref": "resolved",
    }

    if repo is None or license_id is None:
        info = pypi_metadata(package)
        repo = repo or resolve_repo(info, package)
        license_id = license_id or resolve_license(info, package)

    ref = resolve_ref(repo, version, token)
    path = EXAMPLES_PATH_IN_REPO or resolve_path_in_repo(repo, ref, token)
    return {"repo": repo, "ref": ref, "path": path, "license": license_id}, origins


def is_example_file(name):
    return name.endswith(".py") and name not in SKIP_FILES


def list_example_files(source, token):
    url = f"{GITHUB_API}/{source['repo']}/contents/{source['path']}"
    response = requests.get(url, params={"ref": source["ref"]}, headers=github_headers(token), timeout=30)
    response.raise_for_status()
    return [(e["name"], e["download_url"]) for e in response.json() if is_example_file(e["name"])]


def fetch_text(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def blob_url(source, filename):
    return f"https://github.com/{source['repo']}/blob/{source['ref']}/{source['path']}/{filename}"


def download_examples(source, dest_dir, token):
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, download_url in list_example_files(source, token):
        (dest_dir / filename).write_text(fetch_text(download_url), encoding="utf-8")
        files[filename] = blob_url(source, filename)
        print(f"  {filename}")

    manifest = {
        "repo": source["repo"],
        "ref": source["ref"],
        "path_in_repo": source["path"],
        "license": source["license"],
        "files": files,
    }
    (dest_dir / EXAMPLES_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return files


def main():
    version = __import__(parsed_module_name).__version__
    token = load_github_token()

    try:
        source, origins = resolve_source(parsed_module_name, version, token)
    except Unresolved as problem:
        print(f"\n{problem}")
        print(f"\nThis cannot be worked out automatically for {parsed_module_name}.")
        print(f"Find it yourself and set {problem.setting} in config/config.py, then run this again.")
        sys.exit(1)

    print(f"{parsed_module_name} {version}:")
    for field in ("repo", "ref", "path", "license"):
        print(f"  {field:8} {source[field]}  ({origins[field]})")

    dest_dir = examples_src_dir(version)
    files = download_examples(source, dest_dir, token)
    print(f"Downloaded {len(files)} example files to {dest_dir}")


if __name__ == "__main__":
    main()
