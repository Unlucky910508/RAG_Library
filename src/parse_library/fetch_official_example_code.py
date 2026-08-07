"""Download the target library's official example scripts from GitHub.

Which library, which repo, and which path are all read from
config/config.py, so pointing the whole pipeline at a different library
is a config edit, not a code edit. Fetching happens at the tag matching
the installed version (never master), so example code can't drift ahead
of the API records. Test scaffolding (conftest.py, *_test.py) is skipped:
it's not tutorial content.

Downloading only; turning .py files into records is parse_code.py's job,
which reads this directory and never touches the network. A
_manifest.json is written alongside the sources so that step can attach
provenance (upstream URL, license, ref) without knowing about GitHub.
Sources needing a different acquisition method (crawling arbitrary repos,
scraping docs) belong in their own fetch_* script writing the same
directory layout.
"""

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    EXAMPLES_GITHUB_REPO,
    EXAMPLES_LICENSE,
    EXAMPLES_MANIFEST_NAME,
    EXAMPLES_PATH_IN_REPO,
    examples_src_dir,
    parsed_module_name,
)

SKIP_FILES = {"conftest.py"}
SKIP_SUFFIXES = ("_test.py",)


def is_example_file(name):
    return name.endswith(".py") and name not in SKIP_FILES and not name.endswith(SKIP_SUFFIXES)


def list_example_files(ref):
    url = f"https://api.github.com/repos/{EXAMPLES_GITHUB_REPO}/contents/{EXAMPLES_PATH_IN_REPO}"
    response = requests.get(url, params={"ref": ref}, timeout=30)
    response.raise_for_status()
    return [(e["name"], e["download_url"]) for e in response.json() if is_example_file(e["name"])]


def fetch_text(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def blob_url(ref, filename):
    return f"https://github.com/{EXAMPLES_GITHUB_REPO}/blob/{ref}/{EXAMPLES_PATH_IN_REPO}/{filename}"


def download_examples(ref, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, download_url in list_example_files(ref):
        (dest_dir / filename).write_text(fetch_text(download_url), encoding="utf-8")
        files[filename] = blob_url(ref, filename)
        print(f"  {filename}")

    manifest = {
        "repo": EXAMPLES_GITHUB_REPO,
        "ref": ref,
        "path_in_repo": EXAMPLES_PATH_IN_REPO,
        "license": EXAMPLES_LICENSE,
        "files": files,
    }
    (dest_dir / EXAMPLES_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return files


def main():
    version = __import__(parsed_module_name).__version__
    dest_dir = examples_src_dir(version)
    files = download_examples(version, dest_dir)
    print(f"Downloaded {len(files)} example files at ref {version} to {dest_dir}")


if __name__ == "__main__":
    main()
