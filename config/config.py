from pathlib import Path

# Re-exported rather than kept here: which model servers to talk to
# depends on the machine, while the rest of this file describes the
# dataset. Importers still get everything from config, so nothing
# downstream needs to know about the split.
from AI_server_config import (  # noqa: F401
    API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_VERIFY_SSL,
)

parsed_module_name = "pycolmap"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Where the official example scripts live. Left as None, each is worked
# out from the package itself - the repository and licence from its PyPI
# metadata, the directory by looking through the repository for a
# conventionally named one holding .py files - so pointing the pipeline at
# another library is still only a change of parsed_module_name.
#
# Set any of them to take that decision by hand instead. Do that when the
# fetch reports it could not work something out: it stops and says which
# of these to fill in rather than guessing, since a wrong guess would
# quietly fill the dataset with another project's code.
#
# What resolution finds for pycolmap, as a reference for the shape:
#   EXAMPLES_GITHUB_REPO = "colmap/colmap"
#   EXAMPLES_PATH_IN_REPO = "python/examples"
#   EXAMPLES_LICENSE = "BSD-3-Clause"
EXAMPLES_GITHUB_REPO = None
EXAMPLES_PATH_IN_REPO = None
EXAMPLES_LICENSE = None
# Directory names a project might keep its examples under, tried in the
# repository tree when EXAMPLES_PATH_IN_REPO is None.
EXAMPLES_DIR_CONVENTIONS = (
    "examples",
    "example",
    "samples",
    "sample",
    "demos",
    "demo",
    "tutorials",
    "tutorial",
    "cookbook",
    "recipes",
)
# Downloaded .py files land here, with a _manifest.json recording where
# each came from. parse_python_code.py reads this directory; it never
# downloads.
EXAMPLES_MANIFEST_NAME = "_manifest.json"


def examples_src_dir(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_examples_src"


# Community code that uses this library, for fetch_community_code.py.
# Trust here is not taken from stars or reputation: those are only coarse
# prefilters to keep the candidate set small. The decisive test is the static
# one the pipeline already does - resolving every reference against the
# API records of the installed version.
#
# Optional. Without it GitHub allows 60 requests an hour, which covers a
# run or so; with it, 5000. Needs no scopes at all - everything read here
# is public - so a token restricted to public repositories is enough.
GITHUB_TOKEN_FILE = Path(__file__).resolve().parent / "github_token.txt"


def load_github_token():
    if not GITHUB_TOKEN_FILE.exists():
        return None
    return GITHUB_TOKEN_FILE.read_text(encoding="utf-8").strip() or None


COMMUNITY_LICENSE_ALLOWLIST = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "ISC",
    "0BSD",
    "Unlicense",
}
# GitHub reports NOASSERTION when it cannot match a repo's LICENSE to a
# known one, which in practice often means a custom research-only grant.
# Neither auto-accepted nor auto-dropped: flagged for a human to read.
COMMUNITY_LICENSE_REVIEW = {"NOASSERTION"}
COMMUNITY_MIN_STARS = 200
COMMUNITY_MAX_AGE_DAYS = 730
# Measured per function, because that is the unit parse_python_code.py
# turns into a record: the question is whether a file contains at least
# one function that would make a good one. Counting distinct APIs across a
# whole file instead scores a 2100-line grab-bag touching seven APIs the
# same as a 150-line function using eleven, and only the second is worth
# quoting. Three distinct APIs in one function is enough to be showing
# composition rather than a single call.
COMMUNITY_MIN_APIS_PER_FUNCTION = 3
# Any reference that does not resolve against the installed version's API
# records is evidence the file targets a different version.
COMMUNITY_MAX_UNKNOWN_REFS = 0


def community_candidates_path(version):
    """A record of what the fetch kept and what it turned away, written
    alongside the sources so its decisions can be reviewed."""
    return DATA_DIR / f"{parsed_module_name}_{version}_community_candidates.jsonl"


def community_src_dir(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_community_src"


def api_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_api.jsonl"


def examples_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_examples.jsonl"


def community_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_community.jsonl"


def code_sources(version):
    """Directories of .py files parse_python_code.py turns into records,
    each with the prefix its record names carry so a hit is traceable to
    where the code came from. Add a source here once some fetch_* script
    writes the same directory layout."""
    return [
        {
            "src_dir": examples_src_dir(version),
            "jsonl": examples_jsonl_path(version),
            "name_prefix": "examples",
        },
        {
            "src_dir": community_src_dir(version),
            "jsonl": community_jsonl_path(version),
            "name_prefix": "community",
        },
    ]


def record_jsonl_paths(version):
    """All record files that make up the dataset. Downstream steps
    (explanations, chunking, search) iterate whichever of these exist."""
    return [api_jsonl_path(version)] + [s["jsonl"] for s in code_sources(version)]


def chunks_jsonl_path_for(record_jsonl_path):
    """The chunk file paired with a record file, derived from its name:
    ..._api.jsonl -> ..._api_chunks.jsonl. A future record source gets its
    own chunk file automatically, no config edit needed."""
    return record_jsonl_path.with_name(record_jsonl_path.stem + "_chunks.jsonl")


def chunk_jsonl_paths(version):
    """All chunk files. Consumers (load_vectordb) iterate whichever exist."""
    return [chunks_jsonl_path_for(p) for p in record_jsonl_paths(version)]


CHROMA_DIR = DATA_DIR / "chroma"
# BAAI/bge-m3 (like most embedding models) is trained/evaluated for cosine
# similarity, not Chroma's default squared-L2 distance. Only affects a
# collection at creation time - set here so load_vectordb.py and
# search.py can never create it with mismatched metrics.
CHROMA_DISTANCE_METRIC = "cosine"


def chroma_collection_name(version):
    return f"{parsed_module_name}_{version}"


# Ceiling on results per search. Each hit carries a full record's worth of
# text, so a caller asking for many of them floods the agent's context
# with loosely-related matches. Requests above this are clamped, not
# rejected - an over-eager caller still gets an answer.
MAX_TOP_K = 5


# Recipe per chunk_type. All three lists name fields from
# FIELD_EXTRACTORS in src/common/record_fields.py; recombining existing
# fields is a config edit, only a genuinely new field needs code.
#
#   embedding_fields - concatenated into the text that gets embedded and
#                      matched against queries (parse_chunks.py)
#   required         - fields a record must actually have for this
#                      chunk_type to exist at all, so e.g. a class with no
#                      signatures gets no "signature" chunk just because
#                      it has a name
#   return_fields    - concatenated into the text handed back to the
#                      caller on a hit (search.py). Assembled at query
#                      time, so editing this takes effect immediately
#                      without re-embedding anything.
#
# Splitting the two lists means a chunk can be *found* by one kind of text
# and *answered* with another - e.g. match on a prose explanation but hand
# back the source code.
CHUNK_FIELDS = {
    "explanation": {
        "embedding_fields": ["name", "doc", "explanation"],
        "required": ["explanation"],
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "signature": {
        "embedding_fields": ["name", "signatures", "parameter_names"],
        "required": ["signatures"],
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "example_workflow": {
        "embedding_fields": ["name", "apis_used"],
        "required": ["code"],
        "return_fields": ["name", "source", "apis_used"],
    },
    "example": {
        "embedding_fields": ["name", "apis_used", "code"],
        "required": ["code"],
        "return_fields": ["name", "source", "code"],
    },
}


