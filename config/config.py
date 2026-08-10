from pathlib import Path

parsed_module_name = "pycolmap"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
LLM_API_KEY_FILE = Path(__file__).resolve().parent / "key.txt"
# Set to False only for known/trusted internal servers using a self-signed
# certificate - this disables TLS certificate verification entirely.
LLM_VERIFY_SSL = False

# Same server by default (one OpenAI-compatible endpoint serving both
# /chat/completions and /embeddings) - override independently if your
# embedding model is actually hosted elsewhere.
EMBEDDING_BASE_URL = LLM_BASE_URL
EMBEDDING_MODEL = "BAAI/bge-m3"


# Where the official example scripts live: the tag matching the installed
# pycolmap version is fetched, never master, so the code can't drift ahead
# of the API records.
EXAMPLES_GITHUB_REPO = "colmap/colmap"
EXAMPLES_PATH_IN_REPO = "python/examples"
EXAMPLES_LICENSE = "BSD-3-Clause (COLMAP)"
# Downloaded .py files land here, with a _manifest.json recording where
# each came from. parse_code.py reads this directory; it never downloads.
EXAMPLES_MANIFEST_NAME = "_manifest.json"


def examples_src_dir(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_examples_src"


def api_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_api.jsonl"


def examples_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_examples.jsonl"


def record_jsonl_paths(version):
    """All record files that make up the dataset. Downstream steps
    (explanations, chunking, search) iterate whichever of these exist."""
    return [api_jsonl_path(version), examples_jsonl_path(version)]


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
    "example": {
        "embedding_fields": ["name", "apis_used", "code"],
        "required": ["code"],
        "return_fields": ["name", "source", "apis_used", "code"],
    },
}


def load_llm_api_key():
    return LLM_API_KEY_FILE.read_text(encoding="utf-8").strip()
