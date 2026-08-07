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


def api_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_api.jsonl"


def examples_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_examples.jsonl"


def record_jsonl_paths(version):
    """All record files that make up the dataset. Downstream steps
    (explanations, chunking, search) iterate whichever of these exist."""
    return [api_jsonl_path(version), examples_jsonl_path(version)]


def chunks_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_chunks.jsonl"


CHROMA_DIR = DATA_DIR / "chroma"
# BAAI/bge-m3 (like most embedding models) is trained/evaluated for cosine
# similarity, not Chroma's default squared-L2 distance. Only affects a
# collection at creation time - set here so load_vectordb.py and
# search.py can never create it with mismatched metrics.
CHROMA_DISTANCE_METRIC = "cosine"


def chroma_collection_name(version):
    return f"{parsed_module_name}_{version}"


# Recipe for parse_chunks.py: each chunk_type is built by looking up and
# concatenating "fields" (see FIELD_EXTRACTORS in parse_chunks.py) in
# order. "required" lists which of those fields must actually be present
# on a record for this chunk_type to exist at all - e.g. a class with no
# signatures shouldn't get an empty "signature" chunk just because it has
# a name. Add/edit chunk types here without touching any code, as long as
# the field extractors you list already exist.
CHUNK_FIELDS = {
    "explanation": {
        "fields": ["name", "doc", "explanation"],
        "required": ["explanation"],
    },
    "signature": {
        "fields": ["name", "signatures", "parameter_names"],
        "required": ["signatures"],
    },
    "example": {
        "fields": ["name", "apis_used", "code"],
        "required": ["code"],
    },
}


def load_llm_api_key():
    return LLM_API_KEY_FILE.read_text(encoding="utf-8").strip()
