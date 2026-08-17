"""Settings for serving an already-built dataset.

This directory is meant to be copied on its own - handed to someone who
will answer queries but never build anything - so it reads nothing from
the pipeline's config/ and imports nothing from the rest of the tree.
What it needs is stated here as plain values.

Nor does it need the library it describes to be installed: serving reads
a dataset off disk and never introspects anything, so the version is part
of the paths below rather than something looked up at runtime.
"""

# --- The dataset ------------------------------------------------------
# Both directories come from a pipeline run. Copy them alongside this
# folder, or point at wherever they live; absolute paths are fine.
#
#   chroma/    the vector store, from load_vectordb.py
#   raw_text/  the records a hit is answered from, from the parse steps
CHROMA_PATH = "../../data/pycolmap_4.1.0/chroma"
RECORDS_DIR = "../../data/pycolmap_4.1.0/raw_text"
COLLECTION_NAME = "pycolmap_4.1.0"
# Must match what the store was built with. Chroma fixes this when a
# collection is created, so a mismatch here is not corrected, it just
# ranks differently than the data was indexed for.
CHROMA_DISTANCE_METRIC = "cosine"

# --- The embedding server ---------------------------------------------
# A query has to be embedded by the same model the chunks were, or the
# vectors are not comparable.
EMBEDDING_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"
API_KEY = "TYPE_YOUR_API_KEY"
# True verifies normally; a path to a CA bundle verifies against it, which
# is what a TLS-intercepting proxy needs and keeps verification on; False
# disables it entirely.
VERIFY_SSL = False

# --- What the agent sees ----------------------------------------------
LIBRARY_NAME = "pycolmap"
SERVER_NAME = "pycolmap-rag"
# Each hit carries a record's worth of text, so a caller asking for many
# floods its own context. Requests above this are clamped, not refused.
MAX_TOP_K = 5

# --- How a hit is rendered --------------------------------------------
# return_fields per chunk_type, drawn from the vocabulary in
# record_fields.py. Only the return side matters here: what a chunk was
# embedded on was settled when the store was built.
#
# Keep in step with CHUNK_FIELDS in the pipeline's config if you change
# recipes there - this copy is what answers are built from, and nothing
# checks the two agree.
CHUNK_FIELDS = {
    "explanation": {
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "signature": {
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "example_workflow": {
        "return_fields": ["name", "source", "apis_used"],
    },
    "example": {
        "return_fields": ["name", "source", "code"],
    },
}
