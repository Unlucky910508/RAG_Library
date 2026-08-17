"""Settings for serving an already-built dataset.

This directory is meant to be copied on its own - handed to someone who
will answer queries but never build anything - so it reads nothing from
the pipeline's config/ and imports nothing from the rest of the tree.
What it needs is stated here as plain values.

Nor does it need the library it describes to be installed: serving reads
a store off disk and never introspects anything, so the version is part
of the path below rather than something looked up at runtime.
"""

# --- The dataset ------------------------------------------------------
# The Chroma store from load_vectordb.py, which holds the answers as well
# as the vectors - so this is the only thing that travels with this
# folder. Copy it alongside, or point at wherever it lives; absolute
# paths are fine, relative ones are read from this directory.
CHROMA_PATH = "../../data/pycolmap_4.1.0/chroma"
# The collection inside that store, not the store itself: one store can
# hold several. A name that is not there is an error rather than a new
# empty collection - see get_collection in search.py.
COLLECTION_NAME = "pycolmap_4.1.0"

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
