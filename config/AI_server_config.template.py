"""Which model servers to talk to, and the key for them.

Split out from config.py because these belong to whoever is running the
pipeline, while everything there describes the dataset being built.
config.py re-exports them, so scripts still import everything from one
place.

Copy this to AI_server_config.py and fill it in - that file is not
tracked by git, so your key stays local and pulling a newer version of
the pipeline never touches your settings or asks you to reconcile them.

    cp config/AI_server_config.template.py config/AI_server_config.py
"""

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
API_KEY = "TYPE_YOUR_API_KEY"
# Set to False only for known/trusted internal servers using a self-signed
# certificate - this disables TLS certificate verification entirely.
LLM_VERIFY_SSL = False

# Same server by default (one OpenAI-compatible endpoint serving both
# /chat/completions and /embeddings) - override independently if your
# embedding model is actually hosted elsewhere.
EMBEDDING_BASE_URL = LLM_BASE_URL
EMBEDDING_MODEL = "BAAI/bge-m3"
