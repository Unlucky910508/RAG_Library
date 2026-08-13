"""Which model servers to talk to, the key for them, and how to handle TLS.

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

# Same server by default (one OpenAI-compatible endpoint serving both
# /chat/completions and /embeddings) - override independently if your
# embedding model is actually hosted elsewhere.
EMBEDDING_BASE_URL = LLM_BASE_URL
EMBEDDING_MODEL = "BAAI/bge-m3"

# Passed as `verify` to every outbound request the pipeline makes - the
# model servers, and PyPI, GitHub and grep.app during a fetch. It lives
# here because whether verification works is a property of the machine:
# a self-signed model server, or a corporate proxy reissuing certificates
# for everything, breaks it for reasons the pipeline knows nothing about.
#
#   True            verify normally
#   "/path/ca.pem"  verify against a CA bundle - what a corporate proxy
#                   needs, and the better answer if you can get the
#                   certificate, since verification stays on
#   False           do not verify at all
#
# False is a real loss of protection, not a formality: it accepts any
# certificate from anything answering at that address, including the
# public services this fetches code from. Prefer the CA bundle; reach for
# False when you cannot get one.
VERIFY_SSL = False
