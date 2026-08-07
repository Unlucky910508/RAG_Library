# RAG_Library

Builds a JSONL RAG dataset describing the PyColmap Python API (name, kind,
signatures, required/optional parameters, and an LLM-generated explanation
per entry) plus official example code, so a local LLM/agent can look up
API details instead of relying on its own possibly-wrong memory of the
library.

Output (`data/` is gitignored - everything in it is generated locally):
- `data/pycolmap_{version}_api.jsonl` — one record per API.
- `data/pycolmap_{version}_examples.jsonl` — official example code split into per-function records.
- `data/pycolmap_{version}_api_chunks.jsonl` / `..._examples_chunks.jsonl` — that data split into embedding chunks, one chunk file per record file (text only, no vectors).
- `data/chroma/` — a persistent Chroma DB holding each chunk's text, metadata, and embedding vector.

## 1. Environment setup

Create a conda env with the target library (pycolmap) installed, e.g.:

```bash
conda create -n pycolmap python=3.11
conda activate pycolmap
pip install pycolmap
```

Then install this repo's own dependencies into the same env:

```bash
pip install -r requirements.txt
```

All commands below assume this env is active (or call its interpreter
directly, e.g. `/path/to/envs/pycolmap/bin/python`).

## 2. LLM config (only needed for `parse_explanations.py` and `load_vectordb.py`)

- Edit `config/config.py`:
  - `LLM_BASE_URL` — your local OpenAI-compatible server, e.g. `http://localhost:8000/v1`
  - `LLM_MODEL` — the model name your server expects
- Create `config/key.txt` containing your API key on a single line.
  This file is gitignored — it never gets committed.

## 3. Running the pipeline

Three phases: `src/parse_library/` produces the raw per-API data,
`src/rag_ingest/` turns that data into what a RAG system actually indexes,
and `src/rag_query/` serves it to a coding agent. Run in this order from
the repo root — each step reads the previous one's output.

### Phase 1 — raw data (`src/parse_library/`)

```bash
python src/parse_library/parse_api.py
```
Walks the target module (pycolmap) via `dir()`/`inspect` and writes one
record per public API (module/class/function/method/property/enum_member/
constant) to `data/pycolmap_{version}_api.jsonl`.

```bash
python src/parse_library/parse_signatures.py
```
Enriches each record in place with signatures, structured parameters
(required/optional, overload-aware), a `doc` field holding the docstring's
descriptive text beyond the signatures (deprecation notes, per-overload
explanations, class summaries), and kind-specific fields (`writable`,
`members`, `enum_of`, `value`, `type`). Only touches the fields it owns —
anything else already in the jsonl (e.g. `explanation`) survives a rerun,
so it's safe to re-run on an already-explained dataset to pick up new
fields. Only `parse_api.py` rebuilds the file from scratch.

```bash
python src/parse_library/parse_examples.py
```
Fetches the official example scripts from the colmap GitHub repo at the
tag matching the installed pycolmap version (never master), splits each
into per-function/class records plus a module-context record, and writes
`data/pycolmap_{version}_examples.jsonl` — a separate file, so
`parse_api.py`'s rebuild can't wipe it. Every `pycolmap.*` reference in
the code is statically resolved against the API records into `apis_used`;
unresolvable references land in `unknown_refs` (zero for the official
examples — that's the version-alignment check working). Test scaffolding
(`conftest.py`, `*_test.py`) is skipped.

```bash
python src/parse_library/parse_explanations.py
```
Calls the local LLM configured in step 2 to generate a short,
retrieval-friendly explanation per record — API records and example
records both — grounded only in that record's own fields. Resumable —
safe to re-run if interrupted, already-explained records are skipped.

### Phase 2 — into the RAG store (`src/rag_ingest/`)

```bash
python src/rag_ingest/parse_chunks.py
```
Splits each record (from every record file that exists) into one or more
embedding chunks — currently an `explanation` chunk (name + official
docstring text + generated explanation), a `signature` chunk (name +
signatures + parameter names), and an `example` chunk (name + APIs used +
code) — so conceptual, precise/parameter-level, and how-do-I-use-it
queries can each match a chunk suited to that style. Each record file gets
its own paired chunk file, name derived automatically
(`..._api.jsonl` → `..._api_chunks.jsonl`, `..._examples.jsonl` →
`..._examples_chunks.jsonl`), so different data sources stay separate and
a future record source needs no config change. Chunks only carry
`record_id`/`chunk_type`/`text`; the full record is looked up by
`record_id` in the record jsonls once a chunk matches, not duplicated
here. Which fields compose each chunk_type is a declarative recipe
(`CHUNK_FIELDS` in `config/config.py`), not code — new chunk types
built from existing fields need only a config edit.

```bash
python src/rag_ingest/load_vectordb.py
```
Reads every chunk file that exists, calls a local OpenAI-compatible
embeddings endpoint (`EMBEDDING_BASE_URL`/`EMBEDDING_MODEL` in
`config/config.py`, defaults to the same server as `LLM_BASE_URL` with
model `BAAI/bge-m3`) per chunk and loads the resulting
vector straight into a Chroma collection at `data/chroma/` (one collection
per pycolmap version, named via `chroma_collection_name()`, created with
`hnsw:space` set to `CHROMA_DISTANCE_METRIC` — `cosine` by default, matching
what embedding models like `BAAI/bge-m3` are actually trained/evaluated
for, instead of Chroma's default squared-L2 — this only takes effect when
the collection is first created), alongside the chunk's text and
`{record_id, chunk_type, text_hash}` metadata. The raw vector is never
written to the chunks jsonl — a chunk is only skipped if its `chunk_id`
(`record_id::chunk_type`) already exists in Chroma *and* its stored
`text_hash` still matches; otherwise it's (re-)embedded and `upsert()`'d,
so edited explanations or reshaped chunk_types get picked up on a rerun
instead of silently staying stale. Same retry/incremental-flush design as
`parse_explanations.py`.

### Phase 3 — serving queries (`src/rag_query/`)

- `search.py`: the actual RAG search logic (embed a query via the
  configured embeddings endpoint, query the Chroma collection, resolve
  each hit's `record_id` back to its full record, dedupe multiple
  matching chunk_types down to one hit per record). No MCP dependency —
  importable by any consumer, not just an MCP server.
- `mcp_server.py`: thin MCP wrapper around `search.py`, exposing a
  `search_pycolmap_api(query, top_k=5)` tool over stdio for a coding
  agent to call directly:
  ```bash
  python src/rag_query/mcp_server.py
  ```

Scripts must be run by file path (`python src/<folder>/<script>.py`), not
as a module (`-m`) — there is no `__init__.py` under `src/`.

## 4. Notes

- Target library name / output path / LLM settings are centralized in
  `config/config.py` so the three scripts never drift out of sync.
- Known gaps and open issues are tracked locally in `KNOWN_ISSUES.md`
  (gitignored, not part of this repo's public history).

## Maintenance

Keep this file in sync whenever the pipeline, its ordering, or its setup
steps change (e.g. a new script is added, a step's config requirements
change, or the run order changes).
