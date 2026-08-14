# RAG_Library

Builds a JSONL RAG dataset describing the PyColmap Python API (name, kind,
signatures, required/optional parameters, and an LLM-generated explanation
per entry) plus official example code, so a local LLM/agent can look up
API details instead of relying on its own possibly-wrong memory of the
library.

Output (`data/` is gitignored - everything in it is generated locally):
- `data/pycolmap_{version}_api.jsonl` — one record per API.
- `data/pycolmap_{version}_examples_src/` — the downloaded official example `.py` files, plus a `_manifest.json` of where each came from.
- `data/pycolmap_{version}_community_src/` — optional; third-party `.py` files that passed the filters, namespaced per repository.
- `data/pycolmap_{version}_examples.jsonl` / `..._community.jsonl` — that code split into per-function records, one file per source.
- `data/pycolmap_{version}_api_chunks.jsonl` / `..._examples_chunks.jsonl` / `..._community_chunks.jsonl` — that data split into embedding chunks, one chunk file per record file (text only, no vectors).
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

```bash
cp config/AI_server_config.template.py config/AI_server_config.py
```

Then fill it in:
- `LLM_BASE_URL` — your OpenAI-compatible server, e.g. `http://localhost:8000/v1`
- `LLM_MODEL` — the model name your server expects
- `API_KEY` — the key for that server
- `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` — default to the same server, override if your
  embedding model is hosted elsewhere
- `VERIFY_SSL` — TLS verification for **every** outbound request (model servers, PyPI,
  GitHub, grep.app). `True` verifies normally; a path like `"/path/ca.pem"` verifies against
  a CA bundle, which is what a corporate proxy needs and keeps verification on; `False`
  disables it, which also accepts any certificate from the public services this fetches
  code from — prefer the bundle if you can get it

`config/AI_server_config.py` is gitignored: your key stays local, and
pulling a newer version of the pipeline never touches your settings or
asks you to reconcile them. `config/config.py` re-exports everything from
it, so scripts import it all from `config` either way.

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
python src/parse_library/filter_api_records.py --exclude
```
Optional, and only worth it for a library that exposes far more than you
want indexed. Drops records by name prefix: `--exclude` drops what
matches, `--keep` drops what doesn't, `--dry-run` reports without writing.
A prefix matches that exact name and everything beneath it, so
`pycolmap.Camera` drops `Camera.create` but leaves `CameraModelId` alone.

The policy names the file — `--exclude` reads
`filter/<module>_<version>/exclude.py`, `--keep` reads `keep.py` — so
there is no path to pass. Every list of strings in that file is read and
merged, so prefixes can be grouped by reason under whatever names read
best; the variable names are ignored. `filter/` is gitignored — these are
local judgements — so create the file yourself; the step names the exact
path if it is missing, and skipping it entirely is fine.

Run this here rather than later: a record dropped now costs nothing,
while one dropped after `parse_explanations.py` throws away an LLM call.

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
python src/parse_library/fetch_official_example_code.py
```
Downloads the official example scripts at the tag matching the installed
library version — never master, so example code can't drift ahead of the
API records. Writes the `.py` files to
`data/pycolmap_{version}_examples_src/` plus a `_manifest.json` recording
each file's upstream URL, ref, and license. Everything in the directory
is taken except `conftest.py`, which is pytest wiring; tests are kept,
since a test living beside the examples drives them end to end and shows
real usage. Whether a file uses the library at all is decided by
`parse_python_code.py`, which reads it, not guessed from its name.
Downloading only — a source that
needs a different acquisition method gets its own `fetch_*` script writing
the same directory layout.

**Where those examples live is worked out, not configured.** The
repository and licence come from the package's PyPI metadata, and the
directory from scanning the repository tree for a conventionally named one
(`examples/`, `samples/`, `demos/`, …) holding `.py` files. Tag lookup
tries both `{version}` and `v{version}`, since projects differ. So
retargeting the pipeline is still just `parsed_module_name`.

`EXAMPLES_GITHUB_REPO`, `EXAMPLES_PATH_IN_REPO` and `EXAMPLES_LICENSE`
default to `None`, meaning "work it out". Set any of them to decide it by
hand — for examples kept somewhere unusual, a project not on PyPI, or a
ref you want pinned. **Where resolution fails, the run stops and names the
setting to fill in** rather than falling back to a default, since a wrong
repository would quietly fill the dataset with another project's code.
Each run prints what it settled on and whether that came from config or
resolution.

```bash
python src/parse_library/parse_python_code.py
```
Reads `.py` files (never the network) from every source directory listed
in `config.code_sources()` — the official examples above, plus community
code if it has been fetched — and splits each into per-function/class
records plus a module-context record. Each source writes its own jsonl and
prefixes its record names (`examples/…`, `community/…`) so a hit is
traceable to where the code came from, and each stays separate from
`parse_api.py`'s output so its rebuild can't wipe them. Every reference to
the target library is statically resolved via `ast` against the API
records into `apis_used`; unresolvable ones land in `unknown_refs` (zero
for the official examples — that's the version-alignment check working).

Functions and classes that never touch the target library are **left
out** — argument parsers, plotting helpers, plain dataclasses. In a RAG
built around one library they match nothing useful and only dilute
retrieval, and the effect grows with file size. A helper supporting a
workflow is lost along with them; the functions around it that do call
the library still describe that workflow.

`ast` never executes the code, which is what makes it safe over
downloaded sources; the tradeoff is that dynamic references
(`getattr(lib, name)`) are invisible, so `apis_used` is a conservative
lower bound. Python only — other languages would need a sibling
`parse_<language>_code.py`.

#### Optional: finding more example code

The official examples are a small corpus. To pull in third-party code that
uses the library:

```bash
python src/parse_library/fetch_community_code.py
```
Searches grep.app for repositories importing the target library, adds
licence and activity from GitHub's API, downloads the matching files and
scores each one, keeping those that pass in
`data/pycolmap_{version}_community_src/` (namespaced per repository, with
a `_manifest.json` carrying each file's URL and licence). No credentials
needed. Optionally, a GitHub token in `config/github_token.txt`
(gitignored, no scopes required) raises the API limit from 60 requests an
hour to 5000 — useful when re-running while tuning the filters.

Stars and recency are only cheap prefilters. What decides a file is the
same static check as above — every reference must resolve against the
installed version's API records (`unknown_refs` must be empty) — plus
density, measured **per function** rather than per file: a file passes if
it holds at least one function using `COMMUNITY_MIN_APIS_PER_FUNCTION`
distinct APIs. Per function, because a function is what
`parse_python_code.py` turns into a record; counting across a whole file
instead scores a 2000-line grab-bag the same as a focused one. Each
candidate reports `qualifying_functions` and `max_apis_in_function` so
you can see how much a file would actually yield. Thresholds and the
licence allowlist live in `config/config.py`.

Files that import the library without calling into it are left out
entirely rather than listed with an empty `apis_used`.

Only licences on `COMMUNITY_LICENSE_ALLOWLIST` are accepted. GitHub's
`NOASSERTION` — a licence file exists but could not be matched to a known
one — is **turned away**, since unknown terms are not the same as
permissive ones: Tencent's HY-World agreement reads that way and excludes
the EU, UK and South Korea outright. That also costs some genuinely
permissive code (`colmap/colmap`'s own `COPYING.txt` reads `NOASSERTION`
because its BSD text carries a preamble), which is why
`COMMUNITY_SEARCH_PAGES` is set generously: widening the search is
cheaper than auditing licences after the fact.

Every decision is recorded in
`data/pycolmap_{version}_community_candidates.jsonl`. Downloading is not
ingesting: **nothing reaches the dataset until you run
`parse_python_code.py`**, so read that file and delete anything unwanted
from the source directory first.

```bash
python src/parse_library/parse_explanations.py
```
Calls the local LLM configured in step 2 to generate a short,
retrieval-friendly explanation per record, grounded only in that record's
own fields. API records and example records get different prompts:
describing what an introspected API is and describing what a snippet
accomplishes are different jobs. Example code is passed as a raw block
rather than JSON-escaped. Resumable — safe to re-run if interrupted,
already-explained records are skipped.

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
here.

Which fields compose each chunk_type is a declarative recipe
(`CHUNK_FIELDS` in `config/config.py`), not code. Each recipe names three
lists, all drawn from the field vocabulary in
`src/common/record_fields.py`:

- `embedding_fields` — concatenated into the text that gets embedded and matched against queries (used here)
- `required` — fields a record must actually have for this chunk_type to apply at all
- `return_fields` — concatenated into the text handed back on a hit (used by `search.py`)

Splitting embedding from return means a chunk can be *found* by one kind
of text and *answered* with another. Recombining existing fields is a
config edit; only a genuinely new field needs a new renderer.

```bash
python src/rag_ingest/load_vectordb.py
```
Reads every chunk file from `chunk_jsonl_paths()` in `config/config.py`
(the row format contract is just `{record_id, chunk_type, text}` per
line), calls a local OpenAI-compatible embeddings endpoint
(`EMBEDDING_BASE_URL`/`EMBEDDING_MODEL` in `config/config.py`, defaults
to the same server as `LLM_BASE_URL` with
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
  configured embeddings endpoint, query the Chroma collection, dedupe
  multiple matching chunk_types down to one hit per record, then render
  each hit's record through the matched chunk_type's `return_fields`).
  Each hit is `{record_id, matched_chunk_type, distance, text}`. The
  returned text is assembled at query time rather than read from the
  vector DB, so editing a recipe's `return_fields` takes effect on the
  next query with no re-embedding. `top_k` is clamped to
  `MAX_TOP_K` (`config/config.py`, default 5) — every hit carries a full
  record's worth of text, so an unbounded request floods the caller's
  context. Enforced here rather than in the MCP layer so every consumer is
  covered. No MCP dependency — importable by any consumer, not just an MCP
  server.
- `mcp_server.py`: thin MCP wrapper around `search.py`, exposing a
  `search_rag(query, top_k=5)` tool over stdio for a coding
  agent to call directly. The limit is stated in both the tool
  description and the `top_k` parameter's JSON schema description, so a
  calling model sees it; over-limit requests are trimmed, not rejected:
  ```bash
  python src/rag_query/mcp_server.py
  ```

Scripts must be run by file path (`python src/<folder>/<script>.py`), not
as a module (`-m`) — there is no `__init__.py` under `src/`.

## 4. Refreshing stale explanations

Explanations are generated from whatever fields a record had at the time,
so when an enricher starts emitting a new field, explanations written
before that were produced without it. `parse_explanations.py` skips
records that already have one, so they won't refresh on their own — and
deleting the jsonl to force it would regenerate every record, not just
the affected ones.

`invalidate_explanations.py` clears them selectively instead: name the
fields whose arrival makes an explanation stale, and only records
carrying one of them are reset. The next `parse_explanations.py` run then
regenerates exactly those.

```bash
# preview first - writes nothing
python src/parse_library/invalidate_explanations.py data/pycolmap_4.1.0_api.jsonl --when-field doc --dry-run

# clear, then regenerate
python src/parse_library/invalidate_explanations.py data/pycolmap_4.1.0_api.jsonl --when-field doc
python src/parse_library/parse_explanations.py
```

Use `--all` instead of `--when-field` when the prompt itself changed, so
every record's output is stale. Afterwards re-run `parse_chunks.py` and
`load_vectordb.py`; the latter's `text_hash` check re-embeds only the
chunks whose text actually changed.

## 5. Notes

- Target library name / output path / LLM settings are centralized in
  `config/config.py` so the pipeline steps never drift out of sync.
- Known gaps and open issues are tracked locally in `KNOWN_ISSUES.md`
  (gitignored, not part of this repo's public history).

## Maintenance

Keep this file in sync whenever the pipeline, its ordering, or its setup
steps change (e.g. a new script is added, a step's config requirements
change, or the run order changes).
