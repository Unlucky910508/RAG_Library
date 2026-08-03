# RAG_data_creater

Builds a JSONL RAG dataset describing the PyColmap Python API (name, kind,
signatures, required/optional parameters, and an LLM-generated explanation
per entry), so a local LLM/agent can look up API details instead of relying
on its own possibly-wrong memory of the library.

Output: `data/pycolmap_{version}_api.jsonl` (gitignored - generated locally).

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

## 2. LLM config (only needed for step 3, `parse_explanations.py`)

- Edit `config/parse_config.py`:
  - `LLM_BASE_URL` — your local OpenAI-compatible server, e.g. `http://localhost:8000/v1`
  - `LLM_MODEL` — the model name your server expects
- Create `config/key.txt` containing your API key on a single line.
  This file is gitignored — it never gets committed.

## 3. Running the pipeline

Run in this order from the repo root. Each step reads the output of the
previous one and writes back to the same `data/*.jsonl` file.

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
(required/optional, overload-aware), and kind-specific fields (`writable`,
`members`, `enum_of`, `value`, `type`).

```bash
python src/parse_library/parse_explanations.py
```
Calls the local LLM configured in step 2 to generate a short,
retrieval-friendly explanation per record, grounded only in that record's
own fields. Resumable — safe to re-run if interrupted, already-explained
records are skipped.

Scripts must be run by file path (`python src/parse_library/<script>.py`),
not as a module (`-m`) — there is no `__init__.py` under `src/`.

## 4. Notes

- Target library name / output path / LLM settings are centralized in
  `config/parse_config.py` so the three scripts never drift out of sync.
- Known gaps and open issues are tracked locally in `KNOWN_ISSUES.md`
  (gitignored, not part of this repo's public history).

## Maintenance

Keep this file in sync whenever the pipeline, its ordering, or its setup
steps change (e.g. a new script is added, a step's config requirements
change, or the run order changes).
