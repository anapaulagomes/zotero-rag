# Zotero RAG

Local Retrieval-Augmented Generation over your Zotero library.
Runs 100% offline. Chat in any language; cite from your own papers.

**Stack:** LanceDB, docling, Ollama, Chainlit.

## What it does

- Reads your Zotero SQLite database directly.
- Parses every paper's PDF with `docling` (layout-aware, multi-column, tables).
- For items without a PDF, falls back to the abstract; without an abstract, to the title.
- Chunks with header-aware recursive splitting and sentence-aware overlap.
- Embeds via Ollama (`nomic-embed-text`, 768 dims) into LanceDB.
- Serves a chat UI (Chainlit) that streams responses from a local LLM and cites the source papers inline.

**Quick map**

```
ingest/                  # workspace member
├── zotero_reader.py     # Zotero SQLite → polars DataFrame
├── parser.py            # docling PDF/HTML → markdown text
├── chunker.py           # header-aware recursive split + sentence overlap
├── embedder.py          # Ollama embed → LanceDB (idempotent per item_id)
└── main.py              # orchestrator: cascade pdf → abstract → title

retrieval/               # shared library (mounted into app)
├── search.py            # query → embed → LanceDB top-k + metadata
└── prompt.py            # LLM prompt template + inline citations

app/                     # workspace member
└── main.py              # Chainlit chat: search → prompt → Ollama stream + refs
```


## Prerequisites

- macOS on Apple Silicon (M1/M4 tested; both 16GB)
- [Ollama](https://ollama.com) running on the host
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A Zotero library at `~/Zotero/` (default) or set `ZOTERO_DB` / `ZOTERO_STORAGE`

Pull the models once:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct
```

## Setup

```bash
git clone <repo>
cd zotero-rag
cp .env.example .env           # adjust paths if your Zotero isn't at ~/Zotero
uv sync --all-packages         # installs runtime + dev deps for all workspace members
uv run pre-commit install      # optional: wire ruff hooks
```

## Run on host (recommended for dev)

```bash
# 1. Ingestion: populates ./data/lancedb. Idempotent; safe to Ctrl+C and re-run.
uv run python ingest/main.py

# 2. UI
uv run chainlit run app/main.py
# open http://localhost:8000
```

Ingestion takes ~6-7h on M4 / ~10h on M1 for a ~1000-paper library.
The progress bar shows ETA. You can interrupt anytime and resume later.
Already-embedded items are skipped by `item_id`.

## Run on Docker (for deployment / portability)

```bash
docker compose --profile ingest up --build   # one-time: build + ingestion
docker compose --profile app up --build      # UI at http://localhost:8000
```

Container env paths are overridden via the `environment:` block in `docker-compose.yml`
(so the same `.env` works for both host and container without changes). Ollama is
reached via `host.docker.internal:11434` from inside the container.

## Configuration

All knobs in `.env`:

| var | default | notes |
|---|---|---|
| `ZOTERO_DB` | `~/Zotero/zotero.sqlite` | optional override |
| `ZOTERO_STORAGE` | `~/Zotero/storage` | optional override |
| `LANCEDB_PATH` | `./data/lancedb` | vector store on disk |
| `OLLAMA_HOST` | `http://localhost:11434` | replaced inside Docker |
| `EMBED_MODEL` | `nomic-embed-text` | 768-dim, matches schema |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | swap freely, response quality varies |
| `HF_TOKEN` | _(unset)_ | optional: avoids the "unauthenticated requests" warning + throttling when docling pulls models on first ingestion |

## Notes & limitations

- **Apple Silicon PDF parsing**: docling's layout model uses float64, which MPS doesn't support.
  The parser forces CPU (`AcceleratorDevice.CPU`). Slower than GPU, but reliable.
- **Updates / deletes are not tracked**: re-running `ingest/main.py` only catches new items.
  If you edit metadata or delete papers in Zotero, the index doesn't reflect that until you rebuild.
- **No multi-user / auth**: this is a personal tool. The Chainlit server has no login.
  Don't expose to the public internet.
- **PDF links in references** show as paths, not clickable links — browsers block `file://` from web pages.
  Copy/paste into your terminal to open.

## Smoke tests

Individual modules can be exercised standalone:

```bash
# Zotero library stats
uv run python ingest/zotero_reader.py

# Parse a single PDF (or a directory of them)
uv run python ingest/parser.py /path/to/paper.pdf
uv run python ingest/parser.py /path/to/dir/

# Try the chunker on extracted text
uv run python ingest/parser.py paper.pdf > /tmp/parsed.md
uv run python ingest/chunker.py /tmp/parsed.md

# Search the index (requires ingestion done first)
uv run python retrieval/search.py "what is syndromic surveillance"

# Inspect the prompt template
uv run python retrieval/prompt.py
```
