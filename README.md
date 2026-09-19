# ZenLynx RAG API

A FastAPI-based RAG (Retrieval-Augmented Generation) service that answers sustainability and ESG questions about specific organizations, using their documents only. **Strict organization-level data isolation** is the headline requirement.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and (optionally) configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS

# 4. Generate sample PDF documents
python scripts/generate_pdfs.py

# 5. Ingest documents into ChromaDB
python -m app.ingest --reset

# 6. Start the API server
uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`. No API keys are needed — the default configuration uses a local embedding model and an extractive answer fallback.

## Environment Variables

All configuration lives in `.env` (see `.env.example` for defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `LLM_PROVIDER` | `extractive` | `extractive`, `openai`, `anthropic`, or `omniroute` |
| `OPENAI_API_KEY` | — | Required when using OpenAI embeddings or LLM |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic LLM |
| `OMNIROUTE_API_KEY` | — | Required when `LLM_PROVIDER=omniroute` |
| `OMNIROUTE_BASE_URL` | `http://localhost:20128/v1` | OmniRoute gateway base URL (OpenAI-compatible `/v1`) |
| `OMNIROUTE_MODEL` | — | Model name to request via the OmniRoute gateway |
| `CHROMA_PATH` | `chroma_store` | ChromaDB persistent storage directory |
| `SQLITE_PATH` | `zenlynx.db` | SQLite database file path |
| `DATA_DIR` | `data` | Root directory for organization documents |

### Zero-key operation

With default settings (`EMBEDDING_PROVIDER=local`, `LLM_PROVIDER=extractive`), the entire pipeline runs end-to-end without any API keys:

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` runs locally.
- **LLM**: The extractive fallback finds the most relevant chunk via keyword overlap and returns it verbatim — no generation, no hallucination risk.

### Generative answers via an OmniRoute gateway (optional)

To get real LLM-generated answers, point `LLM_PROVIDER=omniroute` at any OpenAI-compatible gateway:

```env
LLM_PROVIDER=omniroute
OMNIROUTE_API_KEY=sk-...
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto/best-coding
```

The OmniRoute provider speaks the OpenAI Chat Completions protocol, so it works against self-hosted gateways (LLM proxies, local model servers) without any vendor-specific SDK. The extractive fallback remains the safe default when no gateway credentials are configured.

## Ingesting Documents

```bash
# Index all organizations
python -m app.ingest

# Index a single organization
python -m app.ingest --org org_001

# Drop and rebuild all indexes
python -m app.ingest --reset
```

Documents are loaded from `data/<org_id>/` and support `.md`, `.txt`, and `.pdf` formats. The organization ID is derived from the directory name, so a document can never be indexed without an owner.

## API Endpoints

### POST /query

Ask an ESG question scoped to one organization.

**Request:**
```json
{
  "organization_id": "org_001",
  "question": "What is the Scope 1 emission for 2025?"
}
```

**Response:**
```json
{
  "organization_id": "org_001",
  "answer": "From **emission_report_2025.md** (## Scope 1 — Direct Emissions):\n\nScope 1 covers direct GHG emissions from sources owned or controlled by Aurora Textiles...\n\nScope 1 (direct): 1,250 tCO2e for FY 2025.",
  "sources": ["emission_report_2025.md"],
  "grounded": true,
  "retrieved_chunks": [
    {
      "document": "emission_report_2025.md",
      "chunk_id": "emission_report_2025.md::chunk_0001",
      "section": "## Scope 1 — Direct Emissions",
      "score": 0.8234,
      "distance": 0.4521,
      "excerpt": "Scope 1 covers direct GHG emissions from sources owned or controlled by Aurora Textiles..."
    }
  ]
}
```

**Error codes:** `404` (unknown org), `422` (malformed input), `503` (empty index).

### GET /organizations

List all registered organizations with document and chunk counts.

### GET /health

Liveness probe — returns status, active embedding provider, and LLM provider.

### POST /ingest

Trigger re-indexing via the API. Accepts optional `organization_id` and `reset` fields.

## RAG Approach

### Pipeline: Load → Chunk → Embed → Retrieve → Generate

1. **Load**: Walk `data/<org_id>/` for `.md`, `.txt`, `.pdf` files. PDF text extraction via `pypdf`.

2. **Chunk**: Split on blank lines (paragraph boundaries) and pack to ~900 characters with 150-character overlap. Each chunk tracks its nearest markdown heading and prefixes it to the embedded text.

3. **Embed**: Batch embed all chunks using the configured provider. The provider name is stored in Chroma collection metadata to prevent querying with a mismatched embedder.

4. **Retrieve**: Fetch ~12 candidates from Chroma, then re-rank with a hybrid score combining vector similarity (70%) and keyword overlap via Jaccard similarity (30%). Drop anything above a distance threshold of 1.2. Return the top 4.

5. **Generate**: Prompt the LLM with numbered context passages and strict instructions to answer only from context. If the LLM cannot find the information, it must say so explicitly.

### Why heading-aware chunking?

ESG documents are full of context-dependent metrics:

> Scope 1 (direct): 1,250 tCO2e

This line is meaningless without knowing it comes from "## Greenhouse Gas Emissions". Our chunker tracks the nearest markdown heading and prefixes it to the embedded text, so the embedding model sees the full context:

> ## Scope 1 — Direct Emissions
> Scope 1 (direct): 1,250 tCO2e for FY 2025.

### Why hybrid re-ranking?

Pure vector search can struggle with exact-match queries common in ESG reporting ("What is Scope 1?"). Adding keyword overlap (Jaccard similarity on lowered tokens) as 30% of the score boosts chunks that contain the literal terms the user asked about, while the 70% vector component preserves semantic matching for paraphrased questions.

## Data Isolation

**Data isolation is the headline requirement.** It is enforced in three independent, redundant layers so that a bug in any single layer cannot leak data across organizations.

### Layer 1 — Physical Isolation

Each organization gets its own ChromaDB collection, named `org__<id>`. Collections are completely separate indexes with no shared query path. An org_001 query physically cannot touch org_002's vectors because they live in different collections.

### Layer 2 — Logical Isolation

Even within a collection, every chunk carries `organization_id` in its metadata, and every query applies a `where={"organization_id": id}` filter. This is a defense-in-depth measure — if a chunk were somehow inserted into the wrong collection, this filter would still exclude it.

### Layer 3 — Assertion (Fail-Closed)

After retrieval, every hit's metadata `organization_id` is verified against the requested org. If any mismatch is detected, the request is immediately blocked with a `DataIsolationError` and an HTTP 500 response. The system fails closed — it never returns potentially leaked data.

### Input Validation

Organization IDs are validated against `^[A-Za-z0-9_\-]{1,64}$` before being used in collection names or file paths. This blocks path-traversal attacks (`../org_002`) and injection at the schema level, before any downstream code sees the value.

Unknown organization IDs return `404` rather than silently searching nothing.

## Project Structure

```
zenlynx-rag-api/
├── app/
│   ├── __init__.py       # Package marker
│   ├── config.py         # All env vars in one place (pydantic-settings)
│   ├── schemas.py        # Pydantic v2 request/response models
│   ├── loaders.py        # Load .md/.txt/.pdf from data/<org_id>/
│   ├── chunking.py       # Heading-aware paragraph chunker
│   ├── embeddings.py     # Pluggable embedding providers
│   ├── vectorstore.py    # Chroma wrapper — isolation enforced here
│   ├── llm.py            # Pluggable LLM + extractive fallback
│   ├── rag.py            # Retrieve → re-rank → generate pipeline
│   ├── db.py             # SQLite org registry + query log
│   ├── ingest.py         # CLI: python -m app.ingest [--reset]
│   └── main.py           # FastAPI application
├── data/
│   ├── org_001/          # Aurora Textiles Pvt Ltd
│   └── org_002/          # Meridian Logistics Ltd
├── scripts/
│   └── generate_pdfs.py  # ReportLab PDF generator
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_isolation.py # Isolation + correctness tests
│   └── test_providers.py # LLM/embedding provider factory tests
├── Assignment.pdf                  # Assignment brief (reference)
├── sustainability_keywords.docx    # Sustainability Radar keyword guide (reference)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Reference Materials

- **Assignment.pdf** — the original assignment brief this implementation satisfies (RAG API, org-scoped Q&A, data isolation).
- **sustainability_keywords.docx** — Zenlynx "Sustainability Radar" keyword master list, monitoring sources, and LinkedIn post prompts. Domain reference material only; it is **not** ingested as organization data.

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use temporary Chroma and SQLite storage so they don't affect your development data. The sentence-transformers model is loaded once per test session.

## Sample Organizations

| ID | Name | Display Name | Scope 1 (2025) |
|----|------|-------------|-----------------|
| `org_001` | aurora_textiles | Aurora Textiles Pvt Ltd | 1,250 tCO2e |
| `org_002` | meridian_logistics | Meridian Logistics Ltd | 2,810 tCO2e |

Both organizations use the **same metric names** with **different values** so that isolation failures are immediately visible rather than plausible.
