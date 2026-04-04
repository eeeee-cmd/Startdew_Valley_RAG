# Stardew Valley RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers Stardew Valley questions using grounded information from the public Stardew Valley Wiki.

## Project Purpose

Build a conversational RAG system grounded in the Stardew Valley Wiki, capable of answering player questions about farming, quests, villagers, fishing, mining, and more.

## System Architecture
```
processed.jsonl → chunker.py → build_index.py → FAISS index
                                                      │
                                              retriever.py
                                                      │
                                     query → app.py (/chat)
                                                      │
                                              llm.py (Qwen3)
                                                      │
                                          answer + sources
```

## Repository Structure
```text
Stardew_Valley_RAG/
├── .env.example
├── .gitignore
├── README.md
├── config.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/                        # raw scraped wiki sections
│   ├── interim/                    # page-level aggregations
│   └── processed/
│       └── stardew_wiki_sections.jsonl   # canonical RAG input (8,674 clean chunks)
├── src2/                           # RAG pipeline (main implementation)
│   ├── app.py                      # FastAPI — /chat and /retrieve endpoints
│   ├── chunker.py                  # load JSONL → LangChain Documents
│   ├── embeddings.py               # LocalEmbedder + LocalEmbeddingsWrapper
│   ├── build_index.py              # embed chunks → save FAISS index (run once)
│   ├── retriever.py                # semantic search over FAISS index
│   ├── llm.py                      # Qwen3 client with reasoning support
│   ├── inspect_data.py             # data inspection helper (local use only)
│   ├── test_llm.py                 # LLM endpoint test (local use only)
│   └── index/                      # FAISS index (not committed — rebuild locally)
├── docs/
├── notebooks/
├── src/                            # original scaffold (unused)
└── tests/
```

## Data

| File | Granularity | Records | Use |
|------|-------------|---------|-----|
| `raw/` | Section-level | 11,748 | Original scrape |
| `interim/` | Page-level | — | Intermediate aggregation |
| `processed/stardew_wiki_sections.jsonl` | Section-level | 8,674 (filtered) | ✅ RAG input |

Filters applied to processed data:
- Removed chunks under 50 characters
- Removed `Modding:` and `Module:` wiki pages
- Removed binary/corrupted records

## Setup
```bash
# 1 — clone and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# 2 — install dependencies
cd src2
pip install -r ../requirements.txt

# 3 — configure environment
cp ../.env.example ../.env
# edit .env and set:
# LLM_BASE_URL=https://rsm-8430-finalproject.bjlkeng.io/v1
# LLM_API_KEY=your-student-id
# LLM_MODEL=qwen3-30b-a3b-fp8

# 4 — build the FAISS index (once, ~25 seconds)
python build_index.py --input ../data/processed/stardew_wiki_sections.jsonl --strategy section_recursive

# 5 — start the API
uvicorn app:app --reload --port 8000
```

## API Endpoints

### `POST /chat` — Full RAG
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I upgrade my watering can?"}'
```

Response includes `answer`, `sources` (page title, heading, URL, score), and `usage`.

### `POST /retrieve` — Retrieval only (no LLM)
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "fishing rod", "top_k": 3}'
```

### `GET /health` — Health check
```bash
curl http://localhost:8000/health
```

## Chunking Strategy

Default: `section_recursive` — `RecursiveCharacterTextSplitter` with `chunk_size=512`, `chunk_overlap=64`.

Each chunk's `page_content` prepends the page title and heading before embedding:
```
'Watering Cans — Upgrades and Water Consumption\n<text>'
```
The original text is stored separately in metadata for clean citation display.

## LLM

Model: `qwen3-30b-a3b-fp8` with reasoning enabled via the course-provided endpoint.
Client uses the OpenAI-compatible API (`openai` Python package).
Chain-of-thought reasoning is enabled by default — set `include_reasoning: true` in `/chat` to expose it in the response.

## Notes

- The `index/` folder is not committed — rebuild it with `build_index.py`
- The `.env` file is not committed — copy from `.env.example` and fill in your student ID
- `inspect_data.py` and `test_llm.py` are local helper scripts, not part of the pipeline