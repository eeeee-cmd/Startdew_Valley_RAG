# Stardew Valley RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers Stardew Valley questions using grounded information from the public Stardew Valley Wiki.

## Project Purpose

Build a conversational RAG system grounded in the Stardew Valley Wiki, capable of answering player questions about farming, quests, villagers, fishing, mining, and more.

## RAG System Overview

**End-to-End Pipeline:**
```
Wiki Data (JSONL)
    │
    ├─→ chunker.py
    │   ├─ Load JSONL documents
    │   └─ Split into 512-char sections with 64-char overlap
    │
    ├─→ build_index.py
    │   ├─ embeddings.py → A2 endpoint (BAAI/bge-base-en-v1.5)
    │   │   ├─ Embed all 8,674 chunks
    │   │   └─ Save vectors to FAISS
    │   │
    │   └─ Save FAISS index to disk (index/section_recursive/)
    │
    └─→ Runtime Pipeline:
        │
        User Query
            │
            ├─→ orchestrator.py (LLM intent routing)
            │   ├─ Classify: CROPS | ITEMS | FRIENDSHIP | UNKNOWN | OFF_TOPIC
            │   └─ Route to appropriate agent
            │
            ├─→ Agent Selection
            │   ├─ CropPlanner
            │   ├─ ItemFinder
            │   ├─ FriendshipFinder
            │   └─ DefaultAgent
            │
            ├─→ retriever.py (FAISS semantic search)
            │   ├─ Embed query using same embeddings
            │   └─ Find top-k similar chunks from index
            │
            ├─→ llm.py (Qwen3-30B generation)
            │   ├─ Augment query with retrieved context
            │   └─ Generate grounded answer
            │
            └─→ app.py (FastAPI)
                └─ Return: answer + sources + intent + confidence
```

**Key Technologies:**
- **Embeddings**: BAAI/bge-base-en-v1.5 (A2 endpoint)
- **Vector Store**: FAISS (8,674 chunks)
- **LLM**: Qwen3-30B (final project endpoint)
- **Framework**: FastAPI + LangChain
- **Intent Routing**: LLM-based classification

## Repository Structure

```text
Stardew_Valley_RAG/
├── README.md                       # Project overview (this file)
├── SETUP.md                        # Installation & running instructions
├── ARCHITECTURE.png                # Visual system diagram
├── .env                            # Configuration (not committed)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/                        # Raw scraped wiki sections
│   ├── interim/                    # Page-level aggregations
│   └── processed/
│       └── stardew_wiki_sections.jsonl   # 8,674 clean wiki chunks (RAG input)
│
├── src2/                           # Main RAG implementation
│   ├── app.py                      # FastAPI server + web UI
│   ├── orchestrator.py             # LLM-based intent routing
│   ├── agents.py                   # 4 domain-specific agents
│   ├── retriever.py                # FAISS vector search
│   ├── llm.py                      # Qwen3 LLM client
│   ├── embeddings.py               # Embedding service (A2 endpoint)
│   ├── chunker.py                  # Document chunking
│   ├── build_index.py              # Build FAISS index
│   ├── index.html                  # Stardew Valley themed UI
│   ├── index/                      # FAISS index (generated, not committed)
│   └── tests/                      # 200+ unit & integration tests
│
├── src/                            # Original scaffold (legacy)
├── docs/                           # Documentation
├── notebooks/                      # Exploration notebooks
└── tests/                          # Top-level tests
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

## Chatbot Architecture

See [ARCHITECTURE.png](ARCHITECTURE.png) for visual system diagram.

The system uses a **multi-agent architecture** where queries are routed to specialized agents based on intent classification:

### Key Components

**1. Orchestrator** (`src2/orchestrator.py`)
- LLM-based intent classification
- Returns: Intent type, confidence score, probabilities for all 5 categories
- Handles ambiguous queries transparently

**2. Agents** (`src2/agents.py`)
- **CropPlanner**: Farming, crops, seasons, profitability
- **ItemFinder**: Items, tools, resources, crafting
- **FriendshipFinder**: Villagers, romance, gifts, marriage
- **DefaultAgent**: General Stardew Valley knowledge
- Each retrieves wiki chunks → Augments with context → Generates answer

**3. Retriever** (`src2/retriever.py`)
- FAISS vector store with 8,674 wiki chunks
- Semantic search using sentence-transformers embeddings
- Returns most similar chunks with relevance scores

**4. LLM Client** (`src2/llm.py`)
- OpenAI-compatible wrapper for Qwen3-30B
- Handles retries, timeouts, errors
- Configurable temperature

**5. FastAPI Server** (`src2/app.py`)
- `/` — Web chat UI
- `/chat` — Multi-agent RAG endpoint
- `/health` — System status

### Intent Categories

| Intent | Agent | Example |
|--------|-------|---------|
| CROPS | CropPlanner | "What crops should I plant in spring?" |
| ITEMS | ItemFinder | "Where can I find copper ore?" |
| FRIENDSHIP | FriendshipFinder | "How do I marry Abigail?" |
| UNKNOWN | DefaultAgent | "Tell me about Stardew Valley" |
| OFF_TOPIC | Rejected | "What's the weather today?" |

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

## Setup & Installation

**For complete step-by-step installation and running instructions, see [SETUP.md](SETUP.md).**

This includes prerequisites, all 7 installation steps, running tests, troubleshooting, and API reference.