# ⚙️ Setup & Installation

Complete step-by-step guide to set up and run the Stardew Valley RAG chatbot from scratch.

## Prerequisites

- **Python 3.8+** (verified with 3.10)
- **pip** or **conda** package manager
- **2GB free disk space** (for FAISS index + data)
- **LLM access**: Qwen3-30B endpoint credentials (student ID)

## Installation & Running Steps

Follow these steps in order. After completing all steps, you'll have the chatbot running and ready to query.

### 1. Clone or Enter Repository

```bash
cd /path/to/Startdew_Valley_RAG
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR on Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file in project root with both endpoints:

```bash
cat > .env << 'EOF'
LLM_BASE_URL=https://rsm-8430-finalproject.bjlkeng.io/v1
LLM_API_KEY=your-student-id
LLM_MODEL=qwen3-30b-a3b-fp8
A2_BASE_URL=https://rsm-8430-a2.bjlkeng.io
EOF
```

Replace `your-student-id` with your actual student ID.

**Endpoint Details:**
- `LLM_BASE_URL`: Final project chat API (qwen3-30b models)
- `A2_BASE_URL`: A2 assignment embeddings endpoint (BAAI/bge-base-en-v1.5)

### 5. Build FAISS Vector Index

```bash
python3 src2/build_index.py --input data/processed/stardew_wiki_sections.jsonl
```

**Runtime:** ~1-2 minutes (one-time only). Uses A2 embeddings endpoint to create the vector search index from wiki chunks.

### 6. Start the Server

```bash
cd src2
uvicorn app:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 7. Access the Chatbot

Open your browser and go to **http://localhost:8000**

You can now ask questions like:
- "What crops should I plant in spring?"
- "Where can I find copper ore?"
- "How do I marry Abigail?"

## Running Tests

In a new terminal (with venv activated):

```bash
# All tests
pytest tests/agent_tests/ -v

# Specific test file
pytest tests/agent_tests/test_orchestrator.py -v

# With coverage report
pytest tests/agent_tests/ --cov=src2 --cov-report=html
```

**Expected:** 200+ tests pass ✅

## Troubleshooting

### Port 8000 Already in Use

Use a different port:
```bash
uvicorn src2/app:app --port 9000 --reload
```

Or find and kill the process using port 8000:
```bash
lsof -i :8000  # Find process ID
kill -9 <PID>  # Kill it
```

### "Index not found" Error

Build the index:
```bash
python3 src2/build_index.py --input data/processed/stardew_wiki_sections.jsonl
```

### LLM Connection Error

1. Verify `.env` file has correct values:
```bash
cat .env
```

2. Check if the LLM endpoint is available:
```bash
curl https://rsm-8430-finalproject.bjlkeng.io/v1/models
```

If it returns an error, the server may be down. Contact your admin.

### "ModuleNotFoundError" When Running Tests

Make sure you're in the project root directory:
```bash
cd /path/to/Startdew_Valley_RAG
pytest tests/agent_tests/ -v
```

### Environment Variables Not Loading

Verify `.env` file exists in project root with correct format:
```bash
cat .env
# Should show:
# LLM_BASE_URL=...
# LLM_API_KEY=...
# LLM_MODEL=...
```

## Quick API Reference

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What crops should I plant in spring?", "top_k": 3}'
```

Response includes: `answer`, `sources`, `intent_type`, `intent_confidence`

### Other Endpoints

- `/health` — System status
- `/docs` — Interactive API docs (Swagger UI)
- `/` — Web chat UI

