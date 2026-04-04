"""
app.py — FastAPI RAG endpoint for the Stardew Valley conversational agent.

LLM  : qwen3-30b-a3b-fp8  (lab / local OpenAI-compatible endpoint)
Index: LangChain FAISS vectorstore  (built with build_index.py)

Endpoints
---------
    POST /chat      — Retrieve + generate (full RAG)
    POST /retrieve  — Retrieval only; useful for debugging chunk quality
    GET  /health    — Health check

Run
---
    export LLM_API_KEY=<your-student-id>
    export LLM_BASE_URL=https://rsm-8430-lab2.bjlkeng.io/v1
    uvicorn app:app --reload --port 8000

Environment variables
---------------------
    LLM_BASE_URL    — LLM endpoint base URL
    LLM_API_KEY     — API key / student ID
    LLM_MODEL       — model name (default: qwen3-30b-a3b-fp8)
    LLM_REASONING   — enable chain-of-thought (default: true)
    INDEX_DIR       — path to FAISS index directory (default: ./index/section_recursive)
    INDEX_STRATEGY  — which sub-directory to load (default: section_recursive)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retriever import Retriever, RetrievedChunk
from llm import LLMClient, LLMResponse, get_llm_client

# ── Config ─────────────────────────────────────────────────────────────────────

_strategy  = os.getenv("INDEX_STRATEGY", "section_recursive")
INDEX_DIR  = Path(os.getenv("INDEX_DIR",  f"index/{_strategy}"))

DEFAULT_TOP_K     = 3
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """\
You are a knowledgeable and friendly guide for the farming simulation game Stardew Valley.
Answer the player's question using ONLY the wiki context provided in the <wiki_context> block.

Rules:
- Be concise and helpful. Use bullet points for multi-step instructions or item lists.
- Always cite the wiki page by name (e.g. "According to the Fishing page...").
- If the context doesn't fully answer the question, say so — do not invent facts.
- If the question is ambiguous, ask one brief clarifying question.
- Never discuss topics unrelated to Stardew Valley.
- Think carefully through the context before answering to ensure accuracy.
"""

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stardew Valley RAG Agent",
    description="RAG agent grounded in the Stardew Valley Wiki — powered by qwen3-30b-a3b-fp8",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_retriever: Optional[Retriever]  = None
_llm:       Optional[LLMClient] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(INDEX_DIR)
    return _retriever


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = get_llm_client()
    return _llm


# ── Schemas ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query:                str          = Field(..., min_length=1, max_length=1000)
    conversation_history: list[Message] = Field(default_factory=list)
    top_k:                int          = Field(DEFAULT_TOP_K, ge=1, le=20)
    min_score:            float        = Field(0.25, ge=0.0, le=1.0)
    include_reasoning:    bool         = Field(False,
        description="Return the model's chain-of-thought in the response")


class SourceRef(BaseModel):
    page_title: str
    heading:    str
    url:        str
    score:      float


class ChatResponse(BaseModel):
    answer:          str
    sources:         list[SourceRef]
    retrieved_count: int
    reasoning:       Optional[str] = Field(None,
        description="Qwen3 chain-of-thought (only when include_reasoning=true)")
    usage:           dict          = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query:     str   = Field(..., min_length=1)
    top_k:     int   = Field(DEFAULT_TOP_K, ge=1, le=20)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class RetrieveResponse(BaseModel):
    query:  str
    chunks: list[dict]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    r   = get_retriever()
    llm = get_llm()
    return {
        "status":        "ok",
        "index_size":    r.index_size,
        "index_dir":     str(INDEX_DIR),
        "llm_model":     llm.model,
        "llm_reasoning": llm.reasoning,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Retrieval only — inspect what chunks are fetched for a query."""
    chunks = get_retriever().retrieve_with_threshold(
        req.query, top_k=req.top_k, min_score=req.min_score
    )
    return RetrieveResponse(
        query=req.query,
        chunks=[
            {
                "doc_id":     c.doc_id,
                "page_title": c.page_title,
                "heading":    c.heading,
                "text":       c.text,
                "url":        c.url,
                "score":      round(c.score, 4),
            }
            for c in chunks
        ],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full RAG pipeline:
      1. Retrieve relevant wiki chunks from LangChain FAISS index
      2. Format as context
      3. Call qwen3-30b-a3b-fp8 with reasoning + conversation history
      4. Return answer + source citations + optional chain-of-thought
    """
    retriever = get_retriever()
    llm       = get_llm()

    # Step 1 — Retrieve
    chunks: list[RetrievedChunk] = retriever.retrieve_with_threshold(
        req.query, top_k=req.top_k, min_score=req.min_score
    )
    if not chunks:
        return ChatResponse(
            answer="I couldn't find anything in the Stardew Valley Wiki relevant to your question. Could you rephrase, or ask something about the game?",
            sources=[],
            retrieved_count=0,
            usage={},
        )

    # Step 2 — Format context
    context_blocks = "\n\n".join(c.as_context_block() for c in chunks)
    user_message   = (
        f"<wiki_context>\n{context_blocks}\n</wiki_context>\n\n"
        f"Player question: {req.query}"
    )

    # Step 3 — Build message list (cap history)
    history  = req.conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [
        *[{"role": m.role, "content": m.content} for m in history],
        {"role": "user", "content": user_message},
    ]

    # Step 4 — Call Qwen3
    try:
        llm_resp: LLMResponse = llm.complete(messages=messages, system=SYSTEM_PROMPT)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM endpoint error: {e}")

    # Step 5 — Return
    return ChatResponse(
        answer=llm_resp.answer,
        sources=[
            SourceRef(page_title=c.page_title, heading=c.heading,
                      url=c.url, score=round(c.score, 4))
            for c in chunks
        ],
        retrieved_count=len(chunks),
        reasoning=llm_resp.reasoning if req.include_reasoning else None,
        usage={
            "input_tokens":  llm_resp.input_tokens,
            "output_tokens": llm_resp.output_tokens,
            "total_tokens":  llm_resp.total_tokens,
        },
    )
