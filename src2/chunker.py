"""
chunker.py — Load processed/raw JSONL and split into LangChain Documents
             using the same splitter patterns from class labs.

Exported helpers
----------------
load_jsonl_documents(path)         -> list[Document]   raw LangChain docs, one per section
chunk_documents(docs, strategy)    -> list[Document]   split docs ready for FAISS

Chunking strategies
-------------------
  "section_recursive"   RecursiveCharacterTextSplitter  (default, best for RAG)
  "section_character"   CharacterTextSplitter
  "section_token"       SentenceTransformersTokenTextSplitter
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# ── Splitter configs (mirrors lab defaults) ───────────────────────────────────

SPLITTER_CONFIGS: dict[str, dict] = {
    "section_recursive": dict(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    ),
    "section_character": dict(
        chunk_size=512,
        chunk_overlap=64,
        separator="\n",
    ),
    "section_token": dict(
        # matches the model used for embeddings
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=256,
        chunk_overlap=30,
    ),
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_jsonl_documents(filepath: str | Path, min_chars: int = 50) -> list[Document]:
    """
    Load processed (or raw) JSONL — one section per line — into LangChain Documents.
    """
    docs: list[Document] = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "").strip()

            # filter 1 — too short
            if len(text) < min_chars:
                skipped += 1
                continue

            # filter 2 — modding/template/binary pages
            page_id = data.get("page_id", "")
            if page_id.startswith("Modding:") or page_id.startswith("Module:"):
                skipped += 1
                continue
            if text.startswith("�PNG"):
                skipped += 1
                continue

            docs.append(Document(
                page_content=f"{data['page_title']} — {data.get('heading', '')}\n{text}",
                metadata={
                    "doc_id":        data["doc_id"],
                    "page_id":       data["page_id"],
                    "page_title":    data["page_title"],
                    "url":           data["url"],
                    "text":          text,
                    "heading":       data.get("heading", ""),
                    "section_index": data.get("section_index", 0),
                    "source":        data.get("source", "Stardew Valley Wiki"),
                },
            ))

    print(f"[chunker] Loaded {len(docs)} documents ({skipped} skipped)")
    return docs


# ── Splitter factory ──────────────────────────────────────────────────────────

def _make_splitter(strategy: str):
    cfg = SPLITTER_CONFIGS.get(strategy)
    if cfg is None:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(SPLITTER_CONFIGS)}"
        )
    if strategy == "section_recursive":
        return RecursiveCharacterTextSplitter(**cfg)
    if strategy == "section_character":
        return CharacterTextSplitter(**cfg)
    if strategy == "section_token":
        return SentenceTransformersTokenTextSplitter(**cfg)
    raise ValueError(strategy)


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    strategy: str = "section_recursive",
) -> list[Document]:
    """
    Split LangChain Documents using the chosen strategy.
    Metadata (page_title, url, heading, …) is propagated to every child chunk.

    Parameters
    ----------
    docs     : output of load_jsonl_documents()
    strategy : one of SPLITTER_CONFIGS keys

    Returns
    -------
    List of LangChain Document objects ready for FAISS.from_documents()
    """
    splitter = _make_splitter(strategy)
    chunks   = splitter.split_documents(docs)

    print(
        f"[chunker] Strategy='{strategy}' → "
        f"{len(docs)} docs → {len(chunks)} chunks"
    )
    return chunks
