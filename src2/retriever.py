"""
retriever.py — Retriever backed by a LangChain FAISS vectorstore.

Usage
-----
    from retriever import Retriever

    r = Retriever("index/section_recursive")
    results = r.retrieve("how do I get a fishing rod?", top_k=5)
    for chunk in results:
        print(f"[{chunk.score:.3f}] {chunk.page_title} — {chunk.heading}")
        print(chunk.text[:120])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS

from embeddings import build_embeddings


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    doc_id:     str
    page_id:    str
    page_title: str
    heading:    str
    text:       str
    url:        str
    score:      float    # cosine similarity ∈ [0, 1]; higher = more relevant

    def as_context_block(self) -> str:
        """Formatted block injected into the LLM prompt."""
        return (
            f"### {self.page_title} — {self.heading}\n"
            f"{self.text}\n"
            f"Source: {self.url}"
        )


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Semantic retriever over a LangChain FAISS vectorstore saved by build_index.py.
    """

    def __init__(
        self,
        index_dir:  str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        index_dir = Path(index_dir)
        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index found at '{index_dir}'. "
                "Run build_index.py first."
            )

        print(f"[retriever] Loading index from: {index_dir}")
        self._emb = build_embeddings(model_name)
        self._vs  = FAISS.load_local(
            str(index_dir),
            self._emb,
            allow_dangerous_deserialization=True,   # safe: we wrote this file ourselves
        )
        print(f"[retriever] Ready — {self._vs.index.ntotal} chunks indexed")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return top-k chunks by cosine similarity."""
        # similarity_search_with_relevance_scores returns (Document, score) pairs
        # where score ∈ [0, 1] with 1 = identical
        pairs = self._vs.similarity_search_with_relevance_scores(query, k=top_k)
        return [self._to_chunk(doc, score) for doc, score in pairs]

    def retrieve_with_threshold(
        self, query: str, top_k: int = 10, min_score: float = 0.30
    ) -> list[RetrievedChunk]:
        """Retrieve and filter by minimum similarity score."""
        return [c for c in self.retrieve(query, top_k=top_k) if c.score >= min_score]

    def build_context(self, query: str, top_k: int = 5, min_score: float = 0.25) -> str:
        """Retrieve and format as a single context string for LLM injection."""
        chunks = self.retrieve_with_threshold(query, top_k=top_k, min_score=min_score)
        if not chunks:
            return "No relevant wiki content found."
        return "\n\n".join(c.as_context_block() for c in chunks)

    @property
    def index_size(self) -> int:
        return self._vs.index.ntotal

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_chunk(doc, score: float) -> RetrievedChunk:
        m = doc.metadata
        return RetrievedChunk(
            doc_id=     m.get("doc_id",     ""),
            page_id=    m.get("page_id",    ""),
            page_title= m.get("page_title", ""),
            heading=    m.get("heading",    ""),
            text=       m.get("text", doc.page_content),
            url=        m.get("url",        ""),
            score=      float(score),
        )
