"""
embeddings.py — LocalEmbedder + LocalEmbeddingsWrapper
                Matches the exact pattern used in class labs.

Usage
-----
    from embeddings import embeddings   # drop-in for FAISS.from_documents()

Or import the components separately:
    from embeddings import LocalEmbedder, LocalEmbeddingsWrapper, build_embeddings
"""

from __future__ import annotations

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# ── Default model (matches lab) ───────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Lab-pattern embedder ──────────────────────────────────────────────────────

class LocalEmbedder:
    """Thin wrapper around SentenceTransformer — identical to the lab version."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[embeddings] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


class LocalEmbeddingsWrapper(Embeddings):
    """
    LangChain Embeddings wrapper — required by FAISS.from_documents().
    Identical structure to the lab's LocalEmbeddingsWrapper.
    """

    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_documents([text])[0]


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> LocalEmbeddingsWrapper:
    """Convenience factory — returns a ready-to-use LangChain embeddings object."""
    return LocalEmbeddingsWrapper(LocalEmbedder(model_name))


# ── Module-level singleton (matches `embeddings = LocalEmbeddingsWrapper(embedder)` in lab) ──

embeddings = build_embeddings()
