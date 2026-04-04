"""
build_index.py — Build and save a LangChain FAISS vector store from JSONL.
                 Mirrors the FAISS.from_documents() pattern from class labs.

Usage
-----
    python build_index.py --input data/processed.jsonl --strategy section_recursive

    # Try all strategies and compare chunk counts:
    python build_index.py --input data/processed.jsonl --all-strategies

Outputs (written to --out-dir, default ./index/<strategy>/)
    faiss.index + index.pkl   — LangChain FAISS save format
    index_info.json           — Build provenance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS

from chunker import load_jsonl_documents, chunk_documents, SPLITTER_CONFIGS
from embeddings import build_embeddings

# ── Builder ───────────────────────────────────────────────────────────────────

def build_index(
    input_path: str,
    strategy:   str  = "section_recursive",
    out_dir:    str  = "index",
    model_name: str  = "sentence-transformers/all-MiniLM-L6-v2",
) -> FAISS:
    """
    Load JSONL → chunk with LangChain splitter → embed → save FAISS index.

    Returns the FAISS vectorstore object (also saved to disk).
    """
    out = Path(out_dir) / strategy
    out.mkdir(parents=True, exist_ok=True)

    # ── Load & chunk ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BUILDING INDEX  strategy='{strategy}'")
    print(f"{'='*60}")

    docs      = load_jsonl_documents(input_path)
    vs_chunks = chunk_documents(docs, strategy=strategy)

    print(f"\nCREATING FAISS VECTOR STORE")
    print(f"Strategy '{strategy}' → {len(vs_chunks)} chunks")

    # ── Embed & build (lab pattern: FAISS.from_documents) ─────────────────────
    emb = build_embeddings(model_name)
    t0  = time.time()
    vs  = FAISS.from_documents(vs_chunks, emb)
    elapsed = time.time() - t0

    print(f"Vector store built in {elapsed:.1f}s")

    # ── Persist ───────────────────────────────────────────────────────────────
    vs.save_local(str(out))
    print(f"Saved → {out}/")

    # ── Write build info ──────────────────────────────────────────────────────
    info = {
        "strategy":      strategy,
        "input":         str(input_path),
        "chunk_count":   len(vs_chunks),
        "model":         model_name,
        "built_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
    }
    (out / "index_info.json").write_text(json.dumps(info, indent=2))

    return vs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LangChain FAISS index from Stardew Valley JSONL"
    )
    parser.add_argument("--input",         required=True,
                        help="Path to processed.jsonl (or raw.jsonl — same schema)")
    parser.add_argument("--strategy",      default="section_recursive",
                        choices=list(SPLITTER_CONFIGS),
                        help="Chunking strategy")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Build an index for every available strategy")
    parser.add_argument("--model",         default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out-dir",       default="index")
    args = parser.parse_args()

    strategies = list(SPLITTER_CONFIGS) if args.all_strategies else [args.strategy]
    for strat in strategies:
        build_index(
            input_path=args.input,
            strategy=strat,
            out_dir=args.out_dir,
            model_name=args.model,
        )

    print("\nDone. Index directories:")
    for strat in strategies:
        print(f"  {args.out_dir}/{strat}/")


if __name__ == "__main__":
    main()
