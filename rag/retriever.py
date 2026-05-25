"""
RAG Retriever
=============
Retrieves relevant chunks from ChromaDB (old knowledge)
AND merges admin training data (new knowledge) into one context.

Agents call format_context() as before — no changes needed in agents.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from config import (CHROMA_API_KEY, CHROMA_TENANT,
                    CHROMA_DATABASE, CHROMA_COLLECTION,
                    EMBEDDING_MODEL, TOP_K)
from rag.training_retriever import get_training_context


_client     = None
_collection = None
_model      = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
        )
        _collection = _client.get_collection(name=CHROMA_COLLECTION)
    return _collection


def _embed(text: str) -> list[float]:
    model = _get_model()
    return model.encode([text], normalize_embeddings=True).tolist()[0]


def retrieve(query: str, k: int = TOP_K,
             category: str = None,
             section: str = None) -> list[dict]:
    col   = _get_collection()
    where = None

    if category and section:
        where = {"$and": [
            {"service_category": {"$eq": category}},
            {"section_type":     {"$eq": section}},
        ]}
    elif category:
        where = {"service_category": {"$eq": category}}
    elif section:
        where = {"section_type": {"$eq": section}}

    query_embedding = _embed(query)

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text":   results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", ""),
            "score":  round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


def format_context(chunks: list[dict]) -> str:
    """
    Build final context string for agents.

    Combines:
    1. ChromaDB RAG chunks (existing knowledge base)
    2. Admin training data — prompts + PDF documents (new knowledge)

    Agents receive both automatically on every request.
    """
    sections = []

    # ── 1. ChromaDB knowledge base ───────────────
    if chunks:
        lines = ["=== KNOWLEDGE BASE ==="]
        for i, c in enumerate(chunks, 1):
            lines.append(f"[{i}] (source: {c['source']}, score: {c['score']})")
            lines.append(c["text"])
            lines.append("")
        sections.append("\n".join(lines))
    else:
        sections.append("=== KNOWLEDGE BASE ===\nNo relevant knowledge found.")

    # ── 2. Admin training data ───────────────────
    try:
        training_ctx = get_training_context()
        if training_ctx:
            sections.append(training_ctx)
    except Exception as e:
        print(f"[Retriever] Training context fetch failed (continuing): {e}")

    return "\n\n".join(sections)