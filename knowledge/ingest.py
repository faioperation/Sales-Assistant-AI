"""
Knowledge base ingestion.
Backend sends document text directly (already extracted from their DB).
"""

import uuid
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

from config import (CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE,
                    CHROMA_COLLECTION, EMBEDDING_MODEL)


# ── Singletons ─────────────────────────────────────────────────
_model      = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
        )
        _collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return _collection


def _chunk(text: str, size: int = 300) -> list[str]:
    """Simple word-based chunking."""
    words  = text.split()
    chunks = [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
    return [c for c in chunks if len(c.split()) > 15]


def ingest_text(text: str, source_name: str) -> dict:
    """
    Process document text end-to-end.

    Args:
        text:        Full document text (backend already extracted)
        source_name: Identifier for this document (used for delete/list)

    Returns:
        {
            "success": bool,
            "source":  str,
            "chunks":  int,
            "error":   str | None
        }
    """
    try:
        if not text or not text.strip():
            return {"success": False, "error": "Empty text",
                    "source": None, "chunks": 0}

        # 1. Chunk
        chunks = _chunk(text)
        if not chunks:
            return {"success": False, "error": "Could not chunk text",
                    "source": None, "chunks": 0}

        # 2. Embed
        model      = _get_model()
        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

        # 3. Store
        collection = _get_collection()
        ids        = [str(uuid.uuid4()) for _ in chunks]
        metadatas  = [{
            "source":      source_name,
            "uploaded_at": datetime.now().isoformat(),
            "chunk_index": i,
        } for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "success": True,
            "source":  source_name,
            "chunks":  len(chunks),
            "error":   None,
        }

    except Exception as e:
        return {
            "success": False,
            "source":  None,
            "chunks":  0,
            "error":   str(e),
        }