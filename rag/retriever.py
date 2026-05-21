import chromadb
from sentence_transformers import SentenceTransformer
from config import (CHROMA_API_KEY, CHROMA_TENANT,
                    CHROMA_DATABASE, CHROMA_COLLECTION,
                    EMBEDDING_MODEL, TOP_K)


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
    if not chunks:
        return "No relevant knowledge found."
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {c['source']}, score: {c['score']})")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)