from __future__ import annotations

from typing import Any, Dict, List

from qdrant_client import QdrantClient

from embedding import EmbeddingModel


def search_products(
    client: QdrantClient,
    embedder: EmbeddingModel,
    *,
    collection_name: str,
    query_text: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    vector = embedder.generate_embeddings(query_text)
    points = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
    ).points
    return [{"id": p.id, "score": p.score, "payload": p.payload} for p in points]
