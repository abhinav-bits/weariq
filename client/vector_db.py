from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from qdrant_client import QdrantClient

from embedding import EmbeddingModel
from qdrant import search_products


class VectorDBClient(ABC):
    @abstractmethod
    def search(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]: ...


class QdrantVectorDBClient(VectorDBClient):
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder: EmbeddingModel,
        collection_name: str,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def search(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        return search_products(
            self.qdrant_client,
            self.embedder,
            collection_name=self.collection_name,
            query_text=query_text,
            limit=limit,
        )
