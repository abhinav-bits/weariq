from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from config import get_settings
from embedding import EmbeddingModel

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _app.state.embedder = EmbeddingModel(
        hf_model_name=settings.embedding_model_name,
        max_length=settings.embedding_max_length,
        hf_token=settings.hf_token,
    )
    _app.state.embedder.load_model(model_dir=settings.hf_model_dir)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


class InsertPoint(BaseModel):
    id: Optional[int] = None
    text: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class InsertRequest(BaseModel):
    collection_name: str = settings.default_collection
    points: List[InsertPoint]


class SearchRequest(BaseModel):
    collection_name: str = settings.default_collection
    query_text: str
    limit: int = 5


def ensure_collection(collection_name: str, vector_size: int) -> None:
    client: QdrantClient = app.state.client
    if client.collection_exists(collection_name):
        collection_info = client.get_collection(collection_name)
        vectors = collection_info.config.params.vectors
        existing_size = getattr(vectors, "size", None)
        if existing_size == vector_size:
            return
        # Recreate to keep collection schema aligned with embedding model output size.
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def get_existing_int_ids(collection_name: str) -> Set[int]:
    client: QdrantClient = app.state.client
    ids: Set[int] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            if isinstance(point.id, int):
                ids.add(point.id)
        if offset is None:
            break

    return ids


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/vectors/insert")
def insert_vectors(request: InsertRequest) -> Dict[str, Any]:
    if not request.points:
        raise HTTPException(status_code=400, detail="points list cannot be empty")

    ensure_collection(request.collection_name, settings.embedding_vector_size)

    client: QdrantClient = app.state.client
    embedder: EmbeddingModel = app.state.embedder
    used_ids = get_existing_int_ids(request.collection_name)
    provided_ids = {point.id for point in request.points if point.id is not None}
    if len(provided_ids) != len([point.id for point in request.points if point.id is not None]):
        raise HTTPException(status_code=400, detail="Duplicate ids found in request")

    used_ids.update(provided_ids)
    next_id = max(used_ids) + 1 if used_ids else 1

    def resolve_id(point: InsertPoint) -> int:
        nonlocal next_id
        if point.id is not None:
            return point.id
        while next_id in used_ids:
            next_id += 1
        generated = next_id
        used_ids.add(generated)
        next_id += 1
        return generated

    qdrant_points = [
        PointStruct(
            id=resolve_id(point),
            vector=embedder.generate_embeddings(point.text),
            payload={**point.payload, "text": point.text},
        )
        for point in request.points
    ]
    operation_info = client.upsert(collection_name=request.collection_name, points=qdrant_points)

    return {
        "status": "inserted",
        "collection_name": request.collection_name,
        "points_count": len(request.points),
        "operation": str(operation_info.status),
    }


@app.post("/vectors/search")
def search_vectors(request: SearchRequest) -> Dict[str, Any]:
    if request.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")

    client: QdrantClient = app.state.client
    embedder: EmbeddingModel = app.state.embedder

    if not client.collection_exists(request.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

    query_vector = embedder.generate_embeddings(request.query_text)

    results = client.query_points(
        collection_name=request.collection_name,
        query=query_vector,
        limit=request.limit,
    ).points

    return {
        "collection_name": request.collection_name,
        "count": len(results),
        "results": [{"id": p.id, "score": p.score, "payload": p.payload} for p in results],
    }
