import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointIdsList, PointStruct, VectorParams

from client.vector_db import QdrantVectorDBClient
from client.storage import S3Uploader
from client.tryon import GeminiTryOnClient
from client.whatapp import WhatsAppClient, extract_all_inbound
from client.llm import LLMClient
from config import get_settings
from embedding import EmbeddingModel
from messagehandler.messagehandler import MessageHandler
from system import OutboundMessage

logger = logging.getLogger(__name__)
settings = get_settings()

# Ensure app INFO logs show when running under uvicorn (only if nothing configured yet).
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
else:
    logging.getLogger(__name__).setLevel(logging.INFO)


def _verify_whatsapp_signature(raw_body: bytes, header: Optional[str], app_secret: Optional[str]) -> bool:
    secret = (app_secret or "").strip()
    if not secret:
        return True
    sig = (header or "").strip()
    if not sig:
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), msg=raw_body, digestmod=hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.lower(), sig.lower())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _app.state.embedder = EmbeddingModel(
        hf_model_name=settings.embedding_model_name,
        max_length=settings.embedding_max_length,
        hf_token=settings.hf_token,
    )
    _app.state.embedder.load_model(model_dir=settings.hf_model_dir)
    tok, pid = settings.whatsapp_access_token, (settings.whatsapp_phone_number_id or "").strip()
    _app.state.whatsapp = (
        WhatsAppClient(tok, pid, settings.whatsapp_graph_api_version) if tok and pid else None
    )
    _app.state.llm = LLMClient()
    _app.state.vector_db = QdrantVectorDBClient(
        _app.state.client,
        _app.state.embedder,
        settings.default_collection,
    )
    _app.state.s3 = S3Uploader(
        bucket=settings.s3_bucket,
        region=settings.s3_region,
        prefix=settings.s3_user_prefix,
    )
    _app.state.tryon = GeminiTryOnClient(model=settings.gemini_tryon_model)
    _app.state.message_handler = (
        MessageHandler(
            _app.state.whatsapp,
            _app.state.llm,
            _app.state.vector_db,
            _app.state.s3,
            _app.state.tryon,
        )
        if _app.state.whatsapp
        else None
    )
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


class ListProductsRequest(BaseModel):
    collection_name: str = settings.default_collection
    limit: int = 100
    offset: Optional[int] = None


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


@app.get("/webhooks/whatsapp", response_class=PlainTextResponse)
def verify_whatsapp_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
) -> str:
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return hub_challenge
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhooks/whatsapp")
async def receive_whatsapp_webhook(request: Request) -> Dict[str, Any]:
    raw_body = await request.body()
    raw_txt = raw_body.decode("utf-8", errors="replace")
    print(f"[whatsapp] webhook raw ({len(raw_body)} bytes): {raw_txt}", flush=True)
    logger.info("WhatsApp webhook raw: %s", raw_txt)

    sig = request.headers.get("x-hub-signature-256") or request.headers.get("X-Hub-Signature-256")
    if not settings.whatsapp_skip_signature_verify and not _verify_whatsapp_signature(
        raw_body, sig, settings.whatsapp_app_secret
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    try:
        payload = json.loads(raw_txt or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    handler: Optional[MessageHandler] = getattr(request.app.state, "message_handler", None)
    if not handler:
        raise HTTPException(
            status_code=503,
            detail="WhatsApp/LLM not configured: set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID",
        )
    response_body = handler.handle_webhook_payload(payload)
    print(f"[whatsapp] response to Meta: {json.dumps(response_body)}", flush=True)
    return response_body


@app.post("/webhooks/fashn")
async def receive_fashn_webhook(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    logger.info("FASHN webhook payload: %s", json.dumps(payload, ensure_ascii=False))
    handler: Optional[MessageHandler] = getattr(request.app.state, "message_handler", None)
    if not handler:
        raise HTTPException(status_code=503, detail="Message handler unavailable")
    return handler.handle_fashn_webhook_payload(payload)


@app.post("/vectors/insert")
def insert_vectors(request: InsertRequest) -> Dict[str, Any]:
    if not request.points:
        raise HTTPException(status_code=400, detail="points list cannot be empty")
    for idx, point in enumerate(request.points, start=1):
        image_url = point.payload.get("image_url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise HTTPException(
                status_code=400,
                detail=f"points[{idx}] payload.image_url is required for product insert",
            )

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


@app.post("/products/list")
def list_products(request: ListProductsRequest) -> Dict[str, Any]:
    if request.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    client: QdrantClient = app.state.client
    if not client.collection_exists(request.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

    points, next_offset = client.scroll(
        collection_name=request.collection_name,
        limit=request.limit,
        offset=request.offset,
        with_payload=True,
        with_vectors=False,
    )
    return {
        "collection_name": request.collection_name,
        "count": len(points),
        "next_offset": next_offset,
        "products": [{"id": p.id, "payload": p.payload} for p in points],
    }


@app.delete("/products/{point_id}")
def delete_product(point_id: int, collection_name: str = settings.default_collection) -> Dict[str, Any]:
    client: QdrantClient = app.state.client
    if not client.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    op = client.delete(
        collection_name=collection_name,
        points_selector=PointIdsList(points=[point_id]),
    )
    return {"status": "deleted", "collection_name": collection_name, "id": point_id, "operation": str(op.status)}
