import os
from dataclasses import dataclass
from functools import lru_cache


def _as_int(env_key: str, default: int) -> int:
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_bool(env_key: str, default: bool = False) -> bool:
    v = os.getenv(env_key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    qdrant_host: str
    qdrant_port: int
    default_collection: str
    embedding_model_name: str
    embedding_max_length: int
    embedding_vector_size: int
    hf_token: str | None
    hf_model_dir: str | None
    whatsapp_access_token: str | None
    whatsapp_phone_number_id: str | None
    whatsapp_verify_token: str
    whatsapp_graph_api_version: str
    whatsapp_app_secret: str | None
    whatsapp_skip_signature_verify: bool
    s3_bucket: str
    s3_region: str
    s3_user_prefix: str
    gemini_tryon_model: str
    app_base_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    wa_secret = os.getenv("WHATSAPP_APP_SECRET")
    if wa_secret is not None and not wa_secret.strip():
        wa_secret = None
    return Settings(
        app_name=os.getenv("APP_NAME", "Qdrant FastAPI Service"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=_as_int("QDRANT_PORT", 6333),
        default_collection=os.getenv("QDRANT_COLLECTION", "products_demo"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "ai4bharat/indic-bert"),
        embedding_max_length=_as_int("EMBEDDING_MAX_LENGTH", 128),
        embedding_vector_size=_as_int("EMBEDDING_VECTOR_SIZE", 768),
        hf_token=os.getenv("HF_TOKEN"),
        hf_model_dir=os.getenv("HF_MODEL_DIR"),
        whatsapp_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN"),
        whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID"),
        whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", "dev_verify_token"),
        whatsapp_graph_api_version=os.getenv("WHATSAPP_GRAPH_API_VERSION", "v22.0"),
        whatsapp_app_secret=wa_secret,
        whatsapp_skip_signature_verify=_as_bool("WHATSAPP_SKIP_SIGNATURE_VERIFY", False),
        s3_bucket=os.getenv("S3_BUCKET", "weariq-storage"),
        s3_region=os.getenv("S3_REGION", "ap-south-1"),
        s3_user_prefix=os.getenv("S3_USER_PREFIX", "user-test"),
        gemini_tryon_model=os.getenv("GEMINI_TRYON_MODEL", "gemini-3.1-flash-image-preview"),
        app_base_url=os.getenv("APP_BASE_URL", "").strip(),
    )
