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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
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
    )
