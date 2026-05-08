from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib import error, parse, request

from system import Product, ProductMetadata

IntentResult = Union[Literal[False], Product]


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _load_google_generative_api_key(
    creds_path: str | Path = "creds.txt",
    env_key: str = "GOOGLE_API_KEY",
) -> str:
    for key_name in (env_key, "GEMINI_API_KEY", "LLM_API_KEY"):
        v = os.getenv(key_name)
        if v and v.strip():
            return v.strip()

    path = Path(creds_path)
    if not path.is_file():
        raise FileNotFoundError(f"Google API key not in env and file missing: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        lower = raw.lower()
        if "google" in lower or "gemini" in lower:
            m = re.search(r"=\s*[\"']?([A-Za-z0-9_\-]+)[\"']?", raw)
            if m:
                return m.group(1).strip()
    raise ValueError(f"Could not parse Google/Gemini API key from {path}")


_TEXT_INTENT_INSTRUCTION = """You are a **query validator and refiner** for a **fashion catalog product search**.
Your job: (1) decide if the user message is a real shopping query for apparel or accessories, and (2) if yes, rewrite it into a clean search line and a short human summary for vector search against a merchant catalog.

Output one JSON object only (no markdown):
{
  "is_product_intent": boolean,
  "confidence": number from 0 to 1,
  "search_query": string,
  "description": string,
  "hints": {
    "category": null or string,
    "color": null or string,
    "material": null or string,
    "fit_or_style": null or string
  }
}
Rules:
- Set **is_product_intent** false and **low confidence** for off-topic text (weather, jokes, abuse, tech support, generic hello with no product idea, non-fashion topics).
- Set **true** when the user is trying to find, compare, or describe **clothing, footwear, bags, jewelry, or other fashion / wearable** items—or clearly shop-like requests that belong in a catalog.
- **search_query**: one dense line for semantic / embedding search. Normalize messy phrasing; keep meaningful terms (garment type, color, fabric, fit, occasion, gender if obvious). Prefer the user's language when it is clear.
- **description**: one plain sentence capturing intent (no invented brand names or SKU codes).
- **hints**: only optional structured tags clearly supported by the message; otherwise null. Do not guess specifics."""


_IMAGE_PRODUCT_INSTRUCTION = """You **validate and refine** what is shown for a **fashion catalog image search**: treat the photo as a catalog query—extract how a shopper would describe this item so we can match similar products.

Output one JSON object only (no markdown):
{
  "search_query": string,
  "title": string,
  "description": string,
  "hints": {
    "category": null or string,
    "color": null or string,
    "material": null or string,
    "fit_or_style": null or string
  }
}
- **search_query**: single dense line for embedding / vector search (garment type, key visuals, color, obvious material or silhouette).
- **title**: very short catalog-style label (a few words).
- **description**: one sentence: what the item looks like and how you'd search for it.
- **hints**: only what you clearly see; null if unknown. Never invent brand names or labels unreadable in the image."""


class LLMClient:
    """Fashion catalog search: validate & refine text queries, extract/refine image queries (Gemini REST)."""

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        creds_path: str | Path = "creds.txt",
    ) -> None:
        self.api_base = (api_base or os.getenv("LLM_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
        self.model = (model or os.getenv("LLM_MODEL") or os.getenv("GOOGLE_PRODUCT_MODEL") or "gemini-3-flash-preview").strip()
        self._api_key_override = api_key.strip() if api_key and api_key.strip() else None
        self._creds_path = Path(creds_path)
        self.timeout = _env_int("LLM_TIMEOUT", 120)
        self.temperature = _env_float("LLM_TEMPERATURE", 0.2)
        self.max_tokens = _env_int("LLM_MAX_TOKENS", 4096)
        self.top_p = _env_float("LLM_TOP_P", 1.0)

    def get_refreshed_api_key(self) -> str:
        if self._api_key_override:
            return self._api_key_override
        k = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if k and k.strip():
            return k.strip()
        return _load_google_generative_api_key(creds_path=self._creds_path, env_key="GOOGLE_API_KEY")

    def _url(self) -> str:
        q = parse.urlencode({"key": self.get_refreshed_api_key()})
        return f"{self.api_base}/v1beta/models/{self.model}:generateContent?{q}"

    @staticmethod
    def _extract_response_text(generate_response: Dict[str, Any]) -> str:
        candidates = generate_response.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

    def _generate_json(self, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": self.top_p,
                "responseMimeType": "application/json",
            },
        }
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            self._url(),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw_text = response.read().decode("utf-8")
                raw = json.loads(raw_text) if raw_text else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"LLM generateContent error ({exc.code}): {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM API connection failed: {exc}") from exc

        text = self._extract_response_text(raw)
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _fetch_image_inline(url: str, timeout: int) -> Dict[str, Any]:
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                ctype = (
                    resp.headers.get_content_type()
                    if hasattr(resp.headers, "get_content_type")
                    else resp.headers.get("Content-Type")
                )
        except error.HTTPError as exc:
            raise RuntimeError(f"Failed to download image: HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to download image: {exc}") from exc
        mime = ctype.split(";")[0].strip() if ctype else None
        if not mime or mime == "application/octet-stream":
            guess, _ = mimetypes.guess_type(url)
            mime = guess or "image/jpeg"
        b64 = base64.standard_b64encode(data).decode("ascii")
        return {"inline_data": {"mime_type": mime, "data": b64}}

    @staticmethod
    def _hints_to_metadata(hints: Any) -> Optional[ProductMetadata]:
        if not isinstance(hints, dict):
            return None
        material = hints.get("material")
        if material is None and hints.get("material_or_fabric") is not None:
            material = hints.get("material_or_fabric")
        style = hints.get("style")
        if style is None and hints.get("fit_or_style") is not None:
            style = hints.get("fit_or_style")
        cleaned: Dict[str, Any] = {}
        for key, val in (
            ("color", hints.get("color")),
            ("material", material),
            ("style", style),
            ("fit", hints.get("fit")),
        ):
            if val is not None and str(val).strip():
                cleaned[key] = str(val).strip()
        if not cleaned:
            return None
        try:
            return ProductMetadata.model_validate(cleaned)
        except Exception:
            return None

    def process_product_text(self, text: str, *, min_confidence: float = 0.35) -> IntentResult:
        """
        If the message is not product-shopping intent, return ``False``.
        Otherwise return a ``Product`` (``id`` = ``llm-text``) built from a short
        ``search_query`` + ``description`` and sparse ``hints`` mapped into ``ProductMetadata``.
        Use :meth:`product_to_search_text` for the Qdrant embedding string.
        """
        stripped = (text or "").strip()
        if not stripped:
            return False

        parts: List[Dict[str, Any]] = [
            {"text": _TEXT_INTENT_INSTRUCTION},
            {"text": f"User message:\n{stripped}"},
        ]
        data = self._generate_json(parts)
        if not data.get("is_product_intent"):
            return False
        conf = data.get("confidence")
        try:
            if conf is not None and float(conf) < min_confidence:
                return False
        except (TypeError, ValueError):
            pass

        hints_raw = data.get("hints")
        category = ""
        if isinstance(hints_raw, dict) and hints_raw.get("category"):
            category = str(hints_raw["category"]).strip()

        search_q = str(data.get("search_query") or "").strip()
        desc = str(data.get("description") or "").strip()
        embed_line = search_q or desc
        if not embed_line:
            return False

        if category:
            name = (
                f"{category} — {search_q}"[:200]
                if search_q
                else f"{category} — {desc}"[:200]
            )
        else:
            name = (search_q or desc)[:120]
        if not desc:
            desc = search_q if search_q else name

        meta = self._hints_to_metadata(hints_raw)
        return Product(
            id="llm-text",
            name=name.strip(),
            description=desc.strip(),
            image_url=None,
            metadata=meta,
        )

    def image_to_product(
        self,
        image_url: str,
        *,
        vendor_text: Optional[str] = None,
    ) -> Product:
        """
        Describe the product in the image as a ``Product`` (``id`` = ``llm-image``).
        Use ``description`` and ``name`` (plus ``metadata``) to form text for Qdrant embedding.
        """
        parts: List[Dict[str, Any]] = [{"text": _IMAGE_PRODUCT_INSTRUCTION}]
        if vendor_text and vendor_text.strip():
            parts.append({"text": f"Vendor note:\n{vendor_text.strip()}"})
        parts.append(self._fetch_image_inline(image_url, self.timeout))

        data = self._generate_json(parts)
        search_q = str(data.get("search_query") or "").strip()
        name = str(data.get("title") or "").strip()
        desc = str(data.get("description") or "").strip()
        if not name and not desc and not search_q:
            raise RuntimeError("LLM returned empty product for image")

        if not name:
            name = (search_q or desc)[:120]
        if not desc:
            desc = search_q or name

        meta = self._hints_to_metadata(data.get("hints"))
        return Product(
            id="llm-image",
            name=name,
            description=desc,
            image_url=image_url,
            metadata=meta,
        )

    def product_to_search_text(self, product: Product) -> str:
        """Flatten ``Product`` (+ metadata) into one string for the embedding model."""
        chunks: List[str] = [product.name, product.description]
        m = product.metadata
        if m:
            for key in ("color", "material", "style", "fit", "occasion", "season", "size", "description"):
                v = getattr(m, key, None)
                if v:
                    chunks.append(str(v))
        return re.sub(r"\s+", " ", " ".join(chunks)).strip()
