from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib import error, parse, request

logger = logging.getLogger(__name__)


TRYON_PROMPT = """Generate a realistic virtual try-on image.
Use IMAGE_1 as the person photo and IMAGE_2 as the garment photo.
Keep person identity, face, body proportions, pose, camera angle, and background unchanged.
Transfer only the garment to the person naturally with realistic fit and cloth drape.
Preserve garment details: color, texture, pattern, sleeves, neckline, and length.
Do not add extra accessories, text, logos, or visual artifacts.
Output one photorealistic final image only."""


def _load_api_key(creds_path: str | Path = "creds.txt") -> str:
    for key_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY"):
        v = os.getenv(key_name)
        if v and v.strip():
            return v.strip()
    path = Path(creds_path)
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
    raise ValueError("Could not load Gemini API key")


def _url_to_inline(image_url: str, timeout: int) -> Dict[str, Any]:
    req = request.Request(image_url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = (
            resp.headers.get_content_type()
            if hasattr(resp.headers, "get_content_type")
            else resp.headers.get("Content-Type")
        )
    mime = (ctype.split(";")[0].strip() if ctype else None) or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode("ascii")}}


class GeminiTryOnClient:
    def __init__(
        self,
        *,
        api_base: str = "https://generativelanguage.googleapis.com",
        model: str = "gemini-2.0-flash-preview-image-generation",
        creds_path: str = "creds.txt",
        timeout_seconds: int = 120,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.creds_path = creds_path
        self.timeout_seconds = timeout_seconds

    def _url(self, model: str) -> str:
        key = parse.urlencode({"key": _load_api_key(self.creds_path)})
        return f"{self.api_base}/v1beta/models/{model}:generateContent?{key}"

    def _candidate_models(self) -> List[str]:
        env_models = os.getenv("GEMINI_TRYON_MODELS", "").strip()
        models: List[str] = []
        if env_models:
            models.extend([m.strip() for m in env_models.split(",") if m.strip()])
        if self.model and self.model not in models:
            models.insert(0, self.model)
        for m in (
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
        ):
            if m not in models:
                models.append(m)
        return models

    def _list_models(self) -> List[str]:
        key = parse.urlencode({"key": _load_api_key(self.creds_path)})
        url = f"{self.api_base}/v1beta/models?{key}"
        req = request.Request(url=url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
        except Exception:
            return []

        preferred: List[str] = []
        others: List[str] = []
        for item in payload.get("models", []) or []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            name = str(item.get("name") or "")
            # API returns names like "models/gemini-1.5-flash"
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if not name:
                continue
            # Prefer likely multimodal families first.
            lname = name.lower()
            if any(k in lname for k in ("image", "vision")):
                preferred.append(name)
            else:
                others.append(name)
        # Keep flash-family early, but still try any generateContent model.
        others.sort(key=lambda x: (0 if "flash" in x.lower() else 1, x))
        return preferred + others

    @staticmethod
    def _extract_inline_image(payload: Dict[str, Any]) -> Tuple[bytes, str] | None:
        for c in payload.get("candidates", []) or []:
            for p in ((c.get("content") or {}).get("parts") or []):
                inline = p.get("inlineData") or p.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    mime = str(inline.get("mimeType") or inline.get("mime_type") or "image/png")
                    return base64.b64decode(inline["data"]), mime
        return None

    def _request_tryon(self, model: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = request.Request(
            self._url(model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def try_on(self, user_image_url: str, product_image_url: str) -> Tuple[bytes, str]:
        body_base = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": TRYON_PROMPT},
                        _url_to_inline(user_image_url, self.timeout_seconds),
                        _url_to_inline(product_image_url, self.timeout_seconds),
                    ],
                }
            ],
        }
        tried: List[str] = []
        discovered_added = False
        last_error: str = ""
        queue: List[str] = self._candidate_models()
        seen: Set[str] = set()
        idx = 0
        while idx < len(queue):
            model = queue[idx]
            idx += 1
            if model in seen:
                continue
            seen.add(model)
            tried.append(model)
            body_variants = [
                {**body_base, "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
                body_base,
            ]
            for variant_idx, body in enumerate(body_variants, start=1):
                try:
                    payload = self._request_tryon(model, body)
                except error.HTTPError as exc:
                    detail = exc.read().decode("utf-8")
                    last_error = f"HTTP {exc.code}: {detail}"
                    logger.warning(
                        "Gemini try-on request failed model=%s variant=%s code=%s detail=%s",
                        model,
                        variant_idx,
                        exc.code,
                        detail[:1200],
                    )
                    if exc.code == 404:
                        # On first model-not-found, discover account-available models and enqueue.
                        if not discovered_added:
                            discovered = self._list_models()
                            for m in discovered:
                                if m not in seen:
                                    queue.append(m)
                            discovered_added = True
                        break
                    # 400 can be model/request-shape mismatch; try next variant/model
                    continue
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning("Gemini try-on request exception model=%s variant=%s err=%s", model, variant_idx, exc)
                    continue

                img = self._extract_inline_image(payload)
                if img is not None:
                    return img
                last_error = "no image output"
                logger.info("Gemini try-on no inline image model=%s variant=%s", model, variant_idx)

        raise RuntimeError(
            f"No supported Gemini try-on model available for this API key "
            f"(tried: {', '.join(tried)}; {last_error})"
        )
