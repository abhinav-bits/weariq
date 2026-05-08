from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib import error, parse, request


def load_fashn_api_key(
    creds_path: str | Path = "creds.txt",
    env_key: str = "FASHN_API_KEY",
) -> str:
    """
    Resolve FASHN API key: env FASHN_API_KEY wins, else parse creds.txt.
    Supports lines like: fashn.ai = "fa-..." or FASHN_API_KEY=fa-...
    """
    from_env = os.getenv(env_key)
    if from_env and from_env.strip():
        return from_env.strip()

    path = Path(creds_path)
    if not path.is_file():
        raise FileNotFoundError(f"FASHN API key not in env {env_key} and creds file missing: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "fashn" in raw.lower() or "FASHN" in raw:
            m = re.search(r"=\s*[\"']?([^\s\"']+)[\"']?", raw)
            if m:
                return m.group(1).strip()
        m = re.match(r"^\s*FASHN_API_KEY\s*=\s*[\"']?([^\"'\s]+)[\"']?\s*$", raw)
        if m:
            return m.group(1).strip()

    raise ValueError(f"Could not parse FASHN API key from {path}")


class FashnClient:
    """
    FASHN API client: POST /v1/run and GET /v1/status/{id}.
    Docs pattern: https://api.fashn.ai/v1/run + polling /v1/status/{id}
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.fashn.ai",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("FASHN API key is empty")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "weariq-fashn-client/1.0",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = response.read().decode("utf-8")
                headers_out = dict(response.headers.items())
                parsed: Dict[str, Any] = json.loads(data) if data else {}
                if "x-fashn-credits-used" in headers_out:
                    parsed["_headers"] = {"x-fashn-credits-used": headers_out.get("x-fashn-credits-used")}
                return parsed
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"FASHN API error ({exc.code}): {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"FASHN API connection failed: {exc}") from exc

    def run(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        *,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/run"
        if webhook_url and webhook_url.strip():
            qs = parse.urlencode({"webhook_url": webhook_url.strip()})
            url = f"{url}?{qs}"
        return self._request_json(
            "POST",
            url,
            {"model_name": model_name, "inputs": inputs},
        )

    def status(self, prediction_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/status/{prediction_id}"
        return self._request_json("GET", url, None)

    def run_tryon_max(
        self,
        product_image: str,
        model_image: str,
        *,
        prompt: Optional[str] = None,
        resolution: Optional[Literal["1k", "2k", "4k"]] = None,
        generation_mode: Optional[Literal["balanced", "quality"]] = None,
        seed: Optional[int] = None,
        num_images: Optional[int] = None,
        output_format: Optional[Literal["png", "jpeg"]] = None,
        return_base64: Optional[bool] = None,
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "product_image": product_image,
            "model_image": model_image,
        }
        if prompt is not None:
            inputs["prompt"] = prompt
        if resolution is not None:
            inputs["resolution"] = resolution
        if generation_mode is not None:
            inputs["generation_mode"] = generation_mode
        if seed is not None:
            inputs["seed"] = seed
        if num_images is not None:
            inputs["num_images"] = num_images
        if output_format is not None:
            inputs["output_format"] = output_format
        if return_base64 is not None:
            inputs["return_base64"] = return_base64
        return self.run("tryon-max", inputs)

    def run_product_to_model(
        self,
        *,
        product_image: str,
        model_image: Optional[str] = None,
        prompt: Optional[str] = None,
        output_format: str = "png",
        resolution: Optional[Literal["1k", "2k", "4k"]] = None,
        generation_mode: Optional[Literal["fast", "balanced", "quality"]] = None,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "product_image": product_image,
            "output_format": output_format,
            "return_base64": False,
        }
        if model_image:
            inputs["model_image"] = model_image
        if prompt:
            inputs["prompt"] = prompt
        if resolution:
            inputs["resolution"] = resolution
        if generation_mode:
            inputs["generation_mode"] = generation_mode
        return self.run("product-to-model", inputs, webhook_url=webhook_url)

    def wait_for_prediction(
        self,
        prediction_id: str,
        *,
        max_wait_seconds: float = 180.0,
        poll_interval_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        import time

        deadline = time.monotonic() + max_wait_seconds
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.status(prediction_id)
            status = last.get("status")
            if status in ("completed", "failed"):
                return last
            time.sleep(poll_interval_seconds)
        return {**last, "status": "timeout", "error": {"message": f"Timed out after {max_wait_seconds}s"}}
