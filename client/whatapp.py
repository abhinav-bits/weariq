from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib import error, request

from system import Customer, InboundMessage, MessageClient, OutboundMessage, Vendor

logger = logging.getLogger(__name__)


def _reply_context(message: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    ctx = message.get("context")
    if not isinstance(ctx, dict):
        return None, None, None
    mid = ctx.get("id")
    reply_id = mid.strip() if isinstance(mid, str) and mid.strip() else None
    rw = ctx.get("from")
    reply_from = str(rw).strip() if rw else None
    return ctx, reply_id, reply_from


def _one_inbound(value: Dict[str, Any], message: Dict[str, Any], all_msgs: List[Dict[str, Any]]) -> InboundMessage:
    metadata = value.get("metadata") or {}
    contacts = list(value.get("contacts") or [])
    from_wa = str(message.get("from") or "").strip() or None
    mtype = str(message.get("type") or "")
    text_body: str | None = None
    image_data: Optional[Dict[str, Any]] = None
    has_image = False
    if mtype == "text":
        text_body = str((message.get("text") or {}).get("body") or "") or None
    elif mtype == "image":
        img = message.get("image")
        if isinstance(img, dict):
            image_data = dict(img)
            has_image = bool(str(image_data.get("id") or "").strip())
            cap = image_data.get("caption")
            if cap is not None and str(cap).strip():
                text_body = str(cap).strip()
    name = ""
    for c in contacts:
        if str(c.get("wa_id", "")) == from_wa:
            name = str((c.get("profile") or {}).get("name") or "")
            break
    customer = (
        Customer(id=from_wa, name=name or "unknown", phone_number=from_wa)
        if from_wa
        else None
    )
    pid = str(metadata.get("phone_number_id") or "")
    disp = str(metadata.get("display_phone_number") or "")
    vendor = (
        Vendor(id=pid or disp, name=disp or pid or "wa", phone_number=disp or pid)
        if (pid or disp)
        else None
    )
    try:
        timestamp = int(message.get("timestamp")) if message.get("timestamp") is not None else None
    except (TypeError, ValueError):
        timestamp = None

    context, reply_to_mid, reply_to_from = _reply_context(message)

    return InboundMessage(
        channel=str(value.get("messaging_product") or "whatsapp"),
        metadata=dict(metadata) if metadata else None,
        has_image=has_image,
        channel_context={"contacts": contacts, "messages": all_msgs},
        customer=customer,
        vendor=vendor,
        message_id=str(message.get("id") or "") or None,
        message_type=mtype or None,
        text_body=text_body,
        timestamp=timestamp,
        sender_id=from_wa,
        reply_route_id=pid or None,
        business_display_label=disp or None,
        image=image_data,
        context=context,
        reply_to_message_id=reply_to_mid,
        reply_to_sender_id=reply_to_from,
        raw_message=dict(message),
    )


def extract_all_inbound(payload: Dict[str, Any]) -> List[InboundMessage]:
    if payload.get("object") != "whatsapp_business_account":
        return []
    out: List[InboundMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            value = change.get("value") or {}
            raw_messages = list(value.get("messages") or [])
            for msg in raw_messages:
                out.append(_one_inbound(value, msg, raw_messages))
    return out


class WhatsAppClient(MessageClient):
    def __init__(self, access_token: str, phone_number_id: str, api_version: str = "v22.0") -> None:
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self.base_url = f"https://graph.facebook.com/{api_version}"

    def extract_payload(self, payload: Dict[str, Any]) -> InboundMessage:
        items = extract_all_inbound(payload)
        if not items:
            raise ValueError("No WhatsApp messages in payload")
        return items[0]

    def extract_all_payloads(self, payload: Dict[str, Any]) -> List[InboundMessage]:
        return extract_all_inbound(payload)

    def send_message(self, message: OutboundMessage) -> Dict[str, Any]:
        mtype = (message.type or "").lower()
        t = message.text or {}
        graph_body: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": message.to,
        }
        if mtype == "text":
            graph_body["type"] = "text"
            graph_body["text"] = {
                "body": str(t.get("body") or ""),
                "preview_url": bool(t.get("preview_url", False)),
            }
        elif mtype == "image":
            media_id = str(t.get("id") or "").strip()
            link = str(t.get("link") or "").strip()
            if media_id:
                image_obj: Dict[str, Any] = {"id": media_id}
            elif link:
                image_obj = {"link": link}
            else:
                raise ValueError("image outbound requires text.id (media id) or text.link (https URL)")
            cap = t.get("caption")
            if cap is not None and str(cap).strip():
                image_obj["caption"] = str(cap).strip()
            graph_body["type"] = "image"
            graph_body["image"] = image_obj
        else:
            raise ValueError(f"unsupported outbound type: {message.type!r}")

        if message.reply_to_message_id:
            graph_body["context"] = {"message_id": message.reply_to_message_id}

        raw_req = json.dumps(graph_body)
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        logger.info("WhatsApp POST %s body=%s", url, raw_req)
        req = request.Request(
            url=url,
            data=raw_req.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                raw_resp = resp.read().decode("utf-8")
                logger.info("WhatsApp response: %s", raw_resp)
                return json.loads(raw_resp) if raw_resp else {}
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            logger.error("WhatsApp HTTP %s: %s", exc.code, err_body)
            raise RuntimeError(f"WhatsApp API error ({exc.code}): {err_body}") from exc

    def get_media_metadata(self, media_id: str) -> Dict[str, Any]:
        """GET /{media-id} — returns `url`, `mime_type`, etc. https://developers.facebook.com/docs/whatsapp/cloud-api/reference/media"""
        mid = media_id.strip()
        url = f"{self.base_url}/{mid}"
        logger.info("WhatsApp GET media metadata %s", url)
        req = request.Request(
            url=url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                logger.info("WhatsApp media metadata response: %s", raw)
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            logger.error("WhatsApp HTTP %s: %s", exc.code, err_body)
            raise RuntimeError(f"WhatsApp media API error ({exc.code}): {err_body}") from exc

    def download_media_bytes(self, media_id: str) -> tuple[bytes, str]:
        meta = self.get_media_metadata(media_id)
        media_url = str(meta.get("url") or "").strip()
        if not media_url:
            raise RuntimeError("WhatsApp media URL missing")
        req = request.Request(
            url=media_url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                ctype = (
                    resp.headers.get_content_type()
                    if hasattr(resp.headers, "get_content_type")
                    else resp.headers.get("Content-Type")
                )
                mime = (ctype.split(";")[0].strip() if ctype else None) or str(meta.get("mime_type") or "image/jpeg")
                return data, mime
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            logger.error("WhatsApp media download HTTP %s: %s", exc.code, err_body)
            raise RuntimeError(f"WhatsApp media download error ({exc.code}): {err_body}") from exc
