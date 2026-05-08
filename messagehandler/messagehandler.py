from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from client.llm import LLMClient
from client.storage import S3Uploader
from client.tryon import GeminiTryOnClient
from client.vector_db import VectorDBClient
from client.whatapp import WhatsAppClient
from system import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    last_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selected_product_image_url: Optional[str] = None
    awaiting_user_photo: bool = False


class MessageHandler:
    """
    Minimal webhook orchestrator:
    webhook payload -> inbound parse -> LLM text->Product -> send reply on WhatsApp.
    """

    def __init__(
        self,
        whatsapp: WhatsAppClient,
        llm: LLMClient,
        vector_db: VectorDBClient,
        s3: S3Uploader,
        tryon: GeminiTryOnClient,
    ) -> None:
        self.whatsapp = whatsapp
        self.llm = llm
        self.vector_db = vector_db
        self.s3 = s3
        self.tryon = tryon
        self.sessions: Dict[str, UserSession] = {}

    def _send_text(self, inbound: InboundMessage, body: str) -> Dict[str, Any]:
        out = OutboundMessage(
            channel="whatsapp",
            to=inbound.sender_id or "",
            type="text",
            text={"body": body},
            reply_to_message_id=inbound.message_id,
        )
        return self.whatsapp.send_message(out)

    def _send_image(self, inbound: InboundMessage, image_url: str, caption: str) -> Dict[str, Any]:
        out = OutboundMessage(
            channel="whatsapp",
            to=inbound.sender_id or "",
            type="image",
            text={"link": image_url, "caption": caption},
            reply_to_message_id=inbound.message_id,
        )
        return self.whatsapp.send_message(out)

    def _send_text_to(self, to: str, body: str, reply_to_message_id: Optional[str]) -> Dict[str, Any]:
        out = OutboundMessage(
            channel="whatsapp",
            to=to,
            type="text",
            text={"body": body},
            reply_to_message_id=reply_to_message_id,
        )
        return self.whatsapp.send_message(out)

    def _send_image_to(self, to: str, image_url: str, caption: str, reply_to_message_id: Optional[str]) -> Dict[str, Any]:
        out = OutboundMessage(
            channel="whatsapp",
            to=to,
            type="image",
            text={"link": image_url, "caption": caption},
            reply_to_message_id=reply_to_message_id,
        )
        return self.whatsapp.send_message(out)

    def _llm_text_to_replies(self, text: str, session: UserSession) -> List[Dict[str, str]]:
        product = self.llm.process_product_text(text)
        if product is False:
            return [{"type": "text", "body": "Please send a fashion product query."}]
        query_text = self.llm.product_to_search_text(product)
        hits = self.vector_db.search(query_text, limit=3)
        if not hits:
            return [{"type": "text", "body": "No matching products found right now."}]
        replies: List[Dict[str, str]] = []
        choice_map: Dict[str, Dict[str, Any]] = {}
        for i, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            name = payload.get("name") or payload.get("text") or "Product"
            price = payload.get("price")
            body = f"{i}. {name}"
            if price is not None:
                body += f" | Price: {price}"
            image_url = payload.get("image_url")
            if isinstance(image_url, str) and image_url.strip():
                replies.append({"type": "image", "image_url": image_url.strip(), "caption": body})
            else:
                replies.append({"type": "text", "body": body})
            choice_map[str(i)] = payload
        session.last_results = choice_map
        replies.append({"type": "text", "body": "Reply with 1/2/3 to select a product for try-on."})
        return replies

    def handle_webhook_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.whatsapp.extract_all_payloads(payload)
        handled = 0
        replies = []
        for msg in messages:
            if not msg.sender_id:
                continue
            session = self.sessions.setdefault(msg.sender_id, UserSession())

            text_in = (msg.text_body or "").strip()
            if text_in in {"1", "2", "3"} and session.last_results.get(text_in):
                payload_pick = session.last_results[text_in]
                image_url = str(payload_pick.get("image_url") or "").strip()
                if not image_url:
                    replies_to_send = [{"type": "text", "body": "Selected product has no image_url."}]
                else:
                    session.selected_product_image_url = image_url
                    session.awaiting_user_photo = True
                    replies_to_send = [
                        {"type": "text", "body": "Great choice. Now send your full-body photo for try-on."}
                    ]
            elif msg.has_image and session.awaiting_user_photo and session.selected_product_image_url:
                media_id = str((msg.image or {}).get("id") or "").strip()
                if not media_id:
                    replies_to_send = [{"type": "text", "body": "Image missing media id. Please send again."}]
                else:
                    try:
                        raw, mime = self.whatsapp.download_media_bytes(media_id)
                        ext = ".jpg"
                        if "png" in mime.lower():
                            ext = ".png"
                        user_url = self.s3.upload_bytes(
                            raw,
                            key_suffix=f"{msg.sender_id}/{msg.message_id or media_id}{ext}",
                            content_type=mime,
                        )
                        out_bytes, out_mime = self.tryon.try_on(
                            user_url,
                            session.selected_product_image_url,
                        )
                        out_ext = ".jpg"
                        if "png" in out_mime.lower():
                            out_ext = ".png"
                        tryon_url = self.s3.upload_bytes(
                            out_bytes,
                            key_suffix=f"{msg.sender_id}/tryon-{msg.message_id or media_id}{out_ext}",
                            content_type=out_mime,
                        )
                        replies_to_send = [
                            {
                                "type": "image",
                                "image_url": tryon_url,
                                "caption": "Here is your try-on preview.",
                            }
                        ]
                    except Exception as exc:
                        logger.exception("Try-on failed")
                        replies_to_send = [
                            {
                                "type": "text",
                                "body": "Try-on is temporarily unavailable. Please try again in a minute.",
                            }
                        ]
                    session.awaiting_user_photo = False
            else:
                replies_to_send = self._llm_text_to_replies(text_in or "image message", session)

            try:
                for reply in replies_to_send:
                    if reply.get("type") == "image" and reply.get("image_url"):
                        sent = self._send_image(msg, reply["image_url"], reply.get("caption", "Product"))
                    else:
                        sent = self._send_text(msg, reply.get("body", ""))
                    replies.append(sent)
                handled += 1
            except Exception as exc:
                logger.exception("Webhook send failed")
                replies.append({"error": str(exc)})
        return {"ok": True, "parsed_messages": len(messages), "handled": handled, "replies": replies}

    def handle_fashn_webhook_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Gemini mode active; FASHN webhook endpoint kept for compatibility.
        return {"ok": True, "ignored": True, "mode": "gemini"}
