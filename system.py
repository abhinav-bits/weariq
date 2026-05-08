from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProductMetadata(BaseModel):
    price: Optional[float] = None
    discount: Optional[float] = None
    size: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    style: Optional[str] = None
    fit: Optional[str] = None
    occasion: Optional[str] = None
    season: Optional[str] = None
    description: Optional[str] = None


class Product(BaseModel):
    id: str
    name: str
    description: str
    image_url: Optional[str] = None
    metadata: Optional[ProductMetadata] = None


class Vendor(BaseModel):
    id: str
    name: str
    phone_number: str
    email: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None


class Customer(BaseModel):
    id: str
    name: str
    phone_number: str
    email: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None


class InboundMessage(BaseModel):
    """
    Parsed webhook message (channel-agnostic shape).

    **WhatsApp Cloud API – inbound image:** webhook `messages[]` with `"type":"image"` includes an
    `image` object. See
    https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#image-messages
    """

    channel: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None
    has_image: bool = False
    channel_context: Dict[str, Any] = Field(default_factory=dict)
    customer: Optional[Customer] = None
    vendor: Optional[Vendor] = None

    message_id: Optional[str] = None
    message_type: Optional[str] = None
    text_body: Optional[str] = None
    timestamp: Optional[int] = None
    sender_id: Optional[str] = None
    reply_route_id: Optional[str] = None
    business_display_label: Optional[str] = None

    image: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "WhatsApp: webhook `image` object — `id` (media id for Graph), `mime_type`, `sha256`, "
            "optional `caption`. Retrieve binary via Media API using `id`."
        ),
    )
    interactive: Optional[Dict[str, Any]] = None

    context: Optional[Dict[str, Any]] = None
    reply_to_message_id: Optional[str] = None
    reply_to_sender_id: Optional[str] = None

    raw_message: Optional[Dict[str, Any]] = None


class OutboundMessage(BaseModel):
    """
    Outbound message mapped to provider APIs.

    **WhatsApp Cloud API – send image:** `type` = `"image"`, and `text` carries:
    - `link`: HTTPS URL to an image (public per Meta requirements), optional `caption`; or
    - `id`: uploaded media id from `POST` media upload.

    See https://developers.facebook.com/docs/whatsapp/cloud-api/guides/send-messages#image-messages
    """

    channel: str = "whatsapp"
    to: str
    type: str
    text: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "WhatsApp: `type=text` → `body`, optional `preview_url`; "
            "`type=image` → `link` or `id`, optional `caption`."
        ),
    )
    customer: Optional[Customer] = None
    vendor: Optional[Vendor] = None
    reply_to_message_id: Optional[str] = None


class MessageClient(ABC):
    """Parse provider webhooks into InboundMessage and send OutboundMessage."""

    @abstractmethod
    def extract_payload(self, payload: Dict[str, Any]) -> InboundMessage: ...

    @abstractmethod
    def extract_all_payloads(self, payload: Dict[str, Any]) -> List[InboundMessage]: ...

    @abstractmethod
    def send_message(self, message: OutboundMessage) -> Dict[str, Any]: ...

    def extract_delivery_statuses(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optional delivery/read receipts etc.; default none."""
        return []


