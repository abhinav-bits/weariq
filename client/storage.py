from __future__ import annotations

import uuid
from typing import Optional

import boto3


class S3Uploader:
    def __init__(self, bucket: str, region: str, prefix: str = "user-test") -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.strip().strip("/")
        self._s3 = boto3.client("s3", region_name=region)

    def upload_bytes(
        self,
        data: bytes,
        *,
        key_suffix: str,
        content_type: Optional[str] = None,
    ) -> str:
        key = f"{self.prefix}/{key_suffix.lstrip('/')}"
        if not key_suffix:
            key = f"{self.prefix}/{uuid.uuid4().hex}.jpg"
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
