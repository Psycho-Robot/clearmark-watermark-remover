
"""
storage/manager.py — Unified storage layer.

Supports:
  - Local filesystem (default, for dev / small deployments)
  - AWS S3 (set STORAGE_BACKEND=s3)

Both backends expose the same interface so the rest of the codebase
never imports boto3 directly.

Auto-delete is handled by:
  - Local: APScheduler job registered at app startup
  - S3:    Bucket lifecycle rule created automatically on first use
"""
from __future__ import annotations
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import time
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Abstract base ────────────────────────────────────────────────────────────

class BaseStorage(ABC):

    @abstractmethod
    def save(self, data: bytes, filename: str, folder: str = "uploads") -> str:
        """Persist *data* and return an opaque storage key."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return file contents for *key*."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object identified by *key*."""

    @abstractmethod
    def get_download_url(self, key: str, ttl: int = 86400) -> str:
        """Return a URL from which the file can be downloaded."""

    @abstractmethod
    def schedule_delete(self, key: str, ttl_seconds: int) -> None:
        """Schedule automatic deletion after *ttl_seconds*."""


# ─── Local storage ────────────────────────────────────────────────────────────

class LocalStorage(BaseStorage):
    """Stores files in UPLOAD_DIR / OUTPUT_DIR on disk."""

    def __init__(self, base_dir: Path, api_base_url: str, ttl_seconds: int = 86400):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.api_base_url = api_base_url.rstrip("/")
        self.ttl_seconds  = ttl_seconds
        self._lock        = threading.Lock()

    # internal helpers
    def _path(self, key: str) -> Path:
        return self.base_dir / key

    # public interface
    def save(self, data: bytes, filename: str, folder: str = "uploads") -> str:
        folder_path = self.base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        unique    = f"{uuid.uuid4().hex}_{filename}"
        key       = f"{folder}/{unique}"
        full_path = self.base_dir / key
        full_path.write_bytes(data)
        logger.info("LocalStorage.save → %s (%d bytes)", key, len(data))
        return key

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()
            logger.info("LocalStorage.delete → %s", key)

    def get_download_url(self, key: str, ttl: int = 86400) -> str:
        # The FastAPI /download/<key> route serves local files
        return f"{self.api_base_url}/api/v1/download/{key}"

    def schedule_delete(self, key: str, ttl_seconds: int) -> None:
        """Fire-and-forget thread that sleeps then deletes."""
        def _delete_after():
            time.sleep(ttl_seconds)
            try:
                self.delete(key)
            except Exception as exc:
                logger.warning("Auto-delete failed for %s: %s", key, exc)

        t = threading.Thread(target=_delete_after, daemon=True)
        t.start()
        logger.debug("Scheduled auto-delete of %s in %ds", key, ttl_seconds)

    def cleanup_expired(self) -> int:
        """Scan base_dir and remove files older than ttl_seconds. Returns count."""
        now     = time.time()
        removed = 0
        for f in self.base_dir.rglob("*"):
            if f.is_file():
                age = now - f.stat().st_mtime
                if age > self.ttl_seconds:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
        if removed:
            logger.info("cleanup_expired removed %d stale files", removed)
        return removed


# ─── S3 storage ───────────────────────────────────────────────────────────────

class S3Storage(BaseStorage):
    """Stores files in AWS S3 with presigned download URLs."""

    def __init__(
        self,
        bucket: str,
        region: str,
        access_key: str,
        secret_key: str,
        presign_expiry: int = 86400,
        ttl_seconds: int = 86400,
    ):
        import boto3  # lazy import — only required for S3 backend
        self.bucket         = bucket
        self.presign_expiry = presign_expiry
        self.ttl_seconds    = ttl_seconds
        self.s3             = boto3.client(
            "s3",
            region_name          = region,
            aws_access_key_id    = access_key,
            aws_secret_access_key= secret_key,
        )
        self._ensure_bucket_lifecycle()

    def _ensure_bucket_lifecycle(self) -> None:
        """Create a lifecycle rule that deletes objects after FILE_TTL_SECONDS."""
        days = max(1, self.ttl_seconds // 86400)
        rule = {
            "Rules": [{
                "ID":     "clearmark-auto-delete",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Expiration": {"Days": days},
            }]
        }
        try:
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=self.bucket, LifecycleConfiguration=rule
            )
            logger.info("S3 lifecycle rule set: delete after %d day(s)", days)
        except Exception as exc:
            logger.warning("Could not set S3 lifecycle rule: %s", exc)

    def save(self, data: bytes, filename: str, folder: str = "uploads") -> str:
        unique = f"{uuid.uuid4().hex}_{filename}"
        key    = f"{folder}/{unique}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        logger.info("S3Storage.save → s3://%s/%s (%d bytes)", self.bucket, key, len(data))
        return key

    def load(self, key: str) -> bytes:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        logger.info("S3Storage.delete → s3://%s/%s", self.bucket, key)

    def get_download_url(self, key: str, ttl: int = 86400) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params        = {"Bucket": self.bucket, "Key": key},
            ExpiresIn     = ttl,
        )

    def schedule_delete(self, key: str, ttl_seconds: int) -> None:
        # S3 lifecycle handles physical deletion; optionally explicit delete earlier
        def _delete_after():
            time.sleep(ttl_seconds)
            try:
                self.delete(key)
            except Exception as exc:
                logger.warning("S3 explicit delete failed for %s: %s", key, exc)
        threading.Thread(target=_delete_after, daemon=True).start()


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_storage() -> BaseStorage:
    """Return the configured storage backend (singleton-style via module cache)."""
    from config.settings import (
        STORAGE_BACKEND, BASE_DIR, API_BASE_URL, FILE_TTL_SECONDS,
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
        S3_BUCKET, S3_PRESIGN_EXPIRY,
    )

    if STORAGE_BACKEND == "s3":
        return S3Storage(
            bucket        = S3_BUCKET,
            region        = AWS_REGION,
            access_key    = AWS_ACCESS_KEY_ID,
            secret_key    = AWS_SECRET_ACCESS_KEY,
            presign_expiry= S3_PRESIGN_EXPIRY,
            ttl_seconds   = FILE_TTL_SECONDS,
        )

    # default: local
    return LocalStorage(
        base_dir     = BASE_DIR / "storage_root",
        api_base_url = API_BASE_URL,
        ttl_seconds  = FILE_TTL_SECONDS,
    )


# Module-level singleton
_storage: Optional[BaseStorage] = None

def storage() -> BaseStorage:
    global _storage
    if _storage is None:
        _storage = get_storage()
    return _storage
