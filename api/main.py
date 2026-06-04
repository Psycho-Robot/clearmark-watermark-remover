"""
api/main.py — FastAPI application.

Routes
──────
POST   /api/v1/upload          Upload file → returns task_id immediately
GET    /api/v1/status/{task_id} Poll Celery task status + progress
GET    /api/v1/download/{key}  Download cleaned file (local storage only)
POST   /api/v1/retry/{task_id} Re-trigger AI track after user flags residual WM
DELETE /api/v1/file/{key}      Manual early deletion
GET    /api/v1/health          Health check
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import (
    MAX_FILE_SIZE_MB,
    FILE_TTL_SECONDS,
    API_HOST,
    API_PORT,
    BASE_DIR,
)

logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ClearMark API",
    description = "Dual-engine PDF watermark removal — Fast Track (PyMuPDF) + AI Track",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve existing frontend if templates dir present
_tpl = BASE_DIR / "templates"
if _tpl.exists():
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(_tpl))

    @app.get("/", include_in_schema=False)
    async def index(request):
        from fastapi import Request
        return templates.TemplateResponse("index.html", {"request": request})


# ─── Allowed MIME types ───────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff"}
MAX_BYTES          = MAX_FILE_SIZE_MB * 1024 * 1024


# ─── Upload ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accept a file upload.  Validates size and type, saves to storage,
    enqueues a Celery task and returns the task_id for polling.
    """
    from storage.manager import storage
    from workers.tasks   import process_document

    # ── Validate ──────────────────────────────────────────────────────────
    filename = file.filename or "upload"
    suffix   = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    data = await file.read()

    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) // 1048576} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
        )

    if len(data) < 64:
        raise HTTPException(status_code=400, detail="File is empty or too small.")

    # ── Store ─────────────────────────────────────────────────────────────
    task_id     = uuid.uuid4().hex
    storage_key = storage().save(data, filename, folder="uploads")

    # ── Enqueue ───────────────────────────────────────────────────────────
    celery_task = process_document.apply_async(
        args   = [task_id, storage_key, filename],
        task_id= task_id,
    )

    logger.info("Queued task %s for file '%s' (%d bytes)", task_id, filename, len(data))

    return {
        "task_id":   task_id,
        "filename":  filename,
        "size_bytes": len(data),
        "message":   "File accepted. Poll /api/v1/status/{task_id} for updates.",
    }


# ─── Status polling ───────────────────────────────────────────────────────────

@app.get("/api/v1/status/{task_id}")
async def get_status(task_id: str) -> Dict[str, Any]:
    """
    Poll the Celery result backend for task progress.

    Returns one of:
      { "status": "PENDING" }
      { "status": "STARTED" }
      { "status": "PROGRESS", "pct": 45, "message": "..." }
      { "status": "SUCCESS",  "download_url": "...", ...result fields }
      { "status": "FAILURE",  "error": "..." }
    """
    from workers.tasks import celery_app as _celery

    result = _celery.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "PENDING", "task_id": task_id}

    if result.state == "STARTED":
        return {"status": "STARTED", "task_id": task_id}

    if result.state == "PROGRESS":
        meta = result.info or {}
        return {
            "status":  "PROGRESS",
            "task_id": task_id,
            "pct":     meta.get("pct",     0),
            "message": meta.get("message", "Processing…"),
        }

    if result.state == "SUCCESS":
        data = result.result or {}
        return {"task_id": task_id, **data}

    if result.state == "FAILURE":
        exc = result.result
        return {
            "status":  "FAILURE",
            "task_id": task_id,
            "error":   str(exc) if exc else "Unknown error",
        }

    # Revoked or unknown
    return {"status": result.state, "task_id": task_id}


# ─── Download ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/download/{folder}/{filename}")
async def download_file(folder: str, filename: str) -> FileResponse:
    """
    Serve a processed file (local storage only).
    For S3 storage the client should use the presigned URL returned in /status.
    """
    from config.settings import STORAGE_BACKEND

    if STORAGE_BACKEND == "s3":
        raise HTTPException(
            status_code=400,
            detail="S3 storage is active. Use the presigned download_url from /status."
        )

    from storage.manager import storage as _storage
    from config.settings  import BASE_DIR

    key  = f"{folder}/{filename}"
    path = BASE_DIR / "storage_root" / key

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found or already deleted.")

    return FileResponse(
        path            = str(path),
        media_type      = "application/pdf",
        filename        = filename,
        headers         = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── Retry with AI track ──────────────────────────────────────────────────────

@app.post("/api/v1/retry/{task_id}")
async def retry_with_ai(task_id: str) -> Dict[str, Any]:
    """
    Called when the user flags that the Fast-Track result still has a watermark.
    Re-queues the same document through the AI engine directly.
    """
    from workers.tasks   import celery_app as _celery, process_document
    from storage.manager import storage

    # Look up the original task result to find the storage key
    result = _celery.AsyncResult(task_id)
    if result.state not in {"SUCCESS", "FAILURE"}:
        raise HTTPException(status_code=409, detail="Original task has not completed yet.")

    data = result.result or {}
    # We need to reload the original file — it was stored with schedule_delete
    # For retry we enqueue a new task with the same storage_key if still available
    original_key = data.get("storage_key") or data.get("output_key")
    if not original_key:
        raise HTTPException(status_code=404, detail="Original file reference not found.")

    new_task_id  = uuid.uuid4().hex
    filename     = data.get("filename", "document.pdf")

    # Force AI track by passing a special marker in the filename
    celery_task = process_document.apply_async(
        args    = [new_task_id, original_key, filename],
        task_id = new_task_id,
        kwargs  = {"force_ai": True},
    )

    return {
        "task_id": new_task_id,
        "message": "Re-queued with AI engine. Poll /api/v1/status/{task_id}.",
    }


# ─── Manual delete ────────────────────────────────────────────────────────────

@app.delete("/api/v1/file/{folder}/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(folder: str, filename: str) -> None:
    """Manually delete a file before the 24-hour TTL expires."""
    from storage.manager import storage
    key = f"{folder}/{filename}"
    try:
        storage().delete(key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Could not delete: {exc}")


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    from workers.tasks import celery_app as _celery
    from config.settings import STORAGE_BACKEND, USE_AI_INPAINTING

    # Ping Celery broker
    try:
        _celery.control.ping(timeout=1)
        broker_ok = True
    except Exception:
        broker_ok = False

    return {
        "status":           "ok",
        "broker_connected": broker_ok,
        "storage_backend":  STORAGE_BACKEND,
        "ai_inpainting":    USE_AI_INPAINTING,
        "max_file_mb":      MAX_FILE_SIZE_MB,
        "file_ttl_hours":   FILE_TTL_SECONDS // 3600,
    }


# ─── Startup: register APScheduler cleanup job ────────────────────────────────

@app.on_event("startup")
async def start_cleanup_scheduler() -> None:
    from config.settings import STORAGE_BACKEND
    if STORAGE_BACKEND != "local":
        return  # S3 handles its own lifecycle
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from storage.manager import storage

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            lambda: storage().cleanup_expired(),
            trigger  = "interval",
            hours    = 1,
            id       = "cleanup_expired_files",
            replace_existing = True,
        )
        scheduler.start()
        logger.info("APScheduler cleanup job started (runs every 1 hour)")
    except Exception as exc:
        logger.warning("Could not start APScheduler: %s", exc)


# ─── Dev server entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = True,
        workers = 1,
        log_level = "info",
    )
