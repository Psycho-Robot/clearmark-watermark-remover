import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os


import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from config.settings import (
    MAX_FILE_SIZE_MB,
    FILE_TTL_SECONDS,
    API_HOST,
    API_PORT,
    BASE_DIR,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ClearMark API", version="2.0.0")

from starlette.middleware.base import BaseHTTPMiddleware
class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST" and "upload" in str(request.url):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_BYTES:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=413, content={"detail": f"File too large. Max is {MAX_FILE_SIZE_MB} MB."})
        return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff"}
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

_tpl_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(_tpl_dir)) if _tpl_dir.exists() else None


@app.get("/", include_in_schema=False)
async def index(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>ClearMark API running</h1><p>Templates folder not found.</p>")


@app.post("/api/v1/upload", status_code=202)
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    from storage.manager import storage
    from workers.tasks import process_document

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    data = await file.read()

    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max is {MAX_FILE_SIZE_MB} MB.")

    if len(data) < 64:
        raise HTTPException(status_code=400, detail="File is empty or too small.")

    task_id = uuid.uuid4().hex
    storage_key = storage().save(data, filename, folder="uploads")

    process_document.apply_async(
        args=[task_id, storage_key, filename],
        task_id=task_id,
    )

    logger.info("Queued task %s for '%s' (%d bytes)", task_id, filename, len(data))
    return {
        "task_id": task_id,
        "filename": filename,
        "size_bytes": len(data),
        "message": "Accepted. Poll /api/v1/status/{task_id}",
    }


@app.get("/api/v1/status/{task_id}")
async def get_status(task_id: str) -> Dict[str, Any]:
    from workers.tasks import celery_app as _celery
    result = _celery.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "PENDING", "task_id": task_id}
    if result.state == "STARTED":
        return {"status": "STARTED", "task_id": task_id}
    if result.state == "PROGRESS":
        meta = result.info or {}
        return {"status": "PROGRESS", "task_id": task_id,
                "pct": meta.get("pct", 0), "message": meta.get("message", "Processing...")}
    if result.state == "SUCCESS":
        return {"task_id": task_id, **(result.result or {})}
    if result.state == "FAILURE":
        return {"status": "FAILURE", "task_id": task_id, "error": str(result.result or "Unknown error")}
    return {"status": result.state, "task_id": task_id}


@app.get("/api/v1/preview/{folder}/{filename}")
async def preview_pdf(folder: str, filename: str, page: int = 0) -> Dict[str, Any]:
    import base64
    import fitz

    key = f"{folder}/{filename}"
    path = BASE_DIR / "storage_root" / key

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        doc = fitz.open(str(path))
        page_count = len(doc)
        if page >= page_count:
            page = 0
        pg  = doc[page]
        mat = fitz.Matrix(1.8, 1.8)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        png = pix.tobytes("png")
        w, h = pix.width, pix.height
        doc.close()
        return {
            "page": page,
            "page_count": page_count,
            "image_b64": base64.b64encode(png).decode(),
            "width": w,
            "height": h,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}")


@app.get("/api/v1/download/{folder}/{filename}")
async def download_file(folder: str, filename: str) -> FileResponse:
    from config.settings import STORAGE_BACKEND
    if STORAGE_BACKEND == "s3":
        raise HTTPException(status_code=400, detail="Use the presigned download_url from /status.")

    key  = f"{folder}/{filename}"
    path = BASE_DIR / "storage_root" / key

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found or already deleted.")

    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/retry/{task_id}")
async def retry_with_ai(task_id: str) -> Dict[str, Any]:
    from workers.tasks import celery_app as _celery, process_document
    result = _celery.AsyncResult(task_id)
    if result.state not in {"SUCCESS", "FAILURE"}:
        raise HTTPException(status_code=409, detail="Original task has not completed yet.")
    data = result.result or {}
    original_key = data.get("upload_key") or data.get("output_key")
    if not original_key:
        raise HTTPException(status_code=404, detail="Original file not found.")
    new_task_id = uuid.uuid4().hex
    process_document.apply_async(
        args=[new_task_id, original_key, data.get("filename", "document.pdf")],
        task_id=new_task_id,
    )
    return {"task_id": new_task_id, "message": "Re-queued with AI engine."}


@app.delete("/api/v1/file/{folder}/{filename}")
async def delete_file(folder: str, filename: str) -> Dict[str, Any]:
    from storage.manager import storage
    key = f"{folder}/{filename}"
    try:
        storage().delete(key)
        return {"status": "deleted", "key": key}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Could not delete: {exc}")


@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    from workers.tasks import celery_app as _celery
    from config.settings import STORAGE_BACKEND, USE_AI_INPAINTING
    try:
        _celery.control.ping(timeout=1)
        broker_ok = True
    except Exception:
        broker_ok = False
    return {
        "status": "ok",
        "broker_connected": broker_ok,
        "storage_backend": STORAGE_BACKEND,
        "ai_inpainting": USE_AI_INPAINTING,
        "max_file_mb": MAX_FILE_SIZE_MB,
        "file_ttl_hours": FILE_TTL_SECONDS // 3600,
    }


@app.on_event("startup")
async def start_cleanup_scheduler() -> None:
    from config.settings import STORAGE_BACKEND
    if STORAGE_BACKEND != "local":
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from storage.manager import storage
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            lambda: storage().cleanup_expired(),
            trigger="interval", hours=1,
            id="cleanup_expired_files", replace_existing=True,
        )
        scheduler.start()
        logger.info("Cleanup scheduler started")
    except Exception as exc:
        logger.warning("APScheduler not started: %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
