"""
workers/tasks.py — Celery task definitions.

Task: process_document
  1. Restore file from storage
  2. Triage → pick engine
  3. Run Fast Track or AI Track
  4. Save output to storage
  5. Schedule auto-delete (input + output)
  6. Return structured result

Task status states:
  PENDING   → queued
  STARTED   → worker picked it up
  PROGRESS  → (custom state) mid-processing with % update
  SUCCESS   → done, download URL attached
  FAILURE   → hard error with message
"""
from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from celery import Celery, states
from celery.utils.log import get_task_logger

from config.settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    FILE_TTL_SECONDS,
    UPLOAD_DIR,
    OUTPUT_DIR,
)

logger = get_task_logger(__name__)

# ─── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "clearmark",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    task_track_started       = True,
    task_acks_late           = True,     # re-queue on worker crash
    worker_prefetch_multiplier = 1,      # one task at a time per worker (heavy tasks)
    result_expires           = FILE_TTL_SECONDS,
    timezone                 = "UTC",
)


# ─── Progress helper ───────────────────────────────────────────────────────────

def _update_progress(task, step: int, total: int, message: str) -> None:
    """Push a PROGRESS custom state so the frontend can poll incremental updates."""
    task.update_state(
        state="PROGRESS",
        meta={
            "step":    step,
            "total":   total,
            "message": message,
            "pct":     int((step / total) * 100),
        },
    )


# ─── Main task ─────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.process_document",
    max_retries=1,
    soft_time_limit=600,    # 10 min soft kill
    time_limit=660,         # 11 min hard kill
)
def process_document(self, task_id: str, storage_key: str, original_filename: str) -> Dict[str, Any]:
    """
    Main watermark-removal pipeline.

    Parameters
    ----------
    task_id           : Unique job identifier (echoed in response)
    storage_key       : Key used to retrieve the uploaded file from storage
    original_filename : Original upload filename (used for output naming)
    """
    from storage.manager    import storage
    from core.triage        import triage, Route, CorruptedPDFError, EncryptedPDFError
    from engines.fast_track import run_fast_track
    from engines.ai_track   import run_ai_track

    start_time = time.time()

    try:
        # ── Step 1: Retrieve file from storage ────────────────────────────
        _update_progress(self, 1, 6, "Retrieving file from storage…")
        file_bytes = storage().load(storage_key)

        stem    = Path(original_filename).stem
        suffix  = Path(original_filename).suffix.lower()
        out_name = f"cleaned_{stem}_{task_id[:8]}{suffix if suffix == '.pdf' else '.pdf'}"

        with tempfile.TemporaryDirectory(prefix="clearmark_") as tmp:
            tmp_dir     = Path(tmp)
            input_path  = tmp_dir / f"input{suffix}"
            output_path = tmp_dir / "output.pdf"

            input_path.write_bytes(file_bytes)
            del file_bytes  # free memory early

            # ── Step 2: Triage ────────────────────────────────────────────
            _update_progress(self, 2, 6, "Analysing document type…")
            try:
                route, meta = triage(input_path)
            except CorruptedPDFError as exc:
                return _fail(task_id, f"Corrupted PDF: {exc}")
            except EncryptedPDFError as exc:
                return _fail(task_id, f"Encrypted PDF: {exc}")

            # ── Step 3: Engine dispatch ───────────────────────────────────
            if route == Route.FAST_TRACK:
                _update_progress(self, 3, 6, "Running Fast-Track engine (PyMuPDF)…")
                result = run_fast_track(input_path, output_path)
                engine = "fast_track"

                if not result.success:
                    # Fallback: try AI track on engine failure
                    logger.warning("Fast-track failed (%s); falling back to AI track", result.error)
                    _update_progress(self, 3, 6, "Fast-track failed; switching to AI engine…")
                    from engines.ai_track import run_ai_track
                    ai_result = run_ai_track(input_path, output_path, image_only=False)
                    if not ai_result.success:
                        return _fail(task_id, ai_result.error)
                    result_meta = {
                        "engine":          "ai_track_fallback",
                        "pages_processed": ai_result.pages_processed,
                        "warnings":        ai_result.warnings,
                    }
                else:
                    result_meta = {
                        "engine":          engine,
                        "hits":            [
                            {"page": h.page, "type": h.type, "detail": h.detail}
                            for h in result.hits if h.removed
                        ],
                        "pages_affected":  result.pages_affected,
                        "watermarks_found": len(result.hits),
                    }

            else:  # AI_TRACK or IMAGE_ONLY
                _update_progress(self, 3, 6, "Running AI engine (rasterise + inpaint)…")
                image_only = (route == Route.FAST_TRACK.__class__.IMAGE_ONLY
                              if hasattr(route, '__class__') else False)
                # Correct check:
                from core.triage import Route as R
                image_only = (route == R.IMAGE_ONLY)
                ai_result  = run_ai_track(input_path, output_path, image_only=image_only)
                if not ai_result.success:
                    return _fail(task_id, ai_result.error)
                result_meta = {
                    "engine":          "ai_track",
                    "pages_processed": ai_result.pages_processed,
                    "warnings":        ai_result.warnings,
                }

            # ── Step 4: Store output ──────────────────────────────────────
            _update_progress(self, 4, 6, "Saving cleaned document…")
            if not output_path.exists():
                return _fail(task_id, "Engine produced no output file.")

            out_bytes  = output_path.read_bytes()
            output_key = storage().save(out_bytes, out_name, folder="processed")

            # ── Step 5: Schedule auto-delete ──────────────────────────────
            _update_progress(self, 5, 6, "Scheduling auto-delete…")
            storage().schedule_delete(storage_key, FILE_TTL_SECONDS)
            storage().schedule_delete(output_key,  FILE_TTL_SECONDS)

            # ── Step 6: Build response ────────────────────────────────────
            _update_progress(self, 6, 6, "Done!")
            download_url = storage().get_download_url(output_key, ttl=FILE_TTL_SECONDS)
            elapsed      = round(time.time() - start_time, 2)

            return {
                "status":        "SUCCESS",
                "task_id":       task_id,
                "download_url":  download_url,
                "output_key":    output_key,
                "filename":      out_name,
                "elapsed_secs":  elapsed,
                "triage":        meta,
                **result_meta,
            }

    except SoftTimeLimitExceeded:
        return _fail(task_id, "Processing timed out (>10 minutes). Try a smaller file.")
    except Exception as exc:
        logger.exception("Unhandled error in process_document task_id=%s", task_id)
        return _fail(task_id, f"Internal error: {exc}")


def _fail(task_id: str, message: str) -> Dict[str, Any]:
    logger.error("Task %s FAILED: %s", task_id, message)
    return {"status": "FAILURE", "task_id": task_id, "error": message}


# Avoid circular import at top level
try:
    from billiard.exceptions import SoftTimeLimitExceeded
except ImportError:
    class SoftTimeLimitExceeded(Exception): pass
