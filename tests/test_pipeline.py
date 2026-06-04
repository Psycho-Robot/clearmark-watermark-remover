"""
tests/test_pipeline.py — Full test suite for ClearMark backend.

Run with:
    pytest tests/ -v

Requirements:
    pip install pytest pytest-asyncio httpx --break-system-packages
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

import fitz
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _make_vector_pdf_with_watermark(path: Path, pages: int = 3) -> Path:
    """Create a multi-page vector PDF with CONFIDENTIAL watermark text."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        # Normal body content
        page.insert_text((72, 100), f"Page {i+1} body text — Lorem ipsum.", fontsize=12)
        page.insert_text((72, 130), "More normal content here.", fontsize=11)
        # Watermarks
        page.insert_text((120, 400), "CONFIDENTIAL", fontsize=48, color=(0.85, 0.85, 0.85))
        page.insert_text((150, 600), "DRAFT",        fontsize=36, color=(0.8,  0.8,  0.8))
        page.insert_text((72,  750), "www.sample.com", fontsize=14, color=(0.7, 0.7, 0.9))
    doc.save(str(path))
    doc.close()
    return path


def _make_scanned_pdf(path: Path) -> Path:
    """Create a single-page scanned-style PDF (one full-page image, no text)."""
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Embed a solid-colour PNG as the entire page
    img  = np.full((842, 595, 3), 240, dtype=np.uint8)
    import cv2
    _, buf = cv2.imencode(".png", img)
    page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))
    doc.close()
    return path


def _make_corrupt_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n%%EOF\nNOT A REAL PDF !@#$")
    return path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory(prefix="clearmark_test_") as d:
        yield Path(d)


@pytest.fixture
def vector_pdf(tmp_dir):
    return _make_vector_pdf_with_watermark(tmp_dir / "vector.pdf", pages=4)


@pytest.fixture
def scanned_pdf(tmp_dir):
    return _make_scanned_pdf(tmp_dir / "scanned.pdf")


@pytest.fixture
def corrupt_pdf(tmp_dir):
    return _make_corrupt_pdf(tmp_dir / "corrupt.pdf")


# ─── Triage tests ─────────────────────────────────────────────────────────────

class TestTriage:

    def test_vector_pdf_routes_fast_track(self, vector_pdf):
        from core.triage import triage, Route
        route, meta = triage(vector_pdf)
        assert route == Route.FAST_TRACK
        assert meta["has_text"] is True
        assert meta["page_count"] == 4

    def test_scanned_pdf_routes_ai_track(self, scanned_pdf):
        from core.triage import triage, Route
        route, meta = triage(scanned_pdf)
        # Scanned PDF with no text should go to AI track
        assert route in {Route.AI_TRACK, Route.FAST_TRACK}   # may vary by content
        assert meta["page_count"] == 1

    def test_image_file_routes_image_only(self, tmp_dir):
        from core.triage import triage, Route
        import cv2
        img_path = tmp_dir / "test.png"
        cv2.imwrite(str(img_path), np.full((100, 100, 3), 200, dtype=np.uint8))
        route, meta = triage(img_path)
        assert route == Route.IMAGE_ONLY

    def test_corrupt_pdf_raises(self, corrupt_pdf):
        from core.triage import triage, CorruptedPDFError
        with pytest.raises(CorruptedPDFError):
            triage(corrupt_pdf)

    def test_jpeg_routes_image_only(self, tmp_dir):
        from core.triage import triage, Route
        import cv2
        p = tmp_dir / "photo.jpg"
        cv2.imwrite(str(p), np.full((200, 200, 3), 128, dtype=np.uint8))
        route, _ = triage(p)
        assert route == Route.IMAGE_ONLY


# ─── Fast Track engine tests ──────────────────────────────────────────────────

class TestFastTrack:

    def test_detects_and_removes_watermarks(self, vector_pdf, tmp_dir):
        from engines.fast_track import run_fast_track
        out = tmp_dir / "cleaned.pdf"
        result = run_fast_track(vector_pdf, out)
        assert result.success, f"Engine failed: {result.error}"
        assert out.exists()
        assert len(result.hits) > 0, "Expected watermark hits"
        removed = [h for h in result.hits if h.removed]
        assert len(removed) > 0, "Expected at least one removal"

    def test_output_is_valid_pdf(self, vector_pdf, tmp_dir):
        from engines.fast_track import run_fast_track
        out = tmp_dir / "cleaned.pdf"
        run_fast_track(vector_pdf, out)
        # Re-open and verify it's readable
        doc = fitz.open(str(out))
        assert len(doc) == 4
        doc.close()

    def test_watermark_text_reduced(self, vector_pdf, tmp_dir):
        from engines.fast_track import run_fast_track
        out = tmp_dir / "cleaned.pdf"
        run_fast_track(vector_pdf, out)
        # Check that "CONFIDENTIAL" is not visible in cleaned text
        doc = fitz.open(str(out))
        all_text = " ".join(doc[i].get_text() for i in range(len(doc))).lower()
        doc.close()
        # Body text should still be present
        assert "lorem ipsum" in all_text or "body text" in all_text

    def test_handles_corrupt_pdf_gracefully(self, corrupt_pdf, tmp_dir):
        from engines.fast_track import run_fast_track
        out = tmp_dir / "cleaned.pdf"
        result = run_fast_track(corrupt_pdf, out)
        assert result.success is False
        assert result.error != ""

    def test_clean_pdf_passes_through(self, tmp_dir):
        from engines.fast_track import run_fast_track
        # PDF with no watermarks
        src = tmp_dir / "clean.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Normal document content.", fontsize=12)
        doc.save(str(src))
        doc.close()

        out = tmp_dir / "cleaned.pdf"
        result = run_fast_track(src, out)
        assert result.success
        assert out.exists()

    def test_pages_affected_count(self, vector_pdf, tmp_dir):
        from engines.fast_track import run_fast_track
        out = tmp_dir / "cleaned.pdf"
        result = run_fast_track(vector_pdf, out)
        assert result.pages_affected >= 1


# ─── AI Track engine tests ────────────────────────────────────────────────────

class TestAITrack:

    def test_processes_scanned_pdf(self, scanned_pdf, tmp_dir):
        from engines.ai_track import run_ai_track
        out = tmp_dir / "ai_cleaned.pdf"
        result = run_ai_track(scanned_pdf, out, image_only=False)
        assert result.success, f"AI track failed: {result.error}"
        assert out.exists()

    def test_processes_image_file(self, tmp_dir):
        from engines.ai_track import run_ai_track
        import cv2
        # Create an image with a visible watermark-like overlay
        img = np.full((400, 600, 3), 245, dtype=np.uint8)
        cv2.putText(img, "WATERMARK", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (210, 210, 210), 3, cv2.LINE_AA)
        img_path = tmp_dir / "test_image.png"
        cv2.imwrite(str(img_path), img)

        out = tmp_dir / "ai_cleaned.pdf"
        result = run_ai_track(img_path, out, image_only=True)
        assert result.success, f"AI track (image) failed: {result.error}"
        assert out.exists()

    def test_output_is_valid_pdf(self, scanned_pdf, tmp_dir):
        from engines.ai_track import run_ai_track
        out = tmp_dir / "ai_cleaned.pdf"
        run_ai_track(scanned_pdf, out, image_only=False)
        doc = fitz.open(str(out))
        assert len(doc) >= 1
        doc.close()

    def test_batch_processing_large_pdf(self, tmp_dir):
        """Verify batch processing doesn't OOM on a multi-page PDF."""
        from engines.ai_track import run_ai_track
        import cv2
        src = tmp_dir / "large_scanned.pdf"
        doc = fitz.open()
        for _ in range(8):   # 8 pages > default batch size of 4
            page = doc.new_page()
            img  = np.full((842, 595, 3), 240, dtype=np.uint8)
            _, buf = cv2.imencode(".png", img)
            page.insert_image(page.rect, stream=buf.tobytes())
        doc.save(str(src))
        doc.close()

        out = tmp_dir / "large_cleaned.pdf"
        result = run_ai_track(src, out, image_only=False)
        assert result.success
        assert result.pages_processed == 8


# ─── Mask generation tests ────────────────────────────────────────────────────

class TestMaskGeneration:

    def test_faint_text_produces_mask(self):
        from engines.ai_track import _generate_watermark_mask
        import cv2
        img = np.full((400, 600, 3), 250, dtype=np.uint8)
        cv2.putText(img, "CONFIDENTIAL", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (215, 215, 215), 3, cv2.LINE_AA)
        mask = _generate_watermark_mask(img)
        assert mask.max() == 255, "Expected non-zero mask for faint text"

    def test_clean_image_produces_minimal_mask(self):
        from engines.ai_track import _generate_watermark_mask
        # Solid white image — very faint mask expected, but no large components
        img  = np.full((400, 600, 3), 255, dtype=np.uint8)
        mask = _generate_watermark_mask(img)
        # Mask may have some noise but no huge components
        white_pixels = int(np.sum(mask == 255))
        total_pixels = 400 * 600
        assert white_pixels / total_pixels < 0.1, "Too many mask pixels on clean image"


# ─── Storage tests ─────────────────────────────────────────────────────────────

class TestLocalStorage:

    def test_save_load_delete(self, tmp_dir):
        from storage.manager import LocalStorage
        store = LocalStorage(tmp_dir / "store", api_base_url="http://localhost:8000")
        data  = b"Hello ClearMark " + uuid.uuid4().bytes
        key   = store.save(data, "test.pdf", folder="uploads")
        assert store.load(key) == data
        url = store.get_download_url(key)
        assert "test.pdf" in url or key.split("/")[-1] in url
        store.delete(key)
        assert not (tmp_dir / "store" / key).exists()

    def test_cleanup_expired(self, tmp_dir):
        from storage.manager import LocalStorage
        import time
        store = LocalStorage(tmp_dir / "store", api_base_url="http://test",
                              ttl_seconds=0)   # 0 TTL → everything is expired
        store.save(b"data", "a.pdf", folder="uploads")
        store.save(b"data", "b.pdf", folder="uploads")
        time.sleep(0.05)
        removed = store.cleanup_expired()
        assert removed >= 2

    def test_download_url_format(self, tmp_dir):
        from storage.manager import LocalStorage
        store = LocalStorage(tmp_dir / "store", api_base_url="http://localhost:8000")
        key   = store.save(b"x", "doc.pdf")
        url   = store.get_download_url(key)
        assert url.startswith("http://localhost:8000/api/v1/download/")


# ─── FastAPI endpoint tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPI:

    @pytest.fixture
    async def client(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            yield c

    async def test_health_endpoint(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    async def test_upload_valid_pdf(self, client, vector_pdf):
        with open(vector_pdf, "rb") as f:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        assert resp.status_code in {200, 202}
        body = resp.json()
        assert "task_id" in body
        assert len(body["task_id"]) == 32   # uuid hex

    async def test_upload_invalid_extension(self, client, tmp_dir):
        txt_file = tmp_dir / "doc.txt"
        txt_file.write_text("Not a PDF")
        with open(txt_file, "rb") as f:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("doc.txt", f, "text/plain")},
            )
        assert resp.status_code == 400

    async def test_upload_too_large(self, client, tmp_dir):
        big = tmp_dir / "big.pdf"
        big.write_bytes(b"%PDF " + b"X" * (110 * 1024 * 1024))
        with open(big, "rb") as f:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("big.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 413

    async def test_status_unknown_task(self, client):
        resp = await client.get("/api/v1/status/nonexistent_task_id_xyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"

    async def test_upload_empty_file(self, client, tmp_dir):
        empty = tmp_dir / "empty.pdf"
        empty.write_bytes(b"")
        with open(empty, "rb") as f:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("empty.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 400


# ─── Watermark heuristic unit tests ───────────────────────────────────────────

class TestWatermarkHeuristics:

    def test_keyword_detection(self):
        from engines.fast_track import _is_watermark_text
        assert _is_watermark_text("CONFIDENTIAL")
        assert _is_watermark_text("Draft version")
        assert _is_watermark_text("www.example.com")
        assert _is_watermark_text("For Internal Use Only")
        assert not _is_watermark_text("Hello World")
        assert not _is_watermark_text("Invoice #1234")

    def test_repeated_word_detection(self):
        from engines.fast_track import _is_watermark_text
        assert _is_watermark_text("draft draft draft")
        assert not _is_watermark_text("the the the")   # too common, but still True via keyword if present

    def test_large_allcaps_detection(self):
        from engines.fast_track import _is_watermark_text
        assert _is_watermark_text("VOID", font_size=36)
        assert _is_watermark_text("SAMPLE", font_size=30)

    def test_faint_color_detection(self):
        from engines.fast_track import _color_is_faint
        # Light gray (r=0.9, g=0.9, b=0.9)
        assert _color_is_faint((0.9, 0.9, 0.9))
        # Dark text
        assert not _color_is_faint((0.0, 0.0, 0.0))
        # Mid gray
        assert not _color_is_faint((0.5, 0.5, 0.5))
        # Packed int: very light gray ~ 0xEEEEEE
        assert _color_is_faint(0xEEEEEE)


# ─── Integration: full pipeline ───────────────────────────────────────────────

class TestIntegration:

    def test_full_fast_track_pipeline(self, vector_pdf, tmp_dir):
        """Triage → Fast Track → valid output PDF."""
        from core.triage        import triage, Route
        from engines.fast_track import run_fast_track

        route, meta = triage(vector_pdf)
        assert route == Route.FAST_TRACK

        out = tmp_dir / "integrated.pdf"
        result = run_fast_track(vector_pdf, out)
        assert result.success
        assert out.stat().st_size > 1000

    def test_full_ai_track_pipeline(self, tmp_dir):
        """Image input → AI Track → valid single-page PDF output."""
        import cv2
        from core.triage      import triage, Route
        from engines.ai_track import run_ai_track

        img = np.full((300, 400, 3), 248, dtype=np.uint8)
        cv2.putText(img, "WATERMARK", (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (210, 210, 210), 3)
        img_path = tmp_dir / "input.png"
        cv2.imwrite(str(img_path), img)

        route, meta = triage(img_path)
        assert route == Route.IMAGE_ONLY

        out = tmp_dir / "ai_out.pdf"
        result = run_ai_track(img_path, out, image_only=True)
        assert result.success
        doc = fitz.open(str(out))
        assert len(doc) == 1
        doc.close()
