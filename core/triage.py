"""
core/triage.py — Triage Router

Inspects a file and decides which processing engine to use:

  Route.FAST_TRACK  → PyMuPDF engine  (vector/text PDF)
  Route.AI_TRACK    → AI image engine (scanned PDF or image file)
  Route.IMAGE_ONLY  → AI track but skip PDF reconstruction

Decision logic
──────────────
1. If the file extension is .jpg / .jpeg / .png → IMAGE_ONLY (AI track)
2. Open with PyMuPDF:
   - If opening fails → raise CorruptedPDFError
   - For the first page (and a sample of subsequent pages):
       * If page.get_text() returns meaningful text
         AND page.get_drawings() returns vector objects
         → FAST_TRACK
       * Otherwise (blank text, single giant image) → AI_TRACK
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum characters on first page to be considered a "text PDF"
MIN_TEXT_CHARS  = 30
# Minimum vector drawing objects on first page
MIN_DRAWINGS    = 3
# How many pages to sample when making the decision
SAMPLE_PAGES    = min(3, 1)   # look at up to 3 pages


class Route(Enum):
    FAST_TRACK = auto()   # PyMuPDF engine
    AI_TRACK   = auto()   # AI image engine (PDF → images → inpaint → PDF)
    IMAGE_ONLY = auto()   # AI track, input is already an image


class CorruptedPDFError(Exception):
    """Raised when PyMuPDF cannot open the file."""


class EncryptedPDFError(Exception):
    """Raised when the PDF is password-protected."""


def triage(file_path: Path) -> Tuple[Route, dict]:
    """
    Inspect *file_path* and return (Route, metadata_dict).

    metadata_dict contains:
      - page_count        : int
      - has_text          : bool
      - has_drawings      : bool
      - is_scanned        : bool
      - sample_text_chars : int
    """
    suffix = file_path.suffix.lower()

    # ── Image files: skip to AI track immediately ─────────────────────────
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
        logger.info("Triage → IMAGE_ONLY  (%s)", file_path.name)
        return Route.IMAGE_ONLY, {"page_count": 1, "has_text": False,
                                   "has_drawings": False, "is_scanned": True,
                                   "sample_text_chars": 0}

    # ── PDF files ─────────────────────────────────────────────────────────
    try:
        doc = fitz.open(str(file_path))
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(f"Cannot open PDF — file may be corrupted: {exc}") from exc
    except Exception as exc:
        raise CorruptedPDFError(f"Unexpected error opening PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            # Try empty password (some "protected" PDFs open with no password)
            if not doc.authenticate(""):
                raise EncryptedPDFError("PDF is password-protected. Please remove the password first.")

        page_count = len(doc)
        if page_count == 0:
            raise CorruptedPDFError("PDF has no pages.")

        # Sample up to SAMPLE_PAGES pages (first + some middle pages)
        indices = [0]
        if page_count > 2:
            indices.append(page_count // 2)
        if page_count > 4:
            indices.append(page_count - 1)

        total_text_chars = 0
        total_drawings   = 0
        scanned_votes    = 0

        for idx in indices:
            page = doc[idx]
            text     = page.get_text("text").strip()
            drawings = page.get_drawings()
            images   = page.get_images(full=True)

            text_chars = len(text)
            draw_count = len(drawings)

            total_text_chars += text_chars
            total_drawings   += draw_count

            # A "scanned" page: very little text, no drawings, has at least one large image
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            is_page_scanned = (
                text_chars < MIN_TEXT_CHARS
                and draw_count < MIN_DRAWINGS
                and _has_full_page_image(doc, page, page_area)
            )
            if is_page_scanned:
                scanned_votes += 1

        has_text     = total_text_chars >= MIN_TEXT_CHARS * len(indices)
        has_drawings = total_drawings   >= MIN_DRAWINGS   * len(indices)
        is_scanned   = scanned_votes >= max(1, len(indices) // 2 + 1)

        meta = {
            "page_count":        page_count,
            "has_text":          has_text,
            "has_drawings":      has_drawings,
            "is_scanned":        is_scanned,
            "sample_text_chars": total_text_chars,
        }

        if is_scanned:
            route = Route.AI_TRACK
            logger.info("Triage → AI_TRACK    (scanned PDF, %d pages)", page_count)
        else:
            route = Route.FAST_TRACK
            logger.info("Triage → FAST_TRACK  (vector PDF, %d pages, %d chars, %d drawings)",
                        page_count, total_text_chars, total_drawings)

        return route, meta

    finally:
        doc.close()


def _has_full_page_image(doc: fitz.Document, page: fitz.Page, page_area: float) -> bool:
    """Return True if the page contains at least one image that covers most of the page."""
    images = page.get_images(full=True)
    if not images:
        return False
    for img in images:
        xref = img[0]
        # Get the image's placement rectangle on the page
        for item in page.get_image_rects(xref):
            img_area = item.width * item.height
            if page_area > 0 and (img_area / page_area) > 0.5:
                return True
    return False
