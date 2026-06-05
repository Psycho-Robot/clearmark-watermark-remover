import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import fitz

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS  = 50
MIN_DRAWINGS    = 2


class Route(Enum):
    FAST_TRACK = auto()
    AI_TRACK   = auto()
    IMAGE_ONLY = auto()


class CorruptedPDFError(Exception):
    pass


class EncryptedPDFError(Exception):
    pass


def triage(file_path: Path) -> Tuple[Route, dict]:
    suffix = file_path.suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
        logger.info("Triage → IMAGE_ONLY (%s)", file_path.name)
        return Route.IMAGE_ONLY, {
            "page_count": 1, "has_text": False,
            "has_drawings": False, "is_scanned": True, "sample_text_chars": 0
        }

    try:
        doc = fitz.open(str(file_path))
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(str(exc)) from exc
    except Exception as exc:
        raise CorruptedPDFError(str(exc)) from exc

    try:
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise EncryptedPDFError("PDF is password-protected.")

        page_count = len(doc)
        if page_count == 0:
            raise CorruptedPDFError("PDF has no pages.")

        # Sample up to 5 pages spread across the document
        indices = list({
            0,
            page_count // 4,
            page_count // 2,
            (3 * page_count) // 4,
            page_count - 1,
        })
        indices = sorted([i for i in indices if 0 <= i < page_count])[:5]

        total_text_chars = 0
        total_drawings   = 0
        scanned_votes    = 0

        for idx in indices:
            page     = doc[idx]
            text     = page.get_text("text").strip()
            drawings = page.get_drawings()
            images   = page.get_images(full=True)

            text_chars = len(text)
            draw_count = len(drawings)
            total_text_chars += text_chars
            total_drawings   += draw_count

            page_area = page.rect.width * page.rect.height
            is_scanned = (
                text_chars < MIN_TEXT_CHARS
                and draw_count < MIN_DRAWINGS
                and _has_full_page_image(doc, page, page_area)
            )
            if is_scanned:
                scanned_votes += 1

        avg_text = total_text_chars / len(indices)
        has_text     = avg_text >= MIN_TEXT_CHARS
        has_drawings = (total_drawings / len(indices)) >= MIN_DRAWINGS
        is_scanned   = scanned_votes >= max(1, len(indices) // 2 + 1)

        meta = {
            "page_count":        page_count,
            "has_text":          has_text,
            "has_drawings":      has_drawings,
            "is_scanned":        is_scanned,
            "sample_text_chars": total_text_chars,
        }

        if is_scanned:
            logger.info("Triage → AI_TRACK (scanned, %d pages)", page_count)
            return Route.AI_TRACK, meta
        else:
            logger.info("Triage → FAST_TRACK (vector, %d pages, ~%d chars/page)",
                        page_count, int(avg_text))
            return Route.FAST_TRACK, meta

    finally:
        doc.close()


def _has_full_page_image(doc: fitz.Document, page: fitz.Page,
                          page_area: float) -> bool:
    images = page.get_images(full=True)
    if not images:
        return False
    for img in images:
        xref = img[0]
        try:
            for item in page.get_image_rects(xref):
                img_area = item.width * item.height
                if page_area > 0 and (img_area / page_area) > 0.40:
                    return True
        except Exception:
            pass
    return False
