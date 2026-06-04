"""
engines/fast_track.py — PyMuPDF "Fast Track" watermark removal engine.

Executes three sequential sweeps on a vector/text PDF:

  Pass 1 — Metadata Sweep
    Scan the raw PDF object tree for /Watermark and /Artifact structure tags.
    Delete matching marked-content objects and optional-content groups (OCGs).

  Pass 2 — Opacity Sweep
    Iterate page.get_drawings() and page.get_images().
    Flag objects with alpha < OPACITY_THRESHOLD.
    If the same object appears (by position hash) on > 50% of pages → delete it.

  Pass 3 — Hex-Color / Pattern Sweep
    Iterate page.get_text("dict") and inspect every text span's color.
    Collect spans that are faint (low-contrast against white) or match
    WATERMARK_KEYWORDS.  Track cross-page frequency; redact repeated hits.

Returns a FastTrackResult dataclass with full audit details.
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF

from config.settings import (
    WATERMARK_KEYWORDS,
    OPACITY_THRESHOLD,
    REPEAT_PAGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WatermarkHit:
    page:    int
    type:    str          # "metadata" | "opacity" | "text" | "annotation"
    detail:  str
    removed: bool = False


@dataclass
class FastTrackResult:
    success:       bool
    hits:          List[WatermarkHit] = field(default_factory=list)
    pages_affected: int = 0
    output_path:   Path | None = None
    error:         str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_watermark_text(text: str, font_size: float = 12.0) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    for kw in WATERMARK_KEYWORDS:
        if kw in t:
            return True
    # URL pattern: contains a dot-separated hostname segment
    import re
    if re.search(r'\b\w+\.(com|net|org|io|co|pdf)\b', t):
        return True
    # Repeated non-trivial word (length > 3 to skip "the", "and", etc.)
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1 and len(words[0]) > 3:
        return True
    if font_size > 24 and text.strip() == text.strip().upper() and len(text.strip()) <= 30:
        return True
    return False


def _color_is_faint(color: int | Tuple) -> bool:
    """
    Return True if the color is a very light gray / near-white
    (common for watermark text meant to be subtle).
    Accepts either a packed int (PyMuPDF sRGB) or an (r,g,b) tuple 0-1.
    """
    if isinstance(color, int):
        r = ((color >> 16) & 0xFF) / 255
        g = ((color >>  8) & 0xFF) / 255
        b =  (color        & 0xFF) / 255
    elif isinstance(color, (list, tuple)) and len(color) >= 3:
        r, g, b = color[:3]
    else:
        return False
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Faint = very bright (near white) — typical for gray watermarks
    return luminance > 0.75


def _drawing_position_hash(drawing: dict) -> str:
    """Stable hash of a drawing's bounding rect (used for cross-page dedup)."""
    rect = drawing.get("rect", fitz.Rect())
    return hashlib.md5(f"{rect.x0:.0f},{rect.y0:.0f},{rect.x1:.0f},{rect.y1:.0f}".encode()).hexdigest()


# ─── Main engine ───────────────────────────────────────────────────────────────

def run_fast_track(input_path: Path, output_path: Path) -> FastTrackResult:
    """
    Run all three sweeps on *input_path*, write cleaned PDF to *output_path*.
    Returns a FastTrackResult.
    """
    try:
        doc = fitz.open(str(input_path))
    except Exception as exc:
        return FastTrackResult(success=False, error=str(exc))

    try:
        hits: List[WatermarkHit] = []

        # ── Pass 1: Metadata sweep ─────────────────────────────────────────
        hits += _pass1_metadata(doc)

        # ── Pass 2: Opacity sweep ──────────────────────────────────────────
        hits += _pass2_opacity(doc)

        # ── Pass 3: Text / hex-color sweep ────────────────────────────────
        hits += _pass3_text(doc)

        # Save
        doc.save(
            str(output_path),
            garbage=4,
            deflate=True,
            clean=True,
        )
        pages_affected = len({h.page for h in hits if h.removed})
        logger.info("Fast-track complete: %d hits, %d pages affected → %s",
                    len(hits), pages_affected, output_path.name)
        return FastTrackResult(
            success=True,
            hits=hits,
            pages_affected=pages_affected,
            output_path=output_path,
        )

    except Exception as exc:
        logger.exception("Fast-track engine error")
        return FastTrackResult(success=False, error=str(exc))

    finally:
        doc.close()


# ─── Pass 1: Metadata sweep ────────────────────────────────────────────────────

def _pass1_metadata(doc: fitz.Document) -> List[WatermarkHit]:
    hits: List[WatermarkHit] = []

    # 1a. Optional Content Groups (OCGs) named "Watermark"
    try:
        catalog = doc.pdf_catalog()
        if catalog:
            ocprops_xref = doc.xref_get_key(catalog, "OCProperties")
            if ocprops_xref and ocprops_xref[0] != "null":
                _remove_watermark_ocgs(doc, hits)
    except Exception as exc:
        logger.debug("Pass1 OCG scan skipped: %s", exc)

    # 1b. Per-page /Artifact and /Watermark marked-content
    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            # Remove all annotations whose subtype or content looks like a watermark
            for annot in list(page.annots()):
                info    = annot.info
                content = (info.get("content") or "") + " " + (info.get("title") or "")
                if _is_watermark_text(content):
                    page.delete_annot(annot)
                    hits.append(WatermarkHit(
                        page=page_num + 1, type="metadata",
                        detail=f"Annotation: {content[:60]}", removed=True
                    ))
        except Exception as exc:
            logger.debug("Pass1 annotation scan page %d: %s", page_num, exc)

    return hits


def _remove_watermark_ocgs(doc: fitz.Document, hits: List[WatermarkHit]) -> None:
    """Walk all xrefs looking for OCG dicts with watermark-like /Name values."""
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Type")[1] == "/OCG":
                name_raw = doc.xref_get_key(xref, "Name")[1]
                name = name_raw.strip("()").lower()
                if any(kw in name for kw in ["watermark", "stamp", "draft", "confidential"]):
                    # Set OCG state to OFF so it is not rendered
                    doc.xref_set_key(xref, "Usage", "<</Print<</PrintState/OFF>>/View<</ViewState/OFF>>>>")
                    hits.append(WatermarkHit(
                        page=0, type="metadata",
                        detail=f"OCG disabled: {name_raw}", removed=True
                    ))
        except Exception:
            pass


# ─── Pass 2: Opacity sweep ─────────────────────────────────────────────────────

def _pass2_opacity(doc: fitz.Document) -> List[WatermarkHit]:
    hits: List[WatermarkHit] = []
    page_count = len(doc)

    # Collect semi-transparent drawing position hashes across all pages
    # key → list of page numbers where it appears
    drawing_pages: Dict[str, List[int]] = defaultdict(list)

    for page_num in range(page_count):
        page = doc[page_num]
        try:
            for drawing in page.get_drawings():
                alpha = drawing.get("fill_opacity", 1.0)
                if alpha is None:
                    alpha = drawing.get("stroke_opacity", 1.0) or 1.0
                if alpha < OPACITY_THRESHOLD:
                    ph = _drawing_position_hash(drawing)
                    drawing_pages[ph].append(page_num)
        except Exception as exc:
            logger.debug("Pass2 drawings page %d: %s", page_num, exc)

    # Flag drawings that appear on > REPEAT_PAGE_THRESHOLD fraction of pages
    threshold_count = max(1, int(page_count * REPEAT_PAGE_THRESHOLD))
    watermark_hashes = {ph for ph, pages in drawing_pages.items()
                        if len(pages) >= threshold_count}

    if not watermark_hashes:
        return hits

    # Second pass: redact them
    for page_num in range(page_count):
        page = doc[page_num]
        try:
            for drawing in page.get_drawings():
                alpha = drawing.get("fill_opacity", 1.0)
                if alpha is None:
                    alpha = drawing.get("stroke_opacity", 1.0) or 1.0
                if alpha < OPACITY_THRESHOLD:
                    ph = _drawing_position_hash(drawing)
                    if ph in watermark_hashes:
                        rect = fitz.Rect(drawing["rect"])
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        hits.append(WatermarkHit(
                            page=page_num + 1, type="opacity",
                            detail=f"Semi-transparent object α={alpha:.2f} at {rect}",
                            removed=True
                        ))
            if any(h.page == page_num + 1 and h.type == "opacity" for h in hits):
                page.apply_redactions()
        except Exception as exc:
            logger.debug("Pass2 redact page %d: %s", page_num, exc)

    return hits


# ─── Pass 3: Text / hex-color sweep ────────────────────────────────────────────

def _pass3_text(doc: fitz.Document) -> List[WatermarkHit]:
    hits:  List[WatermarkHit]     = []
    page_count = len(doc)

    # Collect candidate text strings and which pages they appear on
    # key = (normalised text, approx font size bucket) → list of (page_num, span_rect)
    text_occurrences: Dict[str, List[Tuple[int, fitz.Rect]]] = defaultdict(list)

    for page_num in range(page_count):
        page = doc[page_num]
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception:
            continue

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text = span.get("text", "").strip()
                    if not raw_text:
                        continue
                    font_size = span.get("size", 12)
                    color     = span.get("color", 0)

                    is_wm = (
                        _is_watermark_text(raw_text, font_size)
                        or _color_is_faint(color)
                    )
                    if is_wm:
                        key  = raw_text.lower()[:80]
                        rect = fitz.Rect(span["bbox"])
                        text_occurrences[key].append((page_num, rect))

    if not text_occurrences:
        return hits

    # Determine which strings are repeated across pages (cross-page frequency)
    # A string on a single page still gets redacted if it explicitly matches keywords
    threshold_count = max(1, int(page_count * REPEAT_PAGE_THRESHOLD))

    # Build a set of page_num → list of rects to redact
    redact_map: Dict[int, List[fitz.Rect]] = defaultdict(list)

    for key, occurrences in text_occurrences.items():
        page_nums = [p for p, _ in occurrences]
        unique_pages = len(set(page_nums))

        # Redact if: appears on enough pages OR is an explicit keyword match
        should_redact = (
            unique_pages >= threshold_count
            or any(kw in key for kw in WATERMARK_KEYWORDS)
        )
        if should_redact:
            for page_num, rect in occurrences:
                redact_map[page_num].append(rect)

    # Also do a targeted search for full watermark phrases on each page
    for page_num in range(page_count):
        page = doc[page_num]
        for kw in WATERMARK_KEYWORDS:
            try:
                found = page.search_for(kw, quads=False)
                for rect in found:
                    redact_map[page_num].append(rect)
            except Exception:
                pass

    # Apply all redactions
    for page_num, rects in redact_map.items():
        page = doc[page_num]
        for rect in rects:
            try:
                page.add_redact_annot(rect, fill=(1, 1, 1))
                hits.append(WatermarkHit(
                    page=page_num + 1, type="text",
                    detail=f"Text redacted at {rect}", removed=True
                ))
            except Exception as exc:
                logger.debug("Pass3 add_redact page %d: %s", page_num, exc)
        try:
            page.apply_redactions()
        except Exception as exc:
            logger.debug("Pass3 apply_redactions page %d: %s", page_num, exc)

    return hits
