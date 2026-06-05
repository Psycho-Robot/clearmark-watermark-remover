import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz

from config.settings import WATERMARK_KEYWORDS, OPACITY_THRESHOLD, REPEAT_PAGE_THRESHOLD

logger = logging.getLogger(__name__)


# ─── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WatermarkHit:
    page:    int
    type:    str
    detail:  str
    removed: bool = False


@dataclass
class FastTrackResult:
    success:        bool
    hits:           List[WatermarkHit] = field(default_factory=list)
    pages_affected: int = 0
    output_path:    Optional[Path] = None
    error:          str = ""


# ─── Extended watermark patterns ──────────────────────────────────────────────

# Coaching/institution name patterns (very common in Indian educational PDFs)
INSTITUTION_PATTERNS = [
    r'\b(academy|institute|coaching|classes|education|edu|hub|centre|center|school|college|tutorials?)\b',
    r'\b(study|notes?|material|pdf|download)\b',
    r'\b(hub|group|foundation|publications?)\b',
]

# URL / social media patterns
URL_PATTERNS = [
    r'www\.\S+',
    r'https?://\S+',
    r't\.me/\S+',
    r'@\w+',
    r'\S+\.com\b',
    r'\S+\.in\b',
    r'\S+\.org\b',
    r'\S+\.net\b',
]

# Telegram / WhatsApp / social watermarks
SOCIAL_PATTERNS = [
    r'telegram',
    r'whatsapp',
    r'youtube',
    r'instagram',
    r'facebook',
    r'join\s+(us|our|now)',
    r'subscribe',
    r'follow\s+us',
    r'visit\s+us',
    r'download\s+(from|at|on)',
]


def _is_watermark_text(text: str, font_size: float = 12.0,
                        color=None, opacity: float = 1.0) -> bool:
    """
    Comprehensive watermark text detection.
    Returns True if the text looks like a watermark.
    """
    t = text.strip()
    tl = t.lower()
    if not t or len(t) < 2:
        return False

    # 1. Low opacity is a very strong watermark signal
    if opacity < 0.9:
        return True

    # 2. Faint color (near-white or very light gray)
    if color is not None and _color_is_faint(color):
        return True

    # 3. Known watermark keywords
    for kw in WATERMARK_KEYWORDS:
        if kw.lower() in tl:
            return True

    # 4. URL patterns
    for pat in URL_PATTERNS:
        if re.search(pat, tl, re.IGNORECASE):
            return True

    # 5. Social media patterns
    for pat in SOCIAL_PATTERNS:
        if re.search(pat, tl, re.IGNORECASE):
            return True

    # 6. Institution/coaching patterns (common in Indian PDFs)
    matched_patterns = sum(
        1 for pat in INSTITUTION_PATTERNS
        if re.search(pat, tl, re.IGNORECASE)
    )
    if matched_patterns >= 1 and font_size < 16:
        return True

    # 7. Repeated single word (DRAFT DRAFT DRAFT)
    words = tl.split()
    if len(words) >= 3 and len(set(words)) == 1 and len(words[0]) > 3:
        return True

    # 8. Large all-caps short text (stamp-style)
    if (font_size > 20
            and t == t.upper()
            and re.match(r'^[A-Z\s\-]+$', t)
            and 3 < len(t.strip()) < 25):
        return True

    return False


def _color_is_faint(color) -> bool:
    """Return True for very light colors (near-white watermarks)."""
    if isinstance(color, int):
        r = ((color >> 16) & 0xFF) / 255
        g = ((color >> 8)  & 0xFF) / 255
        b =  (color        & 0xFF) / 255
    elif isinstance(color, (list, tuple)) and len(color) >= 3:
        r, g, b = [c if c <= 1 else c / 255 for c in color[:3]]
    else:
        return False
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.72


def _span_position_bucket(span: dict, page_height: float) -> str:
    """
    Classify span position as 'header', 'footer', 'center', 'left', 'right'.
    Used to detect watermarks that always appear in the same zone.
    """
    bbox = span.get("bbox", [0, 0, 0, 0])
    y_rel = bbox[1] / max(page_height, 1)
    x_rel = (bbox[0] + bbox[2]) / 2

    if y_rel < 0.08:
        return "header"
    if y_rel > 0.88:
        return "footer"
    if 0.35 < y_rel < 0.65 and 0.2 < x_rel < 0.8:
        return "center"
    return "body"


def _rect_position_key(rect) -> str:
    """Rounded position key for cross-page deduplication."""
    return f"{round(rect.x0/10)*10},{round(rect.y0/10)*10}"


# ─── Main engine ───────────────────────────────────────────────────────────────

def run_fast_track(input_path: Path, output_path: Path) -> FastTrackResult:
    try:
        doc = fitz.open(str(input_path))
    except Exception as exc:
        return FastTrackResult(success=False, error=str(exc))

    try:
        hits: List[WatermarkHit] = []
        page_count = len(doc)

        hits += _pass1_annotations(doc)
        hits += _pass2_opacity_drawings(doc, page_count)
        hits += _pass3_text_smart(doc, page_count)

        doc.save(str(output_path), garbage=4, deflate=True, clean=True)
        pages_affected = len({h.page for h in hits if h.removed})

        logger.info("Fast-track: %d hits on %d pages → %s",
                    len(hits), pages_affected, output_path.name)
        return FastTrackResult(
            success=True, hits=hits,
            pages_affected=pages_affected, output_path=output_path,
        )
    except Exception as exc:
        logger.exception("Fast-track error")
        return FastTrackResult(success=False, error=str(exc))
    finally:
        doc.close()


# ─── Pass 1: Annotations ──────────────────────────────────────────────────────

def _pass1_annotations(doc: fitz.Document) -> List[WatermarkHit]:
    hits = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            for annot in list(page.annots()):
                info    = annot.info
                content = (info.get("content") or "") + " " + (info.get("title") or "")
                if _is_watermark_text(content.strip()):
                    page.delete_annot(annot)
                    hits.append(WatermarkHit(
                        page=page_num + 1, type="annotation",
                        detail=content.strip()[:80], removed=True
                    ))
        except Exception as exc:
            logger.debug("Pass1 page %d: %s", page_num, exc)

    # OCG (Optional Content Groups) with watermark names
    try:
        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_get_key(xref, "Type")[1] == "/OCG":
                    name = doc.xref_get_key(xref, "Name")[1].strip("()")
                    if any(kw in name.lower() for kw in ["watermark","stamp","draft","confidential"]):
                        doc.xref_set_key(xref, "Usage",
                            "<</Print<</PrintState/OFF>>/View<</ViewState/OFF>>>>")
                        hits.append(WatermarkHit(
                            page=0, type="annotation",
                            detail=f"OCG: {name}", removed=True
                        ))
            except Exception:
                pass
    except Exception:
        pass

    return hits


# ─── Pass 2: Opacity / drawing sweep ─────────────────────────────────────────

def _pass2_opacity_drawings(doc: fitz.Document, page_count: int) -> List[WatermarkHit]:
    hits = []
    # pos_key → pages it appears on
    drawing_pages: Dict[str, List[int]] = defaultdict(list)

    for page_num in range(page_count):
        page = doc[page_num]
        try:
            for drw in page.get_drawings():
                alpha = drw.get("fill_opacity") or drw.get("stroke_opacity") or 1.0
                if alpha < OPACITY_THRESHOLD:
                    key = _rect_position_key(fitz.Rect(drw["rect"]))
                    drawing_pages[key].append(page_num)
        except Exception:
            pass

    threshold = max(1, int(page_count * REPEAT_PAGE_THRESHOLD))
    wm_keys   = {k for k, pages in drawing_pages.items() if len(pages) >= threshold}

    for page_num in range(page_count):
        page = doc[page_num]
        redacted = False
        try:
            for drw in page.get_drawings():
                alpha = drw.get("fill_opacity") or drw.get("stroke_opacity") or 1.0
                if alpha < OPACITY_THRESHOLD:
                    key = _rect_position_key(fitz.Rect(drw["rect"]))
                    if key in wm_keys:
                        rect = fitz.Rect(drw["rect"])
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        hits.append(WatermarkHit(
                            page=page_num + 1, type="opacity",
                            detail=f"Semi-transparent drawing α={alpha:.2f}",
                            removed=True
                        ))
                        redacted = True
            if redacted:
                page.apply_redactions()
        except Exception as exc:
            logger.debug("Pass2 page %d: %s", page_num, exc)

    return hits


# ─── Pass 3: Smart text sweep (the critical one) ──────────────────────────────

def _pass3_text_smart(doc: fitz.Document, page_count: int) -> List[WatermarkHit]:
    """
    Precision watermark text removal.

    Steps:
    A) Collect ALL text spans with metadata across ALL pages
    B) Score each unique text string:
       - Keyword match score
       - Cross-page frequency score
       - Position score (header/footer = more likely WM)
       - Font size score
       - Color score
    C) Only redact spans that cross a confidence threshold
    D) Redact at SPAN level (not word level) to avoid removing nearby text
    """
    hits = []

    # Structure: text_key → list of (page_num, span_rect, span_info)
    text_index: Dict[str, List[Tuple[int, fitz.Rect, dict]]] = defaultdict(list)
    # page_height cache
    page_heights: Dict[int, float] = {}

    # ── A: Collect all spans ──────────────────────────────────────────────────
    for page_num in range(page_count):
        page = doc[page_num]
        page_heights[page_num] = page.rect.height

        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception:
            continue

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Check for rotated text (diagonal watermarks)
                wmode = line.get("wmode", 0)
                dir_vec = line.get("dir", (1, 0))
                is_diagonal = abs(dir_vec[1]) > 0.3  # not horizontal

                for span in line.get("spans", []):
                    raw = span.get("text", "").strip()
                    if not raw or len(raw) < 2:
                        continue

                    size    = span.get("size", 12)
                    color   = span.get("color", 0)
                    origin  = span.get("origin", (0, 0))
                    bbox    = span.get("bbox", [0, 0, 1, 1])
                    rect    = fitz.Rect(bbox)

                    span_meta = {
                        "size":       size,
                        "color":      color,
                        "diagonal":   is_diagonal,
                        "position":   _span_position_bucket(span, page_heights[page_num]),
                        "rect":       rect,
                    }

                    # Normalise key: lowercase, strip punctuation edges
                    key = re.sub(r'^[\s\W]+|[\s\W]+$', '', raw.lower())
                    if key:
                        text_index[key].append((page_num, rect, span_meta))

    # ── B & C: Score and decide which texts are watermarks ───────────────────
    threshold   = max(1, int(page_count * REPEAT_PAGE_THRESHOLD))
    wm_texts: Dict[str, float] = {}   # key → confidence score 0-1

    for key, occurrences in text_index.items():
        if not key:
            continue
        score = 0.0
        sample_meta = occurrences[0][2]

        # Keyword match
        if _is_watermark_text(key, sample_meta["size"], sample_meta["color"]):
            score += 0.5

        # Cross-page frequency
        unique_pages = len({p for p, _, _ in occurrences})
        if unique_pages >= threshold:
            score += 0.3
        elif unique_pages >= 2:
            score += 0.15

        # Position bonus (header/footer watermarks are very common)
        positions = [m["position"] for _, _, m in occurrences]
        if positions.count("footer") >= threshold or positions.count("header") >= threshold:
            score += 0.25
        if positions.count("center") >= threshold:
            score += 0.15

        # Diagonal text is almost always a watermark
        if any(m["diagonal"] for _, _, m in occurrences):
            score += 0.4

        # Faint color
        if _color_is_faint(sample_meta["color"]):
            score += 0.3

        # URL or social pattern
        for pat in URL_PATTERNS + SOCIAL_PATTERNS:
            if re.search(pat, key, re.IGNORECASE):
                score += 0.4
                break

        if score >= 0.5:
            wm_texts[key] = min(score, 1.0)

    if not wm_texts:
        return hits

    # ── D: Precise span-level redaction ──────────────────────────────────────
    # Group rects to redact per page
    redact_map: Dict[int, List[Tuple[fitz.Rect, str]]] = defaultdict(list)

    for key, confidence in wm_texts.items():
        for page_num, rect, meta in text_index[key]:
            redact_map[page_num].append((rect, key[:60]))

    # Also do direct page.search_for for keyword-matched multi-word strings
    for page_num in range(page_count):
        page = doc[page_num]
        for kw in WATERMARK_KEYWORDS + ["dreamers", "edu hub", "coaching", "academy",
                                         "institute", "classes", "tutorial"]:
            try:
                found = page.search_for(kw, quads=False)
                for rect in found:
                    # Only add if this region isn't already marked for redaction
                    # AND the text at this location is truly isolated (not inside body)
                    if _is_isolated_watermark(page, rect, page_heights[page_num]):
                        redact_map[page_num].append((rect, kw))
            except Exception:
                pass

    # Apply redactions page by page
    for page_num, rects_and_labels in redact_map.items():
        page = doc[page_num]
        applied = False
        for rect, label in rects_and_labels:
            try:
                # Expand rect slightly to catch anti-aliased edges
                padded = fitz.Rect(
                    rect.x0 - 1, rect.y0 - 1,
                    rect.x1 + 1, rect.y1 + 1
                )
                page.add_redact_annot(padded, fill=(1, 1, 1))
                hits.append(WatermarkHit(
                    page=page_num + 1, type="text",
                    detail=label, removed=True
                ))
                applied = True
            except Exception as exc:
                logger.debug("Pass3 redact page %d: %s", page_num, exc)

        if applied:
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception as exc:
                logger.debug("Pass3 apply_redactions page %d: %s", page_num, exc)

    return hits


def _is_isolated_watermark(page: fitz.Page, rect: fitz.Rect, page_height: float) -> bool:
    """
    Return True if the text at *rect* is in a watermark-typical position
    (header, footer, or center diagonal) rather than inside body text flow.
    """
    y_rel = rect.y0 / max(page_height, 1)
    # Header zone
    if y_rel < 0.10:
        return True
    # Footer zone
    if y_rel > 0.88:
        return True
    # Center zone (diagonal watermarks)
    if 0.30 < y_rel < 0.70:
        # Check if there's very little other text around this rect
        clip = fitz.Rect(rect.x0 - 50, rect.y0 - 20, rect.x1 + 50, rect.y1 + 20)
        surrounding = page.get_text("text", clip=clip).strip()
        # If the surrounding area has little text, it's likely isolated watermark
        words = surrounding.split()
        if len(words) <= 6:
            return True
    return False
