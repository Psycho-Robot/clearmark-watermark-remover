import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fitz
import numpy as np

from config.settings import AI_BATCH_PAGES, RASTERISE_DPI, USE_AI_INPAINTING, LAMA_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass
class AITrackResult:
    success:         bool
    pages_processed: int = 0
    output_path:     Optional[Path] = None
    error:           str = ""
    warnings:        List[str] = field(default_factory=list)


# ─── LaMa (optional) ──────────────────────────────────────────────────────────

class LamaInpainter:
    def __init__(self, model_path: str):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self._load(model_path)

    def _load(self, path):
        import torch
        p = Path(path) / "best.pt"
        if not p.exists():
            raise FileNotFoundError(f"LaMa model not found: {p}")
        m = torch.jit.load(str(p), map_location=self.device)
        m.eval()
        return m

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch
        img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        msk  = mask.astype(np.float32) / 255
        it   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        mt   = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(it, mt)
        res = out.squeeze(0).permute(1,2,0).cpu().numpy()
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


_lama: Optional[LamaInpainter] = None

def _get_lama():
    global _lama
    if not USE_AI_INPAINTING:
        return None
    if _lama is None:
        try:
            _lama = LamaInpainter(LAMA_MODEL_PATH)
        except Exception as exc:
            logger.warning("LaMa unavailable: %s", exc)
    return _lama


# ─── Background color detection ───────────────────────────────────────────────

def detect_background_color(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the true background color of the page.
    For scanned notes: almost always near-white (240-255).
    Uses the mode of the brightest 30% of pixels.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Take top 30% brightest pixels as background candidates
    threshold = np.percentile(gray, 70)
    bg_mask = gray >= threshold
    if bg_mask.sum() < 100:
        return np.array([255, 255, 255], dtype=np.uint8)
    bg_pixels = image_bgr[bg_mask]
    # Use median for robustness
    bg_color = np.median(bg_pixels, axis=0).astype(np.uint8)
    return bg_color


# ─── Mask generation ──────────────────────────────────────────────────────────

def generate_watermark_mask(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mask, bg_color):
      mask     : uint8, 255 = watermark pixel
      bg_color : uint8 [B,G,R] detected background
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg_color = detect_background_color(image_bgr)

    # ── Strategy 1: Light-gray faint text (150–235 range) ────────────────────
    # This is the core watermark range for scanned PDFs
    mask_s1 = cv2.inRange(gray, 150, 235)

    # Remove tiny noise blobs (< 15px area)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_s1 = cv2.morphologyEx(mask_s1, cv2.MORPH_OPEN, k_open)

    # Keep only components that are text-sized (not huge image regions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_s1, 8)
    mask_s1_clean = np.zeros((h, w), np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw   = stats[i, cv2.CC_STAT_WIDTH]
        ch   = stats[i, cv2.CC_STAT_HEIGHT]
        # Accept text-character sized components only
        if 10 < area < (h * w * 0.10) and cw < w * 0.7 and ch < h * 0.4:
            mask_s1_clean[labels == i] = 255

    # ── Strategy 2: Color-tinted watermarks ─────────────────────────────────
    # Two sub-strategies:
    #   2a: Semi-transparent blended watermarks (diff 8-80 from bg)
    #   2b: Solid colored watermarks (diff > 80 but gray in mid-range)
    img_float  = image_bgr.astype(np.float32)
    bg_float   = bg_color.astype(np.float32)
    diff       = np.abs(img_float - bg_float)
    diff_max   = diff.max(axis=2)

    mask_s2 = np.zeros((h, w), np.uint8)

    # 2a: Semi-transparent watermarks (small deviation from background)
    s2a = (diff_max >= 8) & (diff_max <= 80) & (gray > 140)
    mask_s2[s2a] = 255

    # 2b: Solid colored watermarks (larger deviation but NOT dark body text)
    # Key: gray between 80-220 means it's neither pure background nor dark text
    # AND it must have a significant color deviation from background
    s2b = (diff_max > 80) & (diff_max <= 220) & (gray >= 55) & (gray <= 225)
    # Extra check: not already covered by dark-text exclusion
    # A colored watermark has HIGH saturation — check if any channel is bright
    channel_max = image_bgr.max(axis=2)
    has_bright_channel = channel_max > 140  # at least one bright channel
    mask_s2[s2b & has_bright_channel] = 255

    # Clean up mask_s2
    mask_s2 = cv2.morphologyEx(mask_s2, cv2.MORPH_OPEN, k_open)
    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask_s2, 8)
    mask_s2_clean = np.zeros((h, w), np.uint8)
    for i in range(1, num2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if 10 < area < (h * w * 0.12):
            mask_s2_clean[labels2 == i] = 255

    # ── Strategy 3: Diagonal structure detection ──────────────────────────────
    # Look for text in non-horizontal orientation (diagonal watermarks)
    mask_s3 = np.zeros((h, w), np.uint8)
    # Use gradient direction analysis
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle  = np.arctan2(np.abs(sobely), np.abs(sobelx) + 1e-6)
    # Diagonal: angle between 25-65 degrees
    diagonal_edges = (angle > 0.44) & (angle < 1.13)  # radians
    # Combined with faint pixel range
    faint = (gray > 160) & (gray < 235)
    diagonal_faint = diagonal_edges & faint
    mask_s3[diagonal_faint] = 255

    # Dilate to connect nearby diagonal pixels
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_s3 = cv2.dilate(mask_s3, k_d, iterations=1)
    # Keep only large enough diagonal components
    num3, labels3, stats3, _ = cv2.connectedComponentsWithStats(mask_s3, 8)
    mask_s3_clean = np.zeros((h, w), np.uint8)
    for i in range(1, num3):
        area = stats3[i, cv2.CC_STAT_AREA]
        if 50 < area < (h * w * 0.20):
            mask_s3_clean[labels3 == i] = 255

    # ── Combine all strategies ────────────────────────────────────────────────
    combined = cv2.bitwise_or(mask_s1_clean, mask_s2_clean)
    combined = cv2.bitwise_or(combined, mask_s3_clean)

    # Final dilate to cover anti-aliased edges
    k_final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined = cv2.dilate(combined, k_final, iterations=1)

    # Remove huge blobs (probably not watermarks)
    num_f, labels_f, stats_f, _ = cv2.connectedComponentsWithStats(combined, 8)
    final = np.zeros((h, w), np.uint8)
    for i in range(1, num_f):
        area = stats_f[i, cv2.CC_STAT_AREA]
        if area < h * w * 0.25:
            final[labels_f == i] = 255

    return final, bg_color


# ─── Smart inpainting ─────────────────────────────────────────────────────────

def smart_inpaint(img_bgr, mask, bg_color, warnings):
    """Stage 1: per-pixel bg estimation. Stage 2: fill. Stage 3: edge smooth. Stage 4: refine."""
    if mask.sum() == 0:
        return img_bgr
    result = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # Stage 1 — estimate per-pixel background using large median blur
    ksize = max(51, (min(h, w) // 8) | 1)
    bg_estimate = cv2.medianBlur(img_bgr, ksize)
    # For white pages use global bg_color (cleaner), colored pages use per-pixel
    if float(bg_color.mean()) > 230:
        fill = np.full_like(img_bgr, bg_color)
    else:
        fill = bg_estimate

    # Stage 2 — fill watermark pixels with estimated background
    result[mask > 0] = fill[mask > 0]

    # Stage 3 — smooth only the mask boundary
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.subtract(cv2.dilate(mask, k, iterations=2),
                        cv2.erode(mask, k, iterations=1))
    if edge.sum() > 0:
        blurred = cv2.GaussianBlur(result, (5, 5), 0)
        result[edge > 0] = blurred[edge > 0]

    # Stage 4 — LaMa or light OpenCV refinement at edges only
    lama = _get_lama()
    if lama:
        try:
            return lama.inpaint(result, mask)
        except Exception as exc:
            warnings.append(f"LaMa failed: {exc}")

    wm_pct = float(mask.sum() / 255) / (h * w)
    if wm_pct < 0.20 and edge.sum() > 0:
        try:
            refined = cv2.inpaint(result, edge, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            result[edge > 0] = refined[edge > 0]
        except Exception as exc:
            warnings.append(f"Edge refinement skipped: {exc}")

    return result


# ─── Rasterise PDF page ───────────────────────────────────────────────────────

def _page_to_bgr(doc: fitz.Document, page_num: int, dpi: int) -> np.ndarray:
    page = doc[page_num]
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


# ─── PDF reconstruction ───────────────────────────────────────────────────────

def _build_pdf(cleaned: List[np.ndarray], output_path: Path,
                orig_doc: Optional[fitz.Document] = None) -> None:
    """
    Reconstruct PDF from cleaned images.
    Uses JPEG at quality=94 for ~10x faster encoding vs PNG.
    Scanned pages are already lossy (camera/scanner) so JPEG is appropriate.
    """
    out = fitz.open()
    for i, img_bgr in enumerate(cleaned):
        # JPEG is 10x faster than PNG for large scanned images
        ok, buf = cv2.imencode(".jpg", img_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            # Fallback to PNG if JPEG fails
            ok, buf = cv2.imencode(".png", img_bgr)
            if not ok:
                raise RuntimeError(f"Could not encode page {i}")

        img_bytes = buf.tobytes()

        if orig_doc and i < len(orig_doc):
            w = orig_doc[i].rect.width
            h = orig_doc[i].rect.height
        else:
            hp, wp = img_bgr.shape[:2]
            w = wp * 72.0 / RASTERISE_DPI
            h = hp * 72.0 / RASTERISE_DPI

        page = out.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=img_bytes)

    out.save(str(output_path), garbage=3, deflate=False)
    out.close()


# ─── Main entry ───────────────────────────────────────────────────────────────

def run_ai_track(input_path: Path, output_path: Path,
                  image_only: bool = False) -> AITrackResult:
    warnings: List[str] = []

    try:
        if image_only:
            src_images = [_load_image(input_path)]
            orig_doc   = None
            page_count = 1
        else:
            try:
                orig_doc   = fitz.open(str(input_path))
            except Exception as exc:
                return AITrackResult(success=False, error=f"Cannot open PDF: {exc}")
            page_count = len(orig_doc)
            src_images = None

        cleaned: List[np.ndarray] = []
        batch_size = AI_BATCH_PAGES

        if image_only:
            batches = [src_images]
        else:
            batches = [
                list(range(i, min(i + batch_size, page_count)))
                for i in range(0, page_count, batch_size)
            ]

        total_wm_pixels = 0

        for batch in batches:
            if image_only:
                pages_bgr = batch
            else:
                logger.debug("Rasterising pages %s at %d DPI", batch, RASTERISE_DPI)
                pages_bgr = [_page_to_bgr(orig_doc, idx, RASTERISE_DPI)
                              for idx in batch]

            for img_bgr in pages_bgr:
                mask, bg_color = generate_watermark_mask(img_bgr)
                wm_pixels = int(mask.sum() / 255)
                total_wm_pixels += wm_pixels

                # Cache bg_color for first non-trivial detection
                if wm_pixels < 30:
                    cleaned.append(img_bgr)
                    continue

                logger.debug("Page WM pixels: %d, bg_color: BGR%s", wm_pixels, bg_color)
                result = smart_inpaint(img_bgr, mask, bg_color, warnings)
                cleaned.append(result)

        _build_pdf(cleaned, output_path, orig_doc if not image_only else None)

        if not image_only and orig_doc:
            orig_doc.close()

        logger.info("AI track: %d pages, %d total WM pixels → %s",
                    page_count, total_wm_pixels, output_path.name)
        return AITrackResult(
            success=True,
            pages_processed=page_count,
            output_path=output_path,
            warnings=warnings,
        )

    except Exception as exc:
        logger.exception("AI track error")
        return AITrackResult(success=False, error=str(exc), warnings=warnings)
