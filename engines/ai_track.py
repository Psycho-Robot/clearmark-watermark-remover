import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
            logger.warning("LaMa not available: %s — using OpenCV fallback", exc)
    return _lama


# ─── Mask generation — multi-strategy ─────────────────────────────────────────

def generate_watermark_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect watermark pixels using 4 strategies combined:

    S1 — Light-gray faint text (most common in scanned PDFs)
    S2 — Diagonal text detection via Hough line transform
    S3 — Colour-cast semi-transparent stamps
    S4 — Repeated texture pattern (tiled watermarks)

    Returns uint8 mask: 255 = watermark pixel, 0 = keep
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── S1: Faint gray text ──────────────────────────────────────────────────
    # Watermarks are typically lighter than body text but darker than background
    # Body text: <80 (dark), Background: >240 (white), Watermark: 160-230
    mask_s1 = np.zeros((h, w), np.uint8)
    # Light gray range — typical watermark zone
    light_gray = cv2.inRange(gray, 150, 235)
    # Remove very small noise
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light_gray = cv2.morphologyEx(light_gray, cv2.MORPH_OPEN, k3)
    # Keep only connected components large enough to be text chars
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(light_gray, 8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw   = stats[i, cv2.CC_STAT_WIDTH]
        ch_  = stats[i, cv2.CC_STAT_HEIGHT]
        # Text characters: reasonable size range
        if 20 < area < (h * w * 0.15) and cw < w * 0.8 and ch_ < h * 0.5:
            mask_s1[labels == i] = 255

    # ── S2: Diagonal text via Radon-like approach ────────────────────────────
    # Convert to binary, look for diagonal line structures
    mask_s2 = np.zeros((h, w), np.uint8)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Detect lines at various angles (watermarks often at 30-60 degrees)
    for angle in [30, 45, 60, -30, -45, -60]:
        kernel_len = max(w, h) // 8
        # Create diagonal structuring element
        rot_mat = cv2.getRotationMatrix2D((kernel_len // 2, kernel_len // 2), angle, 1)
        line_kernel = np.zeros((kernel_len, kernel_len), np.uint8)
        line_kernel[kernel_len // 2, :] = 1
        rotated_kernel = cv2.warpAffine(line_kernel.astype(np.float32), rot_mat,
                                         (kernel_len, kernel_len))
        rotated_kernel = (rotated_kernel > 0.5).astype(np.uint8)
        if rotated_kernel.sum() < 3:
            continue
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, rotated_kernel)
        # Only count pixels that are in the "faint" range (not solid dark text)
        diagonal_and_faint = cv2.bitwise_and(detected, mask_s1)
        mask_s2 = cv2.bitwise_or(mask_s2, diagonal_and_faint)

    # ── S3: Color-cast stamps (blue, red, green tints) ──────────────────────
    mask_s3 = np.zeros((h, w), np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # High value + low-medium saturation = washed-out coloured watermark
    color_mask = cv2.inRange(hsv, np.array([0, 8, 180]), np.array([180, 100, 255]))
    # Intersect with the faint gray mask for precision
    mask_s3 = cv2.bitwise_and(color_mask, mask_s1)

    # ── S4: Tiled / repeated pattern detection ───────────────────────────────
    mask_s4 = np.zeros((h, w), np.uint8)
    # Check if light_gray regions repeat with regularity (tiled watermarks)
    if mask_s1.sum() > 0:
        # Simple approach: if faint pixels are spread across >60% of page area
        # they're likely a repeated watermark pattern
        faint_coverage = mask_s1.sum() / 255 / (h * w)
        if 0.01 < faint_coverage < 0.35:
            mask_s4 = mask_s1.copy()

    # ── Combine all strategies ────────────────────────────────────────────────
    combined = cv2.bitwise_or(mask_s1, mask_s2)
    combined = cv2.bitwise_or(combined, mask_s3)

    # Post-process: dilate slightly to cover anti-aliased edges
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined = cv2.dilate(combined, k_dilate, iterations=1)

    # Final cleanup: remove very large blobs (probably not watermarks)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(combined, 8)
    final = np.zeros_like(combined)
    for i in range(1, num_labels2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if area < h * w * 0.30:   # skip blobs covering >30% of page
            final[labels2 == i] = 255

    return final


# ─── Inpainting ───────────────────────────────────────────────────────────────

def _inpaint(img_bgr: np.ndarray, mask: np.ndarray,
              warnings: List[str]) -> np.ndarray:
    lama = _get_lama()
    if lama:
        try:
            return lama.inpaint(img_bgr, mask)
        except Exception as exc:
            warnings.append(f"LaMa failed ({exc}), using OpenCV")

    # OpenCV Telea — good for thin text watermarks
    # Use INPAINT_NS for larger areas
    wm_area = int(mask.sum() / 255)
    page_area = img_bgr.shape[0] * img_bgr.shape[1]
    if wm_area / page_area > 0.05:
        result = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    else:
        result = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
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
    out = fitz.open()
    for i, img_bgr in enumerate(cleaned):
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError(f"Could not encode page {i}")
        png = buf.tobytes()

        if orig_doc and i < len(orig_doc):
            w = orig_doc[i].rect.width
            h = orig_doc[i].rect.height
        else:
            hp, wp = img_bgr.shape[:2]
            w = wp * 72.0 / RASTERISE_DPI
            h = hp * 72.0 / RASTERISE_DPI

        page = out.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=png)

    out.save(str(output_path), garbage=4, deflate=True)
    out.close()


# ─── Main entry ───────────────────────────────────────────────────────────────

def run_ai_track(input_path: Path, output_path: Path,
                  image_only: bool = False) -> AITrackResult:
    warnings: List[str] = []

    try:
        if image_only:
            src_images   = [_load_image(input_path)]
            orig_doc     = None
            page_count   = 1
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
                pages_bgr = [_page_to_bgr(orig_doc, idx, RASTERISE_DPI) for idx in batch]

            for img_bgr in pages_bgr:
                mask = generate_watermark_mask(img_bgr)
                wm_pixels = int(mask.sum() / 255)
                total_wm_pixels += wm_pixels

                if wm_pixels < 50:
                    # No watermark detected on this page
                    cleaned.append(img_bgr)
                    continue

                result = _inpaint(img_bgr, mask, warnings)
                cleaned.append(result)

        _build_pdf(cleaned, output_path, orig_doc if not image_only else None)

        if not image_only:
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
