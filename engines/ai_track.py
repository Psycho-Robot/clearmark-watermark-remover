"""
engines/ai_track.py — AI "Heavy Track" watermark removal engine.

Pipeline per page:
  1. Rasterise page at 300 DPI → PNG (via pdf2image or direct image load)
  2. Generate a binary inpaint mask via OpenCV (thresholding + morphology)
  3. Inpaint using LaMa model (if available) or OpenCV Telea fallback
  4. Reconstruct PDF from cleaned PNGs via PyMuPDF

Memory guard: pages are processed in batches of AI_BATCH_PAGES to avoid OOM.
"""
from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np

from config.settings import (
    AI_BATCH_PAGES,
    RASTERISE_DPI,
    USE_AI_INPAINTING,
    LAMA_MODEL_PATH,
    INPAINTING_API_URL,
)

logger = logging.getLogger(__name__)


# ─── Result type ───────────────────────────────────────────────────────────────

@dataclass
class AITrackResult:
    success:        bool
    pages_processed: int = 0
    output_path:    Optional[Path] = None
    error:          str = ""
    warnings:       List[str] = field(default_factory=list)


# ─── LaMa inpainting wrapper (optional) ──────────────────────────────────────

class LamaInpainter:
    """Thin wrapper around a local LaMa checkpoint."""

    def __init__(self, model_path: str):
        import torch
        from torchvision import transforms

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self._load_model(model_path)
        self.to_tensor = transforms.ToTensor()
        logger.info("LamaInpainter loaded on %s", self.device)

    def _load_model(self, path: str):
        import torch
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"LaMa model not found at {path}")
        # Standard LaMa checkpoint loading
        model = torch.jit.load(str(model_path / "best.pt"), map_location=self.device)
        model.eval()
        return model

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image_bgr : HxWx3 uint8 BGR
        mask      : HxW   uint8  (255 = inpaint, 0 = keep)
        returns   : HxWx3 uint8 BGR
        """
        import torch

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk     = mask.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        msk_t = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(img_t, msk_t)

        result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# Lazy singleton
_lama: Optional[LamaInpainter] = None

def _get_lama() -> Optional[LamaInpainter]:
    global _lama
    if not USE_AI_INPAINTING:
        return None
    if _lama is None:
        try:
            _lama = LamaInpainter(LAMA_MODEL_PATH)
        except Exception as exc:
            logger.warning("LaMa not available (%s) — falling back to OpenCV Telea", exc)
    return _lama


# ─── Mask generation ──────────────────────────────────────────────────────────

def _generate_watermark_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask (255 = watermark, 0 = keep) using multi-strategy
    OpenCV thresholding tuned for faint / diagonal text watermarks.

    Strategy A — light-gray text on white/light background:
      Convert to grayscale → invert → threshold at a low value to capture
      very faint marks without touching dark body text.

    Strategy B — coloured semi-transparent stamps:
      Convert to HSV → detect low-saturation + high-value pixels (near-white)
      that differ from the surrounding background.

    Both masks are OR-ed together then morphologically cleaned.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── Strategy A: faint gray text ──────────────────────────────────────
    # Invert so watermark (light gray) becomes dark
    inv = cv2.bitwise_not(gray)
    # Binary threshold: anything darker than ~230/255 in the original = candidate
    _, mask_a = cv2.threshold(inv, 25, 255, cv2.THRESH_BINARY)

    # Remove very small noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN, kernel_open)

    # Keep only "large" connected components — watermarks span large areas
    # Small body text characters will also pass, but we use a frequency /
    # size heuristic to exclude them (components > 5% of page area are WM)
    mask_a = _keep_large_components(mask_a, min_area_fraction=0.0002,
                                    max_area_fraction=0.40)

    # ── Strategy B: colour cast (e.g. blue/red stamps) ───────────────────
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Detect high-value (bright) but low-to-medium saturation (washed-out colours)
    lower = np.array([0,  10, 200])
    upper = np.array([180, 80, 255])
    mask_b = cv2.inRange(hsv, lower, upper)

    # ── Combine ───────────────────────────────────────────────────────────
    mask = cv2.bitwise_or(mask_a, mask_b)

    # Dilate slightly to cover anti-aliased edges around watermark letters
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def _keep_large_components(mask: np.ndarray,
                            min_area_fraction: float,
                            max_area_fraction: float) -> np.ndarray:
    """Remove connected components outside the [min, max] fraction of total area."""
    total = mask.shape[0] * mask.shape[1]
    min_a = int(total * min_area_fraction)
    max_a = int(total * max_area_fraction)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_a <= area <= max_a:
            out[labels == i] = 255
    return out


# ─── Inpainting ───────────────────────────────────────────────────────────────

def _inpaint_image(image_bgr: np.ndarray, mask: np.ndarray,
                   warnings: List[str]) -> np.ndarray:
    """
    Apply inpainting.  Try LaMa first; fall back to OpenCV Telea.
    """
    lama = _get_lama()
    if lama is not None:
        try:
            return lama.inpaint(image_bgr, mask)
        except Exception as exc:
            warnings.append(f"LaMa inpainting failed ({exc}), using OpenCV fallback")

    # OpenCV Telea — fast, good for thin text marks
    result = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result


# ─── Rasterisation ────────────────────────────────────────────────────────────

def _pdf_page_to_bgr(doc: fitz.Document, page_num: int, dpi: int) -> np.ndarray:
    """Render a single PDF page to a BGR numpy array."""
    page = doc[page_num]
    zoom  = dpi / 72.0
    mat   = fitz.Matrix(zoom, zoom)
    pix   = page.get_pixmap(matrix=mat, alpha=False)
    # pix.samples is a bytes object: RGBRGB...
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _load_image_file(path: Path) -> np.ndarray:
    """Load a standalone image file as BGR."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read image: {path}")
    return img


# ─── PDF reconstruction ───────────────────────────────────────────────────────

def _build_pdf_from_images(cleaned_images: List[np.ndarray],
                            output_path: Path,
                            original_doc: Optional[fitz.Document] = None) -> None:
    """
    Compile a list of BGR numpy arrays back into a PDF.
    Preserves original page dimensions if original_doc is provided.
    """
    out_doc = fitz.open()

    for i, img_bgr in enumerate(cleaned_images):
        # Encode to PNG in memory
        success, buf = cv2.imencode(".png", img_bgr)
        if not success:
            raise RuntimeError(f"Failed to encode cleaned page {i} as PNG")
        png_bytes = buf.tobytes()

        # Determine target page size (match original if available)
        if original_doc and i < len(original_doc):
            orig_page = original_doc[i]
            w = orig_page.rect.width
            h = orig_page.rect.height
        else:
            # Fall back to image pixel size at 300 DPI
            h_px, w_px = img_bgr.shape[:2]
            w = w_px * 72.0 / 300.0
            h = h_px * 72.0 / 300.0

        page = out_doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=png_bytes)

    out_doc.save(str(output_path), garbage=4, deflate=True)
    out_doc.close()


# ─── Main entry point ──────────────────────────────────────────────────────────

def run_ai_track(input_path: Path, output_path: Path,
                 image_only: bool = False) -> AITrackResult:
    """
    Full AI pipeline.

    image_only=True  → input is a standalone image, output is a single-page PDF.
    image_only=False → input is a (scanned) PDF, output is a cleaned PDF.
    """
    warnings: List[str] = []

    try:
        # ── 1. Load source ─────────────────────────────────────────────────
        if image_only:
            src_images = [_load_image_file(input_path)]
            original_doc = None
            page_count   = 1
        else:
            try:
                original_doc = fitz.open(str(input_path))
            except Exception as exc:
                return AITrackResult(success=False, error=f"Cannot open PDF: {exc}")
            page_count   = len(original_doc)
            src_images   = None   # will be rasterised in batches

        cleaned_images: List[np.ndarray] = []

        # ── 2. Process in batches (OOM guard) ──────────────────────────────
        batch_size  = AI_BATCH_PAGES
        total_pages = page_count

        if image_only:
            batches = [src_images]
        else:
            # Generate page-index batches
            batches = [
                list(range(i, min(i + batch_size, total_pages)))
                for i in range(0, total_pages, batch_size)
            ]

        for batch in batches:
            if image_only:
                pages_bgr = batch  # already numpy arrays
            else:
                logger.debug("Rasterising pages %s at %d DPI", batch, RASTERISE_DPI)
                pages_bgr = [
                    _pdf_page_to_bgr(original_doc, idx, RASTERISE_DPI)
                    for idx in batch
                ]

            for img_bgr in pages_bgr:
                # ── 3. Generate mask ──────────────────────────────────────
                mask = _generate_watermark_mask(img_bgr)

                # Skip inpainting if no watermark pixels detected
                if mask.max() == 0:
                    cleaned_images.append(img_bgr)
                    continue

                # ── 4. Inpaint ────────────────────────────────────────────
                cleaned = _inpaint_image(img_bgr, mask, warnings)
                cleaned_images.append(cleaned)

        # ── 5. Reconstruct PDF ─────────────────────────────────────────────
        if image_only:
            _build_pdf_from_images(cleaned_images, output_path)
        else:
            _build_pdf_from_images(cleaned_images, output_path, original_doc)
            original_doc.close()

        logger.info("AI track complete: %d pages → %s", total_pages, output_path.name)
        return AITrackResult(
            success=True,
            pages_processed=total_pages,
            output_path=output_path,
            warnings=warnings,
        )

    except Exception as exc:
        logger.exception("AI track engine error")
        return AITrackResult(success=False, error=str(exc), warnings=warnings)
