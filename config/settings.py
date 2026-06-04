"""
config/settings.py — Central configuration for ClearMark backend.
All secrets are read from environment variables; sane defaults for local dev.
"""
import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR",  str(BASE_DIR / "uploads")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR",  str(BASE_DIR / "processed")))
TEMP_DIR   = Path(os.getenv("TEMP_DIR",    str(BASE_DIR / "temp")))

for d in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Redis / Celery ────────────────────────────────────────────────────────────
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL  = os.getenv("CELERY_BROKER_URL",  REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# ── Storage backend  ("local" | "s3") ─────────────────────────────────────────
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")

# AWS S3 — only used when STORAGE_BACKEND="s3"
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION",            "us-east-1")
S3_BUCKET             = os.getenv("S3_BUCKET",             "clearmark-files")
S3_PRESIGN_EXPIRY     = int(os.getenv("S3_PRESIGN_EXPIRY", "86400"))  # 24 h

# ── File lifecycle ────────────────────────────────────────────────────────────
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", "86400"))   # 24 hours

# ── Processing ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB      = int(os.getenv("MAX_FILE_SIZE_MB",   "100"))
RASTERISE_DPI         = int(os.getenv("RASTERISE_DPI",      "300"))
AI_BATCH_PAGES        = int(os.getenv("AI_BATCH_PAGES",     "4"))    # OOM guard
OPACITY_THRESHOLD     = float(os.getenv("OPACITY_THRESHOLD","0.85")) # flag if < this
REPEAT_PAGE_THRESHOLD = float(os.getenv("REPEAT_PAGE_THRESHOLD","0.5")) # 50 % pages

# ── AI inpainting ─────────────────────────────────────────────────────────────
# Set USE_AI_INPAINTING=false to skip heavy AI track (useful for low-resource envs)
USE_AI_INPAINTING    = os.getenv("USE_AI_INPAINTING", "false").lower() == "true"
LAMA_MODEL_PATH      = os.getenv("LAMA_MODEL_PATH",  str(BASE_DIR / "models" / "lama"))
INPAINTING_API_URL   = os.getenv("INPAINTING_API_URL", "")   # optional external API

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST",    "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT","8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# ── Watermark detection heuristics ───────────────────────────────────────────
WATERMARK_KEYWORDS = [
    "watermark", "confidential", "draft", "sample", "copy",
    "do not copy", "proprietary", "internal use", "restricted",
    "classified", "top secret", "for review", "preview", "demo",
    "trial", "evaluation", "not for distribution", "paid", "licensed",
    "copyright", "registered", "trademark", "do not distribute",
    "void", "proof", "approved", "rejected",
]
