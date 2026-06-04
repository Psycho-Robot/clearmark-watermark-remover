# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ClearMark"
LABEL description="Dual-engine PDF watermark remover — FastAPI + Celery"

WORKDIR /app

# Runtime system deps
#   poppler-utils → pdf2image
#   libgl1        → OpenCV headless still needs libGL on some distros
#   libglib2.0-0  → OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create non-root user
RUN useradd -m -u 1000 clearmark \
 && chown -R clearmark:clearmark /app

USER clearmark

# Storage directories (will be overridden by volume mount in compose)
RUN mkdir -p /app/storage_root/uploads /app/storage_root/processed /app/temp /app/models

EXPOSE 8000

# Default: run the API server (overridden in docker-compose for worker)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
