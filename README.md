# ClearMark — Dual-Engine Backend

## Project Structure
```
├── config/
│   └── settings.py          # All env-var config (Redis, S3, limits, etc.)
├── core/
│   └── triage.py            # Triage router: Fast Track vs AI Track
├── engines/
│   ├── fast_track.py        # PyMuPDF engine (3 sweep passes)
│   └── ai_track.py          # AI engine (rasterise + OpenCV mask + inpaint)
├── storage/
│   └── manager.py           # LocalStorage + S3Storage abstraction
├── workers/
│   └── tasks.py             # Celery task: process_document
├── api/
│   └── main.py              # FastAPI routes
├── tests/
│   └── test_pipeline.py     # 32 tests (25 run without Redis)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### Local dev (no Docker)
```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 3. Start Celery worker (terminal 1)
celery -A workers.tasks.celery_app worker --loglevel=info

# 4. Start API (terminal 2)
uvicorn api.main:app --reload --port 8000
```

### Docker Compose (full stack)
```bash
cp .env.example .env
docker compose up --build
# API:    http://localhost:8000
# Docs:   http://localhost:8000/docs
# Flower: docker compose --profile monitoring up
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/upload` | Upload PDF/image → returns `task_id` |
| GET | `/api/v1/status/{task_id}` | Poll progress/result |
| GET | `/api/v1/download/{folder}/{file}` | Download cleaned file |
| POST | `/api/v1/retry/{task_id}` | Re-run with AI engine |
| DELETE | `/api/v1/file/{folder}/{file}` | Early delete |
| GET | `/api/v1/health` | Health check |

## Enable AI Inpainting (LaMa)
```env
USE_AI_INPAINTING=true
LAMA_MODEL_PATH=./models/lama   # place best.pt here
```
Without a LaMa checkpoint, the AI track falls back to OpenCV Telea inpainting automatically.

## Run Tests
```bash
pytest tests/ -v -k "not TestAPI and not test_batch"
```
