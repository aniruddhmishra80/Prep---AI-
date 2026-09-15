"""
FastAPI entry point.

Run from the project root:
    uvicorn backend.main:app --reload

Swagger docs: http://127.0.0.1:8000/docs
Test every endpoint there before touching the frontend.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config, db
from backend.routers import interview, report, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on boot: create the SQLite tables if they don't exist yet."""
    db.init_db()
    print(f"[startup] LLM provider: {config.LLM_PROVIDER}")
    print(f"[startup] Embeddings:   {config.EMBEDDING_PROVIDER}")
    print(f"[startup] Database:     {config.DB_PATH}")
    yield


app = FastAPI(
    title="AI Voice Interview Platform",
    description="Upload a resume and a job description, get an adaptive, scored mock interview.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow any origin listed in CORS_ORIGINS env var (defaults to "*" for open APIs).
_origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,  # credentials not compatible with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["Upload & Analysis"])
app.include_router(interview.router, prefix="/interview", tags=["Interview"])
app.include_router(report.router, prefix="/report", tags=["Report"])


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "llm_provider": config.LLM_PROVIDER,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "max_questions": config.MAX_QUESTIONS,
    }
