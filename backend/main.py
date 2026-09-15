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

# Streamlit runs on port 8501, the API on 8000, so the browser needs CORS.
# Locked to localhost rather than "*" - "*" is fine on a laptop and wrong anywhere else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
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
