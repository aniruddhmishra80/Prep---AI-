"""
Every setting the app needs, read once from .env and imported everywhere else.

Interview line: "All configuration lives in one file so nothing else in the
codebase ever calls os.getenv()."
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent

# On Vercel / serverless only /tmp is writable. Locally, use the data/ folder.
_default_data_dir = "/tmp" if os.getenv("VERCEL") else str(BASE_DIR / "data")
DATA_DIR = Path(os.getenv("DATA_DIR", _default_data_dir))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "interview.db")))

for _dir in (UPLOAD_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models
# Which chat model to use: google | mistral | ollama | huggingface
# Default is Gemini. Any of the four works - the rest of the codebase never
# knows which one is active, because llm.py hands back a LangChain Runnable.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

# Which embedding model to use: local | google
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
HF_REPO_ID = os.getenv("HF_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")

# 0.0 makes questions repetitive, 1.0 makes the model drift off-format and break
# the JSON contract. 0.4 is the compromise.
TEMPERATURE = float(os.getenv("TEMPERATURE", "").strip() or "0.4")

# ---------------------------------------------------------------- rag
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50      # so an idea cut at a boundary survives whole in one chunk
RETRIEVE_K = 4

# ---------------------------------------------------------------- interview
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "").strip() or "8")
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
START_DIFFICULTY = 1                 # index into DIFFICULTY_LEVELS -> "medium"
LEVEL_UP_AT = 0.75                   # score above this -> harder next question
LEVEL_DOWN_AT = 0.40                 # score below this -> easier next question

# ---------------------------------------------------------------- scoring
WEIGHT_KEYWORD = 0.20
WEIGHT_COSINE = 0.30
WEIGHT_LLM = 0.50

# ---------------------------------------------------------------- api
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Comma-separated list of allowed CORS origins, e.g. https://myapp.vercel.app
# "*" means all origins - safe for a public API, change if you need auth cookies.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
