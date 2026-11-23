import os
from pathlib import Path
from dotenv import load_dotenv

# Cari .env di direktori yang sama dengan folder app/ (yaitu venv/)
_base = Path(__file__).resolve().parent.parent  # → project root (face-verif-service/)
load_dotenv(_base / ".env")

# ── Database ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL belum diatur di file .env")

# ── Model AI ──────────────────────────────────────────────
MODEL_NAME = "facenet512"
EMBEDDING_DIM = 512  # dimensi output FaceNet512

# ── Keamanan API ──────────────────────────────────────────
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY belum diatur di file .env")

_origins_raw = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _origins_raw.split(",") if o.strip()]