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

# ── Model AI & Verifikasi ─────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "vggface2")
EMBEDDING_DIM = 512  # dimensi output InceptionResnetV1 (VGGFace2)

# Batas minimum kemiripan wajah (default 0.60 atau 60%)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))

# Mode Lingkungan (development vs production)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
# Sembunyikan threshold di response jika production
SHOW_THRESHOLD = (
    os.getenv("SHOW_THRESHOLD", "true").lower() == "true"
    and ENVIRONMENT != "production"
)

# ── Keamanan API ──────────────────────────────────────────
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY belum diatur di file .env")

_origins_raw = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _origins_raw.split(",") if o.strip()]