"""
Entry point FastAPI — Face Verification Service (BPVP).

Jalankan:
    uvicorn app.main:app --reload --port 8001

Port 8001 agar tidak bentrok dengan sistem utama yang biasanya di 8000.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.security import verify_api_key
from app.routers import enrollment, verification

app = FastAPI(
    title="Face Verification Service — BPVP",
    description=(
        "Microservice verifikasi wajah (face detection, recognition, anti-spoofing) "
        "untuk sistem presensi BPVP. Semua endpoint membutuhkan header **X-API-Key**."
    ),
    version="0.1.0",
    # Sembunyikan docs di production jika diperlukan (ganti True → False)
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Hanya origin yang terdaftar di CORS_ORIGINS (.env) yang boleh mengirim request
# dari browser. Request server-to-server (sistem utama → layanan ini) tidak
# terpengaruh CORS, tapi API key tetap diperiksa.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,          # daftar dari .env
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ── Health check (publik, tanpa API key) ─────────────────────────────────────
@app.get("/health", tags=["Sistem"])
def health_check():
    """Cek apakah layanan berjalan. Tidak butuh API key."""
    return {"status": "ok", "service": "face-verif-service-bpvp"}


# ── Contoh endpoint terproteksi ──────────────────────────────────────────────
@app.get("/ping", tags=["Sistem"], dependencies=[Depends(verify_api_key)])
def ping():
    """
    Tes API key. Kirim header 'X-API-Key: <isi>' untuk mendapat pong.
    Gunakan ini dari sistem utama untuk memastikan koneksi dan kunci benar.
    """
    return {"message": "pong", "auth": "API key valid"}


# ── Daftarkan Router Bisnis ──────────────────────────────────────────────────
app.include_router(enrollment.router)
app.include_router(verification.router)
