"""
Autentikasi API key untuk semua endpoint.

Sistem utama (proyek utama) harus menyertakan header berikut di setiap request:
    X-API-Key: <nilai API_KEY di .env>

Cara pakai di endpoint FastAPI:
    from app.security import verify_api_key
    from fastapi import Depends

    @router.post("/contoh")
    def contoh(user=Depends(verify_api_key)):
        ...
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import API_KEY

# FastAPI akan membaca header "X-API-Key" secara otomatis
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(_api_key_header)) -> str:
    """
    Dependency FastAPI. Melempar 403 jika header tidak ada atau nilai salah.
    Kembalikan key yang valid agar bisa dilog jika diperlukan.
    """
    if not key or key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key tidak valid atau tidak disertakan.",
        )
    return key
