"""
Script Verifikasi Endpoint API FastAPI (face-verif-service)
Jalankan:
    python test_api.py
"""

from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.config import API_KEY, SIMILARITY_THRESHOLD

client = TestClient(app)

print("=== MEMULAI TEST ENDPOINT FASTAPI ===")

# 1. Health check (publik)
r = client.get("/health")
assert r.status_code == 200, f"Health check gagal: {r.text}"
print(f"[1] GET /health          -> OK ({r.json()})")

# 2. Ping tanpa API Key -> harus ditolak (403)
r = client.get("/ping")
assert r.status_code == 403, f"Seharusnya ditolak tanpa key, didapat: {r.status_code}"
print(f"[2] GET /ping (no key)   -> Berhasil Ditolak 403 ({r.json()['detail']})")

# 3. Ping dengan API Key -> harus diterima (200)
headers = {"X-API-Key": API_KEY}
r = client.get("/ping", headers=headers)
assert r.status_code == 200, f"Gagal dengan valid key: {r.text}"
print(f"[3] GET /ping (with key) -> Berhasil Diterima 200 ({r.json()})")

# 4. Cek status endpoint dataset user acak
r = client.get("/face-datasets/TEST-99999", headers=headers)
assert r.status_code == 200
print(f"[4] GET /face-datasets   -> OK ({r.json()})")

# 5. Uji coba endpoint verifikasi dengan foto sampel lokal (jika ada)
sample_img = Path("datasets/sample/01_single_frontal_pria.jpg")
if sample_img.exists():
    with open(sample_img, "rb") as f:
        files = {"file": ("selfie.jpg", f, "image/jpeg")}
        data = {"user_id": "NON_EXISTENT_USER"}
        r = client.post("/verify", data=data, files=files, headers=headers)
        # User belum terdaftar -> harus 404
        assert r.status_code == 404
        print(f"[5] POST /verify (user belum daftar) -> Berhasil Ditangani 404 ({r.json()['detail']})")

print("\n[✓] SELURUH UJI COBA ENDPOINT BERHASIL!")
print("Dokumentasi interaktif Swagger siap diakses di: http://localhost:8001/docs")
