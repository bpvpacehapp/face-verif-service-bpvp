# Face Verification Service — BPVP

Microservice verifikasi wajah untuk sistem presensi BPVP.
Dibangun dengan **FastAPI** + **PostgreSQL**, berjalan terpisah dari aplikasi utama.

## Fitur (bertahap)
- [x] Koneksi database PostgreSQL (`face_embeddings`)
- [x] CORS + autentikasi API key (`X-API-Key`)
- [x] Face detection & bounding box crop (YOLOv12n-face + margin 15%)
- [ ] Face alignment (MediaPipe)
- [ ] Face recognition / embedding (FaceNet512)
- [ ] Anti-spoofing / liveness detection
- [ ] Endpoint enrollment (`POST /face-datasets`)
- [ ] Endpoint verifikasi (`POST /verify`)

## Struktur
```
face-verif-service/
  app/
    __init__.py
    config.py       # konfigurasi dari .env
    db.py           # operasi database (psycopg 3)
    main.py         # FastAPI, CORS, health check
    security.py     # API key dependency
    services/       # pipeline AI (detector, aligner, embedder, dsb.)
    routers/        # endpoint FastAPI per fitur
  schema.sql        # DDL tabel face_embeddings
  test_db.py        # tes koneksi database
  requirements.txt
  .env              # TIDAK di-commit (ada di .gitignore)
```

## Menjalankan
```bash
# 1. Buat dan aktifkan venv
python -m venv venv
.\venv\Scripts\activate   # Windows

# 2. Install dependensi
pip install -r requirements.txt

# 3. Salin .env dan isi nilainya
copy .env.example .env    # lalu edit API_KEY dan DATABASE_URL

# 4. Jalankan server
uvicorn app.main:app --reload --port 8001
```

Server berjalan di **http://localhost:8001**
Swagger docs: **http://localhost:8001/docs**

## Autentikasi
Setiap request (kecuali `/health`) harus menyertakan header:
```
X-API-Key: <nilai API_KEY di .env>
```

## Database
PostgreSQL lokal via Laragon. Jalankan `schema.sql` sekali untuk membuat tabel.
Tanpa pgvector untuk sementara — embedding disimpan sebagai `REAL[]`,
pencocokan cosine similarity dihitung di Python (NumPy).
