# Face Verification Service — BPVP (Internal Admin Service)

Microservice verifikasi presensi wajah dan liveness detection untuk sistem internal kantor BPVP.
Layanan ini berjalan di latar belakang (*background service*) untuk kebutuhan audit dan panel kontrol Admin kantor.

## Alur Kerja Sistem (11 Poin Sesuai Arahan Mentor)
1. **User Absen Normal**: Pegawai melakukan absensi di aplikasi seperti biasa tanpa mengetahui atau menjalankan service ini.
2. **Trigger Otomatis**: Sistem utama memicu microservice ini saat ada foto presensi baru diunggah.
3. **Pengecekan Dataset Awal**:
   - Jika pegawai baru dan belum memiliki riwayat foto sebelumnya $\rightarrow$ status **`NO_DATASET`**, foto pertama otomatis disimpan sebagai dataset acuan awal.
   - Jika sudah ada riwayat foto sebelumnya $\rightarrow$ status **`ON_PROCESS`** menuju verifikasi.
4. **Anti-Spoofing (MiniFASNet)**: Mendeteksi apakah foto diambil dari layar monitor/HP atau kertas cetak (*print/replay attack*). Hasilnya berupa `is_spoof: True/False` sebagai pertimbangan internal admin (tidak menggagalkan proses).
5. **Pipeline Verifikasi Wajah**:
   - Deteksi wajah menggunakan **YOLOv12n-face** + Bounding Box Crop (margin 15%).
   - Face Alignment menggunakan **MediaPipe Face Mesh** (rotasi sudut mata horizontal sejajar 0°).
   - Ekstraksi vektor embedding 512 dimensi menggunakan **InceptionResnetV1 (VGGFace2)**.
   - Komparasi 1:1 Cosine Similarity dengan foto-foto sebelumnya milik user untuk menghitung persentase kemiripan (misal `86.4%`).
6. **Waktu Eksekusi Mesin**: Menghitung durasi pemrosesan pada server (dalam detik/milidetik).
7. **Penyimpanan Berkelanjutan (*Incremental Learning*)**: Foto yang berhasil diverifikasi otomatis diekstrak dan disimpan ke database untuk memperkaya dataset biometrik pegawai di masa mendatang.
8. **Siklus Status Terstruktur**: `NO_DATASET`, `ON_PROCESS`, dan `DONE`.
9. **Integrasi Admin Panel**:
   - Endpoint rekap tabel pegawai (tanggal absen, skor kemiripan %, status spoof, durasi mesin).
   - Endpoint Action Detail untuk menampilkan riwayat seluruh foto pegawai bersangkutan berdasarkan tanggal upload.

---

## Struktur Database (`schema.sql`)
- **`face_embeddings`**: Menyimpan dataset biometrik vektor 512-D, tanggal upload (`attendance_date`), dan status keaktifan.
- **`attendance_verifications`**: Menyimpan riwayat audit presensi untuk tabel Admin Panel (`user_id`, `attendance_date`, `is_spoof`, `similarity_percentage`, `verification_status`, `execution_time_seconds`).

---

## Daftar Endpoint API

Semua endpoint dilindungi oleh header:
```
X-API-Key: <nilai API_KEY di .env>
```

| Method | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/trigger-verification` | Trigger otomatis dari sistem utama saat ada upload absen baru (`user_id`, `photo_url`, `attendance_date`) |
| `POST` | `/verify` | Verifikasi presensi upload foto langsung (file JPG/JPEG) |
| `POST` | `/verify/base64` | Verifikasi presensi via JSON string Base64 |
| `GET` | `/admin/verifications` | Menampilkan tabel rekap absensi untuk dashboard Admin Panel |
| `GET` | `/admin/verifications/{user_id}/photos` | Tombol 'Action Detail' menampilkan seluruh foto pegawai berdasarkan tanggal upload |
| `POST` | `/face-datasets` | Pendaftaran batch dataset wajah pegawai dari URL |
| `GET` | `/health` | Pemeriksaan kesehatan layanan server |

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
