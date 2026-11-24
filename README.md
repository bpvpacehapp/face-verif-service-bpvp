# Face Verification Service — BPVP

Microservice independen untuk **verifikasi biometrik wajah (1:1)** dan **passive liveness detection (anti-spoofing)** pada sistem presensi internal BPVP. Berjalan secara asinkron di latar belakang (*background audit service*) untuk menyuplai data analitik ke **Admin Panel** tanpa membebani aplikasi utama.

> 💡 **Interactive Architecture Workbench:** Buka visualisasi interaktif lengkap (Pan/Zoom, Node Inspector, Animasi Alur, & ERD) di [`docs/architecture.html`](docs/architecture.html).

---

## 1. Arsitektur Microservice

Service ini dirancang terpisah (*decoupled*) dari aplikasi presensi utama agar pemrosesan komputasi AI (PyTorch, YOLO, MediaPipe) hemat RAM/CPU dan mudah diskalakan.

```mermaid
flowchart LR
    subgraph Client["1. Aplikasi Utama BPVP"]
        A["Pegawai Absen\n(Selfie Live JPG)"] --> B["Main Backend"]
    end

    subgraph Security["2. Gateway & Auth"]
        C["FastAPI (:8001)\nCORS + X-API-Key"]
    end

    subgraph Pipeline["3. AI Verification Pipeline (Singleton Models)"]
        D["YOLOv12n-face\nDeteksi & Crop (+15% margin)"]
        E["MiniFASNet\nPassive Anti-Spoofing"]
        F["MediaPipe Mesh\nEye Alignment (0° Horizontal)"]
        G["InceptionResnetV1\nVGGFace2 512-D + L2 Norm"]
        H["Cosine Comparator\n1:1 Matching vs Threshold"]
    end

    subgraph Storage["4. PostgreSQL (face_detection_bpvp)"]
        I[("face_embeddings\n(512-D REAL[])")]
        J[("attendance_verifications\n(Audit Log)")]
    end

    subgraph Admin["5. Admin Panel"]
        K["Dashboard Audit\n& Riwayat Foto"]
    end

    B -- "POST /trigger-verification" --> C
    C --> D
    D --> E & F
    F --> G
    G -- "Simpan Vektor Baru\n(Incremental Learning)" --> I
    I -- "Ambil Referensi User" --> H
    G --> H
    E & H -- "Rekap Skor, Spoof & Durasi" --> J
    K -- "GET /admin/verifications" --> C
    C -. "Query Rekap & Detail Foto" .-> J & I
```

### Keunggulan Desain (*Key Insights*)
- **Singleton Model Loading**: Seluruh bobot model AI dimuat satu kali ke memori saat startup, menjaga latensi verifikasi tetap rendah (~0.2–0.6 detik/foto).
- **Passive Auditing (Non-Blocking)**: Deteksi *spoofing* (layar HP/monitor atau kertas cetak) tidak memblokir absensi pegawai di lapangan, melainkan memberi penanda (`is_spoof: true`) agar Admin dapat melakukan audit secara objektif.
- **Incremental Learning**: Setiap foto presensi yang valid otomatis diekstrak menjadi embedding baru di database, sehingga akurasi pengenalan wajah pegawai semakin adaptif terhadap perubahan penampilan seiring waktu.

---

## 2. Alur Verifikasi Presensi

Siklus pemrosesan dari saat pegawai mengunggah foto presensi hingga tampil di dashboard Admin:

```mermaid
sequenceDiagram
    autonumber
    actor P as Pegawai
    participant M as Sistem Utama
    participant F as Face Verif Service
    participant DB as PostgreSQL
    actor A as Admin BPVP

    P->>M: Submit absen + foto selfie live
    M->>F: POST /trigger-verification (user_id, photo_url, date)
    activate F
    F->>F: Download JPG & Validasi Byte
    F->>F: 1. YOLOv12n-face (Deteksi 1 Wajah & Crop 15%)
    F->>F: 2. MiniFASNet (Cek Liveness: REAL / SPOOF)
    F->>F: 3. MediaPipe (Rotasi Mata 0° & Resize 160x160)
    F->>F: 4. VGGFace2 (Ekstrak Vektor 512-D L2-Normalized)
    F->>DB: Cek histori embedding milik user_id
    alt Pegawai Baru (Belum Ada Dataset)
        F->>DB: Simpan vektor perdana ke face_embeddings
        F->>DB: Catat status = NO_DATASET (0.0%)
    else Sudah Ada Dataset Referensi
        F->>F: Hitung 1:1 Cosine Similarity (Vektor Baru vs Histori)
        F->>DB: Simpan vektor baru ke face_embeddings (Incremental)
        F->>DB: Catat status = DONE (VERIFIED / NEEDS_REVIEW)
    end
    F-->>M: 200 OK (similarity_percentage, is_spoof, execution_time)
    deactivate F
    A->>F: GET /admin/verifications & /photos
    F-->>A: Tampilkan tabel audit & modal riwayat foto pegawai
```

### Siklus Status Verifikasi
| Status Proses (`process_status`) | Status Hasil (`verification_status`) | Arti & Kondisi |
| :--- | :--- | :--- |
| `NO_DATASET` | `NO_DATASET` | Pegawai baru pertama kali absen; foto langsung dijadikan referensi awal. |
| `DONE` | `VERIFIED` | Kemiripan wajah $\ge$ threshold (`0.60` / `60%`). |
| `DONE` | `NEEDS_REVIEW` | Kemiripan wajah $<$ threshold atau memerlukan tinjauan manual Admin. |
| `FAILED` | `FAILED` | Foto rusak, tidak ditemukan wajah, atau terdeteksi lebih dari satu wajah. |

---

## 3. Skema Database & ERD

Menggunakan **PostgreSQL** (`face_detection_bpvp`). Vektor disimpan sebagai tipe native `REAL[512]` (siap dimigrasikan ke `pgvector` di masa depan), sedangkan kalkulasi *cosine similarity* dieksekusi di memori menggunakan NumPy.

```mermaid
erDiagram
    FACE_EMBEDDINGS {
        bigserial id PK
        varchar(100) user_id "Index: ID/NIP Pegawai"
        real[512] embedding "Vektor Wajah 512-D (L2-Norm)"
        varchar(50) model_name "Default: vggface2"
        text source_url "URL/Path Foto Sumber"
        real det_confidence "Skor Deteksi YOLO (0.0-1.0)"
        timestamptz attendance_date "Waktu Upload Foto"
        boolean is_active "Status Aktif Dataset"
        timestamptz created_at
    }

    ATTENDANCE_VERIFICATIONS {
        bigserial id PK
        varchar(100) user_id "Index: ID/NIP Pegawai"
        timestamptz attendance_date "Waktu Presensi"
        text photo_url "URL Foto Presensi"
        varchar(20) process_status "NO_DATASET | ON_PROCESS | DONE | FAILED"
        boolean is_spoof "True jika indikasi layar/kertas"
        varchar(10) similarity_percentage "Format UI: 86.4%"
        real similarity_score "Skor Mentah: 0.8640"
        varchar(30) verification_status "VERIFIED | NEEDS_REVIEW | NO_DATASET"
        real execution_time_seconds "Durasi Proses Mesin (detik)"
        text notes "Ringkasan YOLO Conf & Spoof Label"
        timestamptz created_at
    }

    FACE_EMBEDDINGS ||--o{ ATTENDANCE_VERIFICATIONS : "user_id (1:N Logical Relation)"
```

> **Catatan Desain Database:** Atribut `det_confidence` hanya disimpan di `face_embeddings` (dan diringkas pada kolom `notes` di tabel audit) untuk menghindari duplikasi kolom antar-tabel.

---

## 4. Spesifikasi Endpoint API

Seluruh endpoint (kecuali `/health`) wajib menyertakan header autentikasi:
`X-API-Key: <API_KEY>`

| Method | Endpoint | Fungsi Utama |
| :--- | :--- | :--- |
| `POST` | `/trigger-verification` | **Webhook Utama**: Menerima `user_id`, `photo_url`, `attendance_date` dari sistem presensi. |
| `POST` | `/verify` | Uji verifikasi langsung menggunakan unggahan file `multipart/form-data` (JPG/JPEG). |
| `POST` | `/verify/base64` | Verifikasi langsung menggunakan payload JSON string Base64. |
| `POST` | `/face-datasets` | Pendaftaran (*enrollment*) kumpulan foto referensi awal pegawai via daftar URL. |
| `GET` | `/admin/verifications` | Mengambil daftar hasil audit presensi untuk tabel utama di Admin Panel. |
| `GET` | `/admin/verifications/{user_id}/photos` | **Action Detail**: Mengambil seluruh riwayat foto pegawai berdasarkan tanggal upload. |
| `GET` | `/health` | Cek status hidup server (*healthcheck* publik). |

---

## 5. Struktur Proyek & Instalasi Cepat

* `app/`
  * `main.py` — Entry point FastAPI, konfigurasi CORS, & registrasi router
  * `config.py` — Konfigurasi environment (`.env`), dimensi model, & threshold
  * `db.py` — Koneksi PostgreSQL (`psycopg3`) & query repositori
  * `security.py` — Validasi keamanan header `X-API-Key`
  * `routers/` — Endpoint `enrollment.py` & `verification.py`
  * `services/` — Pipeline AI (`detector.py`, `anti_spoofing.py`, `aligner.py`, `embedder.py`, `comparator.py`, `verifier.py`, `downloader.py`)
* `models/` — Bobot model lokal `yolov12n-face.pt` (diabaikan oleh Git)
* `datasets/` — Folder uji lokal `reference/` & `live_test/` (diproteksi penuh oleh `.gitignore`)
* `docs/architecture.html` — Visualisasi arsitektur interaktif mandiri (*Archify*)
* `schema.sql` — DDL inisialisasi tabel & hak akses PostgreSQL

### Menjalankan Layanan (Python 3.11)

```powershell
# 1. Buat & aktifkan virtual environment Python 3.11
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Pasang dependensi & siapkan konfigurasi
pip install -r requirements.txt
copy .env.example .env

# 3. Jalankan database schema di PostgreSQL (DBeaver / psql), lalu nyalakan server
uvicorn app.main:app --reload --port 8001
```

- **Base URL**: `http://localhost:8001`
- **Swagger UI**: `http://localhost:8001/docs`
