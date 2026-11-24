"""
Uji Coba Verifikasi Wajah Pribadi (Simulasi Presensi BPVP)

Struktur Pengujian:
1. datasets/reference/ : Taruh foto wajah master/referensi Anda (1 atau beberapa foto).
2. datasets/live_test/ : Taruh foto selfie Anda yang ingin diuji (seperti kamera live absen).

Jalankan:
    python test_my_face.py
"""

from pathlib import Path
import cv2
import numpy as np

from app.config import SIMILARITY_THRESHOLD, MODEL_NAME
from app.services.detector import detect_and_crop, FaceDetectionError
from app.services.aligner import align_face
from app.services.embedder import extract_embedding
from app.services.comparator import verify_face
from app.db import (
    save_embedding,
    get_embeddings,
    delete_user_embeddings,
    save_verification_log,
)

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / "datasets" / "reference"
LIVE_DIR = ROOT / "datasets" / "live_test"

REF_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)

USER_ID = "PEGAWAI-001"  # ID simulasi untuk pengujian lokal
SUPPORTED = {".jpg", ".jpeg"}

print("=" * 75)
print("   SIMULASI VERIFIKASI PRESENSI WAJAH BPVP (PENGUJIAN LOKAL)")
print("=" * 75)
print(f"Model Pengenal : {MODEL_NAME} (InceptionResnetV1 - 512 Dimensi)")
print(f"Threshold .env : {SIMILARITY_THRESHOLD:.0%} ({SIMILARITY_THRESHOLD})")
print(f"User ID Uji    : {USER_ID}\n")

# ── TAHAP 1: ENROLLMENT (Pendaftaran Foto Referensi ke Database) ──────────────
ref_images = sorted([p for p in REF_DIR.iterdir() if p.suffix.lower() in SUPPORTED])

if not ref_images:
    print(f"[!] FOLDER KOSONG: {REF_DIR}")
    print("    Silakan masukkan minimal 1 foto wajah referensi Anda (.jpg/.jpeg)")
    print("    ke dalam folder 'datasets/reference/' lalu jalankan ulang script ini.\n")
    raise SystemExit(1)

print(f"--- [LANGKAH 1] Mendaftarkan Foto Referensi dari: {REF_DIR.name}/ ---")

# Reset data lama user uji coba agar selalu fresh
delete_user_embeddings(USER_ID)

saved_ref_count = 0
for idx, path in enumerate(ref_images, 1):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[{idx}] Gagal membaca file: {path.name}")
        continue

    try:
        # Pipeline: Deteksi -> Alignment -> Embedding
        det = detect_and_crop(img, margin=0.15)
        aln = align_face(det.face_crop, target_size=(160, 160))
        emb = extract_embedding(aln.aligned_face)

        # Simpan ke PostgreSQL
        emb_id = save_embedding(
            user_id=USER_ID,
            embedding=emb,
            source_url=f"local://reference/{path.name}",
            det_confidence=round(det.confidence, 4),
        )
        saved_ref_count += 1
        print(f"  [✓] {path.name:<25} -> Tersimpan di DB (ID: {emb_id}) | YOLO: {det.confidence:.1%}")

    except FaceDetectionError as e:
        print(f"  [✗] {path.name:<25} -> SKIP DETEKSI: {e}")
    except Exception as e:
        print(f"  [✗] {path.name:<25} -> ERROR: {e}")

print(f"Total {saved_ref_count} foto referensi berhasil didaftarkan ke Database PostgreSQL!\n")


# ── TAHAP 2: VERIFIKASI PRESENSI (Pengujian Foto Selfie Baru) ─────────────────
live_images = sorted([p for p in LIVE_DIR.iterdir() if p.suffix.lower() in SUPPORTED])

if not live_images:
    print(f"[!] FOLDER KOSONG: {LIVE_DIR}")
    print("    Silakan masukkan foto selfie yang ingin diuji (.jpg/.jpeg)")
    print("    ke dalam folder 'datasets/live_test/' lalu jalankan ulang script ini.\n")
    raise SystemExit(1)

print(f"--- [LANGKAH 2] Memverifikasi Foto Presensi dari: {LIVE_DIR.name}/ ---")
print(f"{'No':<3} {'Foto Selfie Live':<25} {'Kemiripan':<12} {'Status':<16} {'Keterangan'}")
print("-" * 75)

# Ambil data embedding referensi dari database PostgreSQL
db_embeddings = get_embeddings(USER_ID)
if db_embeddings.size == 0:
    print("[!] Tidak ada embedding referensi yang ditemukan di database.")
    raise SystemExit(1)

for idx, path in enumerate(live_images, 1):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{idx:<3} {path.name:<25} {'GAGAL BACA':<12}")
        continue

    try:
        # Pipeline Live: Deteksi -> Alignment -> Embedding
        det = detect_and_crop(img, margin=0.15)
        aln = align_face(det.face_crop, target_size=(160, 160))
        live_emb = extract_embedding(aln.aligned_face)

        # Bandingkan dengan referensi di database (1:1 Cosine Similarity)
        verif = verify_face(live_emb, db_embeddings, threshold=SIMILARITY_THRESHOLD)

        # Simpan jejak riwayat verifikasi ke database audit log
        try:
            save_verification_log(
                user_id=USER_ID,
                similarity_score=verif.similarity_score,
                status=verif.status,
                is_match=verif.is_match,
                live_embedding=live_emb,
            )
        except Exception:
            pass

        sim_str = verif.similarity_percentage
        if verif.is_match:
            status_text = "✓ VERIFIED"
            ket = f"Lolos presensi (>= {SIMILARITY_THRESHOLD:.0%})"
        else:
            status_text = "✗ NEEDS_REVIEW"
            ket = f"Di bawah {SIMILARITY_THRESHOLD:.0%}, butuh konfirmasi admin"

        print(f"{idx:<3} {path.name:<25} {sim_str:<12} {status_text:<16} {ket}")

    except FaceDetectionError as e:
        print(f"{idx:<3} {path.name:<25} {'-':<12} {'DITOLAK':<16} Deteksi: {e}")
    except Exception as e:
        print(f"{idx:<3} {path.name:<25} {'-':<12} {'ERROR':<16} {e}")

print("=" * 75)
print("Simulasi selesai. Riwayat verifikasi tercatat di database PostgreSQL.")
