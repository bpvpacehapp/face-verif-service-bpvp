"""
Pengujian Mandiri Alur Verifikasi Presensi Internal Kantor & Admin Panel

Skenario yang Diuji:
1. Pegawai Baru (Belum ada dataset):
   -> Status harus 'NO_DATASET', foto otomatis dijadikan dataset awal.
2. Pegawai Lama (Absen hari berikutnya):
   -> Menjalankan Anti-Spoofing (MiniFASNet).
   -> Menghitung persentase kemiripan wajah (Cosine Similarity).
   -> Mencatat durasi waktu eksekusi mesin (detik).
   -> Status harus 'DONE' & 'VERIFIED' / 'NEEDS_REVIEW'.
3. Simulasi Admin Panel:
   -> Membaca tabel rekap absensi pegawai.
   -> Menampilkan riwayat list foto pegawai (tombol Action Detail).

Jalankan:
    python test_admin_flow.py
"""

from pathlib import Path
import cv2

from app.services.verifier import process_attendance_verification
from app.db import (
    delete_user_embeddings,
    get_attendance_records,
    get_user_photo_history,
)

ROOT = Path(__file__).resolve().parent

# Ambil 2 foto sample untuk simulasi
REF_DIR = ROOT / "datasets" / "reference"
LIVE_DIR = ROOT / "datasets" / "live_test"
SAMPLE_DIR = ROOT / "datasets" / "sample"

# Cari foto yang tersedia di folder datasets
candidate_photos = []
for d in [REF_DIR, LIVE_DIR, SAMPLE_DIR]:
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in [".jpg", ".jpeg"]:
                candidate_photos.append(p)

if len(candidate_photos) < 2:
    print("[!] Dibutuhkan minimal 2 file foto .jpg di folder datasets untuk simulasi alur ini.")
    raise SystemExit(1)

foto_hari_1 = candidate_photos[0]
foto_hari_2 = candidate_photos[1]

USER_TEST = "SIMULASI-PEGAWAI-001"

print("=" * 80)
print("  SIMULASI ALUR VERIFIKASI PRESENSI INTERNAL KANTOR (SESUAI ARAHAN MENTOR)")
print("=" * 80)
print(f"User ID Simulasi : {USER_TEST}")
print(f"Foto Hari ke-1   : {foto_hari_1.name}")
print(f"Foto Hari ke-2   : {foto_hari_2.name}\n")

# Reset data user simulasi agar mulai dari 0
delete_user_embeddings(USER_TEST)

# ── SKENARIO 1: Pegawai Baru (Absen Hari ke-1) ──────────────────────────────
print(">>> [HARI KE-1] Pegawai Baru Pertama Kali Absen...")
img1 = cv2.imread(str(foto_hari_1))
res1 = process_attendance_verification(
    user_id=USER_TEST,
    image_bgr=img1,
    photo_url=f"storage://absen/hari1/{foto_hari_1.name}",
    attendance_date="2026-11-24T08:00:00+07:00",
)

print(f"  Process Status     : {res1.process_status}")
print(f"  Verification Status: {res1.verification_status}")
print(f"  Anti-Spoofing      : Spoof = {res1.is_spoof} ({res1.spoof_label})")
print(f"  Waktu Proses Mesin : {res1.execution_time_seconds:.4f} detik")
print(f"  Pesan Sistem       : {res1.message}\n")

assert res1.process_status == "NO_DATASET", "Seharusnya berstatus NO_DATASET"


# ── SKENARIO 2: Pegawai Yang Sama Absen di Hari ke-2 ─────────────────────────
print(">>> [HARI KE-2] Pegawai Yang Sama Absen Hari Berikutnya...")
img2 = cv2.imread(str(foto_hari_2))
res2 = process_attendance_verification(
    user_id=USER_TEST,
    image_bgr=img2,
    photo_url=f"storage://absen/hari2/{foto_hari_2.name}",
    attendance_date="2026-11-25T08:15:00+07:00",
)

print(f"  Process Status     : {res2.process_status}")
print(f"  Verification Status: {res2.verification_status}")
print(f"  Persentase Kemiripan: {res2.similarity_percentage}")
print(f"  Anti-Spoofing      : Spoof = {res2.is_spoof} ({res2.spoof_label})")
print(f"  Waktu Proses Mesin : {res2.execution_time_seconds:.4f} detik")
print(f"  Pesan Sistem       : {res2.message}\n")

assert res2.process_status == "DONE", "Seharusnya berstatus DONE"


# ── SKENARIO 3: Data untuk Tampilan Admin Panel ──────────────────────────────
print("=" * 80)
print(">>> [ADMIN PANEL] Simulasi Tabel Rekap & Action Detail Foto Pegawai")
print("=" * 80)

# A. Tabel Rekap Pegawai
print("\n[A] Tabel Rekap Absensi Pegawai (Admin Dashboard):")
print(f"{'User ID':<22} {'Tanggal Absen':<24} {'Status':<12} {'Kemiripan':<12} {'Spoof?':<8} {'Waktu (s)'}")
print("-" * 88)
records = get_attendance_records(user_id=USER_TEST)
for r in records:
    tgl = str(r['attendance_date'])[:19]
    spoof_txt = "YA" if r['is_spoof'] else "TIDAK"
    print(f"{r['user_id']:<22} {tgl:<24} {r['verification_status']:<12} {str(r['similarity_percentage']):<12} {spoof_txt:<8} {r['execution_time_seconds']:.4f}")

# B. Action Detail: List Foto Pegawai
print(f"\n[B] List Foto Pegawai Saat Klik Tombol 'Action Detail' ({USER_TEST}):")
photos = get_user_photo_history(USER_TEST)
for idx, p in enumerate(photos, 1):
    tgl = str(p['attendance_date'])[:19]
    print(f"  Foto #{idx}: Tanggal = {tgl} | Sumber = {p['source_url']} | Conf YOLO = {p['det_confidence']:.2%}")

print("\n" + "=" * 80)
print("[✓] SELURUH ALUR INTERNAL KANTOR & ADMIN PANEL BERHASIL DIUJI!")
print("=" * 80)
