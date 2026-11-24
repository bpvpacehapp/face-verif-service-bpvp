"""
Script Otomatis Setup Database PostgreSQL Lokal (Laragon)
Membuat user presensi_app, database face_detection_bpvp, dan seluruh tabelnya.

Jalankan:
    python setup_db_local.py
"""

import sys
import psycopg

# URL superuser default PostgreSQL di Laragon / Windows (user postgres tanpa password atau postgres)
SUPERUSER_URLS = [
    "postgresql://postgres@localhost:5432/postgres",
    "postgresql://postgres:root@localhost:5432/postgres",
    "postgresql://postgres:postgres@localhost:5432/postgres",
    "postgresql://postgres:admin@localhost:5432/postgres",
    "postgresql://postgres:bpvp123@localhost:5432/postgres",
]

print("=== SETUP DATABASE POSTGRESQL (face_detection_bpvp) ===")

conn_super = None
for url in SUPERUSER_URLS:
    try:
        conn_super = psycopg.connect(url, autocommit=True)
        print(f"[✓] Berhasil terhubung ke PostgreSQL menggunakan superuser: {url.split('@')[0]}")
        break
    except Exception:
        continue

if conn_super is None:
    print("[!] Tidak dapat terhubung ke PostgreSQL dengan user 'postgres' default.")
    print("    Silakan buat manual lewat DBeaver menggunakan panduan Cara 2 di bawah.")
    sys.exit(1)

with conn_super:
    # 1. Buat role presensi_app jika belum ada
    with conn_super.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'presensi_app'")
        if not cur.fetchone():
            cur.execute("CREATE ROLE presensi_app WITH LOGIN PASSWORD 'bpvp123' CREATEDB")
            print("[✓] User/Role 'presensi_app' berhasil dibuat (password: bpvp123).")
        else:
            print("[✓] User/Role 'presensi_app' sudah ada.")

    # 2. Buat database face_detection_bpvp jika belum ada
    with conn_super.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'face_detection_bpvp'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE face_detection_bpvp OWNER presensi_app")
            print("[✓] Database 'face_detection_bpvp' berhasil dibuat.")
        else:
            print("[✓] Database 'face_detection_bpvp' sudah ada.")

    with conn_super.cursor() as cur:
        cur.execute("GRANT ALL PRIVILEGES ON DATABASE face_detection_bpvp TO presensi_app")

conn_super.close()

# 3. Hubungkan langsung sebagai presensi_app ke database face_detection_bpvp dan buat tabel
APP_DB_URL = "postgresql://presensi_app:bpvp123@localhost:5432/face_detection_bpvp"
print(f"\n[~] Menghubungkan ke database face_detection_bpvp sebagai 'presensi_app'...")

try:
    with psycopg.connect(APP_DB_URL, autocommit=True) as conn_app:
        from pathlib import Path
        schema_file = Path(__file__).resolve().parent / "schema.sql"
        sql_content = schema_file.read_text(encoding="utf-8")

        # Eksekusi script DDL tabel
        with conn_app.cursor() as cur:
            cur.execute(sql_content)
        print("[✓] Seluruh tabel (face_embeddings, attendance_verifications) berhasil dibuat!")

    print("\n✅ SETUP DATABASE SELESAI & SIAP DIGUNAKAN!")
    print("Silakan jalankan pengujian sekarang:")
    print("  python test_my_face.py")

except Exception as e:
    print(f"[!] Gagal membuat tabel: {e}")
