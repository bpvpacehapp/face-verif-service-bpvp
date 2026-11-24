"""
Script Cek Isi Database Lokal PostgreSQL
Membaca langsung isi tabel face_embeddings dan attendance_verifications.

Jalankan:
    python check_db_data.py
"""

import psycopg
from psycopg.rows import dict_row
from app.config import DATABASE_URL

print("=" * 80)
print("     PEMERIKSAAN DATA LOKAL POSTGRESQL (face_detection_bpvp)")
print("=" * 80)

try:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # 1. Cek Tabel face_embeddings (Foto Master / Dataset)
            cur.execute("""
                SELECT id, user_id, model_name, source_url, det_confidence, 
                       array_length(embedding, 1) as dim, attendance_date 
                FROM face_embeddings 
                ORDER BY id ASC
            """)
            emb_rows = cur.fetchall()

            print(f"\n[1] TABEL: face_embeddings (Total: {len(emb_rows)} baris data)")
            print(f"{'ID':<4} {'User ID':<15} {'Dimensi':<9} {'Conf YOLO':<11} {'Tanggal Upload':<24} {'Sumber Foto'}")
            print("-" * 80)
            for r in emb_rows:
                tgl = str(r['attendance_date'])[:19] if r['attendance_date'] else "-"
                print(f"{r['id']:<4} {r['user_id']:<15} {r['dim']:<9} {r['det_confidence']:<11} {tgl:<24} {r['source_url']}")

            # 2. Cek Tabel attendance_verifications (Riwayat Presensi & Spoofing)
            cur.execute("""
                SELECT id, user_id, attendance_date, verification_status, 
                       similarity_percentage, is_spoof, execution_time_seconds 
                FROM attendance_verifications 
                ORDER BY id DESC
            """)
            verif_rows = cur.fetchall()

            print(f"\n[2] TABEL: attendance_verifications (Total: {len(verif_rows)} riwayat presensi)")
            print(f"{'ID':<4} {'User ID':<15} {'Status':<14} {'Kemiripan':<11} {'Spoof?':<8} {'Waktu (s)':<11} {'Tanggal Absen'}")
            print("-" * 80)
            for r in verif_rows:
                tgl = str(r['attendance_date'])[:19] if r['attendance_date'] else "-"
                spoof_txt = "YA (SPOOF)" if r['is_spoof'] else "ASLI (REAL)"
                print(f"{r['id']:<4} {r['user_id']:<15} {r['verification_status']:<14} {str(r['similarity_percentage']):<11} {spoof_txt:<8} {r['execution_time_seconds']:<11.4f} {tgl}")

    print("\n" + "=" * 80)
    print("KESIMPULAN: Data di atas BUKAN simulasi palsu, melainkan TERSIMPAN FISIK di PostgreSQL Laragon!")
    print("=" * 80)

except Exception as e:
    print(f"[!] Gagal terhubung ke database: {e}")
