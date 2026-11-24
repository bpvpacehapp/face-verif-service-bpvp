"""
Orchestrator Layanan Verifikasi Presensi BPVP (Internal Admin Service)

Mengimplementasikan 11 Poin Alur Baru:
1. Menerima data upload absen (user_id, foto, tanggal upload).
2. Mengecek apakah user sudah memiliki dataset referensi sebelumnya.
   - Jika belum: Status 'NO_DATASET' (foto disimpan sebagai dataset awal).
3. Menjalankan Face Anti-Spoofing (MiniFASNet): menghasilkan status `is_spoof` True/False.
4. Menjalankan Pipeline Verifikasi: YOLO Detection -> Crop -> MediaPipe Alignment -> VGGFace2.
5. Membandingkan dengan dataset foto-foto user sebelumnya (Cosine Similarity %).
6. Mengukur durasi waktu eksekusi mesin (detik).
7. Menyimpan embedding foto hari ini ke database (incremental learning).
8. Mencatat rekap lengkap ke database untuk kebutuhan tabel dan detail foto di Admin Panel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from app.services.detector import detect_and_crop, FaceDetectionError
from app.services.anti_spoofing import check_anti_spoof, AntiSpoofResult
from app.services.aligner import align_face
from app.services.embedder import extract_embedding
from app.services.comparator import verify_face
from app.db import (
    get_embeddings,
    save_embedding,
    save_attendance_record,
    count_user_embeddings,
)


@dataclass
class AttendanceProcessResult:
    user_id: str
    process_status: str             # 'DONE', 'NO_DATASET', 'FAILED'
    verification_status: str        # 'VERIFIED', 'NEEDS_REVIEW', 'NO_DATASET', 'FAILED'
    similarity_percentage: str      # contoh: "84.5%" (atau "0.0%" jika no dataset)
    similarity_score: float         # desimal 0.8450
    is_spoof: bool                  # True jika terindikasi foto layar HP/kertas
    spoof_label: str                # 'REAL' atau 'SPOOF'
    det_confidence: float           # skor keyakinan deteksi wajah (YOLO)
    rotation_angle: float           # sudut derajat rotasi alignment (MediaPipe)
    execution_time_seconds: float   # waktu pemrosesan di mesin (detik)
    attendance_date: str | None
    record_id: int | None = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "process_status": self.process_status,
            "verification_status": self.verification_status,
            "similarity_percentage": self.similarity_percentage,
            "similarity_score": self.similarity_score,
            "is_spoof": self.is_spoof,
            "spoof_label": self.spoof_label,
            "det_confidence": self.det_confidence,
            "rotation_angle": self.rotation_angle,
            "execution_time_seconds": self.execution_time_seconds,
            "attendance_date": self.attendance_date,
            "record_id": self.record_id,
            "message": self.message,
        }


def process_attendance_verification(
    user_id: str,
    image_bgr: np.ndarray,
    photo_url: str | None = None,
    attendance_date: str | None = None,
) -> AttendanceProcessResult:
    """
    Eksekusi alur lengkap verifikasi presensi internal kantor.
    """
    start_time = time.perf_counter()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Gambar absen kosong atau tidak valid.")

    # 1. Deteksi wajah dengan YOLOv12n-face
    det_result = detect_and_crop(image_bgr, margin=0.15)
    bbox = det_result.bbox_original

    # 2. Anti-Spoofing (Liveness Check)
    # Hasil spoofing ini tidak menggagalkan proses, hanya memberikan status True/False
    spoof_result = check_anti_spoof(image_bgr, bbox)

    # 3. Alignment wajah dengan MediaPipe Face Mesh
    align_result = align_face(det_result.face_crop, target_size=(160, 160))

    # 4. Ekstraksi embedding 512 dimensi (VGGFace2)
    new_embedding = extract_embedding(align_result.aligned_face)

    # 5. Cek dataset foto-foto sebelumnya milik user ini di database
    db_embeddings = get_embeddings(user_id)
    has_dataset = db_embeddings.size > 0

    # 6. Simpan foto hari ini ke database (agar menjadi dataset acuan hari berikutnya)
    source_ref = photo_url or f"attendance://{user_id}/{attendance_date or time.time()}"
    save_embedding(
        user_id=user_id,
        embedding=new_embedding,
        source_url=source_ref,
        det_confidence=round(det_result.confidence, 4),
        attendance_date=attendance_date,
    )

    # 7. Evaluasi Verifikasi:
    if not has_dataset:
        # KASUS A: Pegawai baru / belum memiliki foto referensi sebelumnya
        process_status = "NO_DATASET"
        verif_status = "NO_DATASET"
        sim_pct = "0.0%"
        sim_score = 0.0
        msg = "Pegawai baru (belum ada dataset sebelumnya). Foto berhasil disimpan sebagai dataset awal."
    else:
        # KASUS B: Sudah memiliki foto referensi -> hitung tingkat kemiripan (1:1 Cosine Similarity)
        process_status = "DONE"
        verif = verify_face(new_embedding, db_embeddings)
        verif_status = verif.status
        sim_pct = verif.similarity_percentage
        sim_score = verif.similarity_score
        msg = f"Verifikasi selesai: {sim_pct} ({verif_status})."

    # 8. Hitung durasi waktu pemrosesan pada mesin
    duration = round(time.perf_counter() - start_time, 4)

    # 9. Catat riwayat ke tabel attendance_verifications (untuk tabel Admin Panel)
    record_id = None
    try:
        record_id = save_attendance_record(
            user_id=user_id,
            attendance_date=attendance_date,
            photo_url=photo_url,
            process_status=process_status,
            is_spoof=spoof_result.is_spoof,
            similarity_percentage=sim_pct,
            similarity_score=sim_score,
            verification_status=verif_status,
            execution_time_seconds=duration,
            notes=f"YOLO Conf: {det_result.confidence:.2%}, Spoof: {spoof_result.label}",
        )
    except Exception:
        pass

    return AttendanceProcessResult(
        user_id=user_id,
        process_status=process_status,
        verification_status=verif_status,
        similarity_percentage=sim_pct,
        similarity_score=sim_score,
        is_spoof=spoof_result.is_spoof,
        spoof_label=spoof_result.label,
        det_confidence=round(det_result.confidence, 4),
        rotation_angle=round(align_result.rotation_angle, 2),
        execution_time_seconds=duration,
        attendance_date=attendance_date,
        record_id=record_id,
        message=msg,
    )
