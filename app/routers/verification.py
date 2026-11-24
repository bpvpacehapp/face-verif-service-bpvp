"""
Router Endpoint: Verifikasi Presensi Pegawai & Admin Panel Service

Menyediakan:
1. POST /verify               : Verifikasi presensi upload foto langsung (file JPG/JPEG).
2. POST /verify/base64        : Verifikasi presensi kiriman JSON Base64.
3. POST /trigger-verification : Trigger otomatis dari sistem utama saat ada upload absen baru (URL + tanggal).
4. GET  /admin/verifications  : Menampilkan list verifikasi pegawai untuk tabel Admin Panel.
5. GET  /admin/verifications/{user_id}/photos : Action Detail menampilkan riwayat foto pegawai & tanggal upload.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.security import verify_api_key
from app.services.downloader import decode_bytes_to_bgr, download_image_as_bgr, ImageDownloadError
from app.services.detector import FaceDetectionError
from app.services.verifier import process_attendance_verification, AttendanceProcessResult
from app.db import get_attendance_records, get_user_photo_history

router = APIRouter(dependencies=[Depends(verify_api_key)])


class Base64VerifyRequest(BaseModel):
    user_id: str = Field(..., description="ID unik pegawai", example="199201012020121001")
    image_base64: str = Field(..., description="Foto absen format Base64 JPEG")
    attendance_date: Optional[str] = Field(None, description="Tanggal upload absen (opsional)", example="2026-11-24T08:00:00+07:00")


class TriggerVerificationRequest(BaseModel):
    user_id: str = Field(..., description="ID unik pegawai", example="199201012020121001")
    photo_url: str = Field(..., description="URL foto absen yang baru saja diupload pegawai", example="https://storage.bpvp.go.id/absen/foto_01.jpg")
    attendance_date: Optional[str] = Field(None, description="Tanggal & jam pegawai mengupload foto absen", example="2026-11-24T08:00:00+07:00")


# ── 1. Endpoint Verifikasi via Form File ───────────────────────────────────────
@router.post(
    "/verify",
    summary="Verifikasi Presensi (Upload File Foto)",
    tags=["Verifikasi Presensi"]
)
async def verify_attendance_file(
    user_id: str = Form(..., description="ID pegawai"),
    attendance_date: Optional[str] = Form(None, description="Tanggal upload foto absen (opsional)"),
    file: UploadFile = File(..., description="File foto absen (wajib JPG/JPEG)"),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File foto kosong.")

    try:
        image_bgr = decode_bytes_to_bgr(content)
        result = process_attendance_verification(
            user_id=user_id,
            image_bgr=image_bgr,
            photo_url=f"upload://{file.filename}",
            attendance_date=attendance_date,
        )
        return result.to_dict()
    except (ImageDownloadError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FaceDetectionError as exc:
        raise HTTPException(status_code=422, detail=f"Deteksi wajah gagal: {exc}")


# ── 2. Endpoint Verifikasi via JSON Base64 ─────────────────────────────────────
@router.post(
    "/verify/base64",
    summary="Verifikasi Presensi (JSON Base64)",
    tags=["Verifikasi Presensi"]
)
def verify_attendance_base64(payload: Base64VerifyRequest):
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
        image_bgr = decode_bytes_to_bgr(image_bytes)
        result = process_attendance_verification(
            user_id=payload.user_id,
            image_bgr=image_bgr,
            attendance_date=payload.attendance_date,
        )
        return result.to_dict()
    except (binascii.Error, ImageDownloadError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FaceDetectionError as exc:
        raise HTTPException(status_code=422, detail=f"Deteksi wajah gagal: {exc}")


# ── 3. Endpoint Trigger Otomatis dari Database/Sistem Utama ──────────────────
@router.post(
    "/trigger-verification",
    summary="Trigger Verifikasi Presensi dari Database/Sistem Utama (URL)",
    tags=["Verifikasi Presensi"]
)
async def trigger_attendance_verification(payload: TriggerVerificationRequest):
    """
    Dipanggil otomatis oleh sistem utama BPVP saat ada pegawai yang mengupload absen baru.
    Mendownload foto dari URL, menjalankan anti-spoofing, verifikasi kemiripan,
    dan mencatat durasi waktu proses mesin.
    """
    try:
        image_bgr = await download_image_as_bgr(payload.photo_url)
        result = process_attendance_verification(
            user_id=payload.user_id,
            image_bgr=image_bgr,
            photo_url=payload.photo_url,
            attendance_date=payload.attendance_date,
        )
        return result.to_dict()
    except (ImageDownloadError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FaceDetectionError as exc:
        raise HTTPException(status_code=422, detail=f"Deteksi wajah gagal: {exc}")


# ── 4. Endpoint Admin: Tabel Rekap Verifikasi Presensi ────────────────────────
@router.get(
    "/admin/verifications",
    summary="List Rekap Verifikasi Pegawai untuk Tabel Admin Panel",
    tags=["Admin Panel"]
)
def list_admin_verifications(
    user_id: Optional[str] = Query(None, description="Filter berdasarkan ID Pegawai"),
    limit: int = Query(50, ge=1, le=200, description="Jumlah baris data"),
    offset: int = Query(0, ge=0, description="Offset halaman pagination"),
):
    """
    Menampilkan data tabel untuk dashboard Admin Panel:
    - user_id, tanggal absen, foto absen, status spoofing, persentase kemiripan, status proses.
    """
    records = get_attendance_records(user_id=user_id, limit=limit, offset=offset)
    return {
        "total": len(records),
        "limit": limit,
        "offset": offset,
        "data": records,
    }


# ── 5. Endpoint Admin: Action Detail List Foto Pegawai ─────────────────────────
@router.get(
    "/admin/verifications/{user_id}/photos",
    summary="Detail Riwayat Foto Pegawai Berdasarkan Tanggal Upload",
    tags=["Admin Panel"]
)
def get_admin_user_photos(user_id: str):
    """
    Dipanggil saat Admin mengklik tombol 'Action Detail' pada baris pegawai.
    Menampilkan seluruh daftar foto yang pernah diupload oleh pegawai bersangkutan
    lengkap dengan tanggal upload dan tingkat keyakinan deteksi wajahnya.
    """
    photos = get_user_photo_history(user_id)
    return {
        "user_id": user_id,
        "total_photos": len(photos),
        "photos": photos,
    }
