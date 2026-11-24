"""
Komponen Download Gambar untuk Enrollment (Create Face Datasets)

Fungsi:
1. Mengunduh gambar dari daftar URL secara aman & async menggunakan httpx.
2. Memvalidasi bahwa file berformat JPEG/JPG.
3. Mengonversi data bytes internet langsung ke format OpenCV NumPy array (BGR)
   di dalam memori RAM (tanpa menyimpan file sementara di harddisk).
"""

from __future__ import annotations

import cv2
import httpx
import numpy as np


class ImageDownloadError(Exception):
    """Dilempar saat download gambar gagal atau format bukan JPEG/JPG."""


# Byte header (magic numbers) untuk validasi format JPEG/JPG:
# File JPEG selalu diawali dengan byte 0xFF 0xD8 dan diakhiri 0xFF 0xD9
_JPEG_MAGIC = b"\xff\xd8"


def is_jpeg_content(data: bytes) -> bool:
    """Periksa apakah data biner adalah file gambar JPEG/JPG yang valid."""
    return data.startswith(_JPEG_MAGIC)


async def download_image_as_bgr(
    url: str,
    timeout: float = 15.0,
) -> np.ndarray:
    """
    Download gambar dari URL dan ubah menjadi OpenCV BGR numpy array.

    Args:
        url: Tautan HTTP/HTTPS foto referensi pegawai.
        timeout: Batas waktu request dalam detik.

    Returns:
        np.ndarray: Gambar dalam format OpenCV BGR.

    Raises:
        ImageDownloadError: Jika URL tidak dapat diakses atau bukan format JPEG/JPG.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise ImageDownloadError(f"URL tidak valid: {url}")

    headers = {
        "User-Agent": "BPVP-Face-Verification-Service/1.0"
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise ImageDownloadError(
                    f"Gagal mengunduh gambar (HTTP {response.status_code}) dari: {url}"
                )

            data = response.content

    except httpx.RequestError as exc:
        raise ImageDownloadError(f"Koneksi gagal saat mengakses URL {url}: {exc}")

    # 1. Validasi format file: wajib JPEG/JPG
    if not is_jpeg_content(data):
        content_type = response.headers.get("content-type", "unknown")
        raise ImageDownloadError(
            f"Format gambar ditolak ({content_type}). Sistem hanya menerima format JPG/JPEG."
        )

    # 2. Decode bytes menjadi OpenCV BGR numpy array di memori RAM
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        raise ImageDownloadError(f"File gambar dari URL rusak atau tidak dapat di-decode: {url}")

    return image_bgr


def decode_bytes_to_bgr(data: bytes) -> np.ndarray:
    """
    Decode data biner gambar langsung (misal dari upload kamera live) ke BGR.
    Wajib format JPEG/JPG.
    """
    if not is_jpeg_content(data):
        raise ImageDownloadError("Format gambar tidak valid. Hanya file JPEG/JPG yang diizinkan.")

    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ImageDownloadError("Gagal membaca data gambar kamera live.")

    return image_bgr
