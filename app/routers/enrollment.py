"""
Router Endpoint: Enrollment (Create Face Datasets)
Sesuai Diagram Alur Mentor:
Input JSON: {"user_id": "12345", "urls": ["https://...", ...]}
Untuk setiap URL:
1. Download gambar (hanya JPG/JPEG)
2. Face Detection dengan YOLOv12n-face
3. Crop wajah dengan bounding box + margin
4. Face Alignment dengan MediaPipe
5. Ekstraksi Embedding 512-D dengan VGGFace2
6. Simpan Embedding ke Database PostgreSQL
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.security import verify_api_key
from app.services.downloader import download_image_as_bgr, ImageDownloadError
from app.services.detector import detect_and_crop, FaceDetectionError
from app.services.aligner import align_face
from app.services.embedder import extract_embedding
from app.db import save_embedding, get_embeddings, delete_user_embeddings, count_user_embeddings

router = APIRouter(dependencies=[Depends(verify_api_key)])


class EnrollmentRequest(BaseModel):
    user_id: str = Field(..., description="ID unik pegawai dari sistem utama BPVP", example="12345")
    urls: List[str] = Field(
        ...,
        description="Daftar tautan URL foto pegawai (format JPG/JPEG)",
        example=["https://example.com/foto-1.jpg", "https://example.com/foto-2.jpg"]
    )


class UrlResultItem(BaseModel):
    url: str
    status: str             # 'SAVED' atau 'FAILED'
    embedding_id: int | None = None
    confidence: float | None = None
    reason: str | None = None


class EnrollmentResponse(BaseModel):
    user_id: str
    total_requested: int
    total_saved: int
    results: List[UrlResultItem]


@router.post(
    "/face-datasets",
    response_model=EnrollmentResponse,
    summary="Daftarkan Dataset Wajah Pegawai (Enrollment)",
    tags=["Enrollment"]
)
async def create_face_datasets(payload: EnrollmentRequest):
    """
    Menerima daftar URL foto pegawai, mengekstrak fitur biometrik wajah (512-D),
    dan menyimpannya ke database sebagai data acuan verifikasi.
    """
    if not payload.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Daftar 'urls' tidak boleh kosong."
        )

    saved_count = 0
    results: List[UrlResultItem] = []

    for url in payload.urls:
        try:
            # 1. Download gambar dari URL (validasi format JPEG/JPG)
            image_bgr = await download_image_as_bgr(url)

            # 2 & 3. Deteksi wajah & crop dengan margin 15%
            det_result = detect_and_crop(image_bgr, margin=0.15)

            # 4. Alignment wajah (MediaPipe Face Mesh)
            align_result = align_face(det_result.face_crop, target_size=(160, 160))

            # 5. Ekstraksi embedding 512 dimensi (VGGFace2)
            embedding = extract_embedding(align_result.aligned_face)

            # 6. Simpan embedding ke database PostgreSQL
            emb_id = save_embedding(
                user_id=payload.user_id,
                embedding=embedding,
                source_url=url,
                det_confidence=round(det_result.confidence, 4),
            )

            saved_count += 1
            results.append(UrlResultItem(
                url=url,
                status="SAVED",
                embedding_id=emb_id,
                confidence=round(det_result.confidence, 4),
            ))

        except (ImageDownloadError, FaceDetectionError, ValueError) as exc:
            results.append(UrlResultItem(
                url=url,
                status="FAILED",
                reason=str(exc),
            ))
        except Exception as exc:
            results.append(UrlResultItem(
                url=url,
                status="FAILED",
                reason=f"Terjadi kesalahan internal: {exc}",
            ))

    return EnrollmentResponse(
        user_id=payload.user_id,
        total_requested=len(payload.urls),
        total_saved=saved_count,
        results=results,
    )


@router.get(
    "/face-datasets/{user_id}",
    summary="Cek Status Dataset Wajah Pegawai",
    tags=["Enrollment"]
)
def get_user_dataset_status(user_id: str):
    """Cek berapa banyak foto referensi wajah aktif yang dimiliki user."""
    count = count_user_embeddings(user_id)
    return {
        "user_id": user_id,
        "total_active_embeddings": count,
        "is_registered": count > 0,
    }


@router.delete(
    "/face-datasets/{user_id}",
    summary="Hapus Seluruh Dataset Wajah Pegawai",
    tags=["Enrollment"]
)
def delete_user_dataset(user_id: str):
    """Hapus seluruh embedding biometrik user jika ingin registrasi ulang."""
    deleted_rows = delete_user_embeddings(user_id)
    return {
        "user_id": user_id,
        "deleted_count": deleted_rows,
        "message": f"{deleted_rows} data embedding berhasil dihapus.",
    }
