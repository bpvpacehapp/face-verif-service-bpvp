"""
Komponen 3 — Face Embedding Extractor menggunakan InceptionResnetV1 (VGGFace2)

Fungsi:
1. Memuat model pre-trained VGGFace2 (PyTorch) sekali saat startup (singleton).
2. Menerima gambar wajah yang telah di-align berukuran 160x160 piksel.
3. Melakukan inferensi untuk mengekstrak vektor embedding 512 dimensi.
4. Menerapkan L2-Normalization agar perbandingan kemiripan (cosine similarity) akurat & cepat.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from app.config import EMBEDDING_DIM, MODEL_NAME

# ── Singleton Model & Device ──────────────────────────────────────────────────
_device: torch.device | None = None
_embedder_model: InceptionResnetV1 | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_model() -> InceptionResnetV1:
    global _embedder_model
    if _embedder_model is None:
        device = _get_device()
        # pretrained='vggface2' menghasilkan 512 dimensi embedding berkualitas tinggi
        _embedder_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return _embedder_model


def extract_embedding(face_img_bgr: np.ndarray) -> np.ndarray:
    """
    Ekstrak vektor embedding wajah (512 dimensi) dari gambar wajah yang telah di-align.

    Args:
        face_img_bgr: Gambar wajah dalam format numpy BGR (ukuran 160x160).

    Returns:
        Vektor numpy 1D float32 berukuran (512,) yang telah di-L2-normalize.
    """
    if face_img_bgr is None or face_img_bgr.size == 0:
        raise ValueError("Gambar wajah kosong atau tidak valid.")

    # 1. Pastikan ukuran gambar 160x160
    if face_img_bgr.shape[:2] != (160, 160):
        face_img_bgr = cv2.resize(face_img_bgr, (160, 160), interpolation=cv2.INTER_AREA)

    # 2. Konversi BGR -> RGB
    face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Ubah ke tensor PyTorch [C, H, W] dan normalisasi nilai piksel (0.0 - 1.0)
    device = _get_device()
    model = _get_model()

    tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # Tambahkan dimensi batch [1, 3, 160, 160]

    # 4. Inferensi forward-pass (tanpa kalkulasi gradien untuk hemat RAM & CPU)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().flatten().astype(np.float32)

    # 5. L2-Normalization (panjang vektor menjadi 1.0)
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Dimensi embedding tidak cocok: diharapkan {EMBEDDING_DIM}, didapat {embedding.shape[0]}"
        )

    return embedding
