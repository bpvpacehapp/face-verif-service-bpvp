"""
Komponen 4 — Verifikasi dan Komparasi Kemiripan Wajah (1:1)

Menggunakan Cosine Similarity antara embedding foto presensi dengan
embedding referensi user yang tersimpan di database.
Threshold dibaca dari file .env melalui app.config.SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np

from app.config import SIMILARITY_THRESHOLD, SHOW_THRESHOLD


@dataclass
class VerificationResult:
    similarity_score: float         # desimal murni: 0.8342
    similarity_percent: float       # persentase numerik: 83.42 (untuk kalkulasi/grafik)
    similarity_percentage: str      # string ramah UI: "83.4%" (langsung tampil di dashboard)
    is_match: bool                  # True jika similarity >= threshold
    status: str                     # 'VERIFIED' atau 'NEEDS_REVIEW' (perlu tinjauan admin)
    best_index: int                 # indeks foto referensi yang paling cocok
    threshold: float | None = None  # None di production (tidak dibocorkan ke publik)

    def to_dict(self) -> Dict[str, Any]:
        """Konversi ke dictionary JSON API response yang bersih."""
        data = {
            "similarity_score": self.similarity_score,
            "similarity_percent": self.similarity_percent,
            "similarity_percentage": self.similarity_percentage,
            "is_match": self.is_match,
            "status": self.status,
            "best_index": self.best_index,
        }
        # Sertakan threshold HANYA jika diizinkan (mode development)
        if self.threshold is not None:
            data["threshold"] = self.threshold
        return data


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Hitung cosine similarity antara 2 vektor 1D."""
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


def verify_face(
    live_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
    show_threshold: bool = SHOW_THRESHOLD,
) -> VerificationResult:
    """
    Bandingkan embedding foto presensi dengan embedding referensi user (1:1).

    Args:
        live_embedding: Vektor (512,) dari foto presensi saat ini.
        reference_embeddings: Array 2D (N, 512) dari database milik user bersangkutan.
        threshold: Batas minimum kemiripan (default dari .env, misal 0.60).
        show_threshold: Apakah nilai threshold disertakan di hasil (False saat production).

    Returns:
        VerificationResult berisi skor kecocokan, persentase kemiripan, dan status lolos/review.
    """
    if reference_embeddings.size == 0 or len(reference_embeddings.shape) != 2:
        return VerificationResult(
            similarity_score=0.0,
            similarity_percent=0.0,
            similarity_percentage="0.0%",
            is_match=False,
            status="NO_REFERENCE_DATA",
            best_index=-1,
            threshold=threshold if show_threshold else None,
        )

    # Karena vektor sudah di-L2-normalize, dot product per baris adalah cosine similarity
    # reference_embeddings: (N, 512), live_embedding: (512,) -> similarities: (N,)
    similarities = np.dot(reference_embeddings, live_embedding)

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    best_score = max(0.0, min(1.0, best_score))

    is_match = best_score >= threshold

    # Status: VERIFIED jika >= threshold, NEEDS_REVIEW jika di bawahnya untuk konfirmasi admin
    status = "VERIFIED" if is_match else "NEEDS_REVIEW"

    pct = round(best_score * 100, 2)

    return VerificationResult(
        similarity_score=round(best_score, 4),
        similarity_percent=pct,
        similarity_percentage=f"{pct:.1f}%",
        is_match=is_match,
        status=status,
        best_index=best_idx,
        threshold=threshold if show_threshold else None,
    )
