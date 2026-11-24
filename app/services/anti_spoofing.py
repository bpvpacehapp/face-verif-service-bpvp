"""
Komponen Face Anti-Spoofing (Liveness Detection)
Menggunakan arsitektur MiniFASNet (dari uniface / Silent-Face-Anti-Spoofing)

Fungsi:
1. Mendeteksi apakah wajah di foto adalah manusia asli (live/real) atau
   serangan manipulasi (spoof) seperti foto layar HP/monitor atau kertas cetak.
2. Hasil analisis bersifat pasif (tidak menghentikan proses verifikasi),
   melainkan mencatat status `is_spoof: True/False` sebagai audit internal admin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton Model Anti-Spoofing ─────────────────────────────────────────────
_spoofer_model = None
_model_failed = False


def _get_spoofer():
    global _spoofer_model, _model_failed
    if _spoofer_model is None and not _model_failed:
        try:
            # Import MiniFASNet dari uniface
            try:
                from uniface.spoofing import MiniFASNet
                _spoofer_model = MiniFASNet()
            except (ImportError, AttributeError):
                from uniface import MiniFASNet
                _spoofer_model = MiniFASNet()
            logger.info("Model Anti-Spoofing MiniFASNet berhasil dimuat.")
        except Exception as e:
            logger.warning(
                f"Model Anti-Spoofing (uniface) belum siap: {e}. "
                "Mode fallback aktif (semua gambar dianggap valid sementara)."
            )
            _model_failed = True
            _spoofer_model = None

    return _spoofer_model


@dataclass
class AntiSpoofResult:
    is_spoof: bool       # True jika terindikasi foto layar HP/kertas cetak
    is_real: bool        # True jika terindikasi wajah manusia asli
    confidence: float    # Skor kepastian deteksi liveness (0.0 - 1.0)
    label: str           # 'REAL', 'SPOOF', atau 'UNKNOWN'


def check_anti_spoof(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> AntiSpoofResult:
    """
    Periksa keaslian wajah (Anti-Spoofing) berdasarkan gambar asli dan koordinat wajah.

    Args:
        image_bgr: Gambar penuh format BGR.
        bbox: Koordinat bounding box wajah (x1, y1, x2, y2).

    Returns:
        AntiSpoofResult dengan indikator `is_spoof`.
    """
    if image_bgr is None or image_bgr.size == 0:
        return AntiSpoofResult(is_spoof=False, is_real=True, confidence=1.0, label="UNKNOWN")

    spoofer = _get_spoofer()
    if spoofer is None:
        # Fallback jika library belum terinstall / model belum tersedia
        return AntiSpoofResult(
            is_spoof=False,
            is_real=True,
            confidence=1.0,
            label="REAL (FALLBACK)",
        )

    try:
        # MiniFASNet uniface menerima image dan bounding box [x1, y1, x2, y2]
        pred = spoofer.predict(image_bgr, bbox)

        is_real = True
        score = 1.0

        if isinstance(pred, dict):
            # Format resmi uniface / MiniFASNet:
            # {"label": "Real" | "Fake", "score": float, "bbox": [...]}
            label_val = str(pred.get("label", "")).strip().lower()
            score = float(pred.get("score", 1.0))

            if label_val in ("fake", "spoof"):
                is_real = False
            elif label_val in ("real", "live"):
                is_real = True
            elif "is_real" in pred:
                is_real = bool(pred["is_real"])
            else:
                # Jika label tidak ada, gunakan score probabilitas (class 1 = Real)
                is_real = score >= 0.5
        elif isinstance(pred, (bool, np.bool_)):
            is_real = bool(pred)
            score = 1.0
        elif isinstance(pred, (int, float, np.number)):
            score = float(pred)
            is_real = score >= 0.5

        is_spoof = not is_real
        label = "SPOOF" if is_spoof else "REAL"

        return AntiSpoofResult(
            is_spoof=is_spoof,
            is_real=is_real,
            confidence=round(score, 4),
            label=label,
        )

    except Exception as e:
        logger.warning(f"Terjadi kesalahan saat inferensi Anti-Spoofing: {e}")
        # Jangan gagalkan proses verifikasi jika modul spoofing error
        return AntiSpoofResult(
            is_spoof=False,
            is_real=True,
            confidence=1.0,
            label="REAL (SAFE_FALLBACK)",
        )
