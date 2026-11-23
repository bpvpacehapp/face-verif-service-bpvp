"""
Komponen 1 — Face Detection
Menggunakan YOLOv12n-face (file: models/yolov12n-face.pt)

Model dimuat SEKALI saat modul diimpor pertama kali (singleton),
bukan per request — sesuai aturan pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# ── Konstanta ────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov12n-face.pt"
MARGIN = 0.15          # margin 15 % di setiap sisi bounding box
MIN_CONFIDENCE = 0.50  # deteksi di bawah ini diabaikan


# ── Singleton model ───────────────────────────────────────────────────────────
_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {_MODEL_PATH}\n"
                "Unduh yolov12n-face.pt dan taruh di folder models/."
            )
        _model = YOLO(str(_MODEL_PATH))
    return _model


# ── Struktur hasil ────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    face_crop: np.ndarray   # crop wajah dalam format BGR (siap untuk alignment)
    confidence: float       # skor keyakinan YOLO
    bbox_original: tuple    # (x1, y1, x2, y2) sebelum margin, di gambar asli


# ── Fungsi utama ──────────────────────────────────────────────────────────────
class FaceDetectionError(Exception):
    """Dilempar saat deteksi wajah gagal."""


def detect_and_crop(
    image_bgr: np.ndarray,
    margin: float = MARGIN,
    min_conf: float = MIN_CONFIDENCE,
) -> DetectionResult:
    """
    Deteksi wajah dari gambar BGR (numpy array dari cv2.imread atau kamera).

    Returns:
        DetectionResult berisi crop wajah dan metadata.

    Raises:
        FaceDetectionError: Jika tidak ada wajah atau lebih dari satu wajah.
    """
    model = _get_model()
    results = model(image_bgr, verbose=False)[0]

    # Filter berdasarkan confidence
    boxes = [
        b for b in results.boxes
        if float(b.conf[0]) >= min_conf
    ]

    if len(boxes) == 0:
        raise FaceDetectionError("Tidak ada wajah yang terdeteksi.")
    if len(boxes) > 1:
        raise FaceDetectionError(
            f"Ditemukan {len(boxes)} wajah. Pastikan hanya ada satu wajah dalam gambar."
        )

    # Ambil bounding box terbaik
    box = boxes[0]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

    # Tambah margin
    h, w = image_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    x1m = max(0, x1 - mx)
    y1m = max(0, y1 - my)
    x2m = min(w, x2 + mx)
    y2m = min(h, y2 + my)

    face_crop = image_bgr[y1m:y2m, x1m:x2m]

    return DetectionResult(
        face_crop=face_crop,
        confidence=conf,
        bbox_original=(x1, y1, x2, y2),
    )
