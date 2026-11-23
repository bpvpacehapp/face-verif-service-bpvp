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
    face_crop: np.ndarray                     # crop wajah BGR dengan margin (siap untuk alignment)
    confidence: float                         # skor keyakinan deteksi (0.0 - 1.0)
    bbox_original: tuple[int, int, int, int]  # (x1, y1, x2, y2) murni dari YOLO
    bbox_margin: tuple[int, int, int, int]    # (x1m, y1m, x2m, y2m) area setelah margin
    crop_path: Path | None = None             # lokasi file crop jika disimpan ke disk
    boxed_path: Path | None = None            # lokasi file anotasi kotak jika disimpan ke disk


# ── Helper penyimpanan aman ───────────────────────────────────────────────────
def _save_image(filepath: Path, image: np.ndarray) -> bool:
    """Simpan gambar ke disk dengan penanganan kompatibilitas Windows/OpenCV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(filepath), image)
    if not success or not filepath.exists():
        is_ok, buffer = cv2.imencode(".jpg", image)
        if is_ok:
            with open(filepath, "wb") as f:
                f.write(buffer)
            return filepath.exists()
        return False
    return True


# ── Fungsi utama ──────────────────────────────────────────────────────────────
class FaceDetectionError(Exception):
    """Dilempar saat deteksi wajah gagal."""


def detect_and_crop(
    image_bgr: np.ndarray,
    margin: float = MARGIN,
    min_conf: float = MIN_CONFIDENCE,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> DetectionResult:
    """
    Deteksi wajah dan potong (crop) dengan margin 10-20%.
    Dapat langsung menyimpan hasil potongan (crop) dan anotasi kotak ke disk.

    Args:
        image_bgr: Gambar dalam format numpy BGR (dari OpenCV / cv2.imread).
        margin: Persentase margin di sekeliling wajah (default 0.15 = 15%).
        min_conf: Ambang batas confidence deteksi YOLO (default 0.50).
        save_dir: Folder tujuan jika ingin langsung menyimpan hasil (cth: 'datasets/output').
        filename_prefix: Nama awalan file (cth: 'foto_01' -> 'crop_foto_01.jpg').

    Returns:
        DetectionResult berisi potongan wajah, koordinat bounding box, dan path file jika disimpan.

    Raises:
        FaceDetectionError: Jika gambar tidak valid, tidak ada wajah,
                            atau terdapat lebih dari 1 wajah.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise FaceDetectionError("Gambar kosong atau tidak valid.")

    model = _get_model()
    results = model(image_bgr, verbose=False)[0]

    # Filter bounding box berdasarkan threshold confidence
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

    # Ambil bounding box wajah utama
    box = boxes[0]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

    # Hitung margin tambahan (10-20%) agar seluruh dagu, dahi & telinga tidak terpotong
    h, w = image_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    x1m = max(0, x1 - mx)
    y1m = max(0, y1 - my)
    x2m = min(w, x2 + mx)
    y2m = min(h, y2 + my)

    # Potong (crop) area wajah dari gambar asli
    face_crop = image_bgr[y1m:y2m, x1m:x2m].copy()

    if face_crop.size == 0:
        raise FaceDetectionError("Hasil pemotongan wajah kosong (bounding box di luar batas).")

    crop_path: Path | None = None
    boxed_path: Path | None = None

    # Simpan langsung ke output jika folder tujuan diberikan
    if save_dir is not None:
        target_dir = Path(save_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix or "face"

        # 1. Simpan potongan wajah bersih (crop)
        crop_path = target_dir / f"crop_{prefix}.jpg"
        _save_image(crop_path, face_crop)

        # 2. Simpan gambar bergaris bounding box (boxed) untuk dokumentasi/verifikasi
        boxed_img = image_bgr.copy()
        cv2.rectangle(boxed_img, (x1m, y1m), (x2m, y2m), (0, 255, 255), 2)  # kuning = margin
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)        # hijau = yolo bbox
        cv2.putText(
            boxed_img,
            f"Face: {conf:.1%}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        boxed_path = target_dir / f"boxed_{prefix}.jpg"
        _save_image(boxed_path, boxed_img)

    return DetectionResult(
        face_crop=face_crop,
        confidence=conf,
        bbox_original=(x1, y1, x2, y2),
        bbox_margin=(x1m, y1m, x2m, y2m),
        crop_path=crop_path,
        boxed_path=boxed_path,
    )
