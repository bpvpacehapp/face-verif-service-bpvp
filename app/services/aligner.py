"""
Komponen 2 — Face Alignment menggunakan MediaPipe Face Mesh

Fungsi:
1. Menerima gambar crop wajah (hasil dari detector YOLO).
2. Mendeteksi landmark iris/mata kiri & kanan menggunakan MediaPipe.
3. Menghitung sudut kemiringan dan merotasi wajah agar garis mata horizontal (0°).
4. Me-resize wajah ke ukuran standar input model pengenal wajah (160x160 piksel).
5. Fallback toleran jika landmark tidak terbaca: tetap me-resize crop asli ke 160x160.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ── Import MediaPipe Face Mesh dengan Fallback Kompatibilitas ──────────────────
_face_mesh = None
_mediapipe_failed = False


def _get_face_mesh():
    global _face_mesh, _mediapipe_failed
    if _face_mesh is None and not _mediapipe_failed:
        try:
            # Cara 1: Import langsung dari mediapipe.solutions.face_mesh
            import mediapipe.solutions.face_mesh as mp_face_mesh
            _face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception:
            try:
                # Cara 2: Import dari mediapipe.python.solutions
                from mediapipe.python.solutions import face_mesh as mp_face_mesh
                _face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            except Exception as e:
                # Fallback: jika modul solutions mediapipe tidak ditemukan,
                # jangan gagalkan proses, aktifkan mode resize langsung
                _mediapipe_failed = True
                _face_mesh = None

    return _face_mesh


@dataclass
class AlignmentResult:
    aligned_face: np.ndarray        # gambar wajah lurus ukuran (160, 160) BGR
    rotation_angle: float           # sudut derajat rotasi yang diterapkan
    left_eye: Tuple[int, int] | None   # koordinat mata kiri pada crop asli
    right_eye: Tuple[int, int] | None  # koordinat mata kanan pada crop asli
    is_aligned: bool                # True jika berhasil dirotasi berdasarkan landmark
    aligned_path: Path | None = None  # path jika disimpan ke disk


def _save_image(filepath: Path, image: np.ndarray) -> bool:
    """Helper penyimpanan gambar dengan fallback imencode."""
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


def align_face(
    face_crop_bgr: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> AlignmentResult:
    """
    Meluruskan posisi wajah agar kedua mata horizontal, lalu resize ke target_size.

    Args:
        face_crop_bgr: Gambar crop wajah dalam format numpy BGR.
        target_size: Dimensi output (lebar, tinggi), default (160, 160).
        save_dir: Folder tujuan jika ingin menyimpan hasil visual.
        filename_prefix: Nama awalan file saat disimpan.

    Returns:
        AlignmentResult berisi gambar wajah terotasi & metadata sudut.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        raise ValueError("Gambar crop wajah kosong.")

    h, w = face_crop_bgr.shape[:2]

    # MediaPipe membutuhkan format RGB
    rgb_img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    mesh = _get_face_mesh()
    results = None
    if mesh is not None:
        try:
            results = mesh.process(rgb_img)
        except Exception:
            results = None

    left_eye_pt = None
    right_eye_pt = None
    rotation_angle = 0.0
    is_aligned = False
    aligned_img = None

    if results and results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Indeks 468 (iris kiri) dan 473 (iris kanan) dari refine_landmarks
        # Jika indeks 468 tidak ada, fallback ke sudut mata (33 dan 263)
        if len(landmarks) > 473:
            lx, ly = landmarks[468].x * w, landmarks[468].y * h
            rx, ry = landmarks[473].x * w, landmarks[473].y * h
        else:
            lx, ly = landmarks[33].x * w, landmarks[33].y * h
            rx, ry = landmarks[263].x * w, landmarks[263].y * h

        left_eye_pt = (int(lx), int(ly))
        right_eye_pt = (int(rx), int(ry))

        # Hitung sudut kemiringan garis antara kedua mata
        dx = rx - lx
        dy = ry - ly
        rotation_angle = math.degrees(math.atan2(dy, dx))

        # Titik pusat rotasi adalah titik tengah kedua mata
        eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

        # Matriks transformasi rotasi affine
        rot_matrix = cv2.getRotationMatrix2D(eye_center, rotation_angle, scale=1.0)

        # Rotasi gambar crop dengan border refleksi agar tidak ada sudut hitam
        rotated = cv2.warpAffine(
            face_crop_bgr,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        aligned_img = cv2.resize(rotated, target_size, interpolation=cv2.INTER_AREA)
        is_aligned = True
    else:
        # Fallback toleran jika landmark tidak terbaca
        aligned_img = cv2.resize(face_crop_bgr, target_size, interpolation=cv2.INTER_AREA)
        is_aligned = False

    aligned_path = None
    if save_dir is not None:
        target_path = Path(save_dir).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix or "aligned"
        aligned_path = target_path / f"aligned_{prefix}.jpg"
        _save_image(aligned_path, aligned_img)

    return AlignmentResult(
        aligned_face=aligned_img,
        rotation_angle=rotation_angle,
        left_eye=left_eye_pt,
        right_eye=right_eye_pt,
        is_aligned=is_aligned,
        aligned_path=aligned_path,
    )
