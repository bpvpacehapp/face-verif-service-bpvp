"""
Tes face detection dari folder datasets/sample/

Cara pakai:
1. Taruh foto (jpg/png/jpeg) di folder  datasets/sample/
2. Pastikan models/yolov12n-face.pt sudah ada
3. Jalankan:
       .\\venv\\Scripts\\python.exe test_detector.py

Hasil crop wajah disimpan ke  datasets/output/  untuk diperiksa secara visual.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from app.services.detector import detect_and_crop, FaceDetectionError

# Gunakan path absolut berdasarkan lokasi test_detector.py
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "datasets" / "sample"
OUTPUT_DIR = ROOT / "datasets" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted([p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED])

if not images:
    print(f"[!] Tidak ada gambar di {SAMPLE_DIR}/")
    print("    Masukkan file .jpg / .png ke folder tersebut lalu jalankan ulang.")
    raise SystemExit(1)

print(f"Folder Input  : {SAMPLE_DIR}")
print(f"Folder Output : {OUTPUT_DIR}")
print(f"Ditemukan {len(images)} gambar di datasets/sample/\n")
print(f"{'No':<3} {'File':<25} {'Status':<6} {'Conf':<7} {'Ukuran Asli':<13} {'BBox YOLO':<18} {'Hasil Crop':<11}")
print("=" * 90)

ok = fail = 0
saved_files = []

for i, path in enumerate(images, 1):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{i:<3} {path.name:<25} {'GAGAL BACA':<6}")
        fail += 1
        continue

    orig_h, orig_w = img.shape[:2]

    try:
        # Panggil fungsi deteksi & langsung simpan crop + bounding box ke folder output
        result = detect_and_crop(
            img,
            margin=0.15,
            save_dir=OUTPUT_DIR,
            filename_prefix=path.stem,
        )

        crop_h, crop_w = result.face_crop.shape[:2]
        bbox_str = f"[{result.bbox_original[0]},{result.bbox_original[1]} -> {result.bbox_original[2]},{result.bbox_original[3]}]"
        orig_str = f"{orig_w}x{orig_h}px"
        crop_str = f"{crop_w}x{crop_h}px"

        print(
            f"{i:<3} {path.name:<25} {'OK':<6} {result.confidence:>6.1%}  {orig_str:<13} {bbox_str:<18} {crop_str:<11}"
        )
        ok += 1

    except FaceDetectionError as e:
        print(f"{i:<3} {path.name:<25} {'SKIP':<6}  -  ({orig_w}x{orig_h}px)  → {e}")
        fail += 1

print("=" * 90)
print(f"\nHasil: {ok} berhasil dipotong, {fail} dilewati/gagal")

# Verifikasi fisik file di direktori output
actual_files = list(OUTPUT_DIR.glob("*.jpg"))
print(f"\n[Verifikasi Penyimpanan Disk]:")
print(f"Total file tersimpan di {OUTPUT_DIR} : {len(actual_files)} file")
for f in actual_files[:6]:
    size_kb = f.stat().st_size / 1024
    print(f"  -> {f.name} ({size_kb:.1f} KB)")
if len(actual_files) > 6:
    print(f"  ... dan {len(actual_files) - 6} file lainnya")

# Otomatis buka folder output di File Explorer Windows jika di Windows
try:
    if os.name == "nt":
        os.startfile(str(OUTPUT_DIR))
except Exception:
    pass

