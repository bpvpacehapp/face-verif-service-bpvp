"""
Tes face detection dari folder datasets/sample/

Cara pakai:
1. Taruh foto (jpg/png/jpeg) di folder  datasets/sample/
2. Pastikan models/yolov12n-face.pt sudah ada
3. Jalankan:
       .\\venv\\Scripts\\python.exe test_detector.py

Hasil crop wajah disimpan ke  datasets/output/  untuk diperiksa secara visual.
"""

from pathlib import Path

import cv2

from app.services.detector import detect_and_crop, FaceDetectionError

SAMPLE_DIR = Path("datasets/sample")
OUTPUT_DIR = Path("datasets/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted([p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED])

if not images:
    print(f"[!] Tidak ada gambar di {SAMPLE_DIR}/")
    print("    Masukkan file .jpg / .png ke folder tersebut lalu jalankan ulang.")
    raise SystemExit(1)

print(f"Ditemukan {len(images)} gambar di datasets/sample/\n")
print(f"{'No':<4} {'File':<35} {'Status':<12} {'Conf':>6}  Ukuran crop")
print("-" * 72)

ok = fail = 0
for i, path in enumerate(images, 1):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{i:<4} {path.name:<35} {'GAGAL BACA':<12}")
        fail += 1
        continue

    try:
        result = detect_and_crop(img)

        # Simpan crop ke output/
        out_path = OUTPUT_DIR / f"crop_{path.stem}.jpg"
        cv2.imwrite(str(out_path), result.face_crop)

        h, w = result.face_crop.shape[:2]
        print(
            f"{i:<4} {path.name:<35} {'OK':<12} {result.confidence:>6.3f}  {w}x{h}px"
        )
        ok += 1

    except FaceDetectionError as e:
        print(f"{i:<4} {path.name:<35} {'SKIP':<12}  → {e}")
        fail += 1

print("-" * 72)
print(f"\nHasil: {ok} berhasil, {fail} gagal/dilewati")
if ok:
    print(f"Crop wajah tersimpan di: {OUTPUT_DIR.resolve()}")
