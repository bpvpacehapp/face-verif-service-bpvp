"""
Uji Coba Lengkap Pipeline AI:
1. Face Detection & Crop (YOLOv12n-face)
2. Face Alignment (MediaPipe Face Mesh)
3. Face Embedding (InceptionResnetV1 VGGFace2 512-D)
4. Verifikasi Kemiripan (Cosine Similarity dengan threshold .env)

Jalankan:
    python test_pipeline.py
"""

from pathlib import Path
import cv2
import numpy as np

from app.config import SIMILARITY_THRESHOLD, MODEL_NAME
from app.services.detector import detect_and_crop, FaceDetectionError
from app.services.aligner import align_face
from app.services.embedder import extract_embedding
from app.services.comparator import verify_face, compute_cosine_similarity

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "datasets" / "sample"
OUTPUT_DIR = ROOT / "datasets" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
images = sorted([p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED])

if not images:
    print(f"[!] Tidak ada gambar di {SAMPLE_DIR}")
    raise SystemExit(1)

print(f"=== TEST PIPELINE FACE VERIFICATION (BPVP) ===")
print(f"Model Pengenal : {MODEL_NAME} (InceptionResnetV1 - 512 Dimensi)")
print(f"Threshold .env : {SIMILARITY_THRESHOLD:.0%} ({SIMILARITY_THRESHOLD})\n")

processed = []

for idx, img_path in enumerate(images, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    try:
        # Langkah 1: Deteksi & Potong (YOLOv12n-face)
        det_result = detect_and_crop(
            img,
            margin=0.15,
            save_dir=OUTPUT_DIR,
            filename_prefix=img_path.stem,
        )

        # Langkah 2: Alignment Wajah (MediaPipe)
        align_result = align_face(
            det_result.face_crop,
            target_size=(160, 160),
            save_dir=OUTPUT_DIR,
            filename_prefix=img_path.stem,
        )

        # Langkah 3: Ekstraksi Embedding 512-D (VGGFace2)
        embedding = extract_embedding(align_result.aligned_face)

        status_rotasi = f"{align_result.rotation_angle:+.1f}°" if align_result.is_aligned else "Fallback"

        processed.append({
            "name": img_path.name,
            "conf": det_result.confidence,
            "angle": status_rotasi,
            "embedding": embedding,
        })

        print(f"[{idx}] {img_path.name:<28} | YOLO: {det_result.confidence:.1%} | Rotasi: {status_rotasi:<8} | Embedding: (512,)")

    except FaceDetectionError as e:
        print(f"[{idx}] {img_path.name:<28} | SKIP DETEKSI: {e}")
    except Exception as e:
        print(f"[{idx}] {img_path.name:<28} | ERROR: {e}")

print("\n" + "=" * 65)
print("=== PENGUJIAN KEMIRIPAN (SIMILARITY CHECK 1:1) ===")
print(f"Batas Ambang (Threshold): {SIMILARITY_THRESHOLD:.0%}")
print("=" * 65)

if len(processed) >= 2:
    base = processed[0]
    print(f"\nFoto Referensi Acuan: [{base['name']}]\n")
    print(f"{'Dibandingkan dengan':<30} {'Skor Kemiripan':<16} {'Status':<15}")
    print("-" * 65)

    ref_array = np.array([base["embedding"]])

    for target in processed:
        verif = verify_face(target["embedding"], ref_array, threshold=SIMILARITY_THRESHOLD)
        sim_pct = verif.similarity_percentage
        status_lbl = "✓ COCOK" if verif.is_match else "✗ TIDAK COCOK"
        if not verif.is_match and verif.similarity_score >= 0.45:
            status_lbl += " (Perlu Review Admin)"

        print(f"{target['name']:<30} {sim_pct:<16} {status_lbl:<15}")

print("\n[✓] Pengujian selesai! File visual crop dan aligned tersimpan di datasets/output/")
