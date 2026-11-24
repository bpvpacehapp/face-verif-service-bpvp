"""
Diagnostik Khusus Anti-Spoofing (Liveness Check)
Mengecek apakah uniface MiniFASNet terinstall dan membaca output mentah pada setiap foto live_test.

Jalankan:
    python test_spoof_detail.py
"""

from pathlib import Path
import cv2

from app.services.detector import detect_and_crop
from app.services.anti_spoofing import _get_spoofer, check_anti_spoof

ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "datasets" / "live_test"

print("=" * 75)
print("   DIAGNOSTIK ANTI-SPOOFING (MiniFASNet Liveness Detection)")
print("=" * 75)

# 1. Cek Model Spoofer
spoofer = _get_spoofer()
if spoofer is None:
    print("[!] PERINGATAN: Model MiniFASNet belum terpasang!")
    print("    Silakan jalankan: pip install \"uniface[cpu]\" di terminal.\n")
else:
    print(f"[✓] Model Anti-Spoofing MiniFASNet BERHASIL dimuat: {type(spoofer).__name__}\n")

# 2. Periksa Foto-Foto di live_test
images = sorted([p for p in LIVE_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

if not images:
    print(f"[!] Tidak ada foto di {LIVE_DIR}")
    raise SystemExit(1)

print(f"{'Nama File':<28} {'Prediksi':<10} {'Kepastian':<12} {'Raw Output'}")
print("-" * 75)

for path in images:
    img = cv2.imread(str(path))
    if img is None:
        continue

    try:
        det = detect_and_crop(img, margin=0.15)
        bbox = det.bbox_original

        # Raw prediction langsung dari model
        raw_pred = spoofer.predict(img, bbox) if spoofer else "Model Not Loaded"

        # Parsed result
        result = check_anti_spoof(img, bbox)

        color_tag = "🔴 [PALSU/SPOOF]" if result.is_spoof else "🟢 [ASLI/REAL]"
        print(f"{path.name:<28} {color_tag:<10} {result.confidence:>6.1%}      {raw_pred}")

    except Exception as e:
        print(f"{path.name:<28} ERROR: {e}")

print("=" * 75)
