"""
Live-test för snabbmätning av paket via 3D-stereo.

Körs direkt på Raspberry Pi CM5 med:
    python3 scripts/test_stereo_measure_live.py
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.vision_rig import VisionRig
from src.algorithms.stereo_depth import StereoDepthEstimator

# ==============================================================================
# Konfiguration (Ändra dessa värden direkt här vid behov)
# ==============================================================================
BED_Z_MM = 894.0         # Ändra till ditt uppmätta bordsavstånd (ToF) i millimeter
MIN_HEIGHT_MM = 15.0     # Minsta höjd över bordet (mm) för att räknas som paket
MAX_HEIGHT_MM = 400.0    # Max tillåten pakethöjd (mm)
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "quick_measure"


def main():
    print("=" * 70)
    print("  Stereo 3D Snabbmätning (Live-test på hårdvara)")
    print("=" * 70)
    print(f"Bordsreferens (BED_Z_MM): {BED_Z_MM:.1f} mm")
    print(f"Tröskelvärde paket:       > {MIN_HEIGHT_MM:.1f} mm över bordet\n")

    # 1. Initiera stereodjupsberäknaren
    print("[1/4] Initierar StereoDepthEstimator...")
    estimator = StereoDepthEstimator(mode="fast", verbose=True)

    # 2. Starta riggen och ta en bild med full IR-belysning
    print("\n[2/4] Startar VisionRig och tar stereobild (all LEDs on)...")
    with VisionRig(verbose=True) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kameror och belysning.")
            return

        pair = rig.capture_stereo(led_mode="all")

    if not pair.is_valid():
        print("[FEL] Misslyckades med att fånga ett giltigt stereobildpar.")
        return

    # 3. Beräkna stereodjup och 3D-punktmoln
    print("\n[3/4] Beräknar 3D-stereodjup...")
    t_start = time.perf_counter()
    depth_res = estimator.estimate(pair)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    if not depth_res.is_valid():
        print("[FEL] Stereodjupsberäkningen misslyckades.")
        return

    # 4. Paketsegmentering och måttberäkning
    print("\n[4/4] Segmenterar paket och beräknar mått...")
    valid_pts = depth_res.get_valid_points()
    total_pts_count = len(valid_pts)

    if total_pts_count == 0:
        print("[FEL] Inga giltiga 3D-punkter hittades i punktmolnet.")
        return

    # Beräkna höjd över bordet för varje punkt (kameran är i Z=0, bordet i Z=BED_Z_MM)
    heights_mm = BED_Z_MM - valid_pts[:, 2]

    # Maskera ut punkter som tillhör paketet (ovanför bordet)
    package_mask = (heights_mm >= MIN_HEIGHT_MM) & (heights_mm <= MAX_HEIGHT_MM)
    package_pts = valid_pts[package_mask]
    package_heights = heights_mm[package_mask]

    pkg_count = len(package_pts)
    print(f"  Totalt antal 3D-punkter: {total_pts_count:,}")
    print(f"  Punkter på paketet:      {pkg_count:,}")

    if pkg_count < 100:
        print("\n[VARNING] För få punkter detekterades ovanför bordet (< 100 st).")
        print("Ligger det något paket på bordet, eller är BED_Z_MM felkalibrerat?")
        return

    # A. Höjd: Använd 99:e percentilen för att undvika enstaka brus-spikar
    height_mm = float(np.percentile(package_heights, 99.0))

    # B. Längd och Bredd: 2D oriented bounding box på X/Y i millimeter
    pts_2d = package_pts[:, :2].astype(np.float32)

    # Rensa bort extrema 2D-outliers innan bounding box (1:a till 99:e percentilen)
    x_min, x_max = np.percentile(pts_2d[:, 0], [1.0, 99.0])
    y_min, y_max = np.percentile(pts_2d[:, 1], [1.0, 99.0])
    inlier_mask = (
        (pts_2d[:, 0] >= x_min) & (pts_2d[:, 0] <= x_max) &
        (pts_2d[:, 1] >= y_min) & (pts_2d[:, 1] <= y_max)
    )
    inlier_pts_2d = pts_2d[inlier_mask]

    rect = cv2.minAreaRect(inlier_pts_2d)
    _, (dim1, dim2), angle = rect
    length_mm = max(dim1, dim2)
    width_mm = min(dim1, dim2)

    # Spara felsökningsbilder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = depth_res.to_jpeg(OUTPUT_DIR, prefix="quick_")

    print("\n" + "=" * 70)
    print("  UPPSKATTADE DIMENSIONER (Stereo 3D)")
    print("=" * 70)
    print(f"  Längd (L): {length_mm:6.1f} mm")
    print(f"  Bredd (B): {width_mm:6.1f} mm")
    print(f"  Höjd  (H): {height_mm:6.1f} mm")
    print("-" * 70)
    print(f"  Riktning (vinkel):       {angle:5.1f}°")
    print(f"  Total beräkningstid:     {total_time_ms:5.1f} ms")
    if saved_paths.get("disparity"):
        print(f"  Färgkodad djupkarta:     {saved_paths['disparity']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
