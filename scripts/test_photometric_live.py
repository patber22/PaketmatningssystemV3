"""
Live-test för fotometrisk stereo (PhotometricStereoEstimator).

Testar hela svep-pipelinen på Raspberry Pi CM5:
1. Kör ett 4-kanals ljussvep med NoIR-kameran (CAM1).
2. Beräknar ytnormaler, skuggfri albedo och ytgradienter via PhotometricStereoEstimator.
3. Sparar pedagogiska färgbilder:
   - photo_normals_rgb.jpg: 3D-ytans vinklar visualiserade i RGB.
   - photo_albedo.jpg: Reflektanskarta (skuggfri "ren" bild av ytan).
   - photo_slope_color.jpg: Färgkodad ytlutning (blått = plant, rött = branta kanter/svank).

Körs direkt på Raspberry Pi CM5 med:
    python3 scripts/test_photometric_live.py
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
from src.algorithms.photometric_stereo import PhotometricStereoEstimator

# ==============================================================================
# Konfiguration
# ==============================================================================
CAMERA_ID = "left_cam1"    # Primär referenskamera
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "photometric_live"


def main():
    print("=" * 70)
    print("  Fotometrisk Stereo (Live-test på hårdvara)")
    print("=" * 70)
    print(f"Kamera:   {CAMERA_ID}")
    print(f"Målmapp:  {OUTPUT_DIR}\n")

    # 1. Initiera fotometrisk stereoberäknare
    print("[1/3] Initierar PhotometricStereoEstimator...")
    estimator = PhotometricStereoEstimator(camera_id=CAMERA_ID, verbose=True)

    # 2. Kör ett 4-kanals ljussvep via VisionRig
    print("\n[2/3] Startar VisionRig och kör fotometriskt svep (Kanal 1 -> 4)...")
    with VisionRig(verbose=True) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kameror och LED-belysning.")
            return

        sweep = rig.capture_photometric_sweep()

    if not sweep.is_valid():
        print("[FEL] Misslyckades med att fånga ett komplett 4-kanals bildsvep.")
        return

    # 3. Beräkna ytnormaler och albedo
    print("\n[3/3] Beräknar fotometriska ytnormaler och albedo...")
    t_start = time.perf_counter()
    result = estimator.estimate(sweep)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    if not result.is_valid():
        print("[FEL] Fotometrisk stereoberäkning misslyckades.")
        return

    # ==============================================================================
    # Spara pedagogiska bilder
    # ==============================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Standard RGB-normalkarta och Albedobild
    saved = result.to_jpeg(OUTPUT_DIR, prefix="photo_")

    # 2. Färgkodad ytlutning / "relativ höjd-topografi"
    # nz pekar uppåt längs Z-axeln.
    # Lutningsvinkel theta = arccos(nz) * 180 / pi
    # Helt platt yta = 0 grader (blått), branta kanter/svank = 30-45 grader (gult/rött)
    nz = np.clip(result.normals[:, :, 2], 0.0, 1.0)
    slope_deg = np.arccos(nz) * (180.0 / np.pi)
    slope_vis = np.clip(slope_deg / 45.0 * 255.0, 0, 255).astype(np.uint8)
    slope_color = cv2.applyColorMap(slope_vis, cv2.COLORMAP_TURBO)

    valid_mask = result.get_valid_mask()
    slope_color[~valid_mask] = (0, 0, 0)
    slope_path = OUTPUT_DIR / "photo_slope_color.jpg"
    cv2.imwrite(str(slope_path), slope_color)

    # Sammanfattande statistik
    meta = result.to_dict()
    flat_pixels = np.count_nonzero(nz > 0.95)
    total_pixels = valid_mask.size
    flat_percent = (flat_pixels / total_pixels) * 100.0

    print("\n" + "=" * 70)
    print("  RESULTAT FOTOMETRISK STEREO")
    print("=" * 70)
    print(f"  Total beräkningstid:     {total_time_ms:.1f} ms")
    print(f"  Upplösning:              {meta['resolution'][0]} x {meta['resolution'][1]} px")
    print(f"  Giltiga pixlar:          {meta['valid_pixels_count']:,} ({meta['valid_pixels_percent']:.1f}%)")
    print(f"  Ytans medelreflektans:   {meta['albedo_stats']['mean']:.3f}")
    print(f"  Plana ytor (nz > 0.95):  {flat_percent:.1f}% av synfältet")
    print("-" * 70)
    print("  Sparade bilder:")
    print(f"    1. 3D-normalkarta (RGB): {saved['normals_rgb']}")
    print(f"    2. Skuggfri Albedo:      {saved['albedo']}")
    print(f"    3. Ytlutningskarta:      {slope_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
