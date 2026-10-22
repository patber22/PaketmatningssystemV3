"""
Liveskript för att testa Camera-drivrutinen (src/hardware/camera.py) på båda portarna.

Initierar två Camera-instanser:
  - CAM0 (Höger kamera, CSI0)
  - CAM1 (Vänster kamera, CSI1)

Tar en enkel bild (samples=1) och en multi-sample bild (samples=3, median)
för varje kamera och sparar med CameraFrame.to_jpeg() via automatisk namngivning.

Körs på pajen med:
    python3 scripts/test_camera_live.py
"""

import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.camera import Camera
from src.hardware.led_controller import LEDController


def main():
    print("=" * 65)
    print("  Live-test: Dual NoIR Camera (CAM0 Höger & CAM1 Vänster)")
    print("=" * 65)

    output_dir = PROJECT_ROOT / "test_captures"
    saved_files = []

    # Tänd alla 4 IR-lampor under fotograferingen för optimal belysning
    with LEDController(verbose=False) as leds:
        print("\n[Belysning] Tänder alla 4 IR-lampor (all_on)...")
        leds.all_on()
        time.sleep(0.1)

        print("\n[Kameror] Initierar och startar CAM0 och CAM1...")
        with Camera(camera_id="CAM0", verbose=True) as cam0, Camera(camera_id="CAM1", verbose=True) as cam1:
            if not cam0.is_connected or not cam1.is_connected:
                print("\n[Varning] Kunde inte ansluta till båda kamerorna!")
                print(f"  CAM0 (Höger): {'Ansluten' if cam0.is_connected else 'MISSLYCKADES'}")
                print(f"  CAM1 (Vänster): {'Ansluten' if cam1.is_connected else 'MISSLYCKADES'}")
                return

            print("\nBåda kamerorna är anslutna och strömmar i minnet!")

            # -------------------------------------------------------------
            # 1. CAM0 (Höger kamera)
            # -------------------------------------------------------------
            print("\n--- CAM0 (Höger kamera) ---")

            # 1a. Enkel bildruta (samples=1)
            print("  1. Tar enkel bildruta (samples=1)...")
            frame0_single = cam0.capture(samples=1)
            path0_single = frame0_single.to_jpeg(path=output_dir)
            if path0_single:
                saved_files.append(path0_single)
                print(f"     Sparad: {path0_single.name} ({path0_single.stat().st_size / 1024:.1f} KB)")

            time.sleep(1.0)  # Säkerställ unikt tidsstämpelnamn

            # 1b. Multi-sample (samples=3, median)
            print("  2. Tar multi-sample bild (samples=3, median)...")
            frame0_multi = cam0.capture(samples=3, method="median")
            path0_multi = frame0_multi.to_jpeg(path=output_dir)
            if path0_multi:
                saved_files.append(path0_multi)
                print(f"     Sparad: {path0_multi.name} ({path0_multi.stat().st_size / 1024:.1f} KB)")

            time.sleep(1.0)

            # -------------------------------------------------------------
            # 2. CAM1 (Vänster kamera)
            # -------------------------------------------------------------
            print("\n--- CAM1 (Vänster kamera) ---")

            # 2a. Enkel bildruta (samples=1)
            print("  1. Tar enkel bildruta (samples=1)...")
            frame1_single = cam1.capture(samples=1)
            path1_single = frame1_single.to_jpeg(path=output_dir)
            if path1_single:
                saved_files.append(path1_single)
                print(f"     Sparad: {path1_single.name} ({path1_single.stat().st_size / 1024:.1f} KB)")

            time.sleep(1.0)  # Säkerställ unikt tidsstämpelnamn

            # 2b. Multi-sample (samples=3, median)
            print("  2. Tar multi-sample bild (samples=3, median)...")
            frame1_multi = cam1.capture(samples=3, method="median")
            path1_multi = frame1_multi.to_jpeg(path=output_dir)
            if path1_multi:
                saved_files.append(path1_multi)
                print(f"     Sparad: {path1_multi.name} ({path1_multi.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print("  Klart! Båda kamerorna har testats och bilderna är sparade:")
    print("=" * 65)
    for p in saved_files:
        print(f"  * {p.name:<30} ({p.stat().st_size / 1024:.1f} KB)")
    print(f"\nKatalog: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
