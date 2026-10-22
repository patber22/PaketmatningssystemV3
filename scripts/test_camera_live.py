"""
Liveskript för att testa samverkan mellan CameraController och LEDController.

Använder:
  - LEDController för att styra alla 4 IR-lamporna
  - CameraController för att knäppa synkroniserade CameraFramePair (left=CAM1, right=CAM0)
  - Sparar stereoparen automatiskt med pair.to_jpeg(output_dir)

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

from src.hardware.camera_controller import CameraController
from src.hardware.led_controller import LEDController


def main():
    print("=" * 65)
    print("  Live-test: CameraController + LEDController")
    print("=" * 65)

    output_dir = PROJECT_ROOT / "test_captures"

    with LEDController(verbose=True) as leds, CameraController(verbose=True) as cams:
        if not leds.is_connected or not cams.is_connected:
            print("\n[Varning] Kunde inte ansluta till alla hårdvaruenheter!")
            print(f"  LEDController:    {'Ansluten' if leds.is_connected else 'MISSLYCKADES'}")
            print(f"  CameraController: {'Ansluten' if cams.is_connected else 'MISSLYCKADES'}")
            return

        print("\nBåda controllerna är anslutna och redo!")

        # 1. Tänd alla lampor
        print("\n[1/3] Tänder alla lampor (leds.all_on())...")
        leds.all_on()
        time.sleep(0.1)

        # 2. Knäpp ett stereopar (samples=1, snabbast)
        print("\n[2/3] Tar ett synkroniserat stereopar (samples=1)...")
        pair_single = cams.capture_both(samples=1)
        left_path, right_path = pair_single.to_jpeg(path=output_dir)
        print(f"      Vänster (CAM1): {left_path.name if left_path else 'MISSLYCKADES'}")
        print(f"      Höger   (CAM0): {right_path.name if right_path else 'MISSLYCKADES'}")

        time.sleep(1.0)

        # 3. Knäpp ett tidsfiltrerat stereopar (samples=3, median)
        print("\n[3/3] Tar ett tidsfiltrerat stereopar (samples=3, median)...")
        pair_multi = cams.capture_both(samples=3, method="median")
        left_multi, right_multi = pair_multi.to_jpeg(path=output_dir)
        print(f"      Vänster (CAM1): {left_multi.name if left_multi else 'MISSLYCKADES'}")
        print(f"      Höger   (CAM0): {right_multi.name if right_multi else 'MISSLYCKADES'}")

        # 4. Släck lamporna
        print("\n[Släckning] Släcker lamporna...")
        leds.all_off()

    print("\n" + "=" * 65)
    print("  Klart! Båda stereoparen är tagna och sparade till:")
    print(f"  {output_dir.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
