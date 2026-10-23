"""
Live-test för VisionRig fasaden (src/hardware/vision_rig.py).

Testar hela den optiska mätbädden på Raspberry Pi CM5:
1. Full Flash Stereo (12W IR LED belysning) med korta exponeringstiden.
2. Ambient Stereo (släckt belysning) för referens.
3. Fotometriskt ljussvep (sekventiell tändning av Kanal 1 -> 2 -> 3 -> 4).
4. Komplett mätcykel (capture_measurement_cycle) med tidsmätning.
5. Sparar testbilder till data/test_output/vision_rig/{timestamp}/.

Körs på pajen med:
    python3 scripts/test_vision_rig_live.py
"""

import os
import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.vision_rig import VisionRig


def main():
    print("=" * 70)
    print("  Live-test: VisionRig (src/hardware/vision_rig.py)")
    print("=" * 70)

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "test_output" / "vision_rig" / ts_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Bilder sparas till: {output_dir}\n")

    print("[1/5] Initierar VisionRig (CameraController + LEDController)...")
    with VisionRig(verbose=True) as rig:
        if not rig.is_connected:
            print("\n[FEL] Kunde inte ansluta till alla komponenter i VisionRig!")
            print(f"  Kameror anslutna: {rig.cameras.is_connected}")
            print(f"  LEDs anslutna:    {rig.leds.is_connected}")
            print("Kontrollera hårdvaran och rättigheter.")
            return

        print("\n[OK] VisionRig är ansluten och redo!\n")

        # ----------------------------------------------------------------------
        # Test 1: Full Flash Stereo
        # ----------------------------------------------------------------------
        print("-" * 70)
        print("TEST 1: Full Flash Stereo (Alla 4 lampor tända, 15ms exponering)")
        print("-" * 70)
        t0 = time.time()
        stereo_flash = rig.capture_stereo(led_mode="all")
        dt_stereo = (time.time() - t0) * 1000

        if stereo_flash.is_valid():
            l_path, r_path = stereo_flash.to_jpeg(output_dir / "stereo_flash")
            mean_l = float(stereo_flash.left.image.mean())
            mean_r = float(stereo_flash.right.image.mean())
            print(f"  [OK] Stereopar taget på {dt_stereo:.1f} ms")
            print(f"       Medelljusstyrka: Vänster={mean_l:.1f}, Höger={mean_r:.1f}")
            print(f"       Sparat till: {l_path.name} & {r_path.name}")
        else:
            print("  [FEL] Kunde inte ta Full Flash stereopar!")

        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # Test 2: Ambient Stereo (släckta lampor)
        # ----------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST 2: Ambient Stereo (Släckta lampor, rumsljus)")
        print("-" * 70)
        stereo_ambient = rig.capture_stereo(led_mode="ambient")
        if stereo_ambient.is_valid():
            l_path, r_path = stereo_ambient.to_jpeg(output_dir / "stereo_ambient")
            mean_l = float(stereo_ambient.left.image.mean())
            mean_r = float(stereo_ambient.right.image.mean())
            print(f"  [OK] Ambient stereopar sparat:")
            print(f"       Medelljusstyrka: Vänster={mean_l:.1f}, Höger={mean_r:.1f}")
        else:
            print("  [FEL] Kunde inte ta Ambient stereopar!")

        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # Test 3: Fotometriskt ljussvep (Kanal 1 -> 2 -> 3 -> 4)
        # ----------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST 3: Fotometriskt ljussvep (Kanal 1 -> 2 -> 3 -> 4)")
        print("-" * 70)
        t0 = time.time()
        photo_sweep = rig.capture_photometric_sweep()
        dt_sweep = (time.time() - t0) * 1000

        if photo_sweep.is_valid():
            saved_paths = photo_sweep.to_jpeg(output_dir / "photometric")
            print(f"  [OK] Svepet genomfört på {dt_sweep:.1f} ms (alla 4 vinklarna giltiga)")
            for ch in [1, 2, 3, 4]:
                pair = photo_sweep.get_channel(ch)
                mean_l = float(pair.left.image.mean())
                mean_r = float(pair.right.image.mean())
                print(f"       Kanal {ch}: Vänster={mean_l:.1f}, Höger={mean_r:.1f} medelljusstyrka")
        else:
            print("  [FEL] Fotometriska svepet misslyckades eller saknar kanaler!")

        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # Test 4: Komplett Mätcykel (capture_measurement_cycle)
        # ----------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST 4: Komplett mätsekvens (Stereo + 4x Fotometri i en cykel)")
        print("-" * 70)
        t0 = time.time()
        cycle_capture = rig.capture_measurement_cycle()
        dt_cycle = (time.time() - t0) * 1000

        if cycle_capture.is_valid():
            print(f"  [OK] Komplett cykel slutförd på {dt_cycle:.1f} ms!")
            print(f"       Stereo OK:      {cycle_capture.stereo.is_valid()}")
            print(f"       Fotometri OK:   {cycle_capture.photometric.is_valid()}")
            saved = cycle_capture.to_jpeg(output_dir / "complete_cycle")
            print(f"       Bilder sparade under: complete_cycle/")
        else:
            print("  [FEL] Mätsekvensen returnerade ogiltig VisionCapture!")

    print("\n" + "=" * 70)
    print("  LIVETESTET AVSLUTAT! Alla resurser och lampor är släckta.")
    print("=" * 70)


if __name__ == "__main__":
    main()
