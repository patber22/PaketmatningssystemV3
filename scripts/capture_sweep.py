"""
Bildinsamlingsskript: Fånga och spara fotometriskt svep med anpassad gain/exponering
===================================================================================

Detta skript körs direkt på Raspberry Pi CM5 för att samla in testdata:
1. Tar ett 4-kanals ljussvep över LED 1..4 för både CAM1 och CAM0.
2. Spolar hårdvarubufferten efter varje LED-växling så att varken CAM1 eller CAM0
   får gamla/halvexponerade bildrutor från innan lampan tändes.
3. Sparar råbilderna direkt till data/test_data/<dataset>/ så att de enkelt kan pushas till Git.

Användning på CM5:
    python3 scripts/capture_sweep.py --dataset blue_book --gain 6.0 --exposure 50000
    python3 scripts/capture_sweep.py --dataset small_book --gain 6.0 --exposure 50000
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.vision_rig import VisionRig

DATA_DIR = PROJECT_ROOT / "data" / "test_data"
LIVE_OUT_DIR = PROJECT_ROOT / "data" / "test_output" / "photometric_live"


def main():
    parser = argparse.ArgumentParser(description="Fånga fotometriskt svep med anpassad gain och hårdvaruspolning")
    parser.add_argument("--dataset", type=str, default="blue_book", help="Mappnamn under data/test_data/ (t.ex. blue_book, small_book)")
    parser.add_argument("--gain", type=float, default=6.0, help="Kamerans analoga förstärkning (standard: 6.0)")
    parser.add_argument("--exposure", type=int, default=50000, help="Slutartid i mikrosekunder (standard: 50000 = 50 ms)")
    args = parser.parse_args()

    target_dir = DATA_DIR / args.dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    LIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BILDINSAMLING: FOTOMETRISKT SVEP MED HÅRDVARUSPOLNING")
    print("=" * 70)
    print(f"Målmapp:    {target_dir}")
    print(f"Gain:       {args.gain:.2f}")
    print(f"Exponering: {args.exposure} us ({args.exposure / 1000.0:.1f} ms)\n")

    print("[1/2] Ansluter till VisionRig och initierar kameror...")
    with VisionRig(verbose=False) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kameror eller LED-kontroller.")
            return

        # Sätt fast exponering och gain på båda kamerorna
        print(f"      Ställer in sensorer: exp={args.exposure} us, gain={args.gain:.2f}...")
        rig.cameras.set_exposure(args.exposure, args.gain)
        
        # Vänta så sensorns I2C-register hinner uppdateras vid nästa frame boundary
        time.sleep(0.15)
        # Spola ur de gamla bildrutorna som exponerades med tidigare inställningar
        rig.cameras.capture_both(samples=1)
        rig.cameras.capture_both(samples=1)

        print("[2/2] Fångar 4-kanals svep (spolar buffert vid varje LED-växling)...")
        t0 = time.perf_counter()

        captured_pairs = []
        try:
            rig.leds.all_off()
            for ch in [1, 2, 3, 4]:
                # 1. Tänd aktuell LED
                rig.leds.on(ch)
                time.sleep(0.06)  # LED & drivar-stabilisering

                # 2. KRITISKT: Spola bildrutan som redan var i röret/höll på att exponeras
                #    innan lampan tändes. Utan denna spolning får CAM1 en mörk/halvexponerad ruta!
                rig.cameras.capture_both(samples=1)

                # 3. Fånga den äkta bildrutan (exponerad 100% under LED ch)
                pair = rig.cameras.capture_both(samples=1)
                rig.leds.off(ch)

                captured_pairs.append((ch, pair))

        except Exception as e:
            print(f"[FEL] Avbrott under ljussvep: {e}")
            rig.leds.all_off()
            return
        finally:
            rig.leds.all_off()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  [OK] Svep klart på {elapsed_ms:.1f} ms!\n")

    # Spara alla 8 råbilderna (4 kanaler x 2 kameror)
    print("[SPARAR RÅBILDER]")

    for ch_idx, pair in captured_pairs:
        if pair.left and pair.left.image is not None:
            img_l = pair.left.image
            p_l = target_dir / f"raw_ch{ch_idx}_CAM1.jpg"
            cv2.imwrite(str(p_l), img_l)
            cv2.imwrite(str(LIVE_OUT_DIR / f"raw_ch{ch_idx}_CAM1.jpg"), img_l)

            # Ljusstyrkestatistik för CAM1
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape) == 3 else img_l
            p99 = np.percentile(gray_l, 99)
            print(f"  Kanal {ch_idx} CAM1: Min={gray_l.min():3d}, Max={gray_l.max():3d}, Medel={gray_l.mean():5.1f}, 99e perc={p99:5.1f} -> {p_l.name}")

        if pair.right and pair.right.image is not None:
            img_r = pair.right.image
            p_r = target_dir / f"raw_ch{ch_idx}_CAM0.jpg"
            cv2.imwrite(str(p_r), img_r)
            cv2.imwrite(str(LIVE_OUT_DIR / f"raw_ch{ch_idx}_CAM0.jpg"), img_r)

            # Ljusstyrkestatistik för CAM0
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape) == 3 else img_r
            p99_r = np.percentile(gray_r, 99)
            print(f"  Kanal {ch_idx} CAM0: Min={gray_r.min():3d}, Max={gray_r.max():3d}, Medel={gray_r.mean():5.1f}, 99e perc={p99_r:5.1f} -> {p_r.name}")

    print("\n" + "=" * 70)
    print(f"  KLART! 8 bilder sparade i: {target_dir}")
    print("  Nu kan du köra 'git commit' och 'git push' från terminalen!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
