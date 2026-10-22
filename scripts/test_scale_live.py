"""
Livetest för Paketmätningssystemets våg (src/hardware/scale.py).

Ansluter till ESP32-C3 / HX711 via USB-serieport och strömmar rådata,
stabilitetsindikator och eventuell beräknad vikt i realtid.

Körs på pajen med:
    python3 scripts/test_scale_live.py
"""

import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.scale import Scale


def main():
    print("=" * 65)
    print("  Live-test: Scale Driver (src/hardware/scale.py)")
    print("=" * 65)

    print("\nSöker efter ESP32-C3 vågmodul via USB...")
    with Scale(verbose=True) as scale:
        if not scale.is_connected:
            print("\n[FEL] Kunde inte ansluta till någon våg!")
            print("Kontrollera att:")
            print("  1. ESP32-C3 är inkopplad med USB-kabel till Raspberry Pi.")
            print("  2. Användaren har rättigheter till serieporten (t.ex. dialout-gruppen).")
            print("  3. Kör 'ls -l /dev/ttyACM* /dev/ttyUSB*' i terminalen för att felsöka.")
            return

        print(f"\n[OK] Ansluten till våg på port: {scale.port}")
        if scale.is_calibrated:
            print(f"[OK] Kalibrering aktiv:")
            print(f"     Tare Offset : {scale.tare_offset_raw:.0f} counts")
            print(f"     Skalfaktor  : {scale.scale_factor:.3f} counts/g")
        else:
            print("[INFO] Vågen är ännu inte kalibrerad.")
            print("       Visar råa ADC-counts från lastcellerna.")
            print("       Kör 'python3 calibration/calibrate_scale.py' senare för att kalibrera.")

        print("\nStrömmar mätdata i realtid (tryck Ctrl+C för att avsluta)...")
        print("-" * 65)
        print(f"{'Tid':<10} | {'Råvärde (ADC)':<16} | {'Vikt':<14} | {'Status':<10} | {'Prover'}")
        print("-" * 65)

        try:
            # Vänta kort så bakgrundstråden hinner fylla bufferten
            time.sleep(0.5)

            while True:
                reading = scale.read(samples=5)
                now_str = time.strftime("%H:%M:%S")

                raw_str = f"{reading.raw_value:>10} cts"
                if scale.is_calibrated and reading.is_valid():
                    weight_str = f"{reading.weight_g:>8.1f} g"
                elif scale.is_calibrated and reading.weight_g < 0:
                    weight_str = f"{reading.weight_g:>8.1f} g (!)"
                else:
                    weight_str = "Okalibrerad"

                status_str = "[STABIL]" if reading.is_stable else "RÖR SIG "
                samples_str = f"{reading.samples} st"

                print(f"{now_str:<10} | {raw_str:<16} | {weight_str:<14} | {status_str:<10} | {samples_str}")
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n\nAvslutar livetest på användarens begäran...")

    print("Vågen är frånkopplad. Testet klart!")


if __name__ == "__main__":
    main()
