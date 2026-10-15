"""
Kort liveskript för att testa ST VL53L5CX ToF-sensorn på Raspberry Pi.
Körs med: python3 scripts/test_tof_live.py
"""

import time
import sys
from pathlib import Path
import numpy as np

# Säkerställ att projektroten finns i sys.path oavsett varifrån skriptet körs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.tof import ToFSensor


def main():
    print("\n=== Live-test: ST VL53L5CX ToF-sensor ===")

    with ToFSensor(verbose=True) as tof:
        if not tof.is_connected:
            print("\nKunde inte ansluta till ToF-sensorn. Kontrollera I2C-anslutningen (i2cdetect -y 1)!")
            return

        print("\nSensorn är ansluten och mättråden rullar!")
        print("Väntar 0.5s på första mätningarna...\n")
        time.sleep(0.5)

        # 1. Enkel läsning med 2-stegsfiltrering (centerzon, 5 samples)
        print("--- Test 1: Läsning av center-zon (5 samples median) ---")
        reading = tof.read(samples=5, sample_method="median", zone="center", zone_method="median")
        print(f"Resultat: {reading.distance_mm:.1f} mm (is_valid={reading.is_valid()}, Zoner={reading.valid_zones}/4)")

        # 2. Hämta hela den tidsfiltrerade 8x8-matrisen
        print("\n--- Test 2: Hela 8x8-matrisen (mm) ---")
        matrix = tof.read_matrix(samples=5, sample_method="median")
        if matrix is not None:
            for row in matrix:
                print("  ".join(f"{val:6.0f}" if not np.isnan(val) else "   NaN" for val in row))
        else:
            print("Kunde inte hämta 8x8-matris.")

        # 3. Kontinuerlig avläsning under 5 sekunder
        print("\n--- Test 3: Kontinuerlig avläsning under 5 sekunder (rör handen över sensorn) ---")
        for i in range(10):
            r = tof.read(samples=1, zone="center")
            if r.is_valid():
                print(f"[{i+1}/10] Avstånd: {r.distance_mm:6.1f} mm | Zoner: {r.valid_zones}/4")
            else:
                print(f"[{i+1}/10] Ogiltig mätning (is_valid=False)")
            time.sleep(0.5)

    print("\nKlart! ToF-sensorn frånkopplad.\n")


if __name__ == "__main__":
    main()
