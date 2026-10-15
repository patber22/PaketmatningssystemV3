"""
Kort liveskript för att testa Honeywell MS5145-scannern på Raspberry Pi.
Körs med: python scripts/test_scanner_live.py
"""

import time
import sys
from pathlib import Path

# Säkerställ att projektroten finns i sys.path oavsett varifrån skriptet körs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.scanner import BarcodeScanner


def on_scan(reading):
    print(f"\n⚡ [CALLBACK TRIGGAD] Bakgrundstråden fångade koden direkt: '{reading.code}' (is_valid={reading.is_valid()})!")


def main():
    print("\n=== Live-test: Honeywell MS5145 Scanner ===")

    with BarcodeScanner(on_scan_callback=on_scan) as scanner:
        if not scanner.is_connected:
            print("Kunde inte ansluta till scannern. Kontrollera att den är i USB!")
            return

        print("Scannern är ansluten och redo!")
        print("Scanna en streckkod nu (väntar max 10 sekunder)...")

        # Vänta upp till 10 sekunder men avbryt direkt när koden registreras
        scanned = False
        for _ in range(20):
            reading = scanner.peek()
            if reading.is_valid():
                print(f"\n[OK] peek() hittade: '{reading.code}' (is_valid={reading.is_valid()})")
                scanned = True
                break
            time.sleep(0.5)

        if not scanned:
            print("\nTiden gick ut utan att någon kod scannades.")
            return

        print("\n--- Testar consume() ---")
        consumed = scanner.consume()
        print(f"consume() returnerade: '{consumed.code}' (is_valid={consumed.is_valid()})")

        second_try = scanner.consume()
        print(f"Andra försöket gav: '{second_try.code}' (is_valid={second_try.is_valid()}, förväntat: False)")

    print("\nKlart! Scannern frånkopplad.\n")


if __name__ == "__main__":
    main()
