"""
Liveskript för att testa den nya LED-hårdvaruklassen (src/hardware/led.py).
Tänder och släcker lamporna i ordning (Kanal 1 -> 2 -> 3 -> 4) en och en.

Körs på pajen med:
    python3 scripts/test_led_live.py
"""

import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.led import LED


def main():
    print("=" * 60)
    print("  Live-test: LED-drivrutin (Kanal 1 -> 2 -> 3 -> 4)")
    print("=" * 60)

    channels = [1, 2, 3, 4]

    for ch in channels:
        print(f"\n[Test] Förbereder LED Kanal {ch}...")
        with LED(channel=ch, verbose=True) as led:
            if not led.is_connected:
                print(f"[Varning] Kunde inte ansluta till LED {ch}.")
                continue

            print(f"       Tänder {led.description}...")
            led.on()
            time.sleep(0.6)

            print(f"       Släcker {led.description}...")
            led.off()
            time.sleep(0.2)

    print("\n" + "=" * 60)
    print("  Klart! Alla lampor har blinkat och är släckta.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
