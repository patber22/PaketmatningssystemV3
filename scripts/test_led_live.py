"""
Liveskript för att testa LEDController-fasaden (src/hardware/led_controller.py).
Testar både sekventiell styrning (Kanal 1 -> 2 -> 3 -> 4) och gruppstyrning (all_on/all_off).

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

from src.hardware.led_controller import LEDController


def main():
    print("=" * 60)
    print("  Live-test: LEDController (src/hardware/led_controller.py)")
    print("=" * 60)

    with LEDController(verbose=True) as leds:
        if not leds.is_connected:
            print("\n[Varning] Inte alla LED-kanaler kunde ansluta till hårdvaran.")
            print("Kontrollera att koden körs på Raspberry Pi med nödvändiga behörigheter.")
            return

        print("\nAlla 4 LED-kanaler är anslutna!")

        # Test 1: Sekventiell styrning (en och en)
        print("\n--- Test 1: Sekventiell styrning (Kanal 1 -> 2 -> 3 -> 4) ---")
        for ch in [1, 2, 3, 4]:
            print(f"  --> Tänder LED Kanal {ch}...")
            leds.on(ch)
            time.sleep(0.5)

            print(f"  --> Släcker LED Kanal {ch}...")
            leds.off(ch)
            time.sleep(0.2)

        time.sleep(0.5)

        # Test 2: Gruppstyrning (alla samtidigt - stereo-läge)
        print("\n--- Test 2: Gruppstyrning (Alla lampor tänds samtidigt) ---")
        print("  --> leds.all_on()...")
        leds.all_on()
        time.sleep(0.8)

        print("  --> leds.all_off()...")
        leds.all_off()
        time.sleep(0.2)

    print("\n" + "=" * 60)
    print("  Klart! Både enskild och samlad styrning testad felfritt.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
