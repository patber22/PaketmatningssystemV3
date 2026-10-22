"""
Kalibreringsverktyg för Paketmätningssystemets våg (src/hardware/scale.py).

Använder Scale-klassens publika API:
1. Skapar en Scale-instans med en dummy-sökväg och buffer_size=30 (okalibrerat läge).
2. Tömmer vågen -> anropar read(samples=25) -> tar ut råa medianvärdet (tare_offset_raw).
3. Placerar referensvikt -> anropar read(samples=25) -> tar ut råa medianvärdet (raw_with_weight).
4. Beräknar skalfaktorn: (raw_with_weight - tare_offset_raw) / ref_weight_g.
5. Sparar kalibreringen till config/calibration/data/scale.json.
6. Laddar om den riktiga sökvägen via load_calibration() och live-verifierar.

Körs med:
    python3 config/calibration/calibrate_scale.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.scale import Scale

TARGET_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "scale.json",
    )
)
DUMMY_CONFIG_PATH = "/dummy/uncalibrated.json"


def main():
    print("=" * 65)
    print("  Kalibreringsverktyg: Paketvåg (ESP32-C3 / HX711)")
    print("=" * 65)
    print(f"Målfil: {TARGET_CONFIG_PATH}\n")

    # 1. Starta vågen i okalibrerat läge med plats för 30 prover i bufferten
    print("[1/5] Ansluter till vågmodulen...")
    scale = Scale(
        config_path=DUMMY_CONFIG_PATH,
        buffer_size=30,
        auto_connect=True,
        verbose=False,
    )

    if not scale.is_connected:
        print("\n[FEL] Kunde inte ansluta till vågen!")
        print("Kontrollera att ESP32-C3 är inkopplad och körbar.")
        return

    try:
        print(f"[OK] Ansluten på {scale.port}. Startad i okalibrerat läge.\n")

        # 2. Steg 1: Nollställning (Tom mätbädd)
        print("-" * 65)
        print("STEG 1: Nollställning (Tom mätbädd)")
        print("-" * 65)
        print("Se till att vågen är helt tom och fri från vibrationer.")
        input("Tryck ENTER när mätbädden är tom...")

        print("Samlar in 25 mätpunkter för nollpunkten...")
        # Vänta så bufferten hinner fyllas med 25 färska prover (~2.5s vid 10 Hz)
        time.sleep(2.6)
        zero_reading = scale.read(samples=25)

        if zero_reading.samples < 10:
            print(f"\n[FEL] Fick bara {zero_reading.samples} prover. Kontrollera serieporten.")
            return

        tare_offset_raw = float(zero_reading.raw_value)
        print(f"[OK] Nollpunkt fastställd till: {tare_offset_raw:.0f} counts (baserat på {zero_reading.samples} prover)\n")

        # 3. Steg 2: Referensvikt
        print("-" * 65)
        print("STEG 2: Mätning med känd referensvikt")
        print("-" * 65)
        print("Placera ett föremål med känd vikt på vågen (minst 500 g rekommenderas).")

        while True:
            try:
                ref_str = input("Ange referensviktens exakta vikt i gram (t.ex. 2000): ").strip()
                ref_weight_g = float(ref_str)
                if ref_weight_g <= 0:
                    print("Vikten måste vara större än 0 gram. Försök igen.")
                    continue
                break
            except ValueError:
                print("Ogiltigt tal. Ange vikten i siffror (t.ex. 1500 eller 2208.5).")

        input(f"Placera vikten ({ref_weight_g:.1f} g) på vågen och tryck ENTER när den ligger stilla...")

        print("Samlar in 25 mätpunkter med referensvikten...")
        # Vänta så bufferten spolas med 25 färska prover från vikten
        time.sleep(2.6)
        weight_reading = scale.read(samples=25)

        if weight_reading.samples < 10:
            print(f"\n[FEL] Fick bara {weight_reading.samples} prover med referensvikten.")
            return

        raw_with_weight = float(weight_reading.raw_value)
        print(f"[OK] Råvärde med referensvikt: {raw_with_weight:.0f} counts (baserat på {weight_reading.samples} prover)\n")

        # 4. Steg 3: Beräkning & Rimlighetskontroll
        print("-" * 65)
        print("STEG 3: Beräkning av kalibreringsparametrar")
        print("-" * 65)
        delta_raw = raw_with_weight - tare_offset_raw
        if delta_raw <= 0:
            print(f"[FEL] Råvärdet ökade inte när vikten lades på (delta: {delta_raw:.0f}).")
            print("Kontrollera att lastcellerna är rätt kopplade eller att vikten faktiskt lades på.")
            return

        scale_factor = delta_raw / ref_weight_g
        print(f"  Delta råvärde : {delta_raw:.0f} counts")
        print(f"  Referensvikt  : {ref_weight_g:.1f} g")
        print(f"  Skalfaktor    : {scale_factor:.4f} counts/g\n")

        # 5. Steg 4: Spara till JSON-fil
        print("-" * 65)
        print("STEG 4: Sparar till kalibreringsfil")
        print("-" * 65)
        cal_data = {
            "tare_offset_raw": tare_offset_raw,
            "scale_factor": scale_factor,
            "reference_weight_g": ref_weight_g,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(os.path.dirname(TARGET_CONFIG_PATH), exist_ok=True)
        with open(TARGET_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cal_data, f, indent=4)
        print(f"[OK] Kalibreringsfil sparad: {TARGET_CONFIG_PATH}\n")

        # 6. Steg 5: Ladda om och live-verifiera
        print("-" * 65)
        print("STEG 5: Live-verifiering med sparad kalibrering")
        print("-" * 65)
        loaded = scale.load_calibration(TARGET_CONFIG_PATH)
        if not loaded or not scale.is_calibrated:
            print("[VARNING] Kunde inte ladda om den sparade kalibreringen i Scale-objektet.")
            return

        print("Laddade in de nya parametrarna i Scale! Verifierar i 4 sekunder...")
        time.sleep(0.5)

        for _ in range(10):
            reading = scale.read(samples=5)
            status_str = "[STABIL]" if reading.is_stable else "RÖR SIG "
            print(
                f"  Vikt: {reading.weight_g:>8.1f} g | "
                f"Råvärde: {reading.raw_value:>10} cts | "
                f"Status: {status_str}"
            )
            time.sleep(0.3)

        print("\n" + "=" * 65)
        print("  KALIBRERINGEN ÄR SLUTFÖRD OCH VERIFIERAD!")
        print("=" * 65)

    finally:
        scale.disconnect()


if __name__ == "__main__":
    main()
