"""
Kalibreringsverktyg för Paketmätningssystemets våg (src/hardware/scale.py).

Guidad kalibrering i terminalen:
1. Startar vågen okalibrerad (via dummy-sökväg).
2. Samlar in 25 råvärden för tom våg (tara).
3. Samlar in 25 råvärden med en känd referensvikt.
4. Beräknar skalfaktorn (delta_counts / ref_weight_g).
5. Sparar kalibreringen till config/calibration/data/scale.json.
6. Laddar den nya kalibreringen och live-verifierar vikten.

Körs med:
    python3 calibration/calibrate_scale.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List
import numpy as np

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.scale import Scale

TARGET_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "calibration",
        "data",
        "scale.json",
    )
)
DUMMY_CONFIG_PATH = "/dummy/uncalibrated.json"


def collect_raw_samples(scale: Scale, count: int = 25, timeout_s: float = 4.0) -> List[int]:
    """Samlar in 'count' unika råa ADC-mätningar från lastcellerna."""
    collected: List[int] = []
    start_time = time.time()

    while len(collected) < count and (time.time() - start_time) < timeout_s:
        with scale._lock:
            if scale._latest_raw is not None and (not collected or collected[-1] != scale._latest_raw):
                collected.append(scale._latest_raw)
        
        # Visa progressindikator
        print(f"\r  Mäter... [{len(collected)}/{count} prover insamlade]", end="", flush=True)
        time.sleep(0.05)

    print()  # Ny rad efter progress
    return collected


def main():
    print("=" * 65)
    print("  Kalibreringsverktyg: Paketvåg (ESP32-C3 / HX711)")
    print("=" * 65)
    print(f"Målfil: {TARGET_CONFIG_PATH}\n")

    # 1. Starta vågen i ett rent, okalibrerat läge
    print("[1/5] Ansluter till vågmodulen...")
    scale = Scale(
        config_path=DUMMY_CONFIG_PATH,
        auto_connect=True,
        verbose=False,
    )

    if not scale.is_connected:
        print("\n[FEL] Kunde inte ansluta till vågen!")
        print("Kontrollera att ESP32-C3 är inkopplad och körbar.")
        return

    try:
        print(f"[OK] Ansluten på {scale.port}. Startar i okalibrerat läge.\n")
        time.sleep(0.5)

        # 2. Steg 1: Nollställning (Tara)
        print("-" * 65)
        print("STEG 1: Nollställning (Tom mätbädd)")
        print("-" * 65)
        print("Se till att vågen är helt tom och fri från vibrationer.")
        input("Tryck ENTER när mätbädden är tom för att mäta nollpunkten...")

        zero_samples = collect_raw_samples(scale, count=25)
        if len(zero_samples) < 5:
            print("\n[FEL] Kunde inte samla in tillräckligt med rådata för nollpunkten.")
            return

        tare_offset_raw = float(np.median(zero_samples))
        print(f"[OK] Nollpunkt fastställd till: {tare_offset_raw:.0f} counts\n")

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

        weight_samples = collect_raw_samples(scale, count=25)
        if len(weight_samples) < 5:
            print("\n[FEL] Kunde inte samla in tillräckligt med rådata med referensvikten.")
            return

        raw_with_weight = float(np.median(weight_samples))
        print(f"[OK] Råvärde med referensvikt: {raw_with_weight:.0f} counts\n")

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

        print("Laddade in nya parametrarna i Scale! Verifierar i 4 sekunder...")
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
