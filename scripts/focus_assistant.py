"""
Fokusassistent för NoIR-kamerorna (CAM0 och CAM1).

Körs i terminalen på Raspberry Pi:
    python3 scripts/focus_assistant.py

Hjälper användaren att ställa in perfekt manuellt fokus på mätbädden:
- Tänder IR-lamporna så att fokuseringen sker under exakt 850nm våglängd.
- Mäter bildskärpa (Laplacian-varians) i realtid 10-15 ggr/sekund.
- Visar en levande mätare och registrerar högsta uppnådda skärpa (Peak).
- Avslutas med Ctrl+C när båda linserna är justerade till sitt max.
"""

import sys
import time
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.hardware.vision_rig import VisionRig


def calculate_sharpness(img_bgr: np.ndarray) -> float:
    """
    Beräknar skärpa med Laplacian-varians i bildens centrala 50% (ROI).
    Högre värde = skarpare konturer och kanter.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    h, w = gray.shape
    # Fokusera på mitten där kalibrering/paket ligger (undvik kanter/hörn)
    roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def make_bar(value: float, max_val: float = 300.0, bar_len: int = 15) -> str:
    """Skapar en visuell stapel för terminalen."""
    ratio = min(1.0, max(0.0, value / max_val))
    filled = int(ratio * bar_len)
    return "█" * filled + "░" * (bar_len - filled)


def main():
    print("=" * 70)
    print("  FOKUSASSISTENT: NoIR-kameror (CAM0 & CAM1)")
    print("=" * 70)
    print("Tips för bästa resultat:")
    print(" 1. Lägg ett papper med skarp text, streckkod eller schackmönster")
    print("    på mätbädden rakt under båda kamerorna.")
    print(" 2. Tänder 850nm IR-belysningen så att linsen fokuseras för IR-ljus.")
    print(" 3. Vrid försiktigt på linsringen på den kamera du justerar.")
    print("    När värdet når sitt absoluta max (Peak) är linsen i perfekt fokus.")
    print(" 4. Avsluta med Ctrl+C när du är nöjd.\n")

    print("Ansluter till kameror och belysning...")
    with VisionRig(verbose=False) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till VisionRig (kameror eller LED).")
            return

        # Sätt en stabil exponering för fokusering och tänd belysningen
        rig.set_exposure(exposure_us=25000, gain=3.0)
        rig.leds.all_on()

        print("[OK] IR-belysning tänd. Mätning aktiv! Vrid på linserna nu:\n")

        peak_0 = 0.0
        peak_1 = 0.0

        try:
            while True:
                # Hämta bildpar (left=CAM1, right=CAM0)
                pair = rig.cameras.capture_both()
                if not pair.is_valid():
                    time.sleep(0.05)
                    continue

                score_1 = calculate_sharpness(pair.left.image)   # CAM1 (Vänster)
                score_0 = calculate_sharpness(pair.right.image)  # CAM0 (Höger)

                peak_0 = max(peak_0, score_0)
                peak_1 = max(peak_1, score_1)

                bar_0 = make_bar(score_0, max_val=max(200.0, peak_0 * 1.1))
                bar_1 = make_bar(score_1, max_val=max(200.0, peak_1 * 1.1))

                # Uppdatera samma rad i terminalen
                sys.stdout.write(
                    f"\rCAM0 (Höger) : [{bar_0}] {score_0:6.1f} (Peak: {peak_0:6.1f})  |  "
                    f"CAM1 (Vänster): [{bar_1}] {score_1:6.1f} (Peak: {peak_1:6.1f})  "
                )
                sys.stdout.flush()
                time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n\n" + "-" * 70)
            print("Fokusering avslutad!")
            print(f"  Slutgiltig CAM0 Peak: {peak_0:.1f}")
            print(f"  Slutgiltig CAM1 Peak: {peak_1:.1f}")
            print("-" * 70)
        finally:
            rig.leds.all_off()
            print("IR-lampor släckta.")


if __name__ == "__main__":
    main()
